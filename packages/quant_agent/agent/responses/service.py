"""Trusted publication boundary for evidence-bound Agent answers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Final, cast

from quant_agent.agent.runtime import (
    AgentArtifactKind,
    AgentRunEventKind,
    AgentRunSnapshot,
    AgentRunState,
)
from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.config import RuntimeMode
from quant_agent.core.responses import ToolResponse
from quant_agent.core.time import ensure_aware

from .contracts import (
    AGENT_ANSWER_POLICY_VERSION,
    AGENT_ANSWER_SCHEMA_VERSION,
    AgentActionKind,
    AgentAllowedAction,
    AgentAnswer,
    AgentAnswerDraft,
    AgentAnswerStatus,
    AgentCounterEvidence,
    AgentFact,
    AgentInference,
    AgentInvalidationCondition,
    AgentIssueSeverity,
    AgentMetric,
    AgentRiskWarning,
    AgentToolIssueRef,
    EvidenceReference,
    stable_answer_hash,
)

INSUFFICIENT_EVIDENCE_SUMMARY: Final = "证据不足, 无法形成可验证的市场结论。"

_NUMERIC_LITERAL = re.compile(r"[+\-]?\d+(?:[.,]\d+)*%?")
_FORBIDDEN_EXPRESSIONS: Final[tuple[str, ...]] = (
    "稳赚",
    "确定上涨",
    "必买",
    "保证收益",
    "保本保收益",
    "绝对安全",
    "没有风险",
    "guaranteedprofit",
    "guaranteedreturn",
    "riskfree",
    "mustbuy",
    "certainrise",
)
_OUTCOME_KINDS: Final = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    }
)
_SUCCESS_KINDS: Final = frozenset(
    {AgentRunEventKind.TOOL_CALL_SUCCEEDED, AgentRunEventKind.TOOL_CALL_REPLAYED}
)


class AgentAnswerError(RuntimeError):
    """Base failure at the trusted answer publication boundary."""


class AgentAnswerPolicyViolation(AgentAnswerError):
    """Raised when draft language or structure violates a deterministic policy."""


class AgentEvidenceInvalid(AgentAnswerError):
    """Raised internally when a purported source cannot be reproduced exactly."""


def _response_document(response: ToolResponse[object]) -> dict[str, object]:
    document = response.model_dump(mode="json")
    if type(document) is not dict:
        raise AgentEvidenceInvalid("tool response did not serialize to an object")
    return cast(dict[str, object], document)


def tool_response_hash(response: ToolResponse[object]) -> str:
    """Compute the exact response hash used by the Agent runtime."""

    encoded = json.dumps(
        _response_document(response),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return stable_answer_hash(json.loads(encoded))


def _pointer_value(document: object, pointer: str) -> object:
    current = document
    for encoded_part in pointer.split("/")[1:]:
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if type(current) is dict:
            mapping = cast(dict[str, object], current)
            if part not in mapping:
                raise AgentEvidenceInvalid("evidence JSON pointer does not exist")
            current = mapping[part]
        elif type(current) is list:
            if not part.isascii() or not part.isdigit() or (len(part) > 1 and part[0] == "0"):
                raise AgentEvidenceInvalid("evidence array index is not canonical")
            index = int(part)
            sequence = cast(list[object], current)
            if index >= len(sequence):
                raise AgentEvidenceInvalid("evidence array index is out of bounds")
            current = sequence[index]
        else:
            raise AgentEvidenceInvalid("evidence JSON pointer traverses a scalar")
    return current


def build_evidence_reference(
    *,
    tool_name: str,
    response: ToolResponse[object],
    json_pointer: str,
) -> EvidenceReference:
    """Create a reference from an actual response for later runtime verification."""

    document = _response_document(response)
    value = _pointer_value(document, json_pointer)
    data_version = response.provenance.data_version
    if type(data_version) is not str:
        raise AgentEvidenceInvalid("evidence response has no immutable data version")
    return EvidenceReference(
        request_id=response.request_id,
        tool_name=tool_name,
        response_hash=tool_response_hash(response),
        json_pointer=json_pointer,
        value_hash=stable_answer_hash(value),
        data_version=data_version,
    )


def _normalized_policy_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _draft_texts(draft: AgentAnswerDraft) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = [("summary", draft.summary)]
    for collection_name in (
        "facts",
        "inferences",
        "counter_evidence",
        "risks",
        "invalidations",
    ):
        for item in getattr(draft, collection_name):
            values.append((f"{collection_name}.statement", item.statement))
    values.extend(("actions.rationale", item.rationale) for item in draft.actions)
    return tuple(values)


def _validate_language(draft: AgentAnswerDraft) -> None:
    for field, text in _draft_texts(draft):
        normalized = _normalized_policy_text(text)
        if any(expression in normalized for expression in _FORBIDDEN_EXPRESSIONS):
            raise AgentAnswerPolicyViolation(f"{field} contains a prohibited certainty claim")
        if _NUMERIC_LITERAL.search(unicodedata.normalize("NFKC", text)) is not None:
            raise AgentAnswerPolicyViolation(
                f"{field} contains a numeric literal; publish numbers through AgentMetric"
            )


def _all_metrics(draft: AgentAnswerDraft) -> tuple[AgentMetric, ...]:
    metrics: list[AgentMetric] = []
    for name in ("facts", "inferences", "counter_evidence", "risks", "invalidations"):
        for item in getattr(draft, name):
            metrics.extend(item.metrics)
    return tuple(metrics)


def _all_references(draft: AgentAnswerDraft) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = []
    for item in draft.facts:
        references.extend(item.evidence)
    for name in ("counter_evidence", "risks", "invalidations"):
        for item in getattr(draft, name):
            references.extend(item.evidence)
    references.extend(metric.evidence for metric in _all_metrics(draft))
    return tuple(references)


def _validate_unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise AgentAnswerPolicyViolation(f"{label} must be unique")


def _validate_structure(draft: AgentAnswerDraft) -> None:
    fact_ids = tuple(item.fact_id for item in draft.facts)
    inference_ids = tuple(item.inference_id for item in draft.inferences)
    _validate_unique(fact_ids, "fact identifiers")
    _validate_unique(inference_ids, "inference identifiers")
    _validate_unique(
        tuple(item.counter_id for item in draft.counter_evidence), "counter identifiers"
    )
    _validate_unique(tuple(item.risk_id for item in draft.risks), "risk identifiers")
    _validate_unique(
        tuple(item.condition_id for item in draft.invalidations), "invalidation identifiers"
    )
    _validate_unique(tuple(item.metric_id for item in _all_metrics(draft)), "metric identifiers")

    known_facts = set(fact_ids)
    known_inferences = set(inference_ids)
    for inference in draft.inferences:
        if not set(inference.based_on_fact_ids) <= known_facts:
            raise AgentAnswerPolicyViolation("an inference references an unknown fact")
    challenged: set[str] = set()
    for counter in draft.counter_evidence:
        if not set(counter.challenges_inference_ids) <= known_inferences:
            raise AgentAnswerPolicyViolation("counter-evidence references an unknown inference")
        challenged.update(counter.challenges_inference_ids)
    invalidated: set[str] = set()
    for condition in draft.invalidations:
        if not set(condition.invalidates_inference_ids) <= known_inferences:
            raise AgentAnswerPolicyViolation("an invalidation references an unknown inference")
        invalidated.update(condition.invalidates_inference_ids)
    if draft.inferences:
        if not draft.counter_evidence or challenged != known_inferences:
            raise AgentAnswerPolicyViolation("every inference requires counter-evidence")
        if not draft.risks:
            raise AgentAnswerPolicyViolation("market inferences require explicit risks")
        if not draft.invalidations or invalidated != known_inferences:
            raise AgentAnswerPolicyViolation("every inference requires an invalidation condition")
    action_kinds = tuple(item.kind for item in draft.actions)
    if AgentActionKind.NO_ACTION in action_kinds and action_kinds != (AgentActionKind.NO_ACTION,):
        raise AgentAnswerPolicyViolation("NO_ACTION cannot be combined with another action")


def _validate_actions(draft: AgentAnswerDraft, run: AgentRunSnapshot) -> None:
    kinds = {item.kind for item in draft.actions}
    if AgentActionKind.REVIEW_ORDER_DRAFT in kinds and not any(
        artifact.kind is AgentArtifactKind.ORDER_DRAFT for artifact in run.artifacts
    ):
        raise AgentAnswerPolicyViolation("REVIEW_ORDER_DRAFT requires a verified draft artifact")
    if AgentActionKind.REQUEST_HUMAN_APPROVAL in kinds and not (
        run.runtime_mode is RuntimeMode.LIVE_ASSISTED
        and run.state is AgentRunState.PENDING_APPROVAL
    ):
        raise AgentAnswerPolicyViolation(
            "REQUEST_HUMAN_APPROVAL requires a live-assisted pending-approval run"
        )


def _metric_display(value: object) -> str:
    if type(value) is bool or value is None:
        raise AgentEvidenceInvalid("metric evidence must resolve to a numeric scalar")
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise AgentEvidenceInvalid("metric evidence must be finite")
        return json.dumps(value, allow_nan=False)
    if type(value) is str:
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise AgentEvidenceInvalid("metric string is not numeric") from error
        if not parsed.is_finite():
            raise AgentEvidenceInvalid("metric string must be finite")
        return value
    raise AgentEvidenceInvalid("metric evidence must resolve to a numeric JSON scalar")


class AgentAnswerPublisher:
    """Verify model-controlled claims against one immutable run and decision."""

    def publish(
        self,
        *,
        decision: DecisionSnapshot,
        run: AgentRunSnapshot,
        draft: AgentAnswerDraft,
        responses: Mapping[str, ToolResponse[object]],
        generated_at: datetime,
    ) -> AgentAnswer:
        """Publish a supported answer, or fail closed to insufficient evidence."""

        if type(decision) is not DecisionSnapshot:
            raise TypeError("decision must be an exact DecisionSnapshot")
        if type(run) is not AgentRunSnapshot:
            raise TypeError("run must be an exact AgentRunSnapshot")
        if type(draft) is not AgentAnswerDraft:
            raise TypeError("draft must be an exact AgentAnswerDraft")
        try:
            decision = DecisionSnapshot.from_json(decision.to_json())
            run = AgentRunSnapshot.from_json(run.to_json())
            draft = AgentAnswerDraft.model_validate_json(
                draft.model_dump_json(warnings="error"), strict=True
            )
        except (TypeError, ValueError) as error:
            raise AgentAnswerPolicyViolation(
                "answer inputs failed strict integrity validation"
            ) from error
        instant = ensure_aware(generated_at).astimezone(UTC)
        if (
            run.decision_id != decision.decision_id
            or run.decision_snapshot_hash != decision.content_hash
            or run.runtime_mode is not decision.mode
        ):
            raise AgentAnswerPolicyViolation("run does not bind the supplied decision snapshot")
        if instant < decision.as_of:
            raise AgentAnswerPolicyViolation("answer cannot be generated before the data cutoff")

        _validate_language(draft)
        _validate_structure(draft)
        _validate_actions(draft, run)
        issues = self._tool_issues(run)
        try:
            verified_versions = self._verify_evidence(
                decision=decision,
                run=run,
                draft=draft,
                responses=responses,
            )
        except (AgentEvidenceInvalid, AttributeError, TypeError, ValueError):
            return self._insufficient(decision, run, instant, issues)
        if not draft.facts or not _all_references(draft):
            return self._insufficient(decision, run, instant, issues)
        return self._answer(
            decision=decision,
            run=run,
            generated_at=instant,
            status=AgentAnswerStatus.SUPPORTED,
            summary=draft.summary,
            facts=draft.facts,
            inferences=draft.inferences,
            counter_evidence=draft.counter_evidence,
            risks=draft.risks,
            invalidations=draft.invalidations,
            actions=draft.actions,
            issues=issues,
            data_versions=verified_versions,
        )

    @staticmethod
    def _verify_evidence(
        *,
        decision: DecisionSnapshot,
        run: AgentRunSnapshot,
        draft: AgentAnswerDraft,
        responses: Mapping[str, ToolResponse[object]],
    ) -> tuple[str, ...]:
        successful = {
            (event.request_id, event.tool_name, event.response_hash)
            for event in run.events
            if event.kind in _SUCCESS_KINDS
        }
        resolved: dict[EvidenceReference, object] = {}
        for reference in _all_references(draft):
            if (
                reference.request_id,
                reference.tool_name,
                reference.response_hash,
            ) not in successful:
                raise AgentEvidenceInvalid("evidence is absent from successful runtime outcomes")
            response = responses.get(reference.request_id)
            if not isinstance(response, ToolResponse) or not response.ok or response.data is None:
                raise AgentEvidenceInvalid("evidence response is missing, failed, or empty")
            if (
                response.request_id != reference.request_id
                or response.decision_id != decision.decision_id
                or ensure_aware(response.as_of).astimezone(UTC) != decision.as_of
                or response.provenance.data_version != decision.data_version
                or reference.data_version != decision.data_version
                or tool_response_hash(response) != reference.response_hash
            ):
                raise AgentEvidenceInvalid("evidence response identity does not match the run")
            value = _pointer_value(
                _response_document(response), reference.json_pointer
            )
            if stable_answer_hash(value) != reference.value_hash:
                raise AgentEvidenceInvalid("evidence value hash does not match its JSON path")
            resolved[reference] = value
        for metric in _all_metrics(draft):
            if metric.value != _metric_display(resolved[metric.evidence]):
                raise AgentEvidenceInvalid("metric display value does not match tool evidence")
        return tuple(sorted({reference.data_version for reference in resolved}))

    @staticmethod
    def _tool_issues(run: AgentRunSnapshot) -> tuple[AgentToolIssueRef, ...]:
        issues: list[AgentToolIssueRef] = []
        for event in run.events:
            if event.kind not in _OUTCOME_KINDS:
                continue
            if event.request_id is None or event.tool_name is None:
                raise AgentAnswerPolicyViolation("runtime outcome omitted tool identity")
            issues.extend(
                AgentToolIssueRef(
                    severity=AgentIssueSeverity.WARNING,
                    request_id=event.request_id,
                    tool_name=event.tool_name,
                    code=code,
                    response_hash=event.response_hash,
                )
                for code in event.warning_codes
            )
            issues.extend(
                AgentToolIssueRef(
                    severity=AgentIssueSeverity.ERROR,
                    request_id=event.request_id,
                    tool_name=event.tool_name,
                    code=code,
                    response_hash=event.response_hash,
                )
                for code in event.error_codes
            )
        return tuple(dict.fromkeys(issues))

    def _insufficient(
        self,
        decision: DecisionSnapshot,
        run: AgentRunSnapshot,
        generated_at: datetime,
        issues: tuple[AgentToolIssueRef, ...],
    ) -> AgentAnswer:
        return self._answer(
            decision=decision,
            run=run,
            generated_at=generated_at,
            status=AgentAnswerStatus.INSUFFICIENT_EVIDENCE,
            summary=INSUFFICIENT_EVIDENCE_SUMMARY,
            facts=(),
            inferences=(),
            counter_evidence=(),
            risks=(),
            invalidations=(),
            actions=(
                AgentAllowedAction(
                    kind=AgentActionKind.NO_ACTION,
                    rationale="等待可验证的数据与工具证据后再评估。",
                ),
            ),
            issues=issues,
            data_versions=(),
        )

    @staticmethod
    def _answer(
        *,
        decision: DecisionSnapshot,
        run: AgentRunSnapshot,
        generated_at: datetime,
        status: AgentAnswerStatus,
        summary: str,
        facts: tuple[AgentFact, ...],
        inferences: tuple[AgentInference, ...],
        counter_evidence: tuple[AgentCounterEvidence, ...],
        risks: tuple[AgentRiskWarning, ...],
        invalidations: tuple[AgentInvalidationCondition, ...],
        actions: tuple[AgentAllowedAction, ...],
        issues: tuple[AgentToolIssueRef, ...],
        data_versions: tuple[str, ...],
    ) -> AgentAnswer:
        values = {
            "schema_version": AGENT_ANSWER_SCHEMA_VERSION,
            "policy_version": AGENT_ANSWER_POLICY_VERSION,
            "answer_id": f"answer:{run.run_id}:{run.revision}",
            "run_id": run.run_id,
            "decision_id": decision.decision_id,
            "decision_snapshot_hash": decision.content_hash,
            "runtime_mode": run.runtime_mode,
            "goal": run.goal,
            "run_state": run.state,
            "status": status,
            "as_of": decision.as_of,
            "generated_at": generated_at,
            "summary": summary,
            "facts": facts,
            "inferences": inferences,
            "counter_evidence": counter_evidence,
            "risks": risks,
            "invalidations": invalidations,
            "actions": actions,
            "tool_issues": issues,
            "data_versions": data_versions,
        }
        return AgentAnswer.model_validate(
            {**values, "answer_hash": stable_answer_hash(values)}, strict=True
        )


__all__ = [
    "INSUFFICIENT_EVIDENCE_SUMMARY",
    "AgentAnswerError",
    "AgentAnswerPolicyViolation",
    "AgentAnswerPublisher",
    "AgentEvidenceInvalid",
    "build_evidence_reference",
    "tool_response_hash",
]
