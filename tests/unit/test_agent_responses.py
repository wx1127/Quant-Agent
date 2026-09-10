"""Tests for evidence-bound, fail-closed Agent answer publication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from quant_agent.agent.responses import (
    INSUFFICIENT_EVIDENCE_SUMMARY,
    AgentActionKind,
    AgentAllowedAction,
    AgentAnswer,
    AgentAnswerDraft,
    AgentAnswerPolicyViolation,
    AgentAnswerPublisher,
    AgentAnswerStatus,
    AgentCounterEvidence,
    AgentEvidenceInvalid,
    AgentFact,
    AgentInference,
    AgentInvalidationCondition,
    AgentMetric,
    AgentRiskWarning,
    EvidenceReference,
    build_evidence_reference,
    stable_answer_hash,
    tool_response_hash,
)
from quant_agent.agent.runtime import (
    AgentRunGoal,
    AgentRuntime,
    InMemoryAgentRunRepository,
)
from quant_agent.agent.snapshots import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools import ToolCallContext, ToolDescriptor
from quant_agent.agent.tools.research import MarketSnapshotOutput
from quant_agent.config import RuntimeMode
from quant_agent.core import Provenance, ToolResponse
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import ToolIssue

AS_OF = datetime(2026, 9, 8, 7, 0, tzinfo=UTC)
NOW = AS_OF + timedelta(minutes=1)
DECISION_ID = "dec_answer_0123456789abcdef"
REQUEST_ID = "market_snapshot_request"


def _decision() -> DecisionSnapshot:
    reference = StrategySnapshotRef(
        strategy_name="etf-rotation",
        strategy_version="etf-rotation-v1",
        config_hash="1" * 64,
        parameter_version="params-v1",
        parameter_hash="2" * 64,
        registered_at=AS_OF - timedelta(days=1),
    )
    return DecisionSnapshot.build(
        decision_id=DECISION_ID,
        mode=RuntimeMode.RESEARCH,
        market="CN_A",
        as_of=AS_OF,
        data_version="market_20260908_eod_v1",
        data_content_hash="3" * 64,
        strategy_refs=(reference,),
        risk_policy_version="portfolio-risk-v1",
        risk_policy_hash="4" * 64,
        account_id="paper-account-1",
        account_snapshot_id="paper-account-1:20260908T070000",
        account_snapshot_hash="5" * 64,
        code_commit="6" * 40,
        code_artifact_hash="7" * 64,
        agent_version="quant-agent-harness-v1",
        model_version="model-release-v1",
    )


def _response(
    decision: DecisionSnapshot,
    request_id: str,
) -> ToolResponse[MarketSnapshotOutput]:
    data = MarketSnapshotOutput(
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data_version=decision.data_version,
        market="CN_A",
        data_content_hash=decision.data_content_hash,
        created_at=decision.as_of.isoformat(),
        total_file_count=3,
        returned_file_count=0,
        total_row_count=128,
        files_truncated=True,
        files=(),
    )
    return ToolResponse[MarketSnapshotOutput](
        ok=True,
        request_id=request_id,
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data=data,
        warnings=(
            ToolIssue(code=ErrorCode.DATA_UNAVAILABLE, message="some files were omitted"),
        ),
        provenance=Provenance(
            service="agent-answer-test",
            version="1.0.0",
            data_version=decision.data_version,
        ),
    )


@dataclass
class _Registry:
    decision: DecisionSnapshot

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.RESEARCH

    def catalog(self, _context: ToolCallContext) -> tuple[ToolDescriptor, ...]:
        return ()

    def invoke(
        self,
        tool_name: str,
        _raw_arguments: str | dict[str, object],
        context: ToolCallContext,
    ) -> object:
        if tool_name != "get_market_snapshot":
            raise AssertionError("unexpected tool")
        return _response(self.decision, context.request_id)


@dataclass(frozen=True)
class _PublishedInputs:
    decision: DecisionSnapshot
    run: object
    response: ToolResponse[object]
    reference: EvidenceReference


def _inputs() -> _PublishedInputs:
    decision = _decision()
    repository = InMemoryAgentRunRepository()
    runtime = AgentRuntime(
        registry=_Registry(decision),
        repository=repository,
        clock=lambda: NOW,
    )
    started = runtime.start_run(
        decision_snapshot=decision,
        goal=AgentRunGoal.MARKET_RESEARCH,
        run_id="run_agent_answer_test",
    )
    response = runtime.invoke_tool(
        started.run_id,
        "get_market_snapshot",
        {"max_files": 50},
        request_id=REQUEST_ID,
    )
    reference = build_evidence_reference(
        tool_name="get_market_snapshot",
        response=response,
        json_pointer="/data/total_row_count",
    )
    return _PublishedInputs(
        decision=decision,
        run=runtime.get_run(started.run_id),
        response=response,
        reference=reference,
    )


def _draft(reference: EvidenceReference) -> AgentAnswerDraft:
    metric = AgentMetric(
        metric_id="market_rows",
        label="已锁定市场数据行数",
        value="128",
        unit="rows",
        evidence=reference,
    )
    return AgentAnswerDraft(
        summary="现有证据支持继续研究, 但结论仍需审慎解释。",
        facts=(
            AgentFact(
                fact_id="market_snapshot_fact",
                statement="市场快照已锁定并通过运行时留痕。",
                evidence=(reference,),
                metrics=(metric,),
            ),
        ),
        inferences=(
            AgentInference(
                inference_id="research_inference",
                statement="当前数据足以进入后续研究, 不代表未来收益。",
                based_on_fact_ids=("market_snapshot_fact",),
            ),
        ),
        counter_evidence=(
            AgentCounterEvidence(
                counter_id="truncation_counter",
                statement="工具仅返回摘要, 部分文件明细未展开。",
                challenges_inference_ids=("research_inference",),
                evidence=(reference,),
            ),
        ),
        risks=(
            AgentRiskWarning(
                risk_id="coverage_risk",
                statement="摘要覆盖范围可能限制结论外推。",
            ),
        ),
        invalidations=(
            AgentInvalidationCondition(
                condition_id="data_drift",
                statement="若数据版本变化, 当前推断失效并需重新运行。",
                invalidates_inference_ids=("research_inference",),
            ),
        ),
        actions=(
            AgentAllowedAction(
                kind=AgentActionKind.CONTINUE_RESEARCH,
                rationale="先补充数据质量与候选证据。",
            ),
        ),
    )


def _publish(inputs: _PublishedInputs, draft: AgentAnswerDraft | None = None) -> AgentAnswer:
    from quant_agent.agent.runtime import AgentRunSnapshot

    assert isinstance(inputs.run, AgentRunSnapshot)
    return AgentAnswerPublisher().publish(
        decision=inputs.decision,
        run=inputs.run,
        draft=draft or _draft(inputs.reference),
        responses={REQUEST_ID: inputs.response},
        generated_at=NOW,
    )


def test_supported_answer_binds_cutoff_evidence_numbers_and_runtime_issues() -> None:
    inputs = _inputs()
    answer = _publish(inputs)

    assert answer.status is AgentAnswerStatus.SUPPORTED
    assert answer.as_of == AS_OF
    assert answer.decision_id == inputs.decision.decision_id
    assert answer.decision_snapshot_hash == inputs.decision.content_hash
    assert answer.data_versions == (inputs.decision.data_version,)
    assert answer.facts[0].metrics[0].value == "128"
    assert answer.facts[0].metrics[0].evidence.value_hash == stable_answer_hash(128)
    assert [(item.severity.value, item.code) for item in answer.tool_issues] == [
        ("WARNING", "DATA_UNAVAILABLE")
    ]
    assert answer.answer_hash == stable_answer_hash(answer._identity_payload())


def test_answer_json_round_trip_is_strict_and_tamper_evident() -> None:
    answer = _publish(_inputs())

    assert AgentAnswer.from_json(answer.to_json()) == answer
    changed = json.loads(answer.to_json())
    changed["summary"] = "已被修改。"
    with pytest.raises(ValidationError, match="answer_hash"):
        AgentAnswer.from_json(json.dumps(changed))
    changed = json.loads(answer.to_json())
    changed["generated_at"] = "not-a-time"
    with pytest.raises(ValidationError):
        AgentAnswer.from_json(json.dumps(changed))
    duplicate = answer.to_json().replace(
        '"answer_id":', '"answer_id":"duplicate","answer_id":', 1
    )
    with pytest.raises(ValueError, match="duplicate answer JSON field"):
        AgentAnswer.from_json(duplicate)


@pytest.mark.parametrize(
    "responses,reference_update",
    [
        ({}, {}),
        (None, {"response_hash": "f" * 64}),
        (None, {"value_hash": "e" * 64}),
        (None, {"json_pointer": "/data/unknown"}),
        (None, {"data_version": "market_20260907_eod_v1"}),
    ],
)
def test_unverifiable_evidence_fails_closed_to_insufficient(
    responses: dict[str, ToolResponse[object]] | None,
    reference_update: dict[str, str],
) -> None:
    inputs = _inputs()
    reference = inputs.reference.model_copy(update=reference_update)
    draft = _draft(reference)
    from quant_agent.agent.runtime import AgentRunSnapshot

    assert isinstance(inputs.run, AgentRunSnapshot)
    answer = AgentAnswerPublisher().publish(
        decision=inputs.decision,
        run=inputs.run,
        draft=draft,
        responses=responses if responses is not None else {REQUEST_ID: inputs.response},
        generated_at=NOW,
    )

    assert answer.status is AgentAnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.summary == INSUFFICIENT_EVIDENCE_SUMMARY
    assert not answer.facts and not answer.inferences
    assert tuple(item.kind for item in answer.actions) == (AgentActionKind.NO_ACTION,)
    assert answer.data_versions == ()


def test_missing_claims_returns_insufficient_evidence() -> None:
    inputs = _inputs()
    draft = AgentAnswerDraft(
        summary="工具没有提供可发布事实。",
        actions=(
            AgentAllowedAction(kind=AgentActionKind.NO_ACTION, rationale="等待新的证据。"),
        ),
    )

    answer = _publish(inputs, draft)

    assert answer.status is AgentAnswerStatus.INSUFFICIENT_EVIDENCE


def test_metric_must_exactly_match_the_referenced_numeric_scalar() -> None:
    inputs = _inputs()
    draft = _draft(inputs.reference)
    forged_metric = draft.facts[0].metrics[0].model_copy(update={"value": "129"})
    forged_fact = draft.facts[0].model_copy(update={"metrics": (forged_metric,)})

    answer = _publish(inputs, draft.model_copy(update={"facts": (forged_fact,)}))

    assert answer.status is AgentAnswerStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    "summary",
    [
        "该策略稳赚。",
        "该策略稳\u200b赚。",
        "结论是确定 上涨。",
        "This is guaranteed profit.",
        "目标收益为百分之十, 数值为 10%。",
    ],
)
def test_prohibited_or_untraceable_language_is_rejected(summary: str) -> None:
    inputs = _inputs()

    with pytest.raises(AgentAnswerPolicyViolation):
        _publish(inputs, _draft(inputs.reference).model_copy(update={"summary": summary}))


@pytest.mark.parametrize("missing", ["counter_evidence", "risks", "invalidations"])
def test_each_inference_requires_counter_evidence_risk_and_invalidation(missing: str) -> None:
    inputs = _inputs()
    draft = _draft(inputs.reference).model_copy(update={missing: ()})

    with pytest.raises(AgentAnswerPolicyViolation):
        _publish(inputs, draft)


def test_unknown_fact_or_inference_links_are_rejected() -> None:
    inputs = _inputs()
    draft = _draft(inputs.reference)
    inference = draft.inferences[0].model_copy(update={"based_on_fact_ids": ("unknown",)})
    with pytest.raises(AgentAnswerPolicyViolation, match="unknown fact"):
        _publish(inputs, draft.model_copy(update={"inferences": (inference,)}))

    counter = draft.counter_evidence[0].model_copy(
        update={"challenges_inference_ids": ("unknown",)}
    )
    with pytest.raises(AgentAnswerPolicyViolation, match="unknown inference"):
        _publish(inputs, draft.model_copy(update={"counter_evidence": (counter,)}))


def test_no_action_cannot_be_combined_with_another_action() -> None:
    inputs = _inputs()
    draft = _draft(inputs.reference)
    actions = (
        *draft.actions,
        AgentAllowedAction(kind=AgentActionKind.NO_ACTION, rationale="同时停止。"),
    )
    with pytest.raises(AgentAnswerPolicyViolation, match="cannot be combined"):
        _publish(inputs, draft.model_copy(update={"actions": actions}))


@pytest.mark.parametrize(
    "kind,match",
    [
        (AgentActionKind.REVIEW_ORDER_DRAFT, "verified draft artifact"),
        (AgentActionKind.REQUEST_HUMAN_APPROVAL, "pending-approval run"),
    ],
)
def test_privileged_reader_actions_require_runtime_evidence(
    kind: AgentActionKind,
    match: str,
) -> None:
    inputs = _inputs()
    action = AgentAllowedAction(kind=kind, rationale="交由授权人员复核。")
    with pytest.raises(AgentAnswerPolicyViolation, match=match):
        _publish(inputs, _draft(inputs.reference).model_copy(update={"actions": (action,)}))


def test_metric_source_must_be_numeric() -> None:
    inputs = _inputs()
    text_reference = build_evidence_reference(
        tool_name="get_market_snapshot",
        response=inputs.response,
        json_pointer="/data/market",
    )
    metric = AgentMetric(
        metric_id="invalid_metric",
        label="市场标签",
        value="CN_A",
        unit="text",
        evidence=text_reference,
    )
    fact = AgentFact(
        fact_id="market_snapshot_fact",
        statement="市场快照已锁定并通过运行时留痕。",
        evidence=(text_reference,),
        metrics=(metric,),
    )
    draft = _draft(inputs.reference).model_copy(update={"facts": (fact,)})

    assert _publish(inputs, draft).status is AgentAnswerStatus.INSUFFICIENT_EVIDENCE


def test_reference_builder_rejects_missing_or_non_data_pointer() -> None:
    inputs = _inputs()
    with pytest.raises(AgentEvidenceInvalid, match="does not exist"):
        build_evidence_reference(
            tool_name="get_market_snapshot",
            response=inputs.response,
            json_pointer="/data/missing",
        )
    with pytest.raises(ValidationError, match="below /data"):
        build_evidence_reference(
            tool_name="get_market_snapshot",
            response=inputs.response,
            json_pointer="/request_id",
        )


def test_response_hash_matches_runtime_event_identity() -> None:
    inputs = _inputs()
    from quant_agent.agent.runtime import AgentRunEventKind, AgentRunSnapshot

    assert isinstance(inputs.run, AgentRunSnapshot)
    outcome = next(
        event
        for event in inputs.run.events
        if event.kind is AgentRunEventKind.TOOL_CALL_SUCCEEDED
    )
    assert tool_response_hash(inputs.response) == outcome.response_hash


def test_naive_publication_time_and_mismatched_decision_are_rejected() -> None:
    inputs = _inputs()
    from quant_agent.agent.runtime import AgentRunSnapshot

    assert isinstance(inputs.run, AgentRunSnapshot)
    with pytest.raises(ValueError, match="timezone information"):
        AgentAnswerPublisher().publish(
            decision=inputs.decision,
            run=inputs.run,
            draft=_draft(inputs.reference),
            responses={REQUEST_ID: inputs.response},
            generated_at=datetime(2026, 9, 8, 8, 0),
        )

    mismatched = _decision()
    object.__setattr__(mismatched, "decision_id", "different-decision")
    with pytest.raises(AgentAnswerPolicyViolation, match="integrity validation"):
        AgentAnswerPublisher().publish(
            decision=mismatched,
            run=inputs.run,
            draft=_draft(inputs.reference),
            responses={REQUEST_ID: inputs.response},
            generated_at=NOW,
        )
