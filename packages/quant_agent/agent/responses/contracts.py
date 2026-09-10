"""Strict contracts for evidence-bound Agent answers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from quant_agent.agent.runtime import AgentRunGoal, AgentRunState
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware

AGENT_ANSWER_SCHEMA_VERSION: Final = "1"
AGENT_ANSWER_POLICY_VERSION: Final = "agent-answer-policy-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_TOOL_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", re.ASCII)
_POINTER = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_STRICT = ConfigDict(
    extra="forbid",
    frozen=True,
    revalidate_instances="always",
    strict=True,
    validate_default=True,
)

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Text = Annotated[str, Field(min_length=1, max_length=2_000)]
Sha256 = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


def _text(value: str, field: str, maximum: int = 2_000) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isprintable()
    ):
        raise ValueError(f"{field} must be non-empty, trimmed, printable, and bounded")
    return value


def _hash(value: str, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _utc(value: datetime) -> datetime:
    return ensure_aware(value).astimezone(UTC)


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return _utc(value).isoformat(timespec="microseconds")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"unsupported answer value: {type(value).__name__}")


def stable_answer_hash(value: object) -> str:
    """Return the SHA-256 identity of canonical answer JSON."""

    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AgentAnswerStatus(StrEnum):
    """Whether the published answer has enough verified evidence."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class AgentActionKind(StrEnum):
    """Non-executable actions that an answer may recommend to its reader."""

    NO_ACTION = "NO_ACTION"
    CONTINUE_RESEARCH = "CONTINUE_RESEARCH"
    REVIEW_REPORT = "REVIEW_REPORT"
    REVIEW_ORDER_DRAFT = "REVIEW_ORDER_DRAFT"
    REQUEST_HUMAN_APPROVAL = "REQUEST_HUMAN_APPROVAL"


class AgentIssueSeverity(StrEnum):
    """Tool outcome severity surfaced without copying unsafe issue text."""

    WARNING = "WARNING"
    ERROR = "ERROR"


class EvidenceReference(BaseModel):
    """Content-addressed path into one successful tool response."""

    model_config = _STRICT

    request_id: Identifier
    tool_name: Annotated[str, Field(min_length=1, max_length=64)]
    response_hash: Sha256
    json_pointer: Annotated[str, Field(min_length=1, max_length=1_024)]
    value_hash: Sha256
    data_version: Identifier

    @field_validator("request_id", "data_version")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _text(value, getattr(info, "field_name", "identifier"), 256)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if type(value) is not str or _TOOL_NAME.fullmatch(value) is None:
            raise ValueError("tool_name must be lower-ASCII snake-case")
        return value

    @field_validator("response_hash", "value_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: object) -> str:
        return _hash(value, getattr(info, "field_name", "hash"))

    @field_validator("json_pointer")
    @classmethod
    def validate_pointer(cls, value: str) -> str:
        if type(value) is not str or _POINTER.fullmatch(value) is None or not value.startswith(
            "/data/"
        ):
            raise ValueError("json_pointer must be an escaped RFC 6901 path below /data")
        return value


class AgentMetric(BaseModel):
    """A display number whose exact JSON scalar is pinned to one tool response."""

    model_config = _STRICT

    metric_id: Identifier
    label: Text
    value: Annotated[str, Field(min_length=1, max_length=256)]
    unit: Annotated[str, Field(min_length=1, max_length=64)]
    evidence: EvidenceReference

    @field_validator("metric_id", "label", "value", "unit")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        limits = {"metric_id": 256, "label": 2_000, "value": 256, "unit": 64}
        field = getattr(info, "field_name", "text")
        return _text(value, field, limits[field])


class AgentFact(BaseModel):
    """A factual statement supported by one or more exact tool-response paths."""

    model_config = _STRICT

    fact_id: Identifier
    statement: Text
    evidence: Annotated[tuple[EvidenceReference, ...], Field(min_length=1, max_length=128)]
    metrics: Annotated[tuple[AgentMetric, ...], Field(max_length=64)] = ()

    @field_validator("fact_id", "statement")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        return _text(value, field, 256 if field == "fact_id" else 2_000)


class AgentInference(BaseModel):
    """An interpretation explicitly separated from the facts it depends on."""

    model_config = _STRICT

    inference_id: Identifier
    statement: Text
    based_on_fact_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=128)]
    metrics: Annotated[tuple[AgentMetric, ...], Field(max_length=64)] = ()

    @field_validator("inference_id", "statement")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        return _text(value, field, 256 if field == "inference_id" else 2_000)


class AgentCounterEvidence(BaseModel):
    """Verified evidence that weakens or qualifies an inference."""

    model_config = _STRICT

    counter_id: Identifier
    statement: Text
    challenges_inference_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=128)]
    evidence: Annotated[tuple[EvidenceReference, ...], Field(min_length=1, max_length=128)]
    metrics: Annotated[tuple[AgentMetric, ...], Field(max_length=64)] = ()

    @field_validator("counter_id", "statement")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        return _text(value, field, 256 if field == "counter_id" else 2_000)


class AgentRiskWarning(BaseModel):
    """A risk shown separately from supporting facts and conclusions."""

    model_config = _STRICT

    risk_id: Identifier
    statement: Text
    evidence: Annotated[tuple[EvidenceReference, ...], Field(max_length=128)] = ()
    metrics: Annotated[tuple[AgentMetric, ...], Field(max_length=64)] = ()

    @field_validator("risk_id", "statement")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        return _text(value, field, 256 if field == "risk_id" else 2_000)


class AgentInvalidationCondition(BaseModel):
    """An observable condition that invalidates named inferences."""

    model_config = _STRICT

    condition_id: Identifier
    statement: Text
    invalidates_inference_ids: Annotated[
        tuple[Identifier, ...], Field(min_length=1, max_length=128)
    ]
    evidence: Annotated[tuple[EvidenceReference, ...], Field(max_length=128)] = ()
    metrics: Annotated[tuple[AgentMetric, ...], Field(max_length=64)] = ()

    @field_validator("condition_id", "statement")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        return _text(value, field, 256 if field == "condition_id" else 2_000)


class AgentAllowedAction(BaseModel):
    """A non-executable next step; trade submission is intentionally absent."""

    model_config = _STRICT

    kind: AgentActionKind
    rationale: Text

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _text(value, "rationale")


class AgentToolIssueRef(BaseModel):
    """A warning/error identity copied from the immutable runtime event chain."""

    model_config = _STRICT

    severity: AgentIssueSeverity
    request_id: Identifier
    tool_name: Annotated[str, Field(min_length=1, max_length=64)]
    code: Identifier
    response_hash: Sha256 | None = None

    @field_validator("request_id", "code")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _text(value, getattr(info, "field_name", "identifier"), 256)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if type(value) is not str or _TOOL_NAME.fullmatch(value) is None:
            raise ValueError("tool_name must be lower-ASCII snake-case")
        return value

    @field_validator("response_hash")
    @classmethod
    def validate_response_hash(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value, "response_hash")


class AgentAnswerDraft(BaseModel):
    """Model-controlled answer content before trusted evidence verification."""

    model_config = _STRICT

    summary: Text
    facts: Annotated[tuple[AgentFact, ...], Field(max_length=256)] = ()
    inferences: Annotated[tuple[AgentInference, ...], Field(max_length=256)] = ()
    counter_evidence: Annotated[tuple[AgentCounterEvidence, ...], Field(max_length=256)] = ()
    risks: Annotated[tuple[AgentRiskWarning, ...], Field(max_length=256)] = ()
    invalidations: Annotated[tuple[AgentInvalidationCondition, ...], Field(max_length=256)] = ()
    actions: Annotated[tuple[AgentAllowedAction, ...], Field(min_length=1, max_length=32)]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _text(value, "summary")


class AgentAnswer(BaseModel):
    """Published, immutable, tamper-evident answer bound to one Agent run."""

    model_config = _STRICT

    schema_version: Literal["1"] = AGENT_ANSWER_SCHEMA_VERSION
    policy_version: Literal["agent-answer-policy-v1"] = AGENT_ANSWER_POLICY_VERSION
    answer_id: Identifier
    run_id: Identifier
    decision_id: Identifier
    decision_snapshot_hash: Sha256
    runtime_mode: RuntimeMode
    goal: AgentRunGoal
    run_state: AgentRunState
    status: AgentAnswerStatus
    as_of: datetime
    generated_at: datetime
    summary: Text
    facts: Annotated[tuple[AgentFact, ...], Field(max_length=256)] = ()
    inferences: Annotated[tuple[AgentInference, ...], Field(max_length=256)] = ()
    counter_evidence: Annotated[tuple[AgentCounterEvidence, ...], Field(max_length=256)] = ()
    risks: Annotated[tuple[AgentRiskWarning, ...], Field(max_length=256)] = ()
    invalidations: Annotated[tuple[AgentInvalidationCondition, ...], Field(max_length=256)] = ()
    actions: Annotated[tuple[AgentAllowedAction, ...], Field(min_length=1, max_length=32)]
    tool_issues: Annotated[tuple[AgentToolIssueRef, ...], Field(max_length=10_000)] = ()
    data_versions: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()
    answer_hash: Sha256

    @field_validator("answer_id", "run_id", "decision_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _text(value, getattr(info, "field_name", "identifier"), 256)

    @field_validator("decision_snapshot_hash", "answer_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: object) -> str:
        return _hash(value, getattr(info, "field_name", "hash"))

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _text(value, "summary")

    @field_validator("as_of", "generated_at")
    @classmethod
    def normalize_times(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if self.generated_at < self.as_of:
            raise ValueError("generated_at cannot precede the data cutoff")
        if tuple(sorted(set(self.data_versions))) != self.data_versions:
            raise ValueError("data_versions must be unique and sorted")
        if self.status is AgentAnswerStatus.INSUFFICIENT_EVIDENCE:
            if any(
                (self.facts, self.inferences, self.counter_evidence, self.risks, self.invalidations)
            ):
                raise ValueError("insufficient-evidence answers cannot retain unverified claims")
            if tuple(item.kind for item in self.actions) != (AgentActionKind.NO_ACTION,):
                raise ValueError("insufficient-evidence answers must require NO_ACTION")
        elif not self.facts:
            raise ValueError("supported answers require verified facts")
        expected = stable_answer_hash(self._identity_payload())
        if self.answer_hash != expected:
            raise ValueError("answer_hash does not match the complete answer")
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.model_dump(mode="python").items()
            if key != "answer_hash"
        }

    def to_json(self) -> str:
        """Serialize with deterministic keys and strict JSON-compatible values."""

        parsed = json.loads(_ANSWER_ADAPTER.dump_json(self, warnings="error"))
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> AgentAnswer:
        """Strictly revalidate a serialized answer and its content hash."""

        parsed = json.loads(value, object_pairs_hook=_unique_json_object)
        if type(parsed) is not dict:
            raise ValueError("answer JSON must contain an object")
        canonical = json.dumps(
            parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return _ANSWER_ADAPTER.validate_json(canonical, strict=True)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate answer JSON field: {key}")
        result[key] = value
    return result


_ANSWER_ADAPTER = TypeAdapter(AgentAnswer)

__all__ = [
    "AGENT_ANSWER_POLICY_VERSION",
    "AGENT_ANSWER_SCHEMA_VERSION",
    "AgentActionKind",
    "AgentAllowedAction",
    "AgentAnswer",
    "AgentAnswerDraft",
    "AgentAnswerStatus",
    "AgentCounterEvidence",
    "AgentFact",
    "AgentInference",
    "AgentInvalidationCondition",
    "AgentIssueSeverity",
    "AgentMetric",
    "AgentRiskWarning",
    "AgentToolIssueRef",
    "EvidenceReference",
    "stable_answer_hash",
]
