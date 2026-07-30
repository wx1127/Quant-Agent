"""Evidence-grounded final Agent answer contract."""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from quant_agent.core.time import ensure_aware

_FORBIDDEN = ("稳赚", "确定上涨", "必涨", "必买", "guaranteed profit", "risk-free return")
_NUMBER = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?%?")


class AnswerType(StrEnum):
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    CANDIDATE_ANALYSIS = "CANDIDATE_ANALYSIS"
    PORTFOLIO_PROPOSAL = "PORTFOLIO_PROPOSAL"
    EXECUTION_RESULT = "EXECUTION_RESULT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_call_id: str
    tool_name: str
    response_hash: str
    json_pointer: str


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)
    statement: str
    evidence: tuple[EvidenceReference, ...]


class Inference(BaseModel):
    model_config = ConfigDict(frozen=True)
    statement: str
    basis: tuple[EvidenceReference, ...]
    confidence_label: str


class NumericClaim(BaseModel):
    model_config = ConfigDict(frozen=True)
    rendered_value: str
    evidence: EvidenceReference


class AgentAnswer(BaseModel):
    """Facts, inference and risks are separate and numerically traceable."""

    model_config = ConfigDict(frozen=True)

    decision_id: str | None
    as_of: datetime | None
    answer_type: AnswerType
    summary: str
    facts: tuple[Fact, ...] = ()
    inferences: tuple[Inference, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    invalidations: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    data_versions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    numeric_claims: tuple[NumericClaim, ...] = ()

    @field_validator("as_of")
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return ensure_aware(value) if value is not None else None

    @model_validator(mode="after")
    def enforce_evidence_contract(self) -> "AgentAnswer":
        text = " ".join(
            (
                self.summary,
                *(item.statement for item in self.facts),
                *(item.statement for item in self.inferences),
                *self.counter_evidence,
                *self.risks,
                *self.invalidations,
                *self.actions,
                *self.warnings,
            )
        )
        if any(term.casefold() in text.casefold() for term in _FORBIDDEN):
            raise ValueError("answer contains a prohibited certainty claim")
        market_types = {
            AnswerType.MARKET_ANALYSIS,
            AnswerType.CANDIDATE_ANALYSIS,
            AnswerType.PORTFOLIO_PROPOSAL,
        }
        if self.answer_type in market_types and (self.decision_id is None or self.as_of is None):
            raise ValueError("market answers require decision_id and as_of")
        if self.answer_type not in {
            AnswerType.INSUFFICIENT_EVIDENCE,
            AnswerType.REJECTED,
        }:
            if not self.facts:
                raise ValueError("evidence-backed answer requires at least one fact")
            if not self.counter_evidence or not self.risks or not self.invalidations:
                raise ValueError("answer must include counter evidence, risks and invalidations")
        rendered = {item.rendered_value for item in self.numeric_claims}
        untraced = {match.group() for match in _NUMBER.finditer(text)} - rendered
        if untraced:
            raise ValueError(f"numeric claims lack tool evidence: {sorted(untraced)}")
        return self

    @classmethod
    def insufficient(
        cls,
        *,
        decision_id: str | None,
        as_of: datetime | None,
        reason: str = "证据不足, 无法形成可验证的结论。",
    ) -> "AgentAnswer":
        return cls(
            decision_id=decision_id,
            as_of=as_of,
            answer_type=AnswerType.INSUFFICIENT_EVIDENCE,
            summary=reason,
            risks=("缺少可靠证据时不应据此做出投资决定。",),
        )
