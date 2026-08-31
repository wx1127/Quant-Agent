"""Immutable contracts for deterministic, uncalibrated stock candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.leaders.contracts import stable_leader_hash
from quant_agent.regime import MarketRegime


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


class CandidateInputError(ValueError):
    """Raised when candidate inputs cannot be aligned without an unsafe assumption."""


class CandidateTier(StrEnum):
    """Uncalibrated research layers; they are not probabilities or recommendations."""

    A = "A"
    B = "B"
    WATCH = "WATCH"
    EXCLUDED = "EXCLUDED"


class CandidateComponent(StrEnum):
    """Canonical score components in development-guide order."""

    MARKET_REGIME = "MARKET_REGIME"
    THEME_SCORE = "THEME_SCORE"
    LEADER_SCORE = "LEADER_SCORE"
    PRICE_TREND = "PRICE_TREND"
    VOLUME_PRICE_STRUCTURE = "VOLUME_PRICE_STRUCTURE"
    FUNDAMENTAL_OR_EVENT = "FUNDAMENTAL_OR_EVENT"
    VALUATION = "VALUATION"


class CandidateEvidenceSide(StrEnum):
    """Whether a fact supports or opposes the candidate score."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"


class CandidateSignalKind(StrEnum):
    """Optional PIT signals that are not already represented by core features."""

    EVENT = "EVENT"
    VALUATION = "VALUATION"


class CandidateRiskCode(StrEnum):
    """Machine-readable candidate penalty sources."""

    LEADER = "LEADER"
    EVENT = "EVENT"
    VALUATION = "VALUATION"


class CandidateExclusionCode(StrEnum):
    """Hard-gate and upstream exclusion reasons."""

    LEADER_FILTER = "LEADER_FILTER"
    MARKET_DOWNTREND = "MARKET_DOWNTREND"
    UPSTREAM_LEADER_EXCLUSION = "UPSTREAM_LEADER_EXCLUSION"


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """Frozen scoring, tiering, evidence, and risk policy."""

    version: str = "candidate-ranking-v1"
    market_regime_weight: Decimal = Decimal("0.20")
    theme_score_weight: Decimal = Decimal("0.20")
    leader_score_weight: Decimal = Decimal("0.20")
    price_trend_weight: Decimal = Decimal("0.15")
    volume_price_structure_weight: Decimal = Decimal("0.10")
    fundamental_or_event_weight: Decimal = Decimal("0.10")
    valuation_weight: Decimal = Decimal("0.05")
    uptrend_score: Decimal = Decimal("100")
    range_strong_score: Decimal = Decimal("75")
    divergent_score: Decimal = Decimal("45")
    downtrend_score: Decimal = Decimal("10")
    bottom_recovery_score: Decimal = Decimal("55")
    tier_a_minimum: Decimal = Decimal("75")
    tier_b_minimum: Decimal = Decimal("60")
    evidence_support_minimum: Decimal = Decimal("50")
    maximum_risk_penalty: Decimal = Decimal("40")
    allow_downtrend_candidates: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "candidate version"))
        decimal_values = tuple(
            value for value in asdict(self).values() if isinstance(value, Decimal)
        )
        for value in decimal_values:
            canonical_decimal(value, field_name="candidate numeric parameter")
        if any(value < 0 for value in decimal_values):
            raise ValueError("candidate numeric parameters cannot be negative")
        if sum(self.component_weights.values(), Decimal(0)) != Decimal(1):
            raise ValueError("candidate component weights must sum to one")
        score_values = (*self.regime_scores.values(), self.evidence_support_minimum)
        if any(value > 100 for value in score_values):
            raise ValueError("candidate component scores must be within 0..100")
        if not Decimal(0) <= self.tier_b_minimum < self.tier_a_minimum <= Decimal(100):
            raise ValueError("candidate tier thresholds must be ordered within 0..100")
        if self.maximum_risk_penalty > 100:
            raise ValueError("maximum_risk_penalty must be within 0..100")
        if not isinstance(self.allow_downtrend_candidates, bool):
            raise ValueError("allow_downtrend_candidates must be boolean")

    @property
    def component_weights(self) -> dict[CandidateComponent, Decimal]:
        return {
            CandidateComponent.MARKET_REGIME: self.market_regime_weight,
            CandidateComponent.THEME_SCORE: self.theme_score_weight,
            CandidateComponent.LEADER_SCORE: self.leader_score_weight,
            CandidateComponent.PRICE_TREND: self.price_trend_weight,
            CandidateComponent.VOLUME_PRICE_STRUCTURE: self.volume_price_structure_weight,
            CandidateComponent.FUNDAMENTAL_OR_EVENT: self.fundamental_or_event_weight,
            CandidateComponent.VALUATION: self.valuation_weight,
        }

    @property
    def regime_scores(self) -> dict[MarketRegime, Decimal]:
        return {
            MarketRegime.UPTREND: self.uptrend_score,
            MarketRegime.RANGE_STRONG: self.range_strong_score,
            MarketRegime.DIVERGENT: self.divergent_score,
            MarketRegime.DOWNTREND: self.downtrend_score,
            MarketRegime.BOTTOM_RECOVERY: self.bottom_recovery_score,
        }

    @property
    def config_hash(self) -> str:
        return stable_leader_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateSupplementalSignal:
    """One optional event or valuation score known at the decision boundary."""

    instrument_id: str
    kind: CandidateSignalKind
    session_date: date
    as_of: datetime
    data_version: str
    model_version: str
    score: Decimal
    risk_penalty: Decimal
    source_refs: tuple[str, ...]
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        object.__setattr__(
            self,
            "model_version",
            _non_empty(self.model_version, "model_version"),
        )
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.session_date:
            raise ValueError("supplemental signal session_date must match as_of")
        for value in (self.score, self.risk_penalty):
            canonical_decimal(value, field_name="supplemental candidate score")
            if value < 0 or value > 100:
                raise ValueError("supplemental candidate scores must be within 0..100")
        refs = tuple(sorted({_non_empty(value, "source_ref") for value in self.source_refs}))
        if not refs:
            raise ValueError("supplemental candidate signal requires source references")
        object.__setattr__(self, "source_refs", refs)
        _validate_hash(self.input_hash, "supplemental input_hash")
        _validate_hash(self.result_hash, "supplemental result_hash")
        expected_input = stable_leader_hash(
            {
                "as_of": self.as_of,
                "data_version": self.data_version,
                "instrument_id": self.instrument_id,
                "kind": self.kind,
                "session_date": self.session_date,
                "source_refs": list(refs),
            }
        )
        if self.input_hash != expected_input:
            raise ValueError("supplemental input_hash does not match signal inputs")
        expected_result = stable_leader_hash(
            {
                "input_hash": self.input_hash,
                "model_version": self.model_version,
                "risk_penalty": self.risk_penalty,
                "score": self.score,
            }
        )
        if self.result_hash != expected_result:
            raise ValueError("supplemental result_hash does not match signal output")

    @classmethod
    def build(
        cls,
        *,
        instrument_id: str,
        kind: CandidateSignalKind,
        session_date: date,
        as_of: datetime,
        data_version: str,
        model_version: str,
        score: Decimal,
        risk_penalty: Decimal = Decimal(0),
        source_refs: tuple[str, ...],
    ) -> CandidateSupplementalSignal:
        normalized_refs = tuple(sorted({_non_empty(value, "source_ref") for value in source_refs}))
        input_hash = stable_leader_hash(
            {
                "as_of": as_of,
                "data_version": data_version,
                "instrument_id": instrument_id,
                "kind": kind,
                "session_date": session_date,
                "source_refs": list(normalized_refs),
            }
        )
        result_hash = stable_leader_hash(
            {
                "input_hash": input_hash,
                "model_version": model_version,
                "risk_penalty": risk_penalty,
                "score": score,
            }
        )
        return cls(
            instrument_id=instrument_id,
            kind=kind,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            model_version=model_version,
            score=score,
            risk_penalty=risk_penalty,
            source_refs=normalized_refs,
            input_hash=input_hash,
            result_hash=result_hash,
        )

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "instrument_id": self.instrument_id,
            "kind": self.kind.value,
            "model_version": self.model_version,
            "result_hash": self.result_hash,
        }


@dataclass(frozen=True, slots=True)
class CandidateComponentScore:
    """One normalized score and its exact weighted contribution."""

    component: CandidateComponent
    score: Decimal
    weight: Decimal
    contribution: Decimal
    source: str

    def __post_init__(self) -> None:
        for value in (self.score, self.weight, self.contribution):
            canonical_decimal(value, field_name="candidate component value")
        if not Decimal(0) <= self.score <= Decimal(100):
            raise ValueError("candidate component score must be within 0..100")
        if not Decimal(0) <= self.weight <= Decimal(1):
            raise ValueError("candidate component weight must be within 0..1")
        if self.contribution != self.score * self.weight:
            raise ValueError("candidate contribution must equal score times weight")
        object.__setattr__(self, "source", _non_empty(self.source, "candidate component source"))


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Structured support or opposition without predictive-certainty language."""

    feature: str
    value: Decimal
    criterion: str
    side: CandidateEvidenceSide
    rationale: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.feature, "feature"),
            (self.criterion, "criterion"),
            (self.rationale, "rationale"),
        ):
            _non_empty(value, field_name)
        canonical_decimal(self.value, field_name="candidate evidence value")


@dataclass(frozen=True, slots=True)
class CandidateRisk:
    """One applied, capped risk penalty with a machine-readable code."""

    code: CandidateRiskCode
    penalty: Decimal
    rationale: str

    def __post_init__(self) -> None:
        canonical_decimal(self.penalty, field_name="candidate risk penalty")
        if self.penalty <= 0 or self.penalty > 100:
            raise ValueError("candidate risk penalty must be within (0, 100]")
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "risk rationale"))


@dataclass(frozen=True, slots=True)
class CandidateUpstreamIdentity:
    """Exact upstream versions and hashes needed to reproduce one candidate."""

    regime_transition_version: str
    regime_transition_hash: str
    mainline_model_version: str
    mainline_result_hash: str
    leader_feature_version: str
    leader_result_hash: str
    stock_result_hash: str
    tradeability_result_hash: str
    fundamental_result_hash: str
    event_result_hash: str | None
    valuation_result_hash: str | None

    def __post_init__(self) -> None:
        versions = (
            self.regime_transition_version,
            self.mainline_model_version,
            self.leader_feature_version,
        )
        if any(not value.strip() for value in versions):
            raise ValueError("candidate upstream versions must be non-empty")
        hashes = (
            self.regime_transition_hash,
            self.mainline_result_hash,
            self.leader_result_hash,
            self.stock_result_hash,
            self.tradeability_result_hash,
            self.fundamental_result_hash,
        )
        for value in hashes:
            _validate_hash(value, "candidate upstream identity")
        for optional_hash in (self.event_result_hash, self.valuation_result_hash):
            if optional_hash is not None:
                _validate_hash(optional_hash, "candidate supplemental identity")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One scored research candidate or explicitly excluded ranked stock."""

    instrument_id: str
    industry_id: str
    research_rank: int
    eligible_rank: int | None
    tier: CandidateTier
    gross_score: Decimal
    risk_penalty: Decimal
    score: Decimal
    components: tuple[CandidateComponentScore, ...]
    supporting_evidence: tuple[CandidateEvidence, ...]
    counter_evidence: tuple[CandidateEvidence, ...]
    risks: tuple[CandidateRisk, ...]
    exclusion_codes: tuple[CandidateExclusionCode, ...]
    exclusion_reasons: tuple[str, ...]
    observation_conditions: tuple[str, ...]
    invalidations: tuple[str, ...]
    upstream_identity: CandidateUpstreamIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "industry_id", _non_empty(self.industry_id, "industry_id"))
        if self.research_rank < 1 or (self.eligible_rank is not None and self.eligible_rank < 1):
            raise ValueError("candidate ranks must be positive")
        for value in (self.gross_score, self.risk_penalty, self.score):
            canonical_decimal(value, field_name="candidate score")
            if value < 0 or value > 100:
                raise ValueError("candidate scores must be within 0..100")
        if self.score != max(Decimal(0), self.gross_score - self.risk_penalty):
            raise ValueError("candidate score must deduct risk_penalty")
        if tuple(item.component for item in self.components) != tuple(CandidateComponent):
            raise ValueError("candidate components must use canonical order")
        if sum((item.weight for item in self.components), Decimal(0)) != Decimal(1):
            raise ValueError("candidate component weights must sum to one")
        if sum((item.contribution for item in self.components), Decimal(0)) != self.gross_score:
            raise ValueError("candidate gross score must equal component contributions")
        if sum((item.penalty for item in self.risks), Decimal(0)) != self.risk_penalty:
            raise ValueError("candidate risk records must equal risk_penalty")
        excluded = bool(self.exclusion_codes)
        if excluded != bool(self.exclusion_reasons):
            raise ValueError("candidate exclusion codes and reasons must be present together")
        if excluded != (self.tier is CandidateTier.EXCLUDED):
            raise ValueError("candidate EXCLUDED tier must match hard-gate reasons")
        if excluded and self.eligible_rank is not None:
            raise ValueError("excluded candidate cannot have an eligible rank")
        if not excluded and self.eligible_rank is None:
            raise ValueError("eligible candidate requires an eligible rank")
        if not self.observation_conditions or not self.invalidations:
            raise ValueError("candidate output requires observation and invalidation conditions")
        if any(
            not value.strip()
            for value in (
                *self.exclusion_reasons,
                *self.observation_conditions,
                *self.invalidations,
            )
        ):
            raise ValueError("candidate reasons and conditions must be non-empty")

    @property
    def eligible(self) -> bool:
        return self.tier is not CandidateTier.EXCLUDED


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    """One upstream stock that could not receive a candidate score."""

    instrument_id: str
    industry_id: str | None
    code: CandidateExclusionCode
    reason: str
    source_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        if self.industry_id is not None:
            object.__setattr__(
                self,
                "industry_id",
                _non_empty(self.industry_id, "industry_id"),
            )
        object.__setattr__(self, "reason", _non_empty(self.reason, "exclusion reason"))
        _validate_hash(self.source_hash, "candidate exclusion source_hash")


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    """Deterministic, uncalibrated candidate pool for one decision snapshot."""

    session_date: date
    as_of: datetime
    data_version: str
    feature_version: str
    config_hash: str
    regime: MarketRegime
    candidates: tuple[CandidateResult, ...]
    exclusions: tuple[CandidateExclusion, ...]
    calibrated: bool
    calibration_version: str | None
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.session_date:
            raise ValueError("candidate snapshot session_date must match as_of")
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        object.__setattr__(
            self,
            "feature_version",
            _non_empty(self.feature_version, "feature_version"),
        )
        for value in (self.config_hash, self.input_hash, self.result_hash):
            _validate_hash(value, "candidate snapshot identity")
        if self.calibrated or self.calibration_version is not None:
            raise ValueError("first-version candidate scores are explicitly uncalibrated")
        candidate_ids = tuple(item.instrument_id for item in self.candidates)
        exclusion_ids = tuple(item.instrument_id for item in self.exclusions)
        if len(set((*candidate_ids, *exclusion_ids))) != len(candidate_ids) + len(exclusion_ids):
            raise ValueError("candidate and exclusion instrument IDs must be unique")
        if tuple(item.research_rank for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate research ranks must be contiguous and ordered")
        eligible_ranks = tuple(
            item.eligible_rank for item in self.candidates if item.eligible_rank is not None
        )
        if tuple(sorted(eligible_ranks)) != tuple(range(1, len(eligible_ranks) + 1)):
            raise ValueError("candidate eligible ranks must be unique and contiguous")

    @property
    def eligible_candidates(self) -> tuple[CandidateResult, ...]:
        return tuple(item for item in self.candidates if item.eligible)

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "input_hash": self.input_hash,
            "result_hash": self.result_hash,
            "session_date": self.session_date.isoformat(),
        }


__all__ = [
    "CandidateComponent",
    "CandidateComponentScore",
    "CandidateConfig",
    "CandidateEvidence",
    "CandidateEvidenceSide",
    "CandidateExclusion",
    "CandidateExclusionCode",
    "CandidateInputError",
    "CandidateResult",
    "CandidateRisk",
    "CandidateRiskCode",
    "CandidateSignalKind",
    "CandidateSnapshot",
    "CandidateSupplementalSignal",
    "CandidateTier",
    "CandidateUpstreamIdentity",
]
