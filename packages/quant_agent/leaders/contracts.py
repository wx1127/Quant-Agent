"""Immutable contracts for deterministic mainline-leader ranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.features.mainline import MainlineState


def _canonical(value: Any) -> Any:
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return ensure_aware(value).astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def stable_leader_hash(payload: dict[str, Any]) -> str:
    """Return a scale-independent SHA-256 identity for leader artifacts."""

    encoded = json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")


class LeaderInputError(ValueError):
    """Raised when upstream snapshots cannot be aligned without unsafe assumptions."""


class LeaderType(StrEnum):
    """First-version leader classifications."""

    TREND = "TREND"
    CAPACITY = "CAPACITY"
    ELASTICITY = "ELASTICITY"
    OBSERVATION = "OBSERVATION"


class LeaderComponent(StrEnum):
    """Canonical score components in development-guide order."""

    WITHIN_THEME_STRENGTH = "WITHIN_THEME_STRENGTH"
    TREND_QUALITY = "TREND_QUALITY"
    LIQUIDITY = "LIQUIDITY"
    THEME_LEADERSHIP = "THEME_LEADERSHIP"
    DOWNSIDE_RESILIENCE = "DOWNSIDE_RESILIENCE"
    LOGIC_RELEVANCE = "LOGIC_RELEVANCE"
    FUNDAMENTAL_QUALITY = "FUNDAMENTAL_QUALITY"


class LeaderEvidenceSide(StrEnum):
    """Whether an observed fact supports or opposes leadership."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"


class LeaderRiskCode(StrEnum):
    """Machine-readable penalty sources."""

    FUNDAMENTAL = "FUNDAMENTAL"
    MAINLINE_CROWDING = "MAINLINE_CROWDING"
    DEGRADED_STRENGTH = "DEGRADED_STRENGTH"


class LeaderExclusionCode(StrEnum):
    """Why a supplied stock was not ranked."""

    NOT_ACTIVE_MAINLINE_MEMBER = "NOT_ACTIVE_MAINLINE_MEMBER"
    STOCK_STRENGTH_UNAVAILABLE = "STOCK_STRENGTH_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class LeaderConfig:
    """Frozen scoring, classification, and risk-penalty policy."""

    version: str = "leader-ranking-v1"
    within_theme_strength_weight: Decimal = Decimal("0.25")
    trend_quality_weight: Decimal = Decimal("0.15")
    liquidity_weight: Decimal = Decimal("0.15")
    theme_leadership_weight: Decimal = Decimal("0.15")
    downside_resilience_weight: Decimal = Decimal("0.10")
    logic_relevance_weight: Decimal = Decimal("0.10")
    fundamental_quality_weight: Decimal = Decimal("0.10")
    liquidity_percentile_weight: Decimal = Decimal("0.50")
    liquidity_capacity_weight: Decimal = Decimal("0.50")
    theme_return_rank_weight: Decimal = Decimal("0.60")
    theme_volume_price_weight: Decimal = Decimal("0.40")
    downside_relative_weight: Decimal = Decimal("0.50")
    downside_pullback_weight: Decimal = Decimal("0.50")
    capacity_ratio_full_score: Decimal = Decimal("2")
    pullback_tolerance: Decimal = Decimal("0.20")
    capacity_liquidity_min: Decimal = Decimal("70")
    capacity_leadership_min: Decimal = Decimal("55")
    capacity_trend_min: Decimal = Decimal("45")
    trend_quality_min: Decimal = Decimal("60")
    trend_within_strength_min: Decimal = Decimal("55")
    trend_downside_min: Decimal = Decimal("50")
    elasticity_min: Decimal = Decimal("70")
    minimum_candidate_score: Decimal = Decimal("60")
    fundamental_risk_multiplier: Decimal = Decimal("1")
    crowded_penalty_rate: Decimal = Decimal("0.10")
    degraded_strength_penalty: Decimal = Decimal("5")
    maximum_risk_penalty: Decimal = Decimal("30")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "leader version"))
        values = tuple(value for value in asdict(self).values() if isinstance(value, Decimal))
        for value in values:
            canonical_decimal(value, field_name="leader numeric parameter")
        if any(value < 0 for value in values):
            raise ValueError("leader numeric parameters cannot be negative")
        if sum(self.component_weights.values(), Decimal(0)) != 1:
            raise ValueError("leader component weights must sum to one")
        subweights = (
            self.liquidity_percentile_weight + self.liquidity_capacity_weight,
            self.theme_return_rank_weight + self.theme_volume_price_weight,
            self.downside_relative_weight + self.downside_pullback_weight,
        )
        if any(value != 1 for value in subweights):
            raise ValueError("leader subcomponent weights must each sum to one")
        thresholds = (
            self.capacity_liquidity_min,
            self.capacity_leadership_min,
            self.capacity_trend_min,
            self.trend_quality_min,
            self.trend_within_strength_min,
            self.trend_downside_min,
            self.elasticity_min,
            self.minimum_candidate_score,
            self.maximum_risk_penalty,
        )
        if any(value > 100 for value in thresholds):
            raise ValueError("leader thresholds and maximum penalty must be within 0..100")
        if self.capacity_ratio_full_score <= 0 or self.pullback_tolerance <= 0:
            raise ValueError("leader capacity and pullback scales must be positive")
        if self.crowded_penalty_rate > 1:
            raise ValueError("crowded_penalty_rate must be within 0..1")

    @property
    def component_weights(self) -> dict[LeaderComponent, Decimal]:
        return {
            LeaderComponent.WITHIN_THEME_STRENGTH: self.within_theme_strength_weight,
            LeaderComponent.TREND_QUALITY: self.trend_quality_weight,
            LeaderComponent.LIQUIDITY: self.liquidity_weight,
            LeaderComponent.THEME_LEADERSHIP: self.theme_leadership_weight,
            LeaderComponent.DOWNSIDE_RESILIENCE: self.downside_resilience_weight,
            LeaderComponent.LOGIC_RELEVANCE: self.logic_relevance_weight,
            LeaderComponent.FUNDAMENTAL_QUALITY: self.fundamental_quality_weight,
        }

    @property
    def config_hash(self) -> str:
        return stable_leader_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class LeaderUpstreamIdentity:
    """Exact four-feature and mainline identities used for one stock."""

    mainline_model_version: str
    mainline_result_hash: str
    stock_feature_version: str
    stock_config_hash: str
    stock_input_hash: str
    stock_result_hash: str
    tradeability_feature_version: str
    tradeability_config_hash: str
    tradeability_rule_version: str
    tradeability_rule_hash: str
    tradeability_result_hash: str
    fundamental_feature_version: str
    fundamental_config_hash: str
    fundamental_data_version: str
    fundamental_input_hash: str
    fundamental_result_hash: str

    def __post_init__(self) -> None:
        versions = (
            self.mainline_model_version,
            self.stock_feature_version,
            self.tradeability_feature_version,
            self.tradeability_rule_version,
            self.fundamental_feature_version,
            self.fundamental_data_version,
        )
        if any(not value.strip() for value in versions):
            raise ValueError("leader upstream versions must be non-empty")
        hashes = (
            self.mainline_result_hash,
            self.stock_config_hash,
            self.stock_input_hash,
            self.stock_result_hash,
            self.tradeability_config_hash,
            self.tradeability_rule_hash,
            self.tradeability_result_hash,
            self.fundamental_config_hash,
            self.fundamental_input_hash,
            self.fundamental_result_hash,
        )
        for value in hashes:
            _hash(value, "leader upstream identity")


@dataclass(frozen=True, slots=True)
class LeaderComponentScore:
    """One normalized score and its exact weighted contribution."""

    component: LeaderComponent
    score: Decimal
    weight: Decimal
    contribution: Decimal

    def __post_init__(self) -> None:
        values = (self.score, self.weight, self.contribution)
        if any(not value.is_finite() for value in values):
            raise ValueError("leader component values must be finite")
        if not Decimal(0) <= self.score <= Decimal(100):
            raise ValueError("leader component score must be within 0..100")
        if not Decimal(0) <= self.weight <= Decimal(1):
            raise ValueError("leader component weight must be within 0..1")
        if self.contribution != self.score * self.weight:
            raise ValueError("leader component contribution must equal score times weight")


@dataclass(frozen=True, slots=True)
class LeaderEvidence:
    """Structured supporting or opposing score interpretation."""

    feature: str
    value: Decimal
    criterion: str
    side: LeaderEvidenceSide
    rationale: str

    def __post_init__(self) -> None:
        for name, value in (
            ("feature", self.feature),
            ("criterion", self.criterion),
            ("rationale", self.rationale),
        ):
            _non_empty(value, name)
        if not self.value.is_finite():
            raise ValueError("leader evidence value must be finite")


@dataclass(frozen=True, slots=True)
class LeaderRisk:
    """One auditable applied risk penalty."""

    code: LeaderRiskCode
    penalty: Decimal
    rationale: str

    def __post_init__(self) -> None:
        if not self.penalty.is_finite() or not Decimal(0) < self.penalty <= Decimal(100):
            raise ValueError("leader risk penalty must be finite within (0, 100]")
        _non_empty(self.rationale, "risk rationale")


@dataclass(frozen=True, slots=True)
class LeaderResult:
    """Ranked leader with classification, evidence, risks, and candidate gate."""

    instrument_id: str
    industry_id: str
    mainline_state: MainlineState
    overall_rank: int
    theme_rank: int
    leader_type: LeaderType
    gross_score: Decimal
    risk_penalty: Decimal
    score: Decimal
    components: tuple[LeaderComponentScore, ...]
    supporting_evidence: tuple[LeaderEvidence, ...]
    counter_evidence: tuple[LeaderEvidence, ...]
    risks: tuple[LeaderRisk, ...]
    candidate_eligible: bool
    ineligibility_reasons: tuple[str, ...]
    observation_conditions: tuple[str, ...]
    invalidations: tuple[str, ...]
    upstream_identity: LeaderUpstreamIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "industry_id", _non_empty(self.industry_id, "industry_id"))
        if self.mainline_state not in {MainlineState.CONFIRMED, MainlineState.CROWDED}:
            raise ValueError("ranked leaders require CONFIRMED or CROWDED mainlines")
        if self.overall_rank <= 0 or self.theme_rank <= 0:
            raise ValueError("leader ranks must be positive")
        scores = (self.gross_score, self.risk_penalty, self.score)
        if any(
            not value.is_finite() or not Decimal(0) <= value <= Decimal(100) for value in scores
        ):
            raise ValueError("leader scores and penalty must be finite within 0..100")
        if self.score != max(Decimal(0), self.gross_score - self.risk_penalty):
            raise ValueError("leader score must deduct the applied risk penalty")
        if tuple(item.component for item in self.components) != tuple(LeaderComponent):
            raise ValueError("leader components must use canonical order")
        if sum((item.weight for item in self.components), Decimal(0)) != 1:
            raise ValueError("leader component weights must sum to one")
        if sum((item.contribution for item in self.components), Decimal(0)) != self.gross_score:
            raise ValueError("gross leader score must equal component contributions")
        if sum((item.penalty for item in self.risks), Decimal(0)) != self.risk_penalty:
            raise ValueError("leader risk penalty must equal applied risk records")
        candidate_type = self.leader_type in {LeaderType.TREND, LeaderType.CAPACITY}
        if self.candidate_eligible and (not candidate_type or self.ineligibility_reasons):
            raise ValueError("candidate eligibility requires a tradeable trend or capacity leader")
        if not self.candidate_eligible and not self.ineligibility_reasons:
            raise ValueError("ineligible leader requires structured reasons")
        if not self.observation_conditions or not self.invalidations:
            raise ValueError("leader output requires observation and invalidation conditions")
        if any(not value.strip() for value in (*self.observation_conditions, *self.invalidations)):
            raise ValueError("leader conditions must be non-empty")


@dataclass(frozen=True, slots=True)
class LeaderExclusion:
    """A supplied stock excluded before cross-sectional leader ranking."""

    instrument_id: str
    industry_id: str | None
    code: LeaderExclusionCode
    reason: str
    stock_result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        if self.industry_id is not None and not self.industry_id.strip():
            raise ValueError("excluded industry_id must be non-empty when present")
        object.__setattr__(self, "reason", _non_empty(self.reason, "exclusion reason"))
        _hash(self.stock_result_hash, "excluded stock_result_hash")


@dataclass(frozen=True, slots=True)
class LeaderSnapshot:
    """Deterministic cross-sectional leader ranking for one mainline snapshot."""

    session_date: date
    as_of: datetime
    data_version: str
    classification_version: str
    industry_level: int
    feature_version: str
    config_hash: str
    mainline_result_hash: str
    leaders: tuple[LeaderResult, ...]
    exclusions: tuple[LeaderExclusion, ...]
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        versions = (self.data_version, self.classification_version, self.feature_version)
        if any(not value.strip() for value in versions):
            raise ValueError("leader snapshot versions must be non-empty")
        if self.industry_level not in (1, 2, 3):
            raise ValueError("leader snapshot industry_level must be 1, 2, or 3")
        for value in (
            self.config_hash,
            self.mainline_result_hash,
            self.input_hash,
            self.result_hash,
        ):
            _hash(value, "leader snapshot identity")
        leader_ids = tuple(item.instrument_id for item in self.leaders)
        exclusion_ids = tuple(item.instrument_id for item in self.exclusions)
        if len(set((*leader_ids, *exclusion_ids))) != len(leader_ids) + len(exclusion_ids):
            raise ValueError("leader and exclusion instrument identifiers must be unique")
        if tuple(item.overall_rank for item in self.leaders) != tuple(
            range(1, len(self.leaders) + 1)
        ):
            raise ValueError("leader overall ranks must be contiguous and ordered")

    @property
    def candidates(self) -> tuple[LeaderResult, ...]:
        return tuple(item for item in self.leaders if item.candidate_eligible)

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "input_hash": self.input_hash,
            "mainline_result_hash": self.mainline_result_hash,
            "result_hash": self.result_hash,
            "session_date": self.session_date.isoformat(),
        }


__all__ = [
    "LeaderComponent",
    "LeaderComponentScore",
    "LeaderConfig",
    "LeaderEvidence",
    "LeaderEvidenceSide",
    "LeaderExclusion",
    "LeaderExclusionCode",
    "LeaderInputError",
    "LeaderResult",
    "LeaderRisk",
    "LeaderRiskCode",
    "LeaderSnapshot",
    "LeaderType",
    "LeaderUpstreamIdentity",
    "stable_leader_hash",
]
