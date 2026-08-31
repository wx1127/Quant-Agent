"""Versioned contracts for deterministic market-regime classification."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal


def _canonical(value: Any) -> Any:
    """Convert config values into a stable JSON-compatible representation."""

    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        ensure_aware(value)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""

    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class MarketRegime(StrEnum):
    """The five baseline market states exposed to the agent harness."""

    UPTREND = "UPTREND"
    RANGE_STRONG = "RANGE_STRONG"
    DIVERGENT = "DIVERGENT"
    HIGH_LEVEL_DIVERGENCE = "DIVERGENT"
    DOWNTREND = "DOWNTREND"
    BOTTOM_RECOVERY = "BOTTOM_RECOVERY"

    @property
    def display_name(self) -> str:
        """Return the stable Chinese label used in human-facing reports."""

        return {
            MarketRegime.UPTREND: "上升",
            MarketRegime.RANGE_STRONG: "震荡偏强",
            MarketRegime.DIVERGENT: "高位分化",
            MarketRegime.DOWNTREND: "下跌",
            MarketRegime.BOTTOM_RECOVERY: "底部修复",
        }[self]


class EvidenceSide(StrEnum):
    """Whether an evaluated rule supports or opposes the selected state."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"


class ConfidenceMeaning(StrEnum):
    """Machine-readable semantics for the confidence field."""

    RULE_EVIDENCE_CONSISTENCY = "RULE_EVIDENCE_CONSISTENCY"


class InsufficientRegimeData(ValueError):
    """Raised when critical point-in-time inputs are absent or unsafe."""


class RegimeInputMismatch(ValueError):
    """Raised when trend and breadth snapshots are not version/time aligned."""


@dataclass(frozen=True, slots=True)
class RegimeClassifierConfig:
    """Every scoring, mapping, override, and budget threshold for the baseline model."""

    version: str = "market-regime-v1"
    trend_weight: Decimal = Decimal("0.30")
    breadth_weight: Decimal = Decimal("0.25")
    turnover_weight: Decimal = Decimal("0.15")
    new_high_low_weight: Decimal = Decimal("0.20")
    downside_risk_weight: Decimal = Decimal("0.10")
    neutral_component_score: Decimal = Decimal("50")
    uptrend_min_score: Decimal = Decimal("70")
    range_strong_min_score: Decimal = Decimal("55")
    divergent_min_score: Decimal = Decimal("40")
    downside_volatility_ceiling: Decimal = Decimal("0.04")
    minimum_breadth_coverage: Decimal = Decimal("0.95")
    minimum_return_denominator: int = 1
    minimum_moving_average_denominator: int = 1
    minimum_high_low_denominator: int = 1
    minimum_turnover_history_count: int = 1
    minimum_downside_observation_count: int = 1
    divergence_trend_min: Decimal = Decimal("20")
    divergence_above_average_max: Decimal = Decimal("0.45")
    divergence_high_low_spread_max: Decimal = Decimal("0")
    divergence_min_weak_signals: int = 1
    bottom_short_window: int = 20
    bottom_long_window: int = 120
    bottom_long_trend_max: Decimal = Decimal("-15")
    bottom_short_trend_min: Decimal = Decimal("5")
    bottom_advance_decline_min: Decimal = Decimal("0.52")
    bottom_above_average_min: Decimal = Decimal("0.45")
    bottom_new_low_max: Decimal = Decimal("0.08")
    bottom_min_confirming_signals: int = 5
    uptrend_risk_budget_max: Decimal = Decimal("0.80")
    range_strong_risk_budget_max: Decimal = Decimal("0.50")
    divergent_risk_budget_max: Decimal = Decimal("0.35")
    downtrend_risk_budget_max: Decimal = Decimal("0.15")
    bottom_recovery_risk_budget_max: Decimal = Decimal("0.30")
    uptrend_evidence_weights: tuple[Decimal, ...] = (
        Decimal("0.30"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.15"),
        Decimal("0.15"),
    )
    range_strong_evidence_weights: tuple[Decimal, ...] = (
        Decimal("0.30"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.15"),
        Decimal("0.15"),
    )
    divergent_evidence_weights: tuple[Decimal, ...] = (
        Decimal("0.20"),
        Decimal("0.25"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.15"),
    )
    downtrend_evidence_weights: tuple[Decimal, ...] = (
        Decimal("0.30"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.15"),
        Decimal("0.15"),
    )
    bottom_recovery_evidence_weights: tuple[Decimal, ...] = (
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.20"),
        Decimal("0.20"),
    )

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("regime classifier version must be non-empty")
        for value in asdict(self).values():
            decimal_values = value if isinstance(value, tuple) else (value,)
            for candidate in decimal_values:
                if isinstance(candidate, Decimal):
                    canonical_decimal(candidate, field_name="regime numeric parameter")
        weights = (
            self.trend_weight,
            self.breadth_weight,
            self.turnover_weight,
            self.new_high_low_weight,
            self.downside_risk_weight,
        )
        if any(weight < 0 or not weight.is_finite() for weight in weights):
            raise ValueError("regime component weights must be finite and non-negative")
        if sum(weights, Decimal(0)) != Decimal(1):
            raise ValueError("regime component weights must sum to one")
        score_thresholds = (
            self.divergent_min_score,
            self.range_strong_min_score,
            self.uptrend_min_score,
        )
        if (
            not Decimal(0)
            <= score_thresholds[0]
            < score_thresholds[1]
            < score_thresholds[2]
            <= Decimal(100)
        ):
            raise ValueError("regime score thresholds must be ordered within 0..100")
        if not Decimal(0) <= self.neutral_component_score <= Decimal(100):
            raise ValueError("neutral component score must be within 0..100")
        ratios = (
            self.minimum_breadth_coverage,
            self.divergence_above_average_max,
            self.bottom_advance_decline_min,
            self.bottom_above_average_min,
            self.bottom_new_low_max,
        )
        if any(value < 0 or value > 1 or not value.is_finite() for value in ratios):
            raise ValueError("regime ratio thresholds must be finite and within 0..1")
        if not Decimal(-1) <= self.divergence_high_low_spread_max <= Decimal(1):
            raise ValueError("divergence high/low spread threshold must be within -1..1")
        if (
            self.downside_volatility_ceiling <= 0
            or not self.downside_volatility_ceiling.is_finite()
        ):
            raise ValueError("downside volatility ceiling must be finite and positive")
        minimum_counts = (
            self.minimum_return_denominator,
            self.minimum_moving_average_denominator,
            self.minimum_high_low_denominator,
            self.minimum_turnover_history_count,
            self.minimum_downside_observation_count,
        )
        if any(value < 1 for value in minimum_counts):
            raise ValueError("critical breadth minimum counts must be positive")
        if not 1 <= self.divergence_min_weak_signals <= 2:
            raise ValueError("divergence_min_weak_signals must be one or two")
        if self.bottom_short_window < 2 or self.bottom_long_window <= self.bottom_short_window:
            raise ValueError("bottom windows must be ordered and at least two")
        if not 1 <= self.bottom_min_confirming_signals <= 5:
            raise ValueError("bottom_min_confirming_signals must be within one to five")
        trend_thresholds = (
            self.divergence_trend_min,
            self.bottom_long_trend_max,
            self.bottom_short_trend_min,
        )
        if any(value < -100 or value > 100 or not value.is_finite() for value in trend_thresholds):
            raise ValueError("trend thresholds must be finite and within -100..100")
        budgets = (
            self.uptrend_risk_budget_max,
            self.range_strong_risk_budget_max,
            self.divergent_risk_budget_max,
            self.downtrend_risk_budget_max,
            self.bottom_recovery_risk_budget_max,
        )
        if any(value < 0 or value > 1 or not value.is_finite() for value in budgets):
            raise ValueError("risk budget caps must be finite and within 0..1")
        evidence_weights = (
            self.uptrend_evidence_weights,
            self.range_strong_evidence_weights,
            self.divergent_evidence_weights,
            self.downtrend_evidence_weights,
            self.bottom_recovery_evidence_weights,
        )
        if any(len(values) != 5 for values in evidence_weights):
            raise ValueError("each regime requires exactly five evidence weights")
        if any(
            any(value <= 0 or not value.is_finite() for value in values)
            or sum(values, Decimal(0)) != Decimal(1)
            for values in evidence_weights
        ):
            raise ValueError("each regime's evidence weights must be positive and sum to one")

    @property
    def config_hash(self) -> str:
        """Hash the version and all parameters, including state budget caps."""

        return stable_hash(asdict(self))

    def risk_budget_for(self, regime: MarketRegime) -> Decimal:
        """Return the configured maximum risk budget for one state."""

        return {
            MarketRegime.UPTREND: self.uptrend_risk_budget_max,
            MarketRegime.RANGE_STRONG: self.range_strong_risk_budget_max,
            MarketRegime.DIVERGENT: self.divergent_risk_budget_max,
            MarketRegime.DOWNTREND: self.downtrend_risk_budget_max,
            MarketRegime.BOTTOM_RECOVERY: self.bottom_recovery_risk_budget_max,
        }[regime]

    def evidence_weights_for(self, regime: MarketRegime) -> tuple[Decimal, ...]:
        """Return versioned confidence-rule weights for one state."""

        return {
            MarketRegime.UPTREND: self.uptrend_evidence_weights,
            MarketRegime.RANGE_STRONG: self.range_strong_evidence_weights,
            MarketRegime.DIVERGENT: self.divergent_evidence_weights,
            MarketRegime.DOWNTREND: self.downtrend_evidence_weights,
            MarketRegime.BOTTOM_RECOVERY: self.bottom_recovery_evidence_weights,
        }[regime]


@dataclass(frozen=True, slots=True)
class RegimeComponentScores:
    """Auditable normalized inputs and their weighted environment-score terms."""

    trend: Decimal
    breadth: Decimal
    turnover: Decimal
    new_high_low: Decimal
    downside_risk: Decimal
    weighted_trend: Decimal
    weighted_breadth: Decimal
    weighted_turnover: Decimal
    weighted_new_high_low: Decimal
    weighted_downside_risk: Decimal

    def __post_init__(self) -> None:
        components = (
            self.trend,
            self.breadth,
            self.turnover,
            self.new_high_low,
            self.downside_risk,
        )
        weighted = (
            self.weighted_trend,
            self.weighted_breadth,
            self.weighted_turnover,
            self.weighted_new_high_low,
            self.weighted_downside_risk,
        )
        if any(value < 0 or value > 100 or not value.is_finite() for value in components):
            raise ValueError("regime component scores must be finite and within 0..100")
        if any(value < 0 or value > 100 or not value.is_finite() for value in weighted):
            raise ValueError("weighted regime terms must be finite and within 0..100")

    @property
    def total(self) -> Decimal:
        """Return the 0..100 market environment score."""

        return sum(
            (
                self.weighted_trend,
                self.weighted_breadth,
                self.weighted_turnover,
                self.weighted_new_high_low,
                self.weighted_downside_risk,
            ),
            Decimal(0),
        )


@dataclass(frozen=True, slots=True)
class RegimeEvidence:
    """One evaluated rule contributing to classification consistency."""

    feature: str
    value: Decimal
    criterion: str
    rule_weight: Decimal
    contribution: Decimal
    side: EvidenceSide
    rationale: str

    def __post_init__(self) -> None:
        if not self.feature.strip() or not self.criterion.strip() or not self.rationale.strip():
            raise ValueError("regime evidence text fields must be non-empty")
        if not self.value.is_finite():
            raise ValueError("regime evidence value must be finite")
        if self.rule_weight <= 0 or self.rule_weight > 1 or not self.rule_weight.is_finite():
            raise ValueError("regime evidence rule_weight must be within (0, 1]")
        if self.contribution < 0 or not self.contribution.is_finite():
            raise ValueError("regime evidence contribution must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RegimeInputIdentity:
    """Exact upstream feature identities bound into a regime result."""

    trend_feature_version: str
    trend_config_hash: str
    trend_cache_key: str
    breadth_feature_version: str
    breadth_config_hash: str
    breadth_cache_key: str

    def __post_init__(self) -> None:
        if not self.trend_feature_version.strip() or not self.breadth_feature_version.strip():
            raise ValueError("upstream feature versions must be non-empty")
        hashes = (
            self.trend_config_hash,
            self.trend_cache_key,
            self.breadth_config_hash,
            self.breadth_cache_key,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("upstream identities must contain SHA-256 hashes")


@dataclass(frozen=True, slots=True)
class MarketRegimeResult:
    """Harness-ready, point-in-time and model-version-bound regime decision."""

    as_of: datetime
    session_date: date
    data_version: str
    regime: MarketRegime
    score: Decimal
    confidence: Decimal
    risk_budget_max: Decimal
    components: RegimeComponentScores
    evidence: tuple[RegimeEvidence, ...]
    counter_evidence: tuple[RegimeEvidence, ...]
    invalidations: tuple[str, ...]
    model_version: str
    config_hash: str
    input_identity: RegimeInputIdentity
    input_hash: str
    result_hash: str
    confidence_meaning: ConfidenceMeaning = field(
        default=ConfidenceMeaning.RULE_EVIDENCE_CONSISTENCY,
        init=False,
    )
    confidence_definition: str = field(
        default="规则证据一致度; 不是市场上涨概率",
        init=False,
    )

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not self.data_version.strip() or not self.model_version.strip():
            raise ValueError("regime result versions must be non-empty")
        if self.score < 0 or self.score > 100 or not self.score.is_finite():
            raise ValueError("regime score must be finite and within 0..100")
        if self.confidence < 0 or self.confidence > 1 or not self.confidence.is_finite():
            raise ValueError("regime confidence must be finite and within 0..1")
        if self.risk_budget_max < 0 or self.risk_budget_max > 1:
            raise ValueError("regime risk budget cap must be within 0..1")
        if not self.evidence:
            raise ValueError("regime result requires supporting evidence")
        if any(item.side is not EvidenceSide.SUPPORTING for item in self.evidence):
            raise ValueError("evidence must contain only supporting rules")
        if any(item.side is not EvidenceSide.OPPOSING for item in self.counter_evidence):
            raise ValueError("counter_evidence must contain only opposing rules")
        if not self.invalidations or any(not value.strip() for value in self.invalidations):
            raise ValueError("regime result requires non-empty invalidation conditions")
        if any(len(value) != 64 for value in (self.config_hash, self.input_hash, self.result_hash)):
            raise ValueError("regime result hashes must be SHA-256 digests")
        if self.components.total != self.score:
            raise ValueError("regime score must equal the sum of weighted component terms")

    @property
    def raw_score(self) -> Decimal:
        """Compatibility alias for the development-guide output contract."""

        return self.score

    @property
    def max_risk_budget(self) -> Decimal:
        """Compatibility alias for the development-guide output contract."""

        return self.risk_budget_max

    @property
    def supporting_evidence(self) -> tuple[RegimeEvidence, ...]:
        """Return rules consistent with the selected classification."""

        return self.evidence

    @property
    def opposing_evidence(self) -> tuple[RegimeEvidence, ...]:
        """Return rules inconsistent with the selected classification."""

        return self.counter_evidence

    @property
    def confidence_is_probability(self) -> bool:
        """Confidence is evidence consistency, never an upward-price probability."""

        return False

    def identity_payload(self) -> dict[str, str]:
        """Return the stable identities needed to reproduce this decision."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "model_version": self.model_version,
            "result_hash": self.result_hash,
        }
