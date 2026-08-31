"""Versioned contracts for point-in-time fundamental quality and risk features."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


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


def stable_fundamental_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for fundamental feature identities."""

    encoded = json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


class FundamentalQualityInputError(ValueError):
    """Raised when financial records cannot be combined without unsafe assumptions."""


class FundamentalQualityStatus(StrEnum):
    """Coverage status after conservative missing-component handling."""

    READY = "READY"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class FundamentalComponent(StrEnum):
    """Provider-neutral quality dimensions used by the baseline model."""

    PROFITABILITY = "PROFITABILITY"
    OPERATING_CASH_FLOW = "OPERATING_CASH_FLOW"
    LEVERAGE = "LEVERAGE"
    GROWTH_STABILITY = "GROWTH_STABILITY"


class FundamentalEvidenceSide(StrEnum):
    """Whether one component supports or opposes fundamental quality."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"


class FundamentalFlagKind(StrEnum):
    """Whether a machine-readable flag denotes missing data or an observed anomaly."""

    MISSING = "MISSING"
    ANOMALY = "ANOMALY"


class FundamentalFlagCode(StrEnum):
    """Stable missing and risk flag identifiers."""

    MISSING_PROFITABILITY = "MISSING_PROFITABILITY"
    MISSING_OPERATING_CASH_FLOW = "MISSING_OPERATING_CASH_FLOW"
    MISSING_LEVERAGE = "MISSING_LEVERAGE"
    MISSING_GROWTH_HISTORY = "MISSING_GROWTH_HISTORY"
    NEGATIVE_PROFITABILITY = "NEGATIVE_PROFITABILITY"
    NEGATIVE_OPERATING_CASH_FLOW = "NEGATIVE_OPERATING_CASH_FLOW"
    HIGH_LEVERAGE = "HIGH_LEVERAGE"
    UNSTABLE_GROWTH = "UNSTABLE_GROWTH"
    EXTREME_GROWTH = "EXTREME_GROWTH"
    CASH_EARNINGS_MISMATCH = "CASH_EARNINGS_MISMATCH"


@dataclass(frozen=True, slots=True)
class FundamentalMetricMapping:
    """Versioned provider metric names for each provider-neutral component."""

    version: str = "fundamental-metric-map-tushare-v1"
    profitability: tuple[str, ...] = ("roe", "roic")
    operating_cash_flow: tuple[str, ...] = ("ocfps",)
    leverage: tuple[str, ...] = ("debt_to_assets",)
    growth: tuple[str, ...] = ("netprofit_yoy", "dt_netprofit_yoy")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "mapping version"))
        groups = (
            self.profitability,
            self.operating_cash_flow,
            self.leverage,
            self.growth,
        )
        if any(
            not group or any(not value.strip() for value in group) or len(set(group)) != len(group)
            for group in groups
        ):
            raise ValueError("metric mapping groups must be non-empty and unique")
        flattened = tuple(value for group in groups for value in group)
        if len(set(flattened)) != len(flattened):
            raise ValueError("one provider metric cannot map to multiple fundamental components")

    @property
    def mapping_hash(self) -> str:
        """Return the stable identity of all provider-name choices."""

        return stable_fundamental_hash(asdict(self))

    @property
    def all_metric_names(self) -> tuple[str, ...]:
        """Return every configured provider metric name in semantic priority order."""

        return (
            *self.profitability,
            *self.operating_cash_flow,
            *self.leverage,
            *self.growth,
        )


@dataclass(frozen=True, slots=True)
class FundamentalQualityConfig:
    """Versioned thresholds, weights, missing policy, and risk penalties."""

    version: str = "fundamental-quality-v1"
    metric_mapping: FundamentalMetricMapping = field(default_factory=FundamentalMetricMapping)
    growth_window_periods: int = 4
    minimum_growth_periods: int = 3
    profitability_weight: Decimal = Decimal("0.30")
    operating_cash_flow_weight: Decimal = Decimal("0.25")
    leverage_weight: Decimal = Decimal("0.20")
    growth_stability_weight: Decimal = Decimal("0.25")
    profitability_zero_score: Decimal = Decimal("0")
    profitability_full_score: Decimal = Decimal("15")
    cash_flow_zero_score: Decimal = Decimal("0")
    cash_flow_full_score: Decimal = Decimal("1")
    leverage_full_score_max: Decimal = Decimal("40")
    leverage_zero_score_min: Decimal = Decimal("70")
    growth_range_full_score_max: Decimal = Decimal("10")
    growth_range_zero_score_min: Decimal = Decimal("40")
    evidence_support_score_min: Decimal = Decimal("50")
    extreme_growth_absolute_threshold: Decimal = Decimal("100")
    missing_component_penalty: Decimal = Decimal("10")
    negative_profitability_penalty: Decimal = Decimal("15")
    negative_cash_flow_penalty: Decimal = Decimal("15")
    high_leverage_penalty: Decimal = Decimal("20")
    unstable_growth_penalty: Decimal = Decimal("15")
    extreme_growth_penalty: Decimal = Decimal("10")
    cash_earnings_mismatch_penalty: Decimal = Decimal("10")
    maximum_risk_penalty: Decimal = Decimal("100")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "config version"))
        if (
            not isinstance(self.growth_window_periods, int)
            or isinstance(self.growth_window_periods, bool)
            or self.growth_window_periods < 2
        ):
            raise ValueError("growth_window_periods must be an integer of at least two")
        if not 2 <= self.minimum_growth_periods <= self.growth_window_periods:
            raise ValueError("minimum_growth_periods must be within the growth window")
        weights = self.component_weights
        decimal_values = (
            *weights.values(),
            self.profitability_zero_score,
            self.profitability_full_score,
            self.cash_flow_zero_score,
            self.cash_flow_full_score,
            self.leverage_full_score_max,
            self.leverage_zero_score_min,
            self.growth_range_full_score_max,
            self.growth_range_zero_score_min,
            self.evidence_support_score_min,
            self.extreme_growth_absolute_threshold,
            *self.penalties,
            self.maximum_risk_penalty,
        )
        for value in decimal_values:
            canonical_decimal(value, field_name="fundamental quality numeric parameter")
        if any(value < 0 for value in weights.values()) or sum(
            weights.values(), Decimal(0)
        ) != Decimal(1):
            raise ValueError("fundamental quality component weights must sum to one")
        if not self.profitability_zero_score < self.profitability_full_score:
            raise ValueError("profitability scoring thresholds must be increasing")
        if not self.cash_flow_zero_score < self.cash_flow_full_score:
            raise ValueError("cash-flow scoring thresholds must be increasing")
        if not self.leverage_full_score_max < self.leverage_zero_score_min:
            raise ValueError("leverage scoring thresholds must be increasing")
        if not self.growth_range_full_score_max < self.growth_range_zero_score_min:
            raise ValueError("growth-range scoring thresholds must be increasing")
        if not Decimal(0) <= self.evidence_support_score_min <= Decimal(100):
            raise ValueError("evidence support threshold must be within 0..100")
        if self.extreme_growth_absolute_threshold <= 0:
            raise ValueError("extreme growth threshold must be positive")
        if any(value < 0 for value in self.penalties):
            raise ValueError("fundamental risk penalties cannot be negative")
        if not Decimal(0) < self.maximum_risk_penalty <= Decimal(100):
            raise ValueError("maximum risk penalty must be within (0, 100]")

    @property
    def component_weights(self) -> dict[FundamentalComponent, Decimal]:
        """Return the exact provider-neutral score weights."""

        return {
            FundamentalComponent.PROFITABILITY: self.profitability_weight,
            FundamentalComponent.OPERATING_CASH_FLOW: self.operating_cash_flow_weight,
            FundamentalComponent.LEVERAGE: self.leverage_weight,
            FundamentalComponent.GROWTH_STABILITY: self.growth_stability_weight,
        }

    @property
    def penalties(self) -> tuple[Decimal, ...]:
        """Return every configured per-flag risk penalty."""

        return (
            self.missing_component_penalty,
            self.negative_profitability_penalty,
            self.negative_cash_flow_penalty,
            self.high_leverage_penalty,
            self.unstable_growth_penalty,
            self.extreme_growth_penalty,
            self.cash_earnings_mismatch_penalty,
        )

    @property
    def config_hash(self) -> str:
        """Return a stable hash including the metric-name mapping."""

        return stable_fundamental_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class FundamentalQualityRequest:
    """One instrument and point-in-time boundary for feature evaluation."""

    instrument_id: str
    as_of: datetime
    data_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        ensure_aware(self.as_of)

    def fingerprint_payload(self) -> dict[str, str]:
        """Return canonical request fields for cache identities."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "instrument_id": self.instrument_id,
        }


@dataclass(frozen=True, slots=True)
class FundamentalSourceReference:
    """Exact financial record supporting one component or risk flag."""

    metric_name: str
    metric_value: Decimal
    report_period: date
    announced_at: datetime
    available_at: datetime
    provider_revision: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_name", _non_empty(self.metric_name, "metric_name"))
        object.__setattr__(
            self,
            "provider_revision",
            _non_empty(self.provider_revision, "provider_revision"),
        )
        object.__setattr__(self, "source", _non_empty(self.source, "source"))
        canonical_decimal(self.metric_value, field_name="fundamental source metric value")
        ensure_aware(self.announced_at)
        ensure_aware(self.available_at)
        if self.report_period > self.announced_at.astimezone(SHANGHAI_TZ).date():
            raise ValueError("fundamental report_period cannot be after announced_at")
        if self.available_at < self.announced_at:
            raise ValueError("fundamental available_at cannot precede announced_at")

    def fingerprint_payload(self) -> dict[str, str]:
        """Return canonical lineage fields used in the selected-input hash."""

        return {
            "announced_at": self.announced_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "metric_name": self.metric_name,
            "metric_value": canonical_decimal(self.metric_value),
            "provider_revision": self.provider_revision,
            "report_period": self.report_period.isoformat(),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FundamentalComponentResult:
    """One normalized component with conservative missing contribution semantics."""

    component: FundamentalComponent
    raw_value: Decimal | None
    normalized_score: Decimal | None
    configured_weight: Decimal
    contribution: Decimal
    sources: tuple[FundamentalSourceReference, ...]
    missing_reason: str | None

    def __post_init__(self) -> None:
        canonical_decimal(self.configured_weight, field_name="component weight")
        canonical_decimal(self.contribution, field_name="component contribution")
        if not Decimal(0) <= self.configured_weight <= Decimal(1):
            raise ValueError("component weight must be within 0..1")
        if self.raw_value is not None:
            canonical_decimal(self.raw_value, field_name="component raw value")
        if self.normalized_score is None:
            if self.contribution != 0 or not self.missing_reason:
                raise ValueError("missing component must have zero contribution and a reason")
        else:
            canonical_decimal(self.normalized_score, field_name="component score")
            if not Decimal(0) <= self.normalized_score <= Decimal(100):
                raise ValueError("component score must be within 0..100")
            if self.raw_value is None or not self.sources or self.missing_reason is not None:
                raise ValueError("scored component requires value, sources, and no missing reason")
            if self.contribution != self.normalized_score * self.configured_weight:
                raise ValueError("component contribution must equal score times weight")

    @property
    def missing(self) -> bool:
        """Return whether this component was conservatively scored as unavailable."""

        return self.normalized_score is None


@dataclass(frozen=True, slots=True)
class FundamentalEvidence:
    """Structured supporting or opposing interpretation of one component."""

    component: FundamentalComponent
    side: FundamentalEvidenceSide
    observed_value: Decimal | None
    normalized_score: Decimal | None
    criterion: str
    rationale: str
    sources: tuple[FundamentalSourceReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion", _non_empty(self.criterion, "criterion"))
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "rationale"))
        if self.observed_value is not None:
            canonical_decimal(self.observed_value, field_name="evidence observed value")
        if self.normalized_score is not None:
            canonical_decimal(self.normalized_score, field_name="evidence normalized score")
            if not Decimal(0) <= self.normalized_score <= Decimal(100):
                raise ValueError("evidence normalized score must be within 0..100")


@dataclass(frozen=True, slots=True)
class FundamentalFlag:
    """Machine-readable missing or anomaly flag with an explicit risk penalty."""

    code: FundamentalFlagCode
    kind: FundamentalFlagKind
    component: FundamentalComponent | None
    penalty: Decimal
    rationale: str
    sources: tuple[FundamentalSourceReference, ...]

    def __post_init__(self) -> None:
        canonical_decimal(self.penalty, field_name="fundamental flag penalty")
        if self.penalty < 0 or self.penalty > 100:
            raise ValueError("fundamental flag penalty must be within 0..100")
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "flag rationale"))
        if self.kind is FundamentalFlagKind.ANOMALY and not self.sources:
            raise ValueError("observed anomaly flags require financial source records")


@dataclass(frozen=True, slots=True)
class FundamentalQualitySnapshot:
    """PIT quality score, risk penalty, evidence, lineage, and stable identities."""

    instrument_id: str
    as_of: datetime
    data_version: str
    feature_version: str
    metric_mapping_version: str
    metric_mapping_hash: str
    config_hash: str
    status: FundamentalQualityStatus
    quality_score: Decimal
    risk_penalty: Decimal
    adjusted_quality_score: Decimal
    components: tuple[FundamentalComponentResult, ...]
    supporting_evidence: tuple[FundamentalEvidence, ...]
    counter_evidence: tuple[FundamentalEvidence, ...]
    flags: tuple[FundamentalFlag, ...]
    selected_sources: tuple[FundamentalSourceReference, ...]
    input_hash: str
    cache_key: str
    result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        ensure_aware(self.as_of)
        versions = (self.data_version, self.feature_version, self.metric_mapping_version)
        if any(not value.strip() for value in versions):
            raise ValueError("fundamental quality versions must be non-empty")
        for value, name in (
            (self.metric_mapping_hash, "metric_mapping_hash"),
            (self.config_hash, "config_hash"),
            (self.input_hash, "input_hash"),
            (self.cache_key, "cache_key"),
            (self.result_hash, "result_hash"),
        ):
            _validate_hash(value, name)
        scores = (self.quality_score, self.risk_penalty, self.adjusted_quality_score)
        if any(not value.is_finite() or value < 0 or value > 100 for value in scores):
            raise ValueError("fundamental quality scores and penalty must be within 0..100")
        if self.adjusted_quality_score != max(Decimal(0), self.quality_score - self.risk_penalty):
            raise ValueError("adjusted quality score must deduct the risk penalty")
        if tuple(item.component for item in self.components) != tuple(FundamentalComponent):
            raise ValueError("fundamental snapshot requires each component in canonical order")
        if sum((item.configured_weight for item in self.components), Decimal(0)) != Decimal(1):
            raise ValueError("fundamental component weights must sum to one")
        if sum((item.contribution for item in self.components), Decimal(0)) != self.quality_score:
            raise ValueError("quality score must equal component contributions")
        missing_count = sum(item.missing for item in self.components)
        expected_status = (
            FundamentalQualityStatus.READY
            if missing_count == 0
            else (
                FundamentalQualityStatus.INSUFFICIENT_DATA
                if missing_count == len(self.components)
                else FundamentalQualityStatus.PARTIAL
            )
        )
        if self.status is not expected_status:
            raise ValueError("fundamental quality status must reflect component coverage")
        if any(
            item.side is not FundamentalEvidenceSide.SUPPORTING for item in self.supporting_evidence
        ):
            raise ValueError("supporting evidence can contain only supporting items")
        if any(item.side is not FundamentalEvidenceSide.OPPOSING for item in self.counter_evidence):
            raise ValueError("counter evidence can contain only opposing items")
        if len(self.supporting_evidence) + len(self.counter_evidence) != len(self.components):
            raise ValueError("each fundamental component requires exactly one evidence item")
        source_payloads = tuple(
            tuple(sorted(item.fingerprint_payload().items())) for item in self.selected_sources
        )
        if len(set(source_payloads)) != len(source_payloads):
            raise ValueError("selected fundamental source records must be unique")

    @property
    def score(self) -> Decimal:
        """Compatibility alias for downstream candidate scorers."""

        return self.quality_score

    def identity_payload(self) -> dict[str, str]:
        """Return exact version and hash fields for manifests and downstream snapshots."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "cache_key": self.cache_key,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "input_hash": self.input_hash,
            "instrument_id": self.instrument_id,
            "metric_mapping_hash": self.metric_mapping_hash,
            "metric_mapping_version": self.metric_mapping_version,
            "result_hash": self.result_hash,
        }


__all__ = [
    "FundamentalComponent",
    "FundamentalComponentResult",
    "FundamentalEvidence",
    "FundamentalEvidenceSide",
    "FundamentalFlag",
    "FundamentalFlagCode",
    "FundamentalFlagKind",
    "FundamentalMetricMapping",
    "FundamentalQualityConfig",
    "FundamentalQualityInputError",
    "FundamentalQualityRequest",
    "FundamentalQualitySnapshot",
    "FundamentalQualityStatus",
    "FundamentalSourceReference",
    "stable_fundamental_hash",
]
