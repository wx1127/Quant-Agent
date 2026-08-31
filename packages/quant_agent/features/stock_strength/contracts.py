"""Immutable point-in-time contracts for stock trend and relative strength."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.data.domain import DailyBar, IndustryMembership
from quant_agent.features.core import canonical_decimal
from quant_agent.features.core.contracts import stable_feature_hash


class InsufficientStockData(ValueError):
    """Raised when a required stock or reference window is incomplete."""


class IndustryMembershipError(ValueError):
    """Raised when historical industry identity is unavailable or ambiguous."""


class SuspendedStockError(ValueError):
    """Raised when the configured policy rejects a suspended decision session."""


class MissingIndustryPolicy(StrEnum):
    """How scoring behaves when historical industry comparison is incomplete."""

    FAIL = "FAIL"
    DEGRADE = "DEGRADE"


class InsufficientHistoryPolicy(StrEnum):
    """Whether incomplete required history raises or returns an explicit status."""

    FAIL = "FAIL"
    RETURN_STATUS = "RETURN_STATUS"


class SuspensionPolicy(StrEnum):
    """Whether a currently suspended stock raises or returns an explicit status."""

    FAIL = "FAIL"
    RETURN_STATUS = "RETURN_STATUS"


class StockStrengthStatus(StrEnum):
    """Availability and degradation state for one stock feature snapshot."""

    READY = "READY"
    DEGRADED = "DEGRADED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SUSPENDED = "SUSPENDED"


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _finite(value: Decimal, field_name: str) -> Decimal:
    canonical_decimal(value, field_name=field_name)
    return value


@dataclass(frozen=True, slots=True)
class StockBarObservation:
    """One revisable adjusted stock bar wrapping the provider-neutral daily bar."""

    bar: DailyBar
    observed_at: datetime
    adjusted_close: Decimal
    is_suspended: bool
    revision: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.bar.instrument_id, self.bar.source, self.bar.version)
        ):
            raise ValueError("stock bar identifiers and versions must be non-empty")
        ensure_aware(self.observed_at)
        if self.observed_at.astimezone(SHANGHAI_TZ).date() != self.bar.trade_date:
            raise ValueError("stock observed_at must fall on bar.trade_date")
        if self.bar.available_at < self.observed_at:
            raise ValueError("stock available_at cannot precede observed_at")
        if self.adjusted_close <= 0 or not self.adjusted_close.is_finite():
            raise ValueError("adjusted_close must be finite and positive")
        numeric = (
            self.bar.open,
            self.bar.high,
            self.bar.low,
            self.bar.close,
            self.bar.volume,
            self.bar.turnover,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("stock bar numeric fields must be finite")
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))

    @property
    def instrument_id(self) -> str:
        return self.bar.instrument_id

    @property
    def trade_date(self) -> date:
        return self.bar.trade_date

    @property
    def available_at(self) -> datetime:
        return self.bar.available_at

    def fingerprint_payload(self) -> dict[str, str | bool]:
        """Return the exact fields used to bind a snapshot to this revision."""

        return {
            "adjusted_close": canonical_decimal(self.adjusted_close),
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "bar_close": canonical_decimal(self.bar.close),
            "bar_source": self.bar.source,
            "bar_version": self.bar.version,
            "instrument_id": self.instrument_id,
            "is_suspended": self.is_suspended,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "revision": self.revision,
            "trade_date": self.trade_date.isoformat(),
            "turnover": canonical_decimal(self.bar.turnover),
            "volume": canonical_decimal(self.bar.volume),
        }


@dataclass(frozen=True, slots=True)
class ReferenceBarObservation:
    """One revisable industry or benchmark close known at an explicit time."""

    reference_id: str
    trade_date: date
    observed_at: datetime
    available_at: datetime
    close: Decimal
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _non_empty(self.reference_id, "reference_id"),
        )
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        ensure_aware(self.observed_at)
        ensure_aware(self.available_at)
        if self.observed_at.astimezone(SHANGHAI_TZ).date() != self.trade_date:
            raise ValueError("reference observed_at must fall on trade_date")
        if self.available_at < self.observed_at:
            raise ValueError("reference available_at cannot precede observed_at")
        if self.close <= 0 or not self.close.is_finite():
            raise ValueError("reference close must be finite and positive")

    def fingerprint_payload(self) -> dict[str, str]:
        return {
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "close": canonical_decimal(self.close),
            "observed_at": self.observed_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "reference_id": self.reference_id,
            "revision": self.revision,
            "trade_date": self.trade_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class PointInTimeStockIndustryMembership:
    """One historical membership revision at exactly one classification level."""

    membership: IndustryMembership
    level: int
    available_at: datetime
    revision: str

    def __post_init__(self) -> None:
        membership = self.membership
        if any(
            not value.strip()
            for value in (
                membership.instrument_id,
                membership.industry_id,
                membership.source,
                membership.version,
            )
        ):
            raise ValueError("industry membership identifiers must be non-empty")
        if self.level not in (1, 2, 3):
            raise ValueError("industry membership level must be 1, 2, or 3")
        ensure_aware(self.available_at)
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))

    def fingerprint_payload(self) -> dict[str, str | int | None]:
        membership = self.membership
        return {
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "effective_from": membership.effective_from.isoformat(),
            "effective_to": membership.effective_to.isoformat()
            if membership.effective_to
            else None,
            "industry_id": membership.industry_id,
            "instrument_id": membership.instrument_id,
            "level": self.level,
            "revision": self.revision,
            "source": membership.source,
            "version": membership.version,
        }


@dataclass(frozen=True, slots=True)
class StockStrengthConfig:
    """Versioned windows, score normalization, and explicit degradation policy."""

    version: str = "stock-strength-v1"
    horizons: tuple[int, ...] = (5, 20, 60)
    horizon_weights: tuple[Decimal, ...] = (
        Decimal("0.2"),
        Decimal("0.3"),
        Decimal("0.5"),
    )
    moving_average_window: int = 20
    slope_lookback: int = 5
    breakout_window: int = 60
    breakout_hold_sessions: int = 3
    pullback_window: int = 20
    new_high_window: int = 60
    volume_window: int = 20
    maximum_historical_suspensions: int = 3
    absolute_return_weight: Decimal = Decimal("0.25")
    benchmark_relative_weight: Decimal = Decimal("0.25")
    industry_relative_weight: Decimal = Decimal("0.20")
    trend_quality_weight: Decimal = Decimal("0.20")
    volume_confirmation_weight: Decimal = Decimal("0.10")
    trend_position_weight: Decimal = Decimal("0.30")
    trend_slope_weight: Decimal = Decimal("0.25")
    trend_breakout_weight: Decimal = Decimal("0.15")
    trend_pullback_weight: Decimal = Decimal("0.15")
    trend_new_high_weight: Decimal = Decimal("0.15")
    return_score_scale: Decimal = Decimal("400")
    relative_score_scale: Decimal = Decimal("500")
    position_score_scale: Decimal = Decimal("500")
    slope_score_scale: Decimal = Decimal("1000")
    pullback_tolerance: Decimal = Decimal("0.10")
    new_high_distance_tolerance: Decimal = Decimal("0.10")
    volume_ratio_score_scale: Decimal = Decimal("100")
    missing_industry_policy: MissingIndustryPolicy = MissingIndustryPolicy.DEGRADE
    insufficient_history_policy: InsufficientHistoryPolicy = InsufficientHistoryPolicy.FAIL
    suspension_policy: SuspensionPolicy = SuspensionPolicy.RETURN_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("stock strength horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("stock strength horizons must be unique and ascending")
        if len(self.horizons) != len(self.horizon_weights):
            raise ValueError("each stock strength horizon requires one weight")
        windows = (
            self.moving_average_window,
            self.slope_lookback,
            self.breakout_window,
            self.breakout_hold_sessions,
            self.pullback_window,
            self.new_high_window,
            self.volume_window,
        )
        if min(windows) <= 0 or self.moving_average_window < 2:
            raise ValueError("stock strength windows must be positive and meaningful")
        if self.maximum_historical_suspensions < 0:
            raise ValueError("maximum_historical_suspensions cannot be negative")
        score_weights = self.score_weights
        trend_weights = self.trend_weights
        decimal_values = (
            *self.horizon_weights,
            *score_weights.values(),
            *trend_weights.values(),
            self.return_score_scale,
            self.relative_score_scale,
            self.position_score_scale,
            self.slope_score_scale,
            self.pullback_tolerance,
            self.new_high_distance_tolerance,
            self.volume_ratio_score_scale,
        )
        for value in decimal_values:
            _finite(value, "stock strength numeric parameter")
        if (
            any(value < 0 for value in self.horizon_weights)
            or sum(self.horizon_weights, Decimal(0)) <= 0
        ):
            raise ValueError("horizon weights must be non-negative with a positive sum")
        if (
            any(value < 0 for value in score_weights.values())
            or sum(score_weights.values(), Decimal(0)) != 1
        ):
            raise ValueError("stock strength component weights must sum to one")
        if (
            any(value < 0 for value in trend_weights.values())
            or sum(trend_weights.values(), Decimal(0)) != 1
        ):
            raise ValueError("trend quality weights must sum to one")
        scales = (
            self.return_score_scale,
            self.relative_score_scale,
            self.position_score_scale,
            self.slope_score_scale,
            self.pullback_tolerance,
            self.new_high_distance_tolerance,
            self.volume_ratio_score_scale,
        )
        if min(scales) <= 0:
            raise ValueError("stock strength score scales and tolerances must be positive")

    @property
    def score_weights(self) -> dict[str, Decimal]:
        return {
            "absolute_return": self.absolute_return_weight,
            "benchmark_relative": self.benchmark_relative_weight,
            "industry_relative": self.industry_relative_weight,
            "trend_quality": self.trend_quality_weight,
            "volume_confirmation": self.volume_confirmation_weight,
        }

    @property
    def trend_weights(self) -> dict[str, Decimal]:
        return {
            "breakout": self.trend_breakout_weight,
            "new_high": self.trend_new_high_weight,
            "position": self.trend_position_weight,
            "pullback": self.trend_pullback_weight,
            "slope": self.trend_slope_weight,
        }

    @property
    def required_stock_observations(self) -> int:
        return max(
            max(self.horizons) + 1,
            self.moving_average_window + self.slope_lookback,
            self.breakout_window + self.breakout_hold_sessions,
            self.pullback_window,
            self.new_high_window,
            self.volume_window + 1,
        )

    @property
    def config_hash(self) -> str:
        payload: dict[str, Any] = {
            "breakout_hold_sessions": self.breakout_hold_sessions,
            "breakout_window": self.breakout_window,
            "horizon_weights": [canonical_decimal(value) for value in self.horizon_weights],
            "horizons": list(self.horizons),
            "insufficient_history_policy": self.insufficient_history_policy.value,
            "maximum_historical_suspensions": self.maximum_historical_suspensions,
            "missing_industry_policy": self.missing_industry_policy.value,
            "moving_average_window": self.moving_average_window,
            "new_high_distance_tolerance": canonical_decimal(self.new_high_distance_tolerance),
            "new_high_window": self.new_high_window,
            "position_score_scale": canonical_decimal(self.position_score_scale),
            "pullback_tolerance": canonical_decimal(self.pullback_tolerance),
            "pullback_window": self.pullback_window,
            "relative_score_scale": canonical_decimal(self.relative_score_scale),
            "return_score_scale": canonical_decimal(self.return_score_scale),
            "score_weights": {
                key: canonical_decimal(value) for key, value in self.score_weights.items()
            },
            "slope_lookback": self.slope_lookback,
            "slope_score_scale": canonical_decimal(self.slope_score_scale),
            "suspension_policy": self.suspension_policy.value,
            "trend_weights": {
                key: canonical_decimal(value) for key, value in self.trend_weights.items()
            },
            "version": self.version,
            "volume_ratio_score_scale": canonical_decimal(self.volume_ratio_score_scale),
            "volume_window": self.volume_window,
        }
        return stable_feature_hash(payload)


@dataclass(frozen=True, slots=True)
class HorizonStrengthMetrics:
    """Stock and point-in-time reference returns over one aligned horizon."""

    horizon: int
    stock_return: Decimal
    benchmark_return: Decimal
    benchmark_relative_return: Decimal
    industry_return: Decimal | None
    industry_relative_return: Decimal | None
    historical_industry_ids: tuple[str, ...]
    aligned_sessions: int

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.aligned_sessions != self.horizon:
            raise ValueError("horizon strength metrics must be fully aligned")
        required = (self.stock_return, self.benchmark_return, self.benchmark_relative_return)
        if any(not value.is_finite() for value in required):
            raise ValueError("stock and benchmark returns must be finite")
        industry_values = (self.industry_return, self.industry_relative_return)
        if (industry_values[0] is None) != (industry_values[1] is None):
            raise ValueError("industry return fields must be both present or both absent")
        if any(value is not None and not value.is_finite() for value in industry_values):
            raise ValueError("industry returns must be finite")
        if self.industry_return is not None and not self.historical_industry_ids:
            raise ValueError("industry-relative metrics require historical industry identities")


@dataclass(frozen=True, slots=True)
class TrendQualityMetrics:
    """Auditable moving-average, breakout, pullback, and new-high measurements."""

    latest_close: Decimal
    moving_average: Decimal
    position_vs_average: Decimal
    moving_average_slope: Decimal
    breakout_level: Decimal
    breakout_hold_ratio: Decimal
    pullback_depth: Decimal
    distance_to_high: Decimal
    score: Decimal

    def __post_init__(self) -> None:
        numeric = (
            self.latest_close,
            self.moving_average,
            self.position_vs_average,
            self.moving_average_slope,
            self.breakout_level,
            self.breakout_hold_ratio,
            self.pullback_depth,
            self.distance_to_high,
            self.score,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("trend quality metrics must be finite")
        if min(self.latest_close, self.moving_average, self.breakout_level) <= 0:
            raise ValueError("trend quality prices must be positive")
        if not Decimal(0) <= self.breakout_hold_ratio <= Decimal(1):
            raise ValueError("breakout_hold_ratio must be between zero and one")
        if self.pullback_depth < 0 or self.distance_to_high > 0:
            raise ValueError("pullback and new-high distance signs are invalid")
        if not Decimal(-100) <= self.score <= Decimal(100):
            raise ValueError("trend quality score must be within [-100, 100]")


@dataclass(frozen=True, slots=True)
class VolumePriceConfirmation:
    """Current price direction and volume relative to a non-suspended baseline."""

    current_return: Decimal
    current_volume: Decimal
    average_prior_volume: Decimal
    volume_ratio: Decimal
    score: Decimal
    baseline_observations: int

    def __post_init__(self) -> None:
        numeric = (
            self.current_return,
            self.current_volume,
            self.average_prior_volume,
            self.volume_ratio,
            self.score,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("volume-price metrics must be finite")
        if min(self.current_volume, self.average_prior_volume, self.volume_ratio) < 0:
            raise ValueError("volume-price values cannot be negative")
        if self.average_prior_volume <= 0 or self.baseline_observations <= 0:
            raise ValueError("volume-price baseline must be positive and non-empty")
        if not Decimal(-100) <= self.score <= Decimal(100):
            raise ValueError("volume-price score must be within [-100, 100]")


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    """One normalized component and its exact contribution to the final score."""

    component: str
    raw_value: Decimal
    normalized_score: Decimal
    configured_weight: Decimal
    effective_weight: Decimal
    contribution: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _non_empty(self.component, "component"))
        values = (
            self.raw_value,
            self.normalized_score,
            self.configured_weight,
            self.effective_weight,
            self.contribution,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("score contribution values must be finite")
        if not Decimal(-100) <= self.normalized_score <= Decimal(100):
            raise ValueError("normalized score must be within [-100, 100]")
        if not Decimal(0) <= self.configured_weight <= Decimal(1) or not Decimal(
            0
        ) <= self.effective_weight <= Decimal(1):
            raise ValueError("score contribution weights must be within [0, 1]")
        if self.contribution != self.normalized_score * self.effective_weight:
            raise ValueError("contribution must equal normalized_score times effective_weight")


@dataclass(frozen=True, slots=True)
class StockStrengthSnapshot:
    """Version-bound stock strength result with exact input and output hashes."""

    instrument_id: str
    benchmark_id: str
    session_date: date
    as_of: datetime
    data_version: str
    classification_version: str
    industry_level: int
    current_industry_id: str | None
    feature_version: str
    config_hash: str
    status: StockStrengthStatus
    horizons: tuple[HorizonStrengthMetrics, ...]
    trend_quality: TrendQualityMetrics | None
    volume_confirmation: VolumePriceConfirmation | None
    contributions: tuple[ScoreContribution, ...]
    score: Decimal | None
    reason: str | None
    selected_dates: tuple[date, ...]
    input_hash: str
    cache_key: str
    result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "benchmark_id", _non_empty(self.benchmark_id, "benchmark_id"))
        ensure_aware(self.as_of)
        versions = (self.data_version, self.classification_version, self.feature_version)
        if any(not value.strip() for value in versions):
            raise ValueError("stock strength versions must be non-empty")
        if self.industry_level not in (1, 2, 3):
            raise ValueError("industry_level must be 1, 2, or 3")
        if self.current_industry_id is not None and not self.current_industry_id.strip():
            raise ValueError("current_industry_id must be non-empty when present")
        hashes = (self.config_hash, self.input_hash, self.cache_key, self.result_hash)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("stock strength hashes must be SHA-256 digests")
        if tuple(sorted(set(self.selected_dates))) != self.selected_dates:
            raise ValueError("selected_dates must be unique and ascending")
        available = self.status in {StockStrengthStatus.READY, StockStrengthStatus.DEGRADED}
        if available:
            if (
                not self.horizons
                or self.trend_quality is None
                or self.volume_confirmation is None
                or not self.contributions
                or self.score is None
                or not self.score.is_finite()
            ):
                raise ValueError("available stock strength snapshot requires complete metrics")
            if not Decimal(-100) <= self.score <= Decimal(100):
                raise ValueError("stock strength score must be within [-100, 100]")
            if sum((item.effective_weight for item in self.contributions), Decimal(0)) != 1:
                raise ValueError("effective contribution weights must sum to one")
            if sum((item.contribution for item in self.contributions), Decimal(0)) != self.score:
                raise ValueError("score must equal the exact contribution sum")
            if self.status is StockStrengthStatus.READY and self.reason is not None:
                raise ValueError("ready stock strength cannot have a degradation reason")
            if self.status is StockStrengthStatus.DEGRADED and not self.reason:
                raise ValueError("degraded stock strength requires a reason")
        elif self.score is not None or self.contributions:
            raise ValueError("unavailable stock strength cannot have score contributions")
        elif not self.reason:
            raise ValueError("unavailable stock strength requires a reason")

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "cache_key": self.cache_key,
            "classification_version": self.classification_version,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "input_hash": self.input_hash,
            "instrument_id": self.instrument_id,
            "result_hash": self.result_hash,
            "session_date": self.session_date.isoformat(),
        }


__all__ = [
    "HorizonStrengthMetrics",
    "IndustryMembershipError",
    "InsufficientHistoryPolicy",
    "InsufficientStockData",
    "MissingIndustryPolicy",
    "PointInTimeStockIndustryMembership",
    "ReferenceBarObservation",
    "ScoreContribution",
    "StockBarObservation",
    "StockStrengthConfig",
    "StockStrengthSnapshot",
    "StockStrengthStatus",
    "SuspendedStockError",
    "SuspensionPolicy",
    "TrendQualityMetrics",
    "VolumePriceConfirmation",
]
