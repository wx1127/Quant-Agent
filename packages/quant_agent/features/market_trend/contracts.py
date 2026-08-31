"""Configuration and result contracts for market trend features."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal


class MissingIndexPolicy(StrEnum):
    """Whether one unavailable required index blocks the aggregate snapshot."""

    FAIL = "FAIL"
    DEGRADE = "DEGRADE"


class InsufficientMarketData(ValueError):
    """Raised when required index history is absent or too short."""


@dataclass(frozen=True, slots=True)
class MarketTrendConfig:
    """Versioned, transparent parameters for the baseline trend score."""

    version: str = "market-trend-v1"
    windows: tuple[int, ...] = (20, 60, 120)
    window_weights: tuple[Decimal, ...] = (
        Decimal("0.2"),
        Decimal("0.3"),
        Decimal("0.5"),
    )
    slope_lookback: int = 5
    return_weight: Decimal = Decimal("0.60")
    position_weight: Decimal = Decimal("0.25")
    slope_weight: Decimal = Decimal("0.15")
    score_scale: Decimal = Decimal("500")
    missing_index_policy: MissingIndexPolicy = MissingIndexPolicy.FAIL

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("market trend version must be non-empty")
        if not self.windows or any(window < 2 for window in self.windows):
            raise ValueError("trend windows must contain values of at least two")
        if tuple(sorted(set(self.windows))) != self.windows:
            raise ValueError("trend windows must be unique and ascending")
        if len(self.windows) != len(self.window_weights):
            raise ValueError("each trend window requires one aggregation weight")
        decimal_values = (
            *self.window_weights,
            self.return_weight,
            self.position_weight,
            self.slope_weight,
            self.score_scale,
        )
        for value in decimal_values:
            canonical_decimal(value, field_name="market trend numeric parameter")
        if (
            any(weight < 0 for weight in self.window_weights)
            or sum(self.window_weights, Decimal(0)) <= 0
        ):
            raise ValueError("window weights must be non-negative with a positive sum")
        if self.slope_lookback <= 0:
            raise ValueError("slope_lookback must be positive")
        component_weights = (
            self.return_weight,
            self.position_weight,
            self.slope_weight,
        )
        if any(weight < 0 for weight in component_weights):
            raise ValueError("trend component weights cannot be negative")
        if sum(component_weights, Decimal(0)) != Decimal(1):
            raise ValueError("trend component weights must sum to one")
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")

    @property
    def config_hash(self) -> str:
        """Return a stable hash for version and every numeric scoring parameter."""

        payload = {
            "missing_index_policy": self.missing_index_policy.value,
            "position_weight": canonical_decimal(self.position_weight),
            "return_weight": canonical_decimal(self.return_weight),
            "score_scale": canonical_decimal(self.score_scale),
            "slope_lookback": self.slope_lookback,
            "slope_weight": canonical_decimal(self.slope_weight),
            "version": self.version,
            "window_weights": [canonical_decimal(value) for value in self.window_weights],
            "windows": list(self.windows),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class WindowTrendMetrics:
    """Auditable components for one moving-average horizon."""

    window: int
    latest_close: Decimal
    moving_average: Decimal
    period_return: Decimal
    position_vs_average: Decimal
    average_slope: Decimal
    score: Decimal
    input_hash: str

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least two")
        if min(self.latest_close, self.moving_average) <= 0:
            raise ValueError("trend prices must be positive")
        if not all(
            value.is_finite()
            for value in (
                self.period_return,
                self.position_vs_average,
                self.average_slope,
                self.score,
            )
        ):
            raise ValueError("trend metrics must be finite")
        if self.score < -100 or self.score > 100:
            raise ValueError("trend score must be between -100 and 100")
        if len(self.input_hash) != 64:
            raise ValueError("input_hash must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class IndexTrendResult:
    """All horizon metrics and the weighted score for one required index."""

    index_id: str
    windows: tuple[WindowTrendMetrics, ...]
    observation_dates: tuple[date, ...]
    score: Decimal

    def __post_init__(self) -> None:
        if not self.index_id.strip():
            raise ValueError("index_id must be non-empty")
        if not self.windows:
            raise ValueError("index trend requires at least one window")
        if (
            not self.observation_dates
            or tuple(sorted(set(self.observation_dates))) != self.observation_dates
        ):
            raise ValueError("index observation_dates must be unique and ascending")
        if self.score < -100 or self.score > 100 or not self.score.is_finite():
            raise ValueError("index score must be finite and between -100 and 100")


@dataclass(frozen=True, slots=True)
class MarketTrendSnapshot:
    """Version-bound multi-index trend aggregate at one decision time."""

    session_date: date
    as_of: datetime
    data_version: str
    feature_version: str
    config_hash: str
    indices: tuple[IndexTrendResult, ...]
    missing_indices: tuple[str, ...]
    aggregate_score: Decimal
    cache_key: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not self.data_version.strip() or not self.feature_version.strip():
            raise ValueError("snapshot versions must be non-empty")
        if len(self.config_hash) != 64 or len(self.cache_key) != 64:
            raise ValueError("snapshot hashes must be SHA-256 digests")
        if not self.indices:
            raise ValueError("market trend snapshot requires at least one index")
        if self.aggregate_score < -100 or self.aggregate_score > 100:
            raise ValueError("aggregate score must be between -100 and 100")

    def identity_payload(self) -> dict[str, str]:
        """Return the externally useful, stable version identity."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "cache_key": self.cache_key,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "session_date": self.session_date.isoformat(),
        }
