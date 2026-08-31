"""Inputs and auditable outputs for market breadth calculations."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal


class InsufficientBreadthData(ValueError):
    """Raised when current-universe coverage is below the configured safety floor."""


@dataclass(frozen=True, slots=True)
class BreadthObservation:
    """One stock session record with historical lifecycle and suspension state."""

    instrument_id: str
    trade_date: date
    observed_at: datetime
    available_at: datetime
    close: Decimal
    turnover: Decimal
    is_active: bool
    is_suspended: bool
    revision: str

    def __post_init__(self) -> None:
        if not self.instrument_id.strip() or not self.revision.strip():
            raise ValueError("breadth identifiers and revision must be non-empty")
        ensure_aware(self.observed_at)
        ensure_aware(self.available_at)
        if self.observed_at.date() != self.trade_date:
            raise ValueError("observed_at local date must equal trade_date")
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        if self.close <= 0 or not self.close.is_finite():
            raise ValueError("breadth close must be finite and positive")
        if self.turnover < 0 or not self.turnover.is_finite():
            raise ValueError("breadth turnover must be finite and non-negative")

    def fingerprint_payload(self) -> dict[str, str | bool]:
        """Return canonical values used by the snapshot cache identity."""

        return {
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "close": canonical_decimal(self.close),
            "instrument_id": self.instrument_id,
            "is_active": self.is_active,
            "is_suspended": self.is_suspended,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "revision": self.revision,
            "trade_date": self.trade_date.isoformat(),
            "turnover": canonical_decimal(self.turnover),
        }


@dataclass(frozen=True, slots=True)
class MarketBreadthConfig:
    """Versioned window and coverage policy for the baseline breadth feature set."""

    version: str = "market-breadth-v1"
    moving_average_window: int = 20
    new_high_low_window: int = 60
    turnover_percentile_window: int = 60
    downside_volatility_window: int = 20
    min_downside_observations: int = 10
    minimum_current_coverage: Decimal = Decimal("0.95")
    minimum_historical_coverage: Decimal = Decimal("0.95")
    minimum_turnover_observations: int = 20

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("market breadth version must be non-empty")
        windows = (
            self.moving_average_window,
            self.new_high_low_window,
            self.turnover_percentile_window,
            self.downside_volatility_window,
        )
        if any(window < 2 for window in windows):
            raise ValueError("breadth windows must be at least two")
        if not 1 <= self.min_downside_observations <= self.downside_volatility_window:
            raise ValueError("min_downside_observations must be within its window")
        if not 1 <= self.minimum_turnover_observations <= self.turnover_percentile_window:
            raise ValueError("minimum_turnover_observations must be within its window")
        for field_name, value in (
            ("minimum_current_coverage", self.minimum_current_coverage),
            ("minimum_historical_coverage", self.minimum_historical_coverage),
        ):
            canonical_decimal(value, field_name=field_name)
            if not Decimal(0) <= value <= Decimal(1):
                raise ValueError(f"{field_name} must be between zero and one")

    @property
    def config_hash(self) -> str:
        payload = {
            "downside_volatility_window": self.downside_volatility_window,
            "min_downside_observations": self.min_downside_observations,
            "minimum_current_coverage": canonical_decimal(self.minimum_current_coverage),
            "minimum_historical_coverage": canonical_decimal(self.minimum_historical_coverage),
            "minimum_turnover_observations": self.minimum_turnover_observations,
            "moving_average_window": self.moving_average_window,
            "new_high_low_window": self.new_high_low_window,
            "turnover_percentile_window": self.turnover_percentile_window,
            "version": self.version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MarketBreadthSnapshot:
    """Cross-sectional breadth metrics with a denominator for every ratio."""

    session_date: date
    as_of: datetime
    data_version: str
    feature_version: str
    config_hash: str
    expected_active_count: int
    received_active_count: int
    current_coverage: Decimal
    historical_session_count: int
    minimum_historical_coverage_observed: Decimal
    missing_instruments: tuple[str, ...]
    suspended_count: int
    advancing_count: int
    declining_count: int
    unchanged_count: int
    return_denominator: int
    above_average_count: int
    moving_average_denominator: int
    above_average_ratio: Decimal | None
    new_high_count: int
    new_low_count: int
    high_low_denominator: int
    new_high_ratio: Decimal | None
    new_low_ratio: Decimal | None
    total_turnover: Decimal
    turnover_percentile: Decimal | None
    turnover_history_count: int
    downside_volatility: Decimal | None
    downside_observation_count: int
    cache_key: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not self.data_version.strip() or not self.feature_version.strip():
            raise ValueError("breadth snapshot versions must be non-empty")
        if len(self.config_hash) != 64 or len(self.cache_key) != 64:
            raise ValueError("breadth hashes must be SHA-256 digests")
        counts = (
            self.expected_active_count,
            self.received_active_count,
            self.historical_session_count,
            self.suspended_count,
            self.advancing_count,
            self.declining_count,
            self.unchanged_count,
            self.return_denominator,
            self.above_average_count,
            self.moving_average_denominator,
            self.new_high_count,
            self.new_low_count,
            self.high_low_denominator,
            self.turnover_history_count,
            self.downside_observation_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("breadth counts cannot be negative")
        for ratio in (
            self.current_coverage,
            self.minimum_historical_coverage_observed,
            self.above_average_ratio,
            self.new_high_ratio,
            self.new_low_ratio,
            self.turnover_percentile,
        ):
            if ratio is not None and not Decimal(0) <= ratio <= Decimal(1):
                raise ValueError("breadth ratios must be between zero and one")
        if self.total_turnover < 0 or not self.total_turnover.is_finite():
            raise ValueError("total_turnover must be finite and non-negative")
        if self.downside_volatility is not None and (
            self.downside_volatility < 0 or not self.downside_volatility.is_finite()
        ):
            raise ValueError("downside_volatility must be finite and non-negative")

    @property
    def advance_decline_ratio(self) -> Decimal | None:
        """Return advance share over all comparable, non-suspended current stocks."""

        if self.return_denominator == 0:
            return None
        return Decimal(self.advancing_count) / self.return_denominator
