"""Immutable point-in-time contracts for the baseline ETF rotation strategy."""

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.backtest import (
    SignalEvent,
    TargetEvent,
    TradableInstrumentType,
    WeightSignal,
)
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


class ETFRotationInputError(ValueError):
    """Raised when strategy inputs cannot be aligned without lookahead or ambiguity."""


class ETFRotationDecisionStatus(StrEnum):
    """Whether a session emits targets, deliberately holds, or targets all cash."""

    REBALANCE = "REBALANCE"
    NO_REBALANCE = "NO_REBALANCE"
    CASH = "CASH"


class ETFUniverseEligibility(StrEnum):
    """Current PIT eligibility of one known ETF universe member."""

    ELIGIBLE = "ELIGIBLE"
    DISABLED = "DISABLED"
    NOT_LISTED = "NOT_LISTED"
    DELISTED = "DELISTED"
    NO_EFFECTIVE_REVISION = "NO_EFFECTIVE_REVISION"


@dataclass(frozen=True, slots=True)
class ETFUniverseRevision:
    """One revisable historical ETF universe/status interval."""

    instrument_id: str
    listed_on: date
    delisted_on: date | None
    effective_from: date
    effective_to: date | None
    enabled: bool
    available_at: datetime
    revision: str
    data_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        ensure_aware(self.available_at)
        if self.delisted_on is not None and self.delisted_on < self.listed_on:
            raise ValueError("ETF delisted_on cannot precede listed_on")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("ETF universe effective_to cannot precede effective_from")
        if not isinstance(self.enabled, bool):
            raise ValueError("ETF universe enabled must be boolean")

    def applies_on(self, session_date: date) -> bool:
        """Return whether this status interval covers the decision session."""

        return self.effective_from <= session_date and (
            self.effective_to is None or session_date <= self.effective_to
        )

    def fingerprint_payload(self) -> dict[str, str | bool | None]:
        """Return canonical fields used by the universe input hash."""

        return {
            "available_at": self.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "delisted_on": self.delisted_on.isoformat() if self.delisted_on else None,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "enabled": self.enabled,
            "instrument_id": self.instrument_id,
            "listed_on": self.listed_on.isoformat(),
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class ETFRotationConfig:
    """Versioned ranking, trend, exposure, and rebalance policy."""

    version: str = "etf-rotation-v1"
    momentum_horizons: tuple[int, ...] = (20, 60, 120)
    momentum_weights: tuple[Decimal, ...] = (
        Decimal("0.2"),
        Decimal("0.3"),
        Decimal("0.5"),
    )
    volatility_window: int = 20
    volatility_annualization_sessions: int = 252
    volatility_floor: Decimal = Decimal("0.01")
    long_trend_window: int = 120
    minimum_weighted_momentum: Decimal = Decimal("0")
    rebalance_frequency_sessions: int = 5
    top_n: int = 3
    maximum_instrument_weight: Decimal = Decimal("0.50")
    uptrend_exposure: Decimal = Decimal("0.80")
    range_strong_exposure: Decimal = Decimal("0.50")
    divergent_exposure: Decimal = Decimal("0.35")
    downtrend_exposure: Decimal = Decimal("0")
    bottom_recovery_exposure: Decimal = Decimal("0.30")
    force_downtrend_cash: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "strategy version"))
        if (
            not self.momentum_horizons
            or tuple(sorted(set(self.momentum_horizons))) != self.momentum_horizons
            or any(value < 2 for value in self.momentum_horizons)
        ):
            raise ValueError("momentum horizons must be unique, increasing, and at least two")
        if len(self.momentum_horizons) != len(self.momentum_weights):
            raise ValueError("each momentum horizon requires one weight")
        integer_values = (
            self.volatility_window,
            self.volatility_annualization_sessions,
            self.long_trend_window,
            self.rebalance_frequency_sessions,
            self.top_n,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integer_values
        ):
            raise ValueError("ETF rotation windows, frequency, and top_n must be positive integers")
        if self.volatility_window < 2 or self.long_trend_window < 2:
            raise ValueError("volatility and trend windows must be at least two")
        decimals = (
            *self.momentum_weights,
            self.volatility_floor,
            self.minimum_weighted_momentum,
            self.maximum_instrument_weight,
            *self.regime_exposures.values(),
        )
        for value in decimals:
            canonical_decimal(value, field_name="ETF rotation numeric parameter")
        if any(value < 0 for value in self.momentum_weights) or sum(
            self.momentum_weights, Decimal(0)
        ) != Decimal(1):
            raise ValueError("momentum weights must be non-negative and sum to one")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if not Decimal(0) < self.maximum_instrument_weight <= Decimal(1):
            raise ValueError("maximum_instrument_weight must be within (0, 1]")
        if any(value < 0 or value > 1 for value in self.regime_exposures.values()):
            raise ValueError("regime exposure caps must be within 0..1")
        if not isinstance(self.force_downtrend_cash, bool):
            raise ValueError("force_downtrend_cash must be boolean")

    @property
    def regime_exposures(self) -> dict[MarketRegime, Decimal]:
        """Return gross exposure caps for each stabilized market state."""

        return {
            MarketRegime.UPTREND: self.uptrend_exposure,
            MarketRegime.RANGE_STRONG: self.range_strong_exposure,
            MarketRegime.DIVERGENT: self.divergent_exposure,
            MarketRegime.DOWNTREND: self.downtrend_exposure,
            MarketRegime.BOTTOM_RECOVERY: self.bottom_recovery_exposure,
        }

    @property
    def required_price_observations(self) -> int:
        """Return the exact close count required by every configured calculation."""

        return max(
            max(self.momentum_horizons) + 1,
            self.volatility_window + 1,
            self.long_trend_window,
        )

    @property
    def config_hash(self) -> str:
        """Return a stable hash over every strategy parameter."""

        return stable_hash(asdict(self))

    def exposure_for(self, regime: MarketRegime) -> Decimal:
        """Return the configured cap for one stabilized market state."""

        return self.regime_exposures[regime]


@dataclass(frozen=True, slots=True)
class ETFRotationRequest:
    """One deterministic strategy decision boundary."""

    signal_date: date
    as_of: datetime
    data_version: str
    session_index: int

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.signal_date:
            raise ValueError("signal_date must match as_of in Asia/Shanghai")
        if (
            not isinstance(self.session_index, int)
            or isinstance(self.session_index, bool)
            or self.session_index < 0
        ):
            raise ValueError("session_index must be a non-negative integer")

    def fingerprint_payload(self) -> dict[str, str | int]:
        """Return canonical request fields for decision hashes."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "session_index": self.session_index,
            "signal_date": self.signal_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ETFUniverseSelection:
    """Selected PIT status revision and current eligibility for one known ETF."""

    instrument_id: str
    eligibility: ETFUniverseEligibility
    listed_on: date
    delisted_on: date | None
    enabled: bool
    effective_from: date
    effective_to: date | None
    available_at: datetime
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        ensure_aware(self.available_at)

    @property
    def eligible(self) -> bool:
        """Return whether this ETF may enter the ranking set."""

        return self.eligibility is ETFUniverseEligibility.ELIGIBLE


@dataclass(frozen=True, slots=True)
class ETFHorizonMomentum:
    """Close-to-close momentum over one fully aligned observed-session horizon."""

    horizon: int
    return_value: Decimal
    configured_weight: Decimal
    contribution: Decimal

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("momentum horizon must be positive")
        for value in (self.return_value, self.configured_weight, self.contribution):
            canonical_decimal(value, field_name="ETF momentum value")
        if not Decimal(0) <= self.configured_weight <= Decimal(1):
            raise ValueError("momentum configured_weight must be within 0..1")
        if self.contribution != self.return_value * self.configured_weight:
            raise ValueError("momentum contribution must equal return times weight")


@dataclass(frozen=True, slots=True)
class ETFMomentumMetrics:
    """Auditable momentum, volatility, and long-trend calculations."""

    horizons: tuple[ETFHorizonMomentum, ...]
    weighted_momentum: Decimal
    annualized_volatility: Decimal
    risk_adjusted_momentum: Decimal
    latest_close: Decimal
    long_trend_average: Decimal
    above_long_trend: bool
    observation_dates: tuple[date, ...]
    price_revisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("ETF momentum metrics require configured horizons")
        numeric = (
            self.weighted_momentum,
            self.annualized_volatility,
            self.risk_adjusted_momentum,
            self.latest_close,
            self.long_trend_average,
        )
        for value in numeric:
            canonical_decimal(value, field_name="ETF rotation metric")
        if self.annualized_volatility < 0:
            raise ValueError("annualized volatility cannot be negative")
        if self.latest_close <= 0 or self.long_trend_average <= 0:
            raise ValueError("ETF price and trend average must be positive")
        if tuple(sorted(set(self.observation_dates))) != self.observation_dates:
            raise ValueError("ETF observation_dates must be unique and increasing")
        if len(self.observation_dates) != len(self.price_revisions):
            raise ValueError("each ETF observation date requires one selected revision")
        if self.weighted_momentum != sum((item.contribution for item in self.horizons), Decimal(0)):
            raise ValueError("weighted momentum must equal horizon contributions")
        if self.above_long_trend != (self.latest_close > self.long_trend_average):
            raise ValueError("above_long_trend must match latest close versus trend average")


@dataclass(frozen=True, slots=True)
class ETFRotationCandidate:
    """One eligible or rejected known universe member in deterministic rank order."""

    instrument_id: str
    universe_eligibility: ETFUniverseEligibility
    metrics: ETFMomentumMetrics | None
    qualifies: bool
    rank: int | None
    selected: bool
    target_weight: Decimal
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "reason", _non_empty(self.reason, "candidate reason"))
        canonical_decimal(self.target_weight, field_name="ETF candidate target weight")
        if self.target_weight < 0 or self.target_weight > 1:
            raise ValueError("ETF candidate target weight must be within 0..1")
        if self.rank is not None and self.rank < 1:
            raise ValueError("ETF candidate rank must be positive")
        if self.qualifies and (self.metrics is None or self.rank is None):
            raise ValueError("qualifying ETF candidate requires metrics and rank")
        if not self.qualifies and self.rank is not None:
            raise ValueError("non-qualifying ETF candidate cannot retain a rank")
        if self.selected != (self.target_weight > 0):
            raise ValueError("ETF candidate selected flag must match positive target weight")
        if self.selected and not self.qualifies:
            raise ValueError("selected ETF candidate must qualify")


@dataclass(frozen=True, slots=True)
class ETFTargetWeight:
    """One complete long-only target entry, including explicit zero exits."""

    instrument_id: str
    target_weight: Decimal
    instrument_type: TradableInstrumentType = TradableInstrumentType.ETF

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        canonical_decimal(self.target_weight, field_name="ETF target weight")
        if self.instrument_type is not TradableInstrumentType.ETF:
            raise ValueError("ETF rotation targets must use ETF instrument type")
        if self.target_weight < 0 or self.target_weight > 1:
            raise ValueError("ETF target weight must be within 0..1")


@dataclass(frozen=True, slots=True)
class ETFRotationDecision:
    """Version-bound target portfolio before any order or execution logic."""

    signal_date: date
    as_of: datetime
    data_version: str
    strategy_version: str
    config_hash: str
    status: ETFRotationDecisionStatus
    market_regime: MarketRegime
    regime_result_hash: str
    regime_risk_budget: Decimal
    configured_regime_exposure: Decimal
    gross_target_weight: Decimal | None
    cash_target_weight: Decimal | None
    universe: tuple[ETFUniverseSelection, ...]
    candidates: tuple[ETFRotationCandidate, ...]
    targets: tuple[ETFTargetWeight, ...]
    universe_input_hash: str
    price_input_hash: str
    input_hash: str
    result_hash: str
    reason: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.signal_date:
            raise ValueError("ETF decision signal_date must match as_of")
        versions = (self.data_version, self.strategy_version)
        if any(not value.strip() for value in versions):
            raise ValueError("ETF decision versions must be non-empty")
        for hash_value, name in (
            (self.config_hash, "config_hash"),
            (self.regime_result_hash, "regime_result_hash"),
            (self.universe_input_hash, "universe_input_hash"),
            (self.price_input_hash, "price_input_hash"),
            (self.input_hash, "input_hash"),
            (self.result_hash, "result_hash"),
        ):
            _validate_hash(hash_value, name)
        for budget in (self.regime_risk_budget, self.configured_regime_exposure):
            canonical_decimal(budget, field_name="ETF decision risk budget")
            if budget < 0 or budget > 1:
                raise ValueError("ETF decision risk budgets must be within 0..1")
        object.__setattr__(self, "reason", _non_empty(self.reason, "decision reason"))
        universe_ids = tuple(item.instrument_id for item in self.universe)
        if tuple(sorted(set(universe_ids))) != universe_ids:
            raise ValueError("ETF decision universe must be unique and sorted")
        target_ids = tuple(item.instrument_id for item in self.targets)
        if tuple(sorted(set(target_ids))) != target_ids:
            raise ValueError("ETF decision targets must be unique and sorted")
        candidate_ids = tuple(item.instrument_id for item in self.candidates)
        if tuple(sorted(set(candidate_ids))) != candidate_ids:
            raise ValueError("ETF decision candidates must be unique and sorted")

        if self.status is ETFRotationDecisionStatus.NO_REBALANCE:
            if (
                self.gross_target_weight is not None
                or self.cash_target_weight is not None
                or self.targets
                or self.candidates
            ):
                raise ValueError("NO_REBALANCE cannot emit candidates or target weights")
            return
        if self.gross_target_weight is None or self.cash_target_weight is None:
            raise ValueError("targeting ETF decisions require gross and cash weights")
        for target_weight in (self.gross_target_weight, self.cash_target_weight):
            canonical_decimal(target_weight, field_name="ETF decision target weight")
            if target_weight < 0 or target_weight > 1:
                raise ValueError("ETF decision target weights must be within 0..1")
        if self.gross_target_weight + self.cash_target_weight != Decimal(1):
            raise ValueError("ETF decision gross and cash target weights must sum to one")
        if self.gross_target_weight != sum(
            (item.target_weight for item in self.targets), Decimal(0)
        ):
            raise ValueError("ETF gross target must equal instrument target weights")
        if self.gross_target_weight > min(self.regime_risk_budget, self.configured_regime_exposure):
            raise ValueError("ETF gross target exceeds the regime risk budget")
        if target_ids != universe_ids:
            raise ValueError("ETF rebalance targets must cover every known PIT universe member")
        if candidate_ids != universe_ids:
            raise ValueError("ETF rebalance candidates must cover every known PIT universe member")
        universe_by_id = {item.instrument_id: item for item in self.universe}
        targets_by_id = {item.instrument_id: item.target_weight for item in self.targets}
        for candidate in self.candidates:
            if (
                candidate.universe_eligibility
                is not universe_by_id[candidate.instrument_id].eligibility
            ):
                raise ValueError("ETF candidate eligibility must match the selected universe")
            if candidate.target_weight != targets_by_id[candidate.instrument_id]:
                raise ValueError("ETF candidate and portfolio target weights must match")
        ranks = sorted(
            candidate.rank for candidate in self.candidates if candidate.rank is not None
        )
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("qualifying ETF candidate ranks must be unique and contiguous")
        if self.status is ETFRotationDecisionStatus.CASH and self.gross_target_weight != 0:
            raise ValueError("CASH decision must have zero ETF gross target")
        if self.status is ETFRotationDecisionStatus.REBALANCE and self.gross_target_weight <= 0:
            raise ValueError("REBALANCE decision requires positive ETF exposure")

    @property
    def has_targets(self) -> bool:
        """Return whether this decision should be sent to a backtest adapter."""

        return self.status is not ETFRotationDecisionStatus.NO_REBALANCE

    def identity_payload(self) -> dict[str, str]:
        """Return stable strategy/data/config identities for manifests."""

        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "result_hash": self.result_hash,
            "signal_date": self.signal_date.isoformat(),
            "strategy_version": self.strategy_version,
        }


@dataclass(frozen=True, slots=True)
class ETFEventAdaptation:
    """Deterministic prior-session signals and next-session target events."""

    signal_events: tuple[SignalEvent, ...]
    target_events: tuple[TargetEvent, ...]
    execution_semantics: str = "NEXT_TRADING_SESSION"

    def __post_init__(self) -> None:
        if self.execution_semantics != "NEXT_TRADING_SESSION":
            raise ValueError("ETF event adaptation uses next trading session execution")
        if len(self.signal_events) != len(self.target_events):
            raise ValueError("each ETF signal event requires one target event")
        for signal, target in zip(self.signal_events, self.target_events, strict=True):
            if (
                target.signal_event_id != signal.event_id
                or target.signal_time != signal.event_time
                or target.instrument_id != signal.instrument_id
                or target.instrument_type is not signal.instrument_type
            ):
                raise ValueError("ETF target events must bind their matching signal events")
            if target.trading_day <= signal.trading_day:
                raise ValueError("ETF target events must execute on a later trading session")


def target_weight_map(
    values: tuple[ETFTargetWeight, ...] | tuple[WeightSignal, ...] | tuple[TargetEvent, ...],
) -> dict[str, Decimal]:
    """Return comparable target maps across strategy and both backtest adapters."""

    result: dict[str, Decimal] = {}
    for item in values:
        weight = item.target_weight
        if weight is None:  # TargetEvent permits quantity targets; this strategy never emits them.
            raise ValueError("ETF rotation adapters require target_weight values")
        if item.instrument_id in result:
            raise ValueError("ETF target map cannot contain duplicate instruments")
        result[item.instrument_id] = weight
    return result


__all__ = [
    "ETFEventAdaptation",
    "ETFHorizonMomentum",
    "ETFMomentumMetrics",
    "ETFRotationCandidate",
    "ETFRotationConfig",
    "ETFRotationDecision",
    "ETFRotationDecisionStatus",
    "ETFRotationInputError",
    "ETFRotationRequest",
    "ETFTargetWeight",
    "ETFUniverseEligibility",
    "ETFUniverseRevision",
    "ETFUniverseSelection",
    "target_weight_map",
]
