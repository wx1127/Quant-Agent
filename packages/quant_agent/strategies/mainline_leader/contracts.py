"""Immutable contracts for the mainline-leader target portfolio strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.backtest import TradableInstrumentType
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.features.mainline import MainlineState
from quant_agent.leaders import LeaderType
from quant_agent.leaders.candidates import CandidateTier
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


class MainlineLeaderInputError(ValueError):
    """Raised when strategy inputs cannot be aligned without an unsafe assumption."""


class MainlineLeaderWeightingMode(StrEnum):
    """Frozen target-weight allocation methods."""

    EQUAL = "EQUAL"
    RISK_ADJUSTED = "RISK_ADJUSTED"


class MainlineLeaderDecisionStatus(StrEnum):
    """Whether a complete target is changed, unchanged, or entirely cash."""

    REBALANCE = "REBALANCE"
    HOLD = "HOLD"
    CASH = "CASH"


class MainlineLeaderAction(StrEnum):
    """Auditable weight transition for one instrument."""

    ENTER = "ENTER"
    INCREASE = "INCREASE"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    EXCLUDE = "EXCLUDE"


class MainlineLeaderReasonCode(StrEnum):
    """Machine-readable entry, exit, and turnover-control reasons."""

    SELECTED = "SELECTED"
    CANDIDATE_INELIGIBLE = "CANDIDATE_INELIGIBLE"
    TIER_FILTER = "TIER_FILTER"
    SCORE_FILTER = "SCORE_FILTER"
    LEADER_INELIGIBLE = "LEADER_INELIGIBLE"
    LEADER_TYPE_FILTER = "LEADER_TYPE_FILTER"
    MAINLINE_FADING = "MAINLINE_FADING"
    MAINLINE_INACTIVE = "MAINLINE_INACTIVE"
    MARKET_RISK_OFF = "MARKET_RISK_OFF"
    OUTSIDE_TOP_N = "OUTSIDE_TOP_N"
    PRIOR_POSITION_INVALID = "PRIOR_POSITION_INVALID"
    REBALANCE_SCHEDULE = "REBALANCE_SCHEDULE"
    TURNOVER_CAP = "TURNOVER_CAP"
    EXPOSURE_CAP = "EXPOSURE_CAP"


@dataclass(frozen=True, slots=True)
class MainlineLeaderConfig:
    """Versioned entry, allocation, exposure, exit, and turnover policy."""

    version: str = "mainline-leader-v1"
    weighting_mode: MainlineLeaderWeightingMode = MainlineLeaderWeightingMode.EQUAL
    maximum_positions: int = 5
    minimum_candidate_score: Decimal = Decimal("60")
    eligible_tiers: tuple[CandidateTier, ...] = (CandidateTier.A, CandidateTier.B)
    eligible_mainline_states: tuple[MainlineState, ...] = (
        MainlineState.CONFIRMED,
        MainlineState.CROWDED,
    )
    eligible_leader_types: tuple[LeaderType, ...] = (
        LeaderType.TREND,
        LeaderType.CAPACITY,
    )
    maximum_instrument_weight: Decimal = Decimal("0.25")
    rebalance_frequency_sessions: int = 5
    maximum_discretionary_turnover: Decimal = Decimal("0.40")
    crowded_weight_multiplier: Decimal = Decimal("0.70")
    uptrend_exposure: Decimal = Decimal("0.60")
    range_strong_exposure: Decimal = Decimal("0.40")
    divergent_exposure: Decimal = Decimal("0.25")
    downtrend_exposure: Decimal = Decimal(0)
    bottom_recovery_exposure: Decimal = Decimal("0.20")
    force_downtrend_cash: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "strategy version"))
        for count in (self.maximum_positions, self.rebalance_frequency_sessions):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("position count and rebalance frequency must be positive integers")
        decimals = (
            self.minimum_candidate_score,
            self.maximum_instrument_weight,
            self.maximum_discretionary_turnover,
            self.crowded_weight_multiplier,
            *self.regime_exposures.values(),
        )
        for decimal_value in decimals:
            canonical_decimal(decimal_value, field_name="mainline-leader numeric parameter")
        if not Decimal(0) <= self.minimum_candidate_score <= Decimal(100):
            raise ValueError("minimum_candidate_score must be within 0..100")
        if not Decimal(0) < self.maximum_instrument_weight <= Decimal(1):
            raise ValueError("maximum_instrument_weight must be within (0, 1]")
        if not Decimal(0) <= self.maximum_discretionary_turnover <= Decimal(2):
            raise ValueError("maximum_discretionary_turnover must be within 0..2")
        if not Decimal(0) <= self.crowded_weight_multiplier <= Decimal(1):
            raise ValueError("crowded_weight_multiplier must be within 0..1")
        if any(value < 0 or value > 1 for value in self.regime_exposures.values()):
            raise ValueError("regime exposure caps must be within 0..1")
        canonical_tiers = tuple(
            value
            for value in (CandidateTier.A, CandidateTier.B, CandidateTier.WATCH)
            if value in set(self.eligible_tiers)
        )
        if not canonical_tiers or self.eligible_tiers != canonical_tiers:
            raise ValueError("eligible_tiers must be unique and use canonical order")
        canonical_states = tuple(
            value
            for value in (MainlineState.CONFIRMED, MainlineState.CROWDED)
            if value in set(self.eligible_mainline_states)
        )
        if not canonical_states or self.eligible_mainline_states != canonical_states:
            raise ValueError("eligible mainline states are limited to CONFIRMED and CROWDED")
        canonical_types = tuple(
            value
            for value in (LeaderType.TREND, LeaderType.CAPACITY)
            if value in set(self.eligible_leader_types)
        )
        if not canonical_types or self.eligible_leader_types != canonical_types:
            raise ValueError("eligible leader types are limited to TREND and CAPACITY")
        if not isinstance(self.force_downtrend_cash, bool):
            raise ValueError("force_downtrend_cash must be boolean")

    @property
    def regime_exposures(self) -> dict[MarketRegime, Decimal]:
        return {
            MarketRegime.UPTREND: self.uptrend_exposure,
            MarketRegime.RANGE_STRONG: self.range_strong_exposure,
            MarketRegime.DIVERGENT: self.divergent_exposure,
            MarketRegime.DOWNTREND: self.downtrend_exposure,
            MarketRegime.BOTTOM_RECOVERY: self.bottom_recovery_exposure,
        }

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    def exposure_for(self, regime: MarketRegime) -> Decimal:
        return self.regime_exposures[regime]


@dataclass(frozen=True, slots=True)
class MainlineLeaderRequest:
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
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "session_index": self.session_index,
            "signal_date": self.signal_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class MainlineLeaderPortfolioPosition:
    """Prior target weight required to replay turnover-constrained decisions."""

    instrument_id: str
    industry_id: str
    target_weight: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "industry_id", _non_empty(self.industry_id, "industry_id"))
        canonical_decimal(self.target_weight, field_name="prior target weight")
        if not Decimal(0) < self.target_weight <= Decimal(1):
            raise ValueError("prior position target_weight must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class MainlineLeaderPortfolioState:
    """Frozen prior target portfolio; live execution positions belong downstream."""

    as_of: datetime
    data_version: str
    positions: tuple[MainlineLeaderPortfolioPosition, ...] = ()
    source_decision_hash: str | None = None

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        ids = tuple(item.instrument_id for item in self.positions)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("prior portfolio positions must be unique and sorted")
        if sum((item.target_weight for item in self.positions), Decimal(0)) > 1:
            raise ValueError("prior portfolio gross target cannot exceed one")
        if bool(self.positions) != bool(self.source_decision_hash):
            raise ValueError("non-empty prior portfolio requires source_decision_hash")
        if self.source_decision_hash is not None:
            _validate_hash(self.source_decision_hash, "source_decision_hash")

    @property
    def gross_target_weight(self) -> Decimal:
        return sum((item.target_weight for item in self.positions), Decimal(0))

    @property
    def cash_target_weight(self) -> Decimal:
        return Decimal(1) - self.gross_target_weight

    @property
    def input_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class MainlineLeaderSelection:
    """One instrument's aligned qualification and final weight transition."""

    instrument_id: str
    industry_id: str
    candidate_rank: int | None
    candidate_tier: CandidateTier | None
    candidate_score: Decimal | None
    candidate_risk_penalty: Decimal | None
    leader_type: LeaderType | None
    mainline_state: MainlineState | None
    qualified: bool
    desired: bool
    prior_weight: Decimal
    unconstrained_weight: Decimal
    target_weight: Decimal
    action: MainlineLeaderAction
    reason_codes: tuple[MainlineLeaderReasonCode, ...]
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "industry_id", _non_empty(self.industry_id, "industry_id"))
        if self.candidate_rank is not None and self.candidate_rank < 1:
            raise ValueError("candidate rank must be positive")
        if (self.candidate_tier is None) != (self.candidate_score is None):
            raise ValueError("candidate tier and score must be present together")
        if (self.candidate_score is None) != (self.candidate_risk_penalty is None):
            raise ValueError("candidate score and risk penalty must be present together")
        for value in (self.candidate_score, self.candidate_risk_penalty):
            if value is not None:
                canonical_decimal(value, field_name="candidate selection score")
                if value < 0 or value > 100:
                    raise ValueError("candidate selection scores must be within 0..100")
        for value in (self.prior_weight, self.unconstrained_weight, self.target_weight):
            canonical_decimal(value, field_name="mainline-leader selection weight")
            if value < 0 or value > 1:
                raise ValueError("mainline-leader weights must be within 0..1")
        if self.desired and not self.qualified:
            raise ValueError("desired selection must qualify")
        expected_action = _weight_action(self.prior_weight, self.target_weight)
        if self.action is not expected_action:
            raise ValueError("selection action must match prior and target weights")
        if not self.reason_codes or len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("selection reason codes must be non-empty and unique")
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "selection rationale"))


def _weight_action(prior: Decimal, target: Decimal) -> MainlineLeaderAction:
    if prior == 0 and target == 0:
        return MainlineLeaderAction.EXCLUDE
    if prior == 0:
        return MainlineLeaderAction.ENTER
    if target == 0:
        return MainlineLeaderAction.EXIT
    if target > prior:
        return MainlineLeaderAction.INCREASE
    if target < prior:
        return MainlineLeaderAction.REDUCE
    return MainlineLeaderAction.HOLD


@dataclass(frozen=True, slots=True)
class MainlineLeaderTargetWeight:
    """One complete long-only stock target, including explicit zero exits."""

    instrument_id: str
    target_weight: Decimal
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        canonical_decimal(self.target_weight, field_name="stock target weight")
        if self.instrument_type is not TradableInstrumentType.STOCK:
            raise ValueError("mainline-leader targets must use STOCK instrument type")
        if self.target_weight < 0 or self.target_weight > 1:
            raise ValueError("stock target weight must be within 0..1")


@dataclass(frozen=True, slots=True)
class MainlineLeaderDecision:
    """Version-bound complete target portfolio before order or execution logic."""

    signal_date: date
    as_of: datetime
    session_index: int
    data_version: str
    strategy_version: str
    config_hash: str
    status: MainlineLeaderDecisionStatus
    market_regime: MarketRegime
    exposure_cap: Decimal
    gross_target_weight: Decimal
    cash_target_weight: Decimal
    forced_exit_turnover: Decimal
    discretionary_turnover: Decimal
    total_turnover: Decimal
    selections: tuple[MainlineLeaderSelection, ...]
    targets: tuple[MainlineLeaderTargetWeight, ...]
    regime_result_hash: str
    mainline_result_hash: str
    leader_result_hash: str
    candidate_result_hash: str
    portfolio_input_hash: str
    input_hash: str
    result_hash: str
    reason: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.signal_date:
            raise ValueError("strategy decision signal_date must match as_of")
        if (
            not isinstance(self.session_index, int)
            or isinstance(self.session_index, bool)
            or self.session_index < 0
        ):
            raise ValueError("decision session_index must be a non-negative integer")
        if any(not value.strip() for value in (self.data_version, self.strategy_version)):
            raise ValueError("strategy decision versions must be non-empty")
        for hash_value, name in (
            (self.config_hash, "config_hash"),
            (self.regime_result_hash, "regime_result_hash"),
            (self.mainline_result_hash, "mainline_result_hash"),
            (self.leader_result_hash, "leader_result_hash"),
            (self.candidate_result_hash, "candidate_result_hash"),
            (self.portfolio_input_hash, "portfolio_input_hash"),
            (self.input_hash, "input_hash"),
            (self.result_hash, "result_hash"),
        ):
            _validate_hash(hash_value, name)
        for metric in (
            self.exposure_cap,
            self.gross_target_weight,
            self.cash_target_weight,
            self.forced_exit_turnover,
            self.discretionary_turnover,
            self.total_turnover,
        ):
            canonical_decimal(metric, field_name="strategy portfolio metric")
            if metric < 0:
                raise ValueError("strategy portfolio metrics cannot be negative")
        if self.exposure_cap > 1 or self.gross_target_weight > self.exposure_cap:
            raise ValueError("gross target cannot exceed the exposure cap")
        if self.gross_target_weight + self.cash_target_weight != Decimal(1):
            raise ValueError("gross and cash target weights must sum to one")
        if self.total_turnover != self.forced_exit_turnover + self.discretionary_turnover:
            raise ValueError("total turnover must equal forced plus discretionary turnover")
        target_ids = tuple(item.instrument_id for item in self.targets)
        selection_ids = tuple(item.instrument_id for item in self.selections)
        if tuple(sorted(set(target_ids))) != target_ids or selection_ids != target_ids:
            raise ValueError("strategy targets and selections must share a sorted unique universe")
        if self.gross_target_weight != sum(
            (item.target_weight for item in self.targets), Decimal(0)
        ):
            raise ValueError("gross target must equal instrument target weights")
        for selection, target in zip(self.selections, self.targets, strict=True):
            if selection.target_weight != target.target_weight:
                raise ValueError("selection and portfolio target weights must match")
        expected_status = (
            MainlineLeaderDecisionStatus.CASH
            if self.gross_target_weight == 0
            else (
                MainlineLeaderDecisionStatus.HOLD
                if self.total_turnover == 0
                else MainlineLeaderDecisionStatus.REBALANCE
            )
        )
        if self.status is not expected_status:
            raise ValueError("decision status must match target exposure and turnover")
        object.__setattr__(self, "reason", _non_empty(self.reason, "decision reason"))
        expected_input_hash = stable_hash(
            {
                "candidate_result_hash": self.candidate_result_hash,
                "config_hash": self.config_hash,
                "leader_result_hash": self.leader_result_hash,
                "mainline_result_hash": self.mainline_result_hash,
                "portfolio_input_hash": self.portfolio_input_hash,
                "regime_result_hash": self.regime_result_hash,
                "request": {
                    "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
                    "data_version": self.data_version,
                    "session_index": self.session_index,
                    "signal_date": self.signal_date.isoformat(),
                },
            }
        )
        if self.input_hash != expected_input_hash:
            raise ValueError("strategy input hash does not match frozen inputs")
        expected_result_hash = stable_hash(
            {
                "cash_target_weight": self.cash_target_weight,
                "discretionary_turnover": self.discretionary_turnover,
                "exposure_cap": self.exposure_cap,
                "forced_exit_turnover": self.forced_exit_turnover,
                "gross_target_weight": self.gross_target_weight,
                "input_hash": self.input_hash,
                "reason": self.reason,
                "selections": [asdict(item) for item in self.selections],
                "status": self.status,
                "targets": [asdict(item) for item in self.targets],
                "total_turnover": self.total_turnover,
            }
        )
        if self.result_hash != expected_result_hash:
            raise ValueError("strategy result hash does not match decision output")

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "result_hash": self.result_hash,
            "signal_date": self.signal_date.isoformat(),
            "strategy_version": self.strategy_version,
        }


__all__ = [
    "MainlineLeaderAction",
    "MainlineLeaderConfig",
    "MainlineLeaderDecision",
    "MainlineLeaderDecisionStatus",
    "MainlineLeaderInputError",
    "MainlineLeaderPortfolioPosition",
    "MainlineLeaderPortfolioState",
    "MainlineLeaderReasonCode",
    "MainlineLeaderRequest",
    "MainlineLeaderSelection",
    "MainlineLeaderTargetWeight",
    "MainlineLeaderWeightingMode",
]
