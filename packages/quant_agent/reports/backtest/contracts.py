"""Immutable contracts for unified, reproducible backtest acceptance reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.backtest.validation import ParameterStabilitySummary
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


class BacktestReportInputError(ValueError):
    """Raised when report inputs cannot be aligned without an unsafe assumption."""


class FailureReasonCode(StrEnum):
    """Machine-readable reasons why a historical period is reported as weak."""

    DRAWDOWN = "DRAWDOWN"
    UNDERPERFORMANCE = "UNDERPERFORMANCE"


class GrossReturnMethod(StrEnum):
    """Frozen cost-before counterfactual used by the report adapter."""

    SAME_FILL_DAILY_FEE_ADDBACK = "SAME_FILL_DAILY_FEE_ADDBACK"


@dataclass(frozen=True, slots=True)
class BacktestReportConfig:
    """Versioned risk, annualization, and failure-period reporting semantics."""

    version: str = "backtest-report-v1"
    gross_return_method: GrossReturnMethod = GrossReturnMethod.SAME_FILL_DAILY_FEE_ADDBACK
    annualization_sessions: int = 252
    annual_risk_free_rate: Decimal = Decimal(0)
    failure_window_sessions: int = 20
    failure_drawdown_threshold: Decimal = Decimal("-0.10")
    failure_excess_return_threshold: Decimal = Decimal("-0.05")
    maximum_failure_periods: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "report config version"))
        for count in (
            self.annualization_sessions,
            self.failure_window_sessions,
            self.maximum_failure_periods,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("report session counts must be positive integers")
        for value in (
            self.annual_risk_free_rate,
            self.failure_drawdown_threshold,
            self.failure_excess_return_threshold,
        ):
            canonical_decimal(value, field_name="backtest report config value")
        if self.annual_risk_free_rate <= Decimal(-1):
            raise ValueError("annual risk-free rate must be greater than -1")
        if self.failure_drawdown_threshold >= 0:
            raise ValueError("failure drawdown threshold must be negative")
        if self.failure_excess_return_threshold > 0:
            raise ValueError("failure excess-return threshold must be non-positive")
        if self.gross_return_method is not GrossReturnMethod.SAME_FILL_DAILY_FEE_ADDBACK:
            raise ValueError("unsupported gross return method")

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class BacktestReportIdentity:
    """Exact snapshot, code, strategy, parameter, engine, and market-rule identity."""

    run_id: str
    as_of: datetime
    currency: str
    snapshot_id: str
    snapshot_hash: str
    data_version: str
    code_version: str
    code_hash: str
    strategy_name: str
    strategy_version: str
    strategy_config_hash: str
    parameter_version: str
    parameter_hash: str
    parameter_freeze_hashes: tuple[str, ...]
    engine_version: str
    market_rule_version: str
    market_rule_hash: str
    fee_rule_hash: str
    slippage_rule_hash: str
    benchmark_id: str
    source_input_hash: str
    source_result_hash: str
    walk_forward_result_hash: str | None = None
    event_log_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "currency",
            "snapshot_id",
            "data_version",
            "code_version",
            "strategy_name",
            "strategy_version",
            "parameter_version",
            "engine_version",
            "market_rule_version",
            "benchmark_id",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.as_of)
        for hash_value, field_name in (
            (self.snapshot_hash, "snapshot_hash"),
            (self.code_hash, "code_hash"),
            (self.strategy_config_hash, "strategy_config_hash"),
            (self.parameter_hash, "parameter_hash"),
            (self.market_rule_hash, "market_rule_hash"),
            (self.fee_rule_hash, "fee_rule_hash"),
            (self.slippage_rule_hash, "slippage_rule_hash"),
            (self.source_input_hash, "source_input_hash"),
            (self.source_result_hash, "source_result_hash"),
        ):
            _validate_hash(hash_value, field_name)
        if tuple(sorted(set(self.parameter_freeze_hashes))) != self.parameter_freeze_hashes:
            raise ValueError("parameter_freeze_hashes must be non-empty, unique, and sorted")
        if not self.parameter_freeze_hashes:
            raise ValueError("parameter_freeze_hashes must be non-empty")
        for freeze_hash in self.parameter_freeze_hashes:
            _validate_hash(freeze_hash, "parameter_freeze_hash")
        for optional_hash, field_name in (
            (self.walk_forward_result_hash, "walk_forward_result_hash"),
            (self.event_log_hash, "event_log_hash"),
        ):
            if optional_hash is not None:
                _validate_hash(optional_hash, field_name)

    @property
    def identity_hash(self) -> str:
        return stable_hash(asdict(self))

    def identity_payload(self) -> dict[str, str]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "code_version": self.code_version,
            "data_version": self.data_version,
            "identity_hash": self.identity_hash,
            "parameter_version": self.parameter_version,
            "snapshot_id": self.snapshot_id,
            "strategy_version": self.strategy_version,
        }


@dataclass(frozen=True, slots=True)
class SessionRegimeAttribution:
    """Prior-session stabilized regime used to attribute one future return session."""

    return_session: date
    source_session: date
    source_as_of: datetime
    regime: MarketRegime
    data_version: str
    model_version: str
    source_result_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.source_as_of)
        if self.source_session >= self.return_session:
            raise ValueError("regime source session must precede the attributed return session")
        if self.source_as_of.astimezone(SHANGHAI_TZ).date() != self.source_session:
            raise ValueError("regime source_as_of must fall on source_session")
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        object.__setattr__(self, "model_version", _non_empty(self.model_version, "model_version"))
        _validate_hash(self.source_result_hash, "regime source_result_hash")


@dataclass(frozen=True, slots=True)
class BacktestDailyInput:
    """One normalized daily source row with explicit gross and cost-after NAV."""

    trading_day: date
    available_at: datetime
    gross_nav: Decimal
    net_nav: Decimal
    benchmark_nav: Decimal
    turnover: Decimal
    transaction_cost: Decimal
    regime_attribution: SessionRegimeAttribution
    fillable_order_count: int = 0
    unfillable_order_count: int = 0

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.available_at.astimezone(SHANGHAI_TZ).date() < self.trading_day:
            raise ValueError("daily report input cannot be available before its trading day")
        if self.regime_attribution.return_session != self.trading_day:
            raise ValueError("regime attribution must bind the daily return session")
        if self.regime_attribution.source_as_of >= self.available_at:
            raise ValueError("regime attribution must be known before the daily result")
        for value in (
            self.gross_nav,
            self.net_nav,
            self.benchmark_nav,
            self.turnover,
            self.transaction_cost,
        ):
            canonical_decimal(value, field_name="daily backtest report input")
        if min(self.gross_nav, self.net_nav, self.benchmark_nav) <= 0:
            raise ValueError("daily report NAV values must be positive")
        if self.net_nav > self.gross_nav:
            raise ValueError("cost-after NAV cannot exceed gross NAV")
        if self.turnover < 0 or self.transaction_cost < 0:
            raise ValueError("daily report turnover and transaction cost cannot be negative")
        for count in (self.fillable_order_count, self.unfillable_order_count):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("daily report order counts must be non-negative integers")

    @property
    def input_hash(self) -> str:
        return stable_hash(asdict(self))

    @property
    def regime(self) -> MarketRegime:
        return self.regime_attribution.regime


@dataclass(frozen=True, slots=True)
class BacktestTableRow:
    """Canonical daily table row and sole source for rendered chart series."""

    trading_day: date
    gross_nav: Decimal
    net_nav: Decimal
    benchmark_nav: Decimal
    gross_return: Decimal
    net_return: Decimal
    benchmark_return: Decimal
    gross_drawdown: Decimal
    net_drawdown: Decimal
    turnover: Decimal
    transaction_cost: Decimal
    cumulative_transaction_cost: Decimal
    regime: MarketRegime
    regime_source_session: date
    regime_source_as_of: datetime
    regime_model_version: str
    regime_result_hash: str
    fillable_order_count: int
    unfillable_order_count: int

    def __post_init__(self) -> None:
        for value in (
            self.gross_nav,
            self.net_nav,
            self.benchmark_nav,
            self.gross_return,
            self.net_return,
            self.benchmark_return,
            self.gross_drawdown,
            self.net_drawdown,
            self.turnover,
            self.transaction_cost,
            self.cumulative_transaction_cost,
        ):
            canonical_decimal(value, field_name="backtest table value")
        if min(self.gross_nav, self.net_nav, self.benchmark_nav) <= 0:
            raise ValueError("backtest table NAV values must be positive")
        if self.gross_drawdown > 0 or self.net_drawdown > 0:
            raise ValueError("backtest table drawdowns must be non-positive")
        if min(self.turnover, self.transaction_cost, self.cumulative_transaction_cost) < 0:
            raise ValueError("backtest table turnover and costs cannot be negative")
        if min(self.fillable_order_count, self.unfillable_order_count) < 0:
            raise ValueError("backtest table order counts cannot be negative")
        ensure_aware(self.regime_source_as_of)
        if self.regime_source_session >= self.trading_day:
            raise ValueError("table regime source session must precede its return session")
        if self.regime_source_as_of.astimezone(SHANGHAI_TZ).date() != self.regime_source_session:
            raise ValueError("table regime source_as_of must fall on source session")
        object.__setattr__(
            self,
            "regime_model_version",
            _non_empty(self.regime_model_version, "regime_model_version"),
        )
        _validate_hash(self.regime_result_hash, "regime_result_hash")


@dataclass(frozen=True, slots=True)
class BacktestChartSeries:
    """Plot-ready arrays that must be an exact projection of report table rows."""

    trading_days: tuple[date, ...]
    gross_nav: tuple[Decimal, ...]
    net_nav: tuple[Decimal, ...]
    benchmark_nav: tuple[Decimal, ...]
    gross_drawdown: tuple[Decimal, ...]
    net_drawdown: tuple[Decimal, ...]
    turnover: tuple[Decimal, ...]
    cumulative_transaction_cost: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.trading_days),
            len(self.gross_nav),
            len(self.net_nav),
            len(self.benchmark_nav),
            len(self.gross_drawdown),
            len(self.net_drawdown),
            len(self.turnover),
            len(self.cumulative_transaction_cost),
        }
        if lengths != {len(self.trading_days)} or not self.trading_days:
            raise ValueError("backtest chart arrays must be non-empty and equally sized")
        if tuple(sorted(set(self.trading_days))) != self.trading_days:
            raise ValueError("backtest chart trading days must be unique and increasing")


@dataclass(frozen=True, slots=True)
class BacktestSummaryMetrics:
    """Balanced return, risk, turnover, cost, and execution summary."""

    observation_count: int
    gross_total_return: Decimal
    net_total_return: Decimal
    benchmark_total_return: Decimal
    gross_excess_return: Decimal
    net_excess_return: Decimal
    gross_annualized_return: Decimal
    net_annualized_return: Decimal
    annualized_volatility: Decimal
    sharpe_ratio: Decimal | None
    sortino_ratio: Decimal | None
    gross_maximum_drawdown: Decimal
    net_maximum_drawdown: Decimal
    total_turnover: Decimal
    average_daily_turnover: Decimal
    total_transaction_cost: Decimal
    transaction_cost_to_initial_nav: Decimal
    monthly_win_rate: Decimal | None
    unfillable_order_ratio: Decimal

    def __post_init__(self) -> None:
        if self.observation_count < 1:
            raise ValueError("backtest summary observation_count must be positive")
        values = (
            self.gross_total_return,
            self.net_total_return,
            self.benchmark_total_return,
            self.gross_excess_return,
            self.net_excess_return,
            self.gross_annualized_return,
            self.net_annualized_return,
            self.annualized_volatility,
            self.gross_maximum_drawdown,
            self.net_maximum_drawdown,
            self.total_turnover,
            self.average_daily_turnover,
            self.total_transaction_cost,
            self.transaction_cost_to_initial_nav,
            self.unfillable_order_ratio,
        )
        for value in values:
            canonical_decimal(value, field_name="backtest summary metric")
        for optional_metric in (
            self.sharpe_ratio,
            self.sortino_ratio,
            self.monthly_win_rate,
        ):
            if optional_metric is not None:
                canonical_decimal(
                    optional_metric,
                    field_name="optional backtest summary metric",
                )
        if self.annualized_volatility < 0:
            raise ValueError("annualized volatility cannot be negative")
        if self.gross_maximum_drawdown > 0 or self.net_maximum_drawdown > 0:
            raise ValueError("summary maximum drawdowns must be non-positive")
        if (
            min(
                self.total_turnover,
                self.average_daily_turnover,
                self.total_transaction_cost,
                self.transaction_cost_to_initial_nav,
            )
            < 0
        ):
            raise ValueError("summary turnover and cost metrics cannot be negative")
        if not Decimal(0) <= self.unfillable_order_ratio <= Decimal(1):
            raise ValueError("unfillable_order_ratio must be within 0..1")
        if self.monthly_win_rate is not None and not Decimal(0) <= self.monthly_win_rate <= 1:
            raise ValueError("monthly_win_rate must be within 0..1")


@dataclass(frozen=True, slots=True)
class RegimePerformanceSlice:
    """Compounded cost-before/after performance for one stabilized market regime."""

    regime: MarketRegime
    observation_count: int
    gross_return: Decimal
    net_return: Decimal
    benchmark_return: Decimal
    net_excess_return: Decimal
    net_maximum_drawdown: Decimal
    turnover: Decimal
    transaction_cost: Decimal
    unfillable_order_ratio: Decimal

    def __post_init__(self) -> None:
        if self.observation_count < 1:
            raise ValueError("regime slice observation_count must be positive")
        for value in (
            self.gross_return,
            self.net_return,
            self.benchmark_return,
            self.net_excess_return,
            self.net_maximum_drawdown,
            self.turnover,
            self.transaction_cost,
            self.unfillable_order_ratio,
        ):
            canonical_decimal(value, field_name="regime performance metric")
        if self.net_maximum_drawdown > 0:
            raise ValueError("regime maximum drawdown must be non-positive")
        if self.turnover < 0 or self.transaction_cost < 0:
            raise ValueError("regime turnover and cost cannot be negative")
        if not Decimal(0) <= self.unfillable_order_ratio <= 1:
            raise ValueError("regime unfillable ratio must be within 0..1")


@dataclass(frozen=True, slots=True)
class BacktestFailurePeriod:
    """One contiguous weak period with explicit drawdown/underperformance reasons."""

    start_date: date
    end_date: date
    observation_count: int
    gross_return: Decimal
    net_return: Decimal
    benchmark_return: Decimal
    net_excess_return: Decimal
    net_maximum_drawdown: Decimal
    regimes: tuple[MarketRegime, ...]
    reason_codes: tuple[FailureReasonCode, ...]

    def __post_init__(self) -> None:
        if self.end_date < self.start_date or self.observation_count < 1:
            raise ValueError("failure period dates and observation_count are invalid")
        for value in (
            self.gross_return,
            self.net_return,
            self.benchmark_return,
            self.net_excess_return,
            self.net_maximum_drawdown,
        ):
            canonical_decimal(value, field_name="failure period metric")
        if self.net_maximum_drawdown > 0:
            raise ValueError("failure period maximum drawdown must be non-positive")
        expected_regimes = tuple(value for value in MarketRegime if value in set(self.regimes))
        if not expected_regimes or self.regimes != expected_regimes:
            raise ValueError("failure period regimes must use canonical enum order")
        expected_reasons = tuple(
            value for value in FailureReasonCode if value in set(self.reason_codes)
        )
        if not expected_reasons or self.reason_codes != expected_reasons:
            raise ValueError("failure period reasons must use canonical enum order")


@dataclass(frozen=True, slots=True)
class BacktestAcceptanceReport:
    """Canonical P4-T08 report with table/chart and version-bound analytics."""

    identity: BacktestReportIdentity
    report_config_version: str
    report_config_hash: str
    initial_nav: Decimal
    period_start: date
    period_end: date
    daily_input_hashes: tuple[str, ...]
    table_rows: tuple[BacktestTableRow, ...]
    chart_series: BacktestChartSeries
    summary: BacktestSummaryMetrics
    regime_slices: tuple[RegimePerformanceSlice, ...]
    parameter_stability: tuple[ParameterStabilitySummary, ...]
    failure_periods: tuple[BacktestFailurePeriod, ...]
    input_hash: str
    result_hash: str
    interpretation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_config_version",
            _non_empty(self.report_config_version, "report config version"),
        )
        canonical_decimal(self.initial_nav, field_name="report initial_nav")
        if self.initial_nav <= 0:
            raise ValueError("report initial_nav must be positive")
        for value, field_name in (
            (self.report_config_hash, "report_config_hash"),
            (self.input_hash, "report input_hash"),
            (self.result_hash, "report result_hash"),
        ):
            _validate_hash(value, field_name)
        if not self.table_rows:
            raise ValueError("backtest report requires daily table rows")
        if len(self.daily_input_hashes) != len(self.table_rows):
            raise ValueError("each report table row requires one daily input hash")
        for value in self.daily_input_hashes:
            _validate_hash(value, "daily_input_hash")
        dates = tuple(item.trading_day for item in self.table_rows)
        if tuple(sorted(set(dates))) != dates:
            raise ValueError("backtest report table dates must be unique and increasing")
        if self.period_start != dates[0] or self.period_end != dates[-1]:
            raise ValueError("backtest report period must match table boundaries")
        if self.identity.as_of.astimezone(SHANGHAI_TZ).date() < self.period_end:
            raise ValueError("report identity as_of cannot precede the report period")
        if self.summary.observation_count != len(self.table_rows):
            raise ValueError("report summary count must match table rows")
        expected_regimes = tuple(
            value for value in MarketRegime if any(item.regime is value for item in self.table_rows)
        )
        if tuple(item.regime for item in self.regime_slices) != expected_regimes:
            raise ValueError("report regime slices must cover observed regimes in enum order")
        stability_keys = tuple(
            (item.parameter_freeze_hash, item.parameter_name) for item in self.parameter_stability
        )
        if not stability_keys or tuple(sorted(set(stability_keys))) != stability_keys:
            raise ValueError("report parameter stability must be unique and sorted")
        observed_freezes = {item.parameter_freeze_hash for item in self.parameter_stability}
        if observed_freezes != set(self.identity.parameter_freeze_hashes):
            raise ValueError("report sensitivity must cover every frozen parameter assignment")
        if tuple(item.start_date for item in self.failure_periods) != tuple(
            sorted(item.start_date for item in self.failure_periods)
        ):
            raise ValueError("failure periods must use chronological order")
        self._validate_chart_projection()
        object.__setattr__(
            self, "interpretation", _non_empty(self.interpretation, "interpretation")
        )
        expected_input_hash = stable_hash(
            {
                "daily_input_hashes": self.daily_input_hashes,
                "identity_hash": self.identity.identity_hash,
                "initial_nav": self.initial_nav,
                "parameter_stability": [item.result_hash for item in self.parameter_stability],
                "report_config_hash": self.report_config_hash,
            }
        )
        if self.input_hash != expected_input_hash:
            raise ValueError("backtest report input_hash does not match frozen inputs")
        expected_result_hash = stable_hash(
            {
                "chart_series": asdict(self.chart_series),
                "failure_periods": [asdict(item) for item in self.failure_periods],
                "identity_hash": self.identity.identity_hash,
                "input_hash": self.input_hash,
                "interpretation": self.interpretation,
                "parameter_stability": [item.result_hash for item in self.parameter_stability],
                "regime_slices": [asdict(item) for item in self.regime_slices],
                "summary": asdict(self.summary),
                "table_rows": [asdict(item) for item in self.table_rows],
            }
        )
        if self.result_hash != expected_result_hash:
            raise ValueError("backtest report result_hash does not match report output")

    def _validate_chart_projection(self) -> None:
        rows = self.table_rows
        expected = BacktestChartSeries(
            trading_days=tuple(item.trading_day for item in rows),
            gross_nav=tuple(item.gross_nav for item in rows),
            net_nav=tuple(item.net_nav for item in rows),
            benchmark_nav=tuple(item.benchmark_nav for item in rows),
            gross_drawdown=tuple(item.gross_drawdown for item in rows),
            net_drawdown=tuple(item.net_drawdown for item in rows),
            turnover=tuple(item.turnover for item in rows),
            cumulative_transaction_cost=tuple(item.cumulative_transaction_cost for item in rows),
        )
        if self.chart_series != expected:
            raise ValueError("backtest chart series must exactly project report table rows")

    def identity_payload(self) -> dict[str, str]:
        return {
            "identity_hash": self.identity.identity_hash,
            "input_hash": self.input_hash,
            "report_config_hash": self.report_config_hash,
            "result_hash": self.result_hash,
        }


__all__ = [
    "BacktestAcceptanceReport",
    "BacktestChartSeries",
    "BacktestDailyInput",
    "BacktestFailurePeriod",
    "BacktestReportConfig",
    "BacktestReportIdentity",
    "BacktestReportInputError",
    "BacktestSummaryMetrics",
    "BacktestTableRow",
    "FailureReasonCode",
    "GrossReturnMethod",
    "RegimePerformanceSlice",
    "SessionRegimeAttribution",
]
