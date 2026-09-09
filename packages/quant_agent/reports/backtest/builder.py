"""Deterministic construction of unified backtest acceptance reports."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from quant_agent.backtest import BacktestResult
from quant_agent.backtest.validation import ParameterStabilitySummary
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash
from quant_agent.reports.backtest.contracts import (
    BacktestAcceptanceReport,
    BacktestChartSeries,
    BacktestDailyInput,
    BacktestFailurePeriod,
    BacktestReportConfig,
    BacktestReportIdentity,
    BacktestReportInputError,
    BacktestSummaryMetrics,
    BacktestTableRow,
    FailureReasonCode,
    RegimePerformanceSlice,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)


def _divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        raise ZeroDivisionError("backtest report denominator cannot be zero")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return numerator / denominator


def _compound(returns: Iterable[Decimal]) -> Decimal:
    result = _ONE
    for value in returns:
        result *= _ONE + value
    return result - _ONE


def _annualized_return(
    total_return: Decimal,
    period_count: int,
    annualization_sessions: int,
) -> Decimal:
    if period_count == 0 or total_return == 0:
        return _ZERO
    growth = _ONE + total_return
    if growth <= 0:
        raise BacktestReportInputError("annualized return requires positive terminal growth")
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        exponent = Decimal(annualization_sessions) / Decimal(period_count)
        return (growth.ln() * exponent).exp() - _ONE


def _annualized_volatility(
    returns: tuple[Decimal, ...],
    annualization_sessions: int,
) -> Decimal:
    if len(returns) < 2:
        return _ZERO
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        mean = sum(returns, _ZERO) / Decimal(len(returns))
        variance = sum(((value - mean) ** 2 for value in returns), _ZERO) / Decimal(
            len(returns) - 1
        )
        return (variance * Decimal(annualization_sessions)).sqrt()


def _sharpe_ratio(
    returns: tuple[Decimal, ...],
    volatility: Decimal,
    config: BacktestReportConfig,
) -> Decimal | None:
    if not returns or volatility == 0:
        return None
    periodic_risk_free = _divide(
        config.annual_risk_free_rate,
        Decimal(config.annualization_sessions),
    )
    mean_return = _divide(sum(returns, _ZERO), Decimal(len(returns)))
    return _divide(
        (mean_return - periodic_risk_free) * Decimal(config.annualization_sessions),
        volatility,
    )


def _sortino_ratio(
    returns: tuple[Decimal, ...],
    config: BacktestReportConfig,
) -> Decimal | None:
    if not returns:
        return None
    periodic_risk_free = _divide(
        config.annual_risk_free_rate,
        Decimal(config.annualization_sessions),
    )
    downside = tuple(min(_ZERO, value - periodic_risk_free) for value in returns)
    with localcontext() as context:
        context.prec = 34
        downside_deviation = (
            _divide(sum((value**2 for value in downside), _ZERO), Decimal(len(downside)))
            * Decimal(config.annualization_sessions)
        ).sqrt()
    if downside_deviation == 0:
        return None
    mean_return = _divide(sum(returns, _ZERO), Decimal(len(returns)))
    return _divide(
        (mean_return - periodic_risk_free) * Decimal(config.annualization_sessions),
        downside_deviation,
    )


def _drawdown_from_returns(returns: Iterable[Decimal]) -> Decimal:
    nav = _ONE
    peak = _ONE
    worst = _ZERO
    for value in returns:
        nav *= _ONE + value
        peak = max(peak, nav)
        worst = min(worst, _divide(nav, peak) - _ONE)
    return worst


def _unfillable_ratio(fillable: int, unfillable: int) -> Decimal:
    total = fillable + unfillable
    return _divide(Decimal(unfillable), Decimal(total)) if total else _ZERO


def _table_rows(
    daily_inputs: tuple[BacktestDailyInput, ...],
    initial_nav: Decimal,
) -> tuple[BacktestTableRow, ...]:
    previous_gross = initial_nav
    previous_net = initial_nav
    previous_benchmark = initial_nav
    gross_peak = initial_nav
    net_peak = initial_nav
    cumulative_cost = _ZERO
    rows: list[BacktestTableRow] = []
    for item in daily_inputs:
        gross_return = _divide(item.gross_nav, previous_gross) - _ONE
        net_return = _divide(item.net_nav, previous_net) - _ONE
        benchmark_return = _divide(item.benchmark_nav, previous_benchmark) - _ONE
        gross_peak = max(gross_peak, item.gross_nav)
        net_peak = max(net_peak, item.net_nav)
        cumulative_cost += item.transaction_cost
        rows.append(
            BacktestTableRow(
                trading_day=item.trading_day,
                gross_nav=item.gross_nav,
                net_nav=item.net_nav,
                benchmark_nav=item.benchmark_nav,
                gross_return=gross_return,
                net_return=net_return,
                benchmark_return=benchmark_return,
                gross_drawdown=_divide(item.gross_nav, gross_peak) - _ONE,
                net_drawdown=_divide(item.net_nav, net_peak) - _ONE,
                turnover=item.turnover,
                transaction_cost=item.transaction_cost,
                cumulative_transaction_cost=cumulative_cost,
                regime=item.regime,
                regime_source_session=item.regime_attribution.source_session,
                regime_source_as_of=item.regime_attribution.source_as_of,
                regime_model_version=item.regime_attribution.model_version,
                regime_result_hash=item.regime_attribution.source_result_hash,
                fillable_order_count=item.fillable_order_count,
                unfillable_order_count=item.unfillable_order_count,
            )
        )
        previous_gross = item.gross_nav
        previous_net = item.net_nav
        previous_benchmark = item.benchmark_nav
    return tuple(rows)


def _monthly_win_rate(
    rows: tuple[BacktestTableRow, ...],
    initial_nav: Decimal,
) -> Decimal | None:
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row.trading_day.year, row.trading_day.month), []).append(index)
    if not grouped:
        return None
    wins = 0
    for indexes in grouped.values():
        start_index = indexes[0]
        end_index = indexes[-1]
        start_net = rows[start_index - 1].net_nav if start_index else initial_nav
        start_benchmark = rows[start_index - 1].benchmark_nav if start_index else initial_nav
        net_return = _divide(rows[end_index].net_nav, start_net) - _ONE
        benchmark_return = _divide(rows[end_index].benchmark_nav, start_benchmark) - _ONE
        wins += net_return > benchmark_return
    return _divide(Decimal(wins), Decimal(len(grouped)))


def _summary(
    rows: tuple[BacktestTableRow, ...],
    initial_nav: Decimal,
    config: BacktestReportConfig,
) -> BacktestSummaryMetrics:
    gross_total = _divide(rows[-1].gross_nav, initial_nav) - _ONE
    net_total = _divide(rows[-1].net_nav, initial_nav) - _ONE
    benchmark_total = _divide(rows[-1].benchmark_nav, initial_nav) - _ONE
    net_returns = tuple(item.net_return for item in rows)
    volatility = _annualized_volatility(net_returns, config.annualization_sessions)
    turnover = sum((item.turnover for item in rows), _ZERO)
    cost = rows[-1].cumulative_transaction_cost
    fillable = sum(item.fillable_order_count for item in rows)
    unfillable = sum(item.unfillable_order_count for item in rows)
    return BacktestSummaryMetrics(
        observation_count=len(rows),
        gross_total_return=gross_total,
        net_total_return=net_total,
        benchmark_total_return=benchmark_total,
        gross_excess_return=gross_total - benchmark_total,
        net_excess_return=net_total - benchmark_total,
        gross_annualized_return=_annualized_return(
            gross_total,
            len(rows),
            config.annualization_sessions,
        ),
        net_annualized_return=_annualized_return(
            net_total,
            len(rows),
            config.annualization_sessions,
        ),
        annualized_volatility=volatility,
        sharpe_ratio=_sharpe_ratio(net_returns, volatility, config),
        sortino_ratio=_sortino_ratio(net_returns, config),
        gross_maximum_drawdown=min(item.gross_drawdown for item in rows),
        net_maximum_drawdown=min(item.net_drawdown for item in rows),
        total_turnover=turnover,
        average_daily_turnover=_divide(turnover, Decimal(len(rows))),
        total_transaction_cost=cost,
        transaction_cost_to_initial_nav=_divide(cost, initial_nav),
        monthly_win_rate=_monthly_win_rate(rows, initial_nav),
        unfillable_order_ratio=_unfillable_ratio(fillable, unfillable),
    )


def _regime_slices(
    rows: tuple[BacktestTableRow, ...],
) -> tuple[RegimePerformanceSlice, ...]:
    result: list[RegimePerformanceSlice] = []
    for regime in MarketRegime:
        selected = tuple(item for item in rows if item.regime is regime)
        if not selected:
            continue
        gross = _compound(item.gross_return for item in selected)
        net = _compound(item.net_return for item in selected)
        benchmark = _compound(item.benchmark_return for item in selected)
        fillable = sum(item.fillable_order_count for item in selected)
        unfillable = sum(item.unfillable_order_count for item in selected)
        result.append(
            RegimePerformanceSlice(
                regime=regime,
                observation_count=len(selected),
                gross_return=gross,
                net_return=net,
                benchmark_return=benchmark,
                net_excess_return=net - benchmark,
                net_maximum_drawdown=_drawdown_from_returns(item.net_return for item in selected),
                turnover=sum((item.turnover for item in selected), _ZERO),
                transaction_cost=sum(
                    (item.transaction_cost for item in selected),
                    _ZERO,
                ),
                unfillable_order_ratio=_unfillable_ratio(fillable, unfillable),
            )
        )
    return tuple(result)


def _failure_periods(
    rows: tuple[BacktestTableRow, ...],
    initial_nav: Decimal,
    config: BacktestReportConfig,
) -> tuple[BacktestFailurePeriod, ...]:
    reasons_by_index: dict[int, tuple[FailureReasonCode, ...]] = {}
    for index, row in enumerate(rows):
        reasons: list[FailureReasonCode] = []
        if row.net_drawdown <= config.failure_drawdown_threshold:
            reasons.append(FailureReasonCode.DRAWDOWN)
        if index + 1 >= config.failure_window_sessions:
            window = rows[index + 1 - config.failure_window_sessions : index + 1]
            net = _compound(item.net_return for item in window)
            benchmark = _compound(item.benchmark_return for item in window)
            if net - benchmark <= config.failure_excess_return_threshold:
                reasons.append(FailureReasonCode.UNDERPERFORMANCE)
        if reasons:
            reasons_by_index[index] = tuple(reasons)

    groups: list[tuple[int, int, tuple[FailureReasonCode, ...]]] = []
    start: int | None = None
    last: int | None = None
    active_reasons: set[FailureReasonCode] = set()
    for index in sorted(reasons_by_index):
        if start is None or last is None or index != last + 1:
            if start is not None and last is not None:
                groups.append(
                    (
                        start,
                        last,
                        tuple(value for value in FailureReasonCode if value in active_reasons),
                    )
                )
            start = index
            active_reasons = set()
        active_reasons.update(reasons_by_index[index])
        last = index
    if start is not None and last is not None:
        groups.append(
            (
                start,
                last,
                tuple(value for value in FailureReasonCode if value in active_reasons),
            )
        )

    periods: list[BacktestFailurePeriod] = []
    for start_index, end_index, reason_codes in groups:
        selected = rows[start_index : end_index + 1]
        start_gross = rows[start_index - 1].gross_nav if start_index else initial_nav
        start_net = rows[start_index - 1].net_nav if start_index else initial_nav
        start_benchmark = rows[start_index - 1].benchmark_nav if start_index else initial_nav
        gross_return = _divide(selected[-1].gross_nav, start_gross) - _ONE
        net_return = _divide(selected[-1].net_nav, start_net) - _ONE
        benchmark_return = _divide(selected[-1].benchmark_nav, start_benchmark) - _ONE
        periods.append(
            BacktestFailurePeriod(
                start_date=selected[0].trading_day,
                end_date=selected[-1].trading_day,
                observation_count=len(selected),
                gross_return=gross_return,
                net_return=net_return,
                benchmark_return=benchmark_return,
                net_excess_return=net_return - benchmark_return,
                net_maximum_drawdown=min(item.net_drawdown for item in selected),
                regimes=tuple(
                    value
                    for value in MarketRegime
                    if any(item.regime is value for item in selected)
                ),
                reason_codes=reason_codes,
            )
        )
    if len(periods) > config.maximum_failure_periods:
        periods = sorted(
            periods,
            key=lambda item: (
                item.net_excess_return,
                item.net_maximum_drawdown,
                item.start_date,
            ),
        )[: config.maximum_failure_periods]
    return tuple(sorted(periods, key=lambda item: item.start_date))


def build_backtest_acceptance_report(
    *,
    identity: BacktestReportIdentity,
    daily_inputs: Iterable[BacktestDailyInput],
    initial_nav: Decimal,
    parameter_stability: Iterable[ParameterStabilitySummary] = (),
    config: BacktestReportConfig | None = None,
    event_result: BacktestResult | None = None,
) -> BacktestAcceptanceReport:
    """Build one table/chart-consistent report from frozen normalized daily inputs."""

    active_config = config or BacktestReportConfig()
    if not initial_nav.is_finite() or initial_nav <= 0:
        raise ValueError("report initial_nav must be finite and positive")
    frozen_inputs = tuple(daily_inputs)
    if not frozen_inputs:
        raise BacktestReportInputError("backtest report requires daily inputs")
    dates = tuple(item.trading_day for item in frozen_inputs)
    if tuple(sorted(set(dates))) != dates:
        raise BacktestReportInputError(
            "backtest report daily inputs must already be unique and chronological"
        )
    if any(item.available_at > identity.as_of for item in frozen_inputs):
        raise BacktestReportInputError("future daily input is not available at report as_of")
    if identity.event_log_hash is None:
        if event_result is not None:
            raise BacktestReportInputError(
                "event replay result requires event_log_hash in report identity"
            )
    else:
        if event_result is None:
            raise BacktestReportInputError("event-bound report identity requires the replay result")
        recomputed_event_hash = hashlib.sha256(event_result.event_log.encode("utf-8")).hexdigest()
        if (
            event_result.event_log_hash != recomputed_event_hash
            or identity.event_log_hash != recomputed_event_hash
            or event_result.run_id != identity.run_id
            or event_result.currency != identity.currency
        ):
            raise BacktestReportInputError(
                "event replay result does not bind identity, currency, and canonical log"
            )
    stability = tuple(
        sorted(
            parameter_stability,
            key=lambda item: (item.parameter_freeze_hash, item.parameter_name),
        )
    )
    stability_keys = tuple((item.parameter_freeze_hash, item.parameter_name) for item in stability)
    if not stability_keys:
        raise BacktestReportInputError(
            "backtest acceptance report requires parameter sensitivity evidence"
        )
    if len(set(stability_keys)) != len(stability_keys):
        raise BacktestReportInputError("parameter stability summaries must be unique")
    if {item.parameter_freeze_hash for item in stability} != set(identity.parameter_freeze_hashes):
        raise BacktestReportInputError(
            "parameter sensitivity must cover every identity freeze hash"
        )

    rows = _table_rows(frozen_inputs, initial_nav)
    chart = BacktestChartSeries(
        trading_days=tuple(item.trading_day for item in rows),
        gross_nav=tuple(item.gross_nav for item in rows),
        net_nav=tuple(item.net_nav for item in rows),
        benchmark_nav=tuple(item.benchmark_nav for item in rows),
        gross_drawdown=tuple(item.gross_drawdown for item in rows),
        net_drawdown=tuple(item.net_drawdown for item in rows),
        turnover=tuple(item.turnover for item in rows),
        cumulative_transaction_cost=tuple(item.cumulative_transaction_cost for item in rows),
    )
    summary = _summary(rows, initial_nav, active_config)
    regime_slices = _regime_slices(rows)
    failure_periods = _failure_periods(rows, initial_nav, active_config)
    daily_hashes = tuple(item.input_hash for item in frozen_inputs)
    input_hash = stable_hash(
        {
            "daily_input_hashes": daily_hashes,
            "identity_hash": identity.identity_hash,
            "initial_nav": initial_nav,
            "parameter_stability": [item.result_hash for item in stability],
            "report_config_hash": active_config.config_hash,
        }
    )
    execution_scope = (
        "event replay evidence is bound"
        if identity.event_log_hash is not None
        else "research/vectorized evidence only; final event-driven acceptance is not bound"
    )
    interpretation = (
        "descriptive backtest acceptance report with SAME_FILL_DAILY_FEE_ADDBACK gross "
        f"and cost-after results; {execution_scope}; historical performance, regime slices, "
        "and sensitivity do not guarantee future returns"
    )
    result_payload = {
        "chart_series": asdict(chart),
        "failure_periods": [asdict(item) for item in failure_periods],
        "identity_hash": identity.identity_hash,
        "input_hash": input_hash,
        "interpretation": interpretation,
        "parameter_stability": [item.result_hash for item in stability],
        "regime_slices": [asdict(item) for item in regime_slices],
        "summary": asdict(summary),
        "table_rows": [asdict(item) for item in rows],
    }
    return BacktestAcceptanceReport(
        identity=identity,
        report_config_version=active_config.version,
        report_config_hash=active_config.config_hash,
        initial_nav=initial_nav,
        period_start=rows[0].trading_day,
        period_end=rows[-1].trading_day,
        daily_input_hashes=daily_hashes,
        table_rows=rows,
        chart_series=chart,
        summary=summary,
        regime_slices=regime_slices,
        parameter_stability=stability,
        failure_periods=failure_periods,
        input_hash=input_hash,
        result_hash=stable_hash(result_payload),
        interpretation=interpretation,
    )


__all__ = ["build_backtest_acceptance_report"]
