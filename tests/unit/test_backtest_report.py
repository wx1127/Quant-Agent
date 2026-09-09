"""Unified version-bound backtest report and vectorized-engine adapter tests."""

import hashlib
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
from tests.unit import test_vectorized_backtest as vector_fixtures

from quant_agent.backtest import BacktestResult, VectorizedBacktestEngine
from quant_agent.backtest.validation import (
    build_parameter_perturbation,
    evaluate_parameter_stability,
)
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash
from quant_agent.reports.backtest import (
    BacktestDailyInput,
    BacktestReportConfig,
    BacktestReportIdentity,
    BacktestReportInputError,
    FailureReasonCode,
    SessionRegimeAttribution,
    build_backtest_acceptance_report,
    vectorized_daily_inputs,
)


def _identity(as_of: datetime) -> BacktestReportIdentity:
    return BacktestReportIdentity(
        run_id="acceptance-run-1",
        as_of=as_of,
        currency="CNY",
        snapshot_id="snapshot-2026-02-04",
        snapshot_hash=stable_hash({"snapshot": "2026-02-04"}),
        data_version="snapshot-v1",
        code_version="git:abc123",
        code_hash=stable_hash({"code": "abc123"}),
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        strategy_config_hash=stable_hash({"strategy": "mainline-leader-v1"}),
        parameter_version="params-v2",
        parameter_hash=stable_hash({"parameters": "params-v2"}),
        parameter_freeze_hashes=("f" * 64,),
        engine_version="vectorized-v1",
        market_rule_version="cn-market-v1",
        market_rule_hash=stable_hash({"market_rule": "cn-market-v1"}),
        fee_rule_hash=stable_hash({"fee_rule": "cn-fee-v1"}),
        slippage_rule_hash=stable_hash({"slippage_rule": "close-v1"}),
        benchmark_id="CN.SH.000300",
        source_input_hash=stable_hash({"source": "input"}),
        source_result_hash=stable_hash({"source": "result"}),
        walk_forward_result_hash=stable_hash({"walk_forward": "result"}),
    )


def _regime_attribution(
    return_session: date,
    regime: MarketRegime,
) -> SessionRegimeAttribution:
    source_session = return_session - timedelta(days=1)
    return SessionRegimeAttribution(
        return_session=return_session,
        source_session=source_session,
        source_as_of=datetime.combine(source_session, time(18), tzinfo=SHANGHAI_TZ),
        regime=regime,
        data_version="snapshot-v1",
        model_version="regime-transition-v1",
        source_result_hash=stable_hash({"regime": regime, "source_session": source_session}),
    )


def _daily_inputs() -> tuple[BacktestDailyInput, ...]:
    start = date(2026, 1, 28)
    gross_navs = ("102", "104", "100", "95", "94", "96", "98", "101")
    net_navs = ("101", "102", "97", "90", "88", "90", "93", "96")
    benchmark_navs = ("100", "101", "100", "99", "98", "99", "100", "102")
    result = []
    for index, values in enumerate(zip(gross_navs, net_navs, benchmark_navs, strict=True)):
        trading_day = start + timedelta(days=index)
        result.append(
            BacktestDailyInput(
                trading_day=trading_day,
                available_at=datetime.combine(
                    trading_day,
                    time(18),
                    tzinfo=SHANGHAI_TZ,
                ),
                gross_nav=Decimal(values[0]),
                net_nav=Decimal(values[1]),
                benchmark_nav=Decimal(values[2]),
                turnover=Decimal("0.10"),
                transaction_cost=Decimal(1),
                regime_attribution=_regime_attribution(
                    trading_day,
                    MarketRegime.UPTREND if index < 4 else MarketRegime.DOWNTREND,
                ),
                fillable_order_count=1,
                unfillable_order_count=1 if index in {2, 3} else 0,
            )
        )
    return tuple(result)


def _stability():  # type: ignore[no-untyped-def]
    freeze_hash = "f" * 64
    perturbations = tuple(
        build_parameter_perturbation(
            parameter_freeze_hash=freeze_hash,
            parameter_name="maximum_positions",
            relative_change=change,
            net_metric=metric,
        )
        for change, metric in (
            (Decimal("-0.10"), Decimal("0.07")),
            (Decimal("0.10"), Decimal("0.075")),
        )
    )
    return evaluate_parameter_stability(
        parameter_freeze_hash=freeze_hash,
        parameter_name="maximum_positions",
        baseline_net_metric=Decimal("0.08"),
        perturbations=perturbations,
    )


def _report():  # type: ignore[no-untyped-def]
    inputs = _daily_inputs()
    return build_backtest_acceptance_report(
        identity=_identity(inputs[-1].available_at),
        daily_inputs=inputs,
        initial_nav=Decimal(100),
        parameter_stability=(_stability(),),
        config=BacktestReportConfig(
            failure_window_sessions=3,
            failure_drawdown_threshold=Decimal("-0.08"),
            failure_excess_return_threshold=Decimal("-0.03"),
        ),
    )


def test_report_is_reproducible_version_bound_and_balances_return_risk_cost() -> None:
    report = _report()
    repeated = _report()

    assert report == repeated
    assert report.identity.data_version == "snapshot-v1"
    assert report.identity.code_version == "git:abc123"
    assert report.identity.strategy_version == "mainline-leader-v1"
    assert report.identity.parameter_version == "params-v2"
    assert report.summary.gross_total_return == Decimal("0.01")
    assert report.summary.net_total_return == Decimal("-0.04")
    assert report.summary.benchmark_total_return == Decimal("0.02")
    assert report.summary.gross_excess_return == Decimal("-0.01")
    assert report.summary.net_excess_return == Decimal("-0.06")
    assert report.summary.gross_maximum_drawdown < 0
    assert report.summary.net_maximum_drawdown < report.summary.gross_maximum_drawdown
    assert report.summary.total_turnover == Decimal("0.80")
    assert report.summary.average_daily_turnover == Decimal("0.10")
    assert report.summary.total_transaction_cost == Decimal(8)
    assert report.summary.transaction_cost_to_initial_nav == Decimal("0.08")
    assert report.summary.monthly_win_rate == Decimal("0.5")
    assert report.summary.unfillable_order_ratio == Decimal("0.2")
    assert report.summary.sharpe_ratio is not None
    assert report.summary.sortino_ratio is not None
    assert "do not guarantee future returns" in report.interpretation
    assert len(report.input_hash) == len(report.result_hash) == 64


def test_chart_series_is_an_exact_projection_of_the_canonical_table() -> None:
    report = _report()

    assert report.chart_series.trading_days == tuple(item.trading_day for item in report.table_rows)
    assert report.chart_series.gross_nav == tuple(item.gross_nav for item in report.table_rows)
    assert report.chart_series.net_nav == tuple(item.net_nav for item in report.table_rows)
    assert report.chart_series.benchmark_nav == tuple(
        item.benchmark_nav for item in report.table_rows
    )
    assert report.chart_series.net_drawdown == tuple(
        item.net_drawdown for item in report.table_rows
    )
    assert report.chart_series.cumulative_transaction_cost[-1] == Decimal(8)
    assert all(item.regime_source_session < item.trading_day for item in report.table_rows)
    assert all(len(item.regime_result_hash) == 64 for item in report.table_rows)

    changed_chart = replace(
        report.chart_series,
        net_nav=(Decimal(999), *report.chart_series.net_nav[1:]),
    )
    with pytest.raises(ValueError, match="exactly project"):
        replace(report, chart_series=changed_chart)


def test_regime_slices_include_cost_and_execution_capacity_in_enum_order() -> None:
    report = _report()

    assert tuple(item.regime for item in report.regime_slices) == (
        MarketRegime.UPTREND,
        MarketRegime.DOWNTREND,
    )
    uptrend, downtrend = report.regime_slices
    assert uptrend.observation_count == downtrend.observation_count == 4
    assert uptrend.transaction_cost == downtrend.transaction_cost == Decimal(4)
    assert uptrend.unfillable_order_ratio == Decimal(
        "0.33333333333333333333333333333333333333333333333333"
    )
    assert downtrend.unfillable_order_ratio == 0
    assert downtrend.net_return > 0


def test_failure_periods_are_contiguous_explainable_and_chronological() -> None:
    report = _report()

    assert len(report.failure_periods) == 1
    failure = report.failure_periods[0]
    assert failure.start_date == date(2026, 1, 30)
    assert failure.end_date == date(2026, 2, 3)
    assert failure.observation_count == 5
    assert failure.net_return < failure.benchmark_return
    assert failure.net_maximum_drawdown <= Decimal("-0.08")
    assert failure.reason_codes == (
        FailureReasonCode.DRAWDOWN,
        FailureReasonCode.UNDERPERFORMANCE,
    )
    assert failure.regimes == (MarketRegime.UPTREND, MarketRegime.DOWNTREND)


def test_report_rejects_future_unordered_duplicate_or_incoherent_daily_inputs() -> None:
    inputs = _daily_inputs()
    identity = _identity(inputs[-1].available_at)
    with pytest.raises(BacktestReportInputError, match="unique and chronological"):
        build_backtest_acceptance_report(
            identity=identity,
            daily_inputs=tuple(reversed(inputs)),
            initial_nav=Decimal(100),
        )
    with pytest.raises(BacktestReportInputError, match="unique and chronological"):
        build_backtest_acceptance_report(
            identity=identity,
            daily_inputs=(inputs[0], inputs[0], *inputs[1:]),
            initial_nav=Decimal(100),
        )
    future = replace(
        inputs[-1],
        available_at=identity.as_of + timedelta(seconds=1),
    )
    with pytest.raises(BacktestReportInputError, match="future daily input"):
        build_backtest_acceptance_report(
            identity=identity,
            daily_inputs=(*inputs[:-1], future),
            initial_nav=Decimal(100),
        )
    with pytest.raises(ValueError, match="cannot exceed gross NAV"):
        replace(inputs[0], net_nav=Decimal(103))
    with pytest.raises(ValueError, match="must precede"):
        replace(
            inputs[0].regime_attribution,
            source_session=inputs[0].trading_day,
            source_as_of=inputs[0].available_at,
        )


def test_report_hashes_reject_input_output_and_version_tampering() -> None:
    report = _report()

    with pytest.raises(ValueError, match="input_hash does not match"):
        replace(report, daily_input_hashes=("e" * 64, *report.daily_input_hashes[1:]))
    with pytest.raises(ValueError, match="result_hash does not match"):
        replace(report, result_hash="f" * 64)
    changed_identity = replace(report.identity, code_version="git:different")
    with pytest.raises(ValueError, match="input_hash does not match"):
        replace(report, identity=changed_identity)


def test_vectorized_adapter_builds_gross_path_by_adding_back_daily_fees() -> None:
    result = VectorizedBacktestEngine().run(
        request=vector_fixtures._request(),
        signals=[vector_fixtures._signal("A", vector_fixtures.CALENDAR[0], "0.5")],
        prices=[
            vector_fixtures._bar("A", vector_fixtures.CALENDAR[0], "10", "10"),
            vector_fixtures._bar("A", vector_fixtures.CALENDAR[1], "10", "11"),
            vector_fixtures._bar("A", vector_fixtures.CALENDAR[2], "11", "12"),
        ],
        benchmark=vector_fixtures._benchmark(),
    )
    regimes = {
        day: _regime_attribution(
            day,
            MarketRegime.UPTREND if index < 2 else MarketRegime.RANGE_STRONG,
        )
        for index, day in enumerate(vector_fixtures.CALENDAR)
    }
    normalized = vectorized_daily_inputs(result, regimes)

    assert tuple(item.trading_day for item in normalized) == vector_fixtures.CALENDAR
    assert normalized[-1].gross_nav >= normalized[-1].net_nav
    assert (
        sum((item.transaction_cost for item in normalized), Decimal(0)) == result.metrics.total_fees
    )
    identity = replace(
        _identity(result.request.as_of),
        run_id=result.request.run_id,
        data_version=result.request.data_version,
        strategy_version=result.request.strategy_version,
        source_input_hash=result.input_hash,
        source_result_hash=result.result_hash,
        benchmark_id=result.benchmark_id,
    )
    report = build_backtest_acceptance_report(
        identity=identity,
        daily_inputs=normalized,
        initial_nav=result.request.initial_nav,
        parameter_stability=(_stability(),),
    )
    assert report.summary.net_total_return == result.metrics.total_return
    assert report.summary.benchmark_total_return == result.metrics.benchmark_total_return
    assert report.summary.total_transaction_cost == result.metrics.total_fees

    tampered_point = replace(result.points[-1], nav=result.points[-1].nav + Decimal(1))
    tampered_result = replace(result, points=(*result.points[:-1], tampered_point))
    with pytest.raises(BacktestReportInputError, match="economic output"):
        vectorized_daily_inputs(tampered_result, regimes)

    with pytest.raises(BacktestReportInputError, match="cover the result calendar exactly"):
        vectorized_daily_inputs(
            result,
            {
                vector_fixtures.CALENDAR[0]: _regime_attribution(
                    vector_fixtures.CALENDAR[0],
                    MarketRegime.UPTREND,
                )
            },
        )


def test_config_and_identity_contracts_reject_unsafe_values() -> None:
    with pytest.raises(ValueError, match="must be negative"):
        BacktestReportConfig(failure_drawdown_threshold=Decimal(0))
    with pytest.raises(ValueError, match="must be non-positive"):
        BacktestReportConfig(failure_excess_return_threshold=Decimal("0.01"))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_identity(_daily_inputs()[-1].available_at), parameter_hash="bad")

    stability = _stability()
    with pytest.raises(BacktestReportInputError, match="must be unique"):
        build_backtest_acceptance_report(
            identity=_identity(_daily_inputs()[-1].available_at),
            daily_inputs=_daily_inputs(),
            initial_nav=Decimal(100),
            parameter_stability=(stability, stability),
        )
    with pytest.raises(BacktestReportInputError, match="requires parameter sensitivity"):
        build_backtest_acceptance_report(
            identity=_identity(_daily_inputs()[-1].available_at),
            daily_inputs=_daily_inputs(),
            initial_nav=Decimal(100),
        )


def test_report_distinguishes_research_only_from_event_replay_evidence() -> None:
    research = _report()
    assert "research/vectorized evidence only" in research.interpretation

    identity = replace(
        research.identity,
        event_log_hash=hashlib.sha256(b"").hexdigest(),
    )
    replay = BacktestResult(
        run_id=identity.run_id,
        account_id="research-account",
        currency="CNY",
        initial_cash=Decimal(100),
        cash_balance=Decimal(100),
        positions=(),
        orders=(),
        events=(),
        event_log="",
        event_log_hash=identity.event_log_hash,
    )
    event_bound = build_backtest_acceptance_report(
        identity=identity,
        daily_inputs=_daily_inputs(),
        initial_nav=Decimal(100),
        parameter_stability=(_stability(),),
        config=BacktestReportConfig(
            failure_window_sessions=3,
            failure_drawdown_threshold=Decimal("-0.08"),
            failure_excess_return_threshold=Decimal("-0.03"),
        ),
        event_result=replay,
    )
    assert "event replay evidence is bound" in event_bound.interpretation
    with pytest.raises(BacktestReportInputError, match="canonical log"):
        build_backtest_acceptance_report(
            identity=identity,
            daily_inputs=_daily_inputs(),
            initial_nav=Decimal(100),
            parameter_stability=(_stability(),),
            event_result=replace(replay, event_log_hash="e" * 64),
        )
