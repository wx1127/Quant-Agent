from datetime import date, timedelta

import pytest

from quant_agent.regime.models import MarketRegime
from quant_agent.reports.backtest import BacktestReportBuilder, EquityPoint


def test_backtest_report_is_version_bound_cost_aware_and_reproducible() -> None:
    start = date(2025, 1, 1)
    equities = [100, 105, 90, 110]
    points = [
        EquityPoint(
            start + timedelta(days=index),
            value + index,
            value,
            100 + index,
            0.1,
            1.0,
            MarketRegime.UPTREND if index < 2 else MarketRegime.DIVERGENT,
        )
        for index, value in enumerate(equities)
    ]
    builder = BacktestReportBuilder()
    report = builder.build(
        points,
        snapshot_version="snap",
        code_version="code",
        strategy_version="strategy",
        parameter_version="params",
        parameter_sensitivity={"window-10%": 0.08, "window+10%": 0.12},
    )
    again = builder.build(
        list(reversed(points)),
        snapshot_version="snap",
        code_version="code",
        strategy_version="strategy",
        parameter_version="params",
        parameter_sensitivity={"window-10%": 0.08, "window+10%": 0.12},
    )
    assert report.content_hash == again.content_hash
    assert report.total_return_after_cost == pytest.approx(0.1)
    assert report.total_return_before_cost > report.total_return_after_cost
    assert report.maximum_drawdown < -0.1
    assert report.failure_periods
    assert set(report.regime_returns) == {"DIVERGENT", "UPTREND"}
    assert report.total_cost == 4
    presentation = builder.present(report, points)
    table = dict(presentation.summary_table)
    assert table["return_after_cost"] == report.total_return_after_cost
    assert presentation.equity_chart[-1][2] == points[-1].equity_after_cost


def test_backtest_report_rejects_missing_versions_and_bad_equity() -> None:
    builder = BacktestReportBuilder()
    point = EquityPoint(date.today(), 100, 100, 100, 0, 0, MarketRegime.UPTREND)
    with pytest.raises(ValueError, match="versions"):
        builder.build(
            [point, point],
            snapshot_version="",
            code_version="c",
            strategy_version="s",
            parameter_version="p",
            parameter_sensitivity={},
        )
    with pytest.raises(ValueError):
        EquityPoint(date.today(), 0, 1, 1, 0, 0, MarketRegime.UPTREND)
