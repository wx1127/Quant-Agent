"""Unified, version-bound backtest acceptance reports."""

from quant_agent.reports.backtest.adapters import vectorized_daily_inputs
from quant_agent.reports.backtest.builder import build_backtest_acceptance_report
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
    GrossReturnMethod,
    RegimePerformanceSlice,
    SessionRegimeAttribution,
)

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
    "build_backtest_acceptance_report",
    "vectorized_daily_inputs",
]
