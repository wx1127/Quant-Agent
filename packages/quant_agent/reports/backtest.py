"""Version-bound backtest acceptance report metrics."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from quant_agent.regime.models import MarketRegime


@dataclass(frozen=True, slots=True)
class EquityPoint:
    trade_date: date
    equity_before_cost: float
    equity_after_cost: float
    benchmark_equity: float
    turnover: float
    cost: float
    regime: MarketRegime

    def __post_init__(self) -> None:
        if (
            min(
                self.equity_before_cost,
                self.equity_after_cost,
                self.benchmark_equity,
            )
            <= 0
        ):
            raise ValueError("equity values must be positive")


@dataclass(frozen=True, slots=True)
class BacktestAcceptanceReport:
    snapshot_version: str
    code_version: str
    strategy_version: str
    parameter_version: str
    total_return_before_cost: float
    total_return_after_cost: float
    benchmark_return: float
    annualized_return_after_cost: float
    maximum_drawdown: float
    sharpe: float
    sortino: float
    turnover: float
    total_cost: float
    cost_ratio: float
    regime_returns: dict[str, float]
    parameter_sensitivity: dict[str, float]
    failure_periods: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class BacktestReportPresentation:
    """Table and chart values derived from the same acceptance result."""

    summary_table: tuple[tuple[str, float], ...]
    equity_chart: tuple[tuple[str, float, float, float], ...]


class BacktestReportBuilder:
    def build(
        self,
        points: list[EquityPoint],
        *,
        snapshot_version: str,
        code_version: str,
        strategy_version: str,
        parameter_version: str,
        parameter_sensitivity: dict[str, float],
    ) -> BacktestAcceptanceReport:
        versions = (
            snapshot_version,
            code_version,
            strategy_version,
            parameter_version,
        )
        if any(not value for value in versions):
            raise ValueError("all report versions are required")
        ordered = sorted(points, key=lambda item: item.trade_date)
        if len(ordered) < 2 or len({item.trade_date for item in ordered}) != len(ordered):
            raise ValueError("report requires unique chronological equity points")
        before = ordered[-1].equity_before_cost / ordered[0].equity_before_cost - 1
        after = ordered[-1].equity_after_cost / ordered[0].equity_after_cost - 1
        benchmark = ordered[-1].benchmark_equity / ordered[0].benchmark_equity - 1
        periods = len(ordered) - 1
        annualized = (1 + after) ** (252 / periods) - 1
        returns = [
            current.equity_after_cost / previous.equity_after_cost - 1
            for previous, current in pairwise(ordered)
        ]
        mean_return = statistics.fmean(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        downside = [min(value, 0.0) for value in returns]
        downside_deviation = math.sqrt(statistics.fmean(value * value for value in downside))
        sharpe = mean_return / volatility * math.sqrt(252) if volatility else 0.0
        sortino = mean_return / downside_deviation * math.sqrt(252) if downside_deviation else 0.0
        peak = ordered[0].equity_after_cost
        maximum_drawdown = 0.0
        failures = []
        for item in ordered:
            peak = max(peak, item.equity_after_cost)
            drawdown = item.equity_after_cost / peak - 1
            maximum_drawdown = min(maximum_drawdown, drawdown)
            if drawdown <= -0.10:
                failures.append(f"{item.trade_date.isoformat()} drawdown {drawdown:.4f}")
        grouped: dict[str, list[float]] = {}
        for previous, current in pairwise(ordered):
            grouped.setdefault(current.regime.value, []).append(
                current.equity_after_cost / previous.equity_after_cost - 1
            )
        regime_returns = {
            regime: statistics.fmean(values) for regime, values in sorted(grouped.items())
        }
        total_cost = sum(item.cost for item in ordered)
        payload = {
            "snapshot_version": snapshot_version,
            "code_version": code_version,
            "strategy_version": strategy_version,
            "parameter_version": parameter_version,
            "total_return_before_cost": before,
            "total_return_after_cost": after,
            "benchmark_return": benchmark,
            "annualized_return_after_cost": annualized,
            "maximum_drawdown": maximum_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "turnover": sum(item.turnover for item in ordered),
            "total_cost": total_cost,
            "cost_ratio": total_cost / ordered[0].equity_after_cost,
            "regime_returns": regime_returns,
            "parameter_sensitivity": parameter_sensitivity,
            "failure_periods": tuple(failures[:10]),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return BacktestAcceptanceReport(**payload, content_hash=digest)

    def present(
        self,
        report: BacktestAcceptanceReport,
        points: list[EquityPoint],
    ) -> BacktestReportPresentation:
        ordered = sorted(points, key=lambda item: item.trade_date)
        if len(ordered) < 2:
            raise ValueError("presentation requires at least two equity points")
        chart_return = ordered[-1].equity_after_cost / ordered[0].equity_after_cost - 1
        if abs(chart_return - report.total_return_after_cost) > 1e-10:
            raise ValueError("chart and report table inputs are inconsistent")
        return BacktestReportPresentation(
            summary_table=(
                ("return_before_cost", report.total_return_before_cost),
                ("return_after_cost", report.total_return_after_cost),
                ("benchmark_return", report.benchmark_return),
                ("maximum_drawdown", report.maximum_drawdown),
            ),
            equity_chart=tuple(
                (
                    item.trade_date.isoformat(),
                    item.equity_before_cost,
                    item.equity_after_cost,
                    item.benchmark_equity,
                )
                for item in ordered
            ),
        )
