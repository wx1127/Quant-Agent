"""Versioned portfolio limits with hard rejection and drawdown controls."""

from dataclasses import dataclass
from datetime import datetime

from quant_agent.backtest.contracts import AssetType
from quant_agent.core.time import ensure_aware
from quant_agent.risk.contracts import (
    RiskDecision,
    RiskFinding,
    RiskRequest,
    RiskSeverity,
)


@dataclass(frozen=True, slots=True)
class PortfolioRiskPolicy:
    version: str = "portfolio_risk_v1"
    maximum_stock_weight: float = 0.05
    maximum_etf_weight: float = 0.40
    maximum_industry_weight: float = 0.25
    maximum_total_equity_weight: float = 0.80
    maximum_turnover: float = 0.20
    drawdown_warning: float = 0.10
    drawdown_stop: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.maximum_stock_weight,
            self.maximum_etf_weight,
            self.maximum_industry_weight,
            self.maximum_total_equity_weight,
            self.maximum_turnover,
            self.drawdown_warning,
            self.drawdown_stop,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("risk policy thresholds must be in [0, 1]")
        if self.drawdown_warning > self.drawdown_stop:
            raise ValueError("drawdown warning cannot exceed stop threshold")


class PortfolioRiskEngine:
    def __init__(self, policy: PortfolioRiskPolicy | None = None) -> None:
        self.policy = policy or PortfolioRiskPolicy()

    def check(self, request: RiskRequest, *, checked_at: datetime) -> RiskDecision:
        ensure_aware(checked_at)
        hard: list[RiskFinding] = []
        warnings: list[RiskFinding] = []
        if not request.data_complete:
            hard.append(
                RiskFinding("DATA_INCOMPLETE", RiskSeverity.HARD, "required data is incomplete")
            )
        if not request.service_dependencies_healthy:
            hard.append(
                RiskFinding(
                    "DEPENDENCY_UNHEALTHY",
                    RiskSeverity.HARD,
                    "risk dependency is unavailable",
                )
            )
        target_total = sum(item.target_weight for item in request.portfolio.lines)
        total_limit = min(
            self.policy.maximum_total_equity_weight,
            request.market_risk_budget,
        )
        if target_total > total_limit + 1e-10:
            hard.append(
                RiskFinding(
                    "TOTAL_EQUITY_WEIGHT",
                    RiskSeverity.HARD,
                    "target equity weight exceeds risk budget",
                    observed_value=target_total,
                    limit_value=total_limit,
                )
            )
        for item in request.portfolio.lines:
            limit = (
                self.policy.maximum_stock_weight
                if item.asset_type is AssetType.STOCK
                else self.policy.maximum_etf_weight
            )
            if item.target_weight > limit + 1e-10:
                hard.append(
                    RiskFinding(
                        "INSTRUMENT_WEIGHT",
                        RiskSeverity.HARD,
                        "instrument target exceeds concentration limit",
                        instrument_id=item.instrument_id,
                        observed_value=item.target_weight,
                        limit_value=limit,
                    )
                )
        for industry_id, exposure in request.portfolio.industry_exposure.items():
            if exposure > self.policy.maximum_industry_weight + 1e-10:
                hard.append(
                    RiskFinding(
                        "INDUSTRY_WEIGHT",
                        RiskSeverity.HARD,
                        f"industry {industry_id} exceeds concentration limit",
                        observed_value=exposure,
                        limit_value=self.policy.maximum_industry_weight,
                    )
                )
        if request.portfolio.expected_turnover > self.policy.maximum_turnover:
            hard.append(
                RiskFinding(
                    "TURNOVER_LIMIT",
                    RiskSeverity.HARD,
                    "planned turnover exceeds daily limit",
                    observed_value=request.portfolio.expected_turnover,
                    limit_value=self.policy.maximum_turnover,
                )
            )
        increases_risk = any(item.delta_weight > 1e-10 for item in request.portfolio.lines)
        if request.current_drawdown >= self.policy.drawdown_stop and increases_risk:
            hard.append(
                RiskFinding(
                    "DRAWDOWN_STOP",
                    RiskSeverity.HARD,
                    "drawdown stop forbids new risk",
                    observed_value=request.current_drawdown,
                    limit_value=self.policy.drawdown_stop,
                )
            )
        elif request.current_drawdown >= self.policy.drawdown_warning:
            warnings.append(
                RiskFinding(
                    "DRAWDOWN_WARNING",
                    RiskSeverity.WARNING,
                    "portfolio drawdown warning threshold reached",
                    observed_value=request.current_drawdown,
                    limit_value=self.policy.drawdown_warning,
                )
            )
        return RiskDecision(
            request_id=request.request_id,
            passed=not hard,
            violations=tuple(hard),
            warnings=tuple(warnings),
            checked_policy_version=self.policy.version,
            checked_at=checked_at,
        )
