"""Natural-language-independent risk request and decision contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.portfolio.builder import PortfolioBuildResult
from quant_agent.portfolio.snapshots import AccountSnapshot


class RiskSeverity(StrEnum):
    WARNING = "WARNING"
    HARD = "HARD"


@dataclass(frozen=True, slots=True)
class RiskFinding:
    rule_id: str
    severity: RiskSeverity
    message: str
    instrument_id: str | None = None
    observed_value: float | None = None
    limit_value: float | None = None


@dataclass(frozen=True, slots=True)
class RiskRequest:
    request_id: str
    decision_id: str
    account: AccountSnapshot
    portfolio: PortfolioBuildResult
    current_drawdown: float
    market_risk_budget: float
    data_complete: bool
    service_dependencies_healthy: bool

    def __post_init__(self) -> None:
        if self.decision_id != self.portfolio.decision_id:
            raise ValueError("risk request decision id mismatch")
        if self.account.snapshot_id != self.portfolio.account_snapshot_id:
            raise ValueError("risk request account snapshot mismatch")
        if not 0 <= self.current_drawdown <= 1:
            raise ValueError("drawdown must be in [0, 1]")
        if not 0 <= self.market_risk_budget <= 1:
            raise ValueError("market risk budget must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    request_id: str
    passed: bool
    violations: tuple[RiskFinding, ...]
    warnings: tuple[RiskFinding, ...]
    checked_policy_version: str
    checked_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.checked_at)
        if self.passed and self.violations:
            raise ValueError("passed risk decision cannot contain hard violations")

    @classmethod
    def fail_closed(
        cls,
        request_id: str,
        *,
        checked_at: datetime,
        policy_version: str,
        reason: str,
    ) -> "RiskDecision":
        return cls(
            request_id=request_id,
            passed=False,
            violations=(
                RiskFinding(
                    "RISK_SERVICE_FAILURE",
                    RiskSeverity.HARD,
                    reason,
                ),
            ),
            warnings=(),
            checked_policy_version=policy_version,
            checked_at=checked_at,
        )
