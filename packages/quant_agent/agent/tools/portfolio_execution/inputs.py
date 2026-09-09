"""Trusted input boundary for portfolio, risk, draft, paper, and reconciliation tools.

None of the values in this module are model supplied.  A composition root must
resolve them from the same immutable decision and from independently observed
execution state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import ConfigDict

from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.backtest.cn_market import MarketSessionState
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.execution.paper import PaperExecutionReceipt
from quant_agent.portfolio import (
    AccountSnapshot,
    PITIndustryClassification,
    StrategySleeve,
    TargetPortfolio,
)
from quant_agent.reconciliation import (
    ObservedExecutionEvidence,
    ReconciliationPolicy,
)
from quant_agent.risk import PortfolioRiskContext, RiskCheckResult


class PortfolioExecutionInputError(RuntimeError):
    """Base class for expected trusted-boundary failures."""


class PortfolioExecutionInputUnavailable(PortfolioExecutionInputError):
    """A required authoritative input or execution observation is absent."""


class PortfolioExecutionInputInvalid(PortfolioExecutionInputError):
    """An authoritative input fails its decision or account binding."""


class PortfolioRiskBlocked(PortfolioExecutionInputError):
    """A complete REJECT or ERROR risk result prevented draft creation."""

    def __init__(self, result: RiskCheckResult) -> None:
        super().__init__("portfolio risk result does not allow order-draft creation")
        self.result = result


@dataclass(frozen=True, slots=True)
class ReconciliationInputs:
    """Independent actual observations and the expected paper receipt."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    receipt: PaperExecutionReceipt
    observed: ObservedExecutionEvidence
    policy: ReconciliationPolicy
    previous_result_hash: str | None = None


class PortfolioExecutionInputSource(Protocol):
    """Read-only provider of authoritative inputs for one frozen decision."""

    def load_account_snapshot(self, decision: DecisionSnapshot) -> AccountSnapshot:
        """Return the exact account snapshot bound to ``decision``."""

        ...

    def load_strategy_sleeves(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[StrategySleeve, ...]:
        """Return deterministic strategy targets bound to decision strategy refs."""

        ...

    def load_industry_classifications(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[PITIndustryClassification, ...]:
        """Return PIT classifications needed by portfolio constraints."""

        ...

    def load_risk_context(self, decision: DecisionSnapshot) -> PortfolioRiskContext:
        """Return independently sourced drawdown, valuation, and regime context."""

        ...

    def load_market_states(
        self,
        decision: DecisionSnapshot,
        target: TargetPortfolio,
    ) -> tuple[MarketSessionState, ...]:
        """Return exact PIT market states for every target-portfolio line."""

        ...

    def load_reconciliation_inputs(
        self,
        decision: DecisionSnapshot,
        draft: OrderDraftBatch,
    ) -> ReconciliationInputs:
        """Return independent observed evidence for one already-submitted draft."""

        ...


__all__ = [
    "PortfolioExecutionInputError",
    "PortfolioExecutionInputInvalid",
    "PortfolioExecutionInputSource",
    "PortfolioExecutionInputUnavailable",
    "PortfolioRiskBlocked",
    "ReconciliationInputs",
]
