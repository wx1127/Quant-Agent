"""Aggregate isolated strategy budgets into an explainable target portfolio."""

from dataclasses import dataclass
from datetime import datetime

from quant_agent.backtest.contracts import AssetType
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.strategies.contracts import TargetPortfolio


@dataclass(frozen=True, slots=True)
class PortfolioConstraintConfig:
    version: str = "portfolio_builder_v1"
    maximum_stock_weight: float = 0.05
    maximum_etf_weight: float = 0.40
    maximum_industry_weight: float = 0.25
    maximum_invested_weight: float = 0.80

    def __post_init__(self) -> None:
        values = (
            self.maximum_stock_weight,
            self.maximum_etf_weight,
            self.maximum_industry_weight,
            self.maximum_invested_weight,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("portfolio limits must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StrategyAllocation:
    strategy_id: str
    capital_budget: float
    target: TargetPortfolio
    asset_types: dict[str, AssetType]
    industries: dict[str, str | None]

    def __post_init__(self) -> None:
        if not 0 <= self.capital_budget <= 1:
            raise ValueError("strategy capital budget must be in [0, 1]")
        if self.strategy_id != self.target.strategy_id:
            raise ValueError("allocation and target strategy ids must match")


@dataclass(frozen=True, slots=True)
class PortfolioLine:
    instrument_id: str
    asset_type: AssetType
    industry_id: str | None
    current_weight: float
    target_weight: float
    delta_weight: float
    strategy_sources: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class PortfolioBuildResult:
    decision_id: str
    account_snapshot_id: str
    as_of: datetime
    lines: tuple[PortfolioLine, ...]
    cash_target_weight: float
    expected_turnover: float
    industry_exposure: dict[str, float]
    strategy_budgets: dict[str, float]
    data_version: str
    builder_version: str
    warnings: tuple[str, ...]


class PortfolioBuilder:
    def __init__(self, config: PortfolioConstraintConfig | None = None) -> None:
        self.config = config or PortfolioConstraintConfig()

    def build(
        self,
        allocations: list[StrategyAllocation],
        *,
        account: AccountSnapshot,
        decision_id: str,
        data_version: str,
    ) -> PortfolioBuildResult:
        if not allocations:
            raise ValueError("portfolio build requires strategy allocations")
        dates = {item.target.as_of for item in allocations}
        if dates != {account.as_of}:
            raise ValueError("strategy targets and account snapshot must share as_of")
        total_budget = sum(item.capital_budget for item in allocations)
        if total_budget > self.config.maximum_invested_weight + 1e-10:
            raise ValueError("isolated strategy budgets exceed invested-weight limit")
        target_weights: dict[str, float] = {}
        asset_types: dict[str, AssetType] = {}
        industries: dict[str, str | None] = {}
        sources: dict[str, list[str]] = {}
        reasons: dict[str, list[str]] = {}
        warnings: list[str] = []
        for allocation in allocations:
            for target in allocation.target.targets:
                instrument_id = target.instrument_id
                if instrument_id not in allocation.asset_types:
                    raise ValueError(f"asset type missing for {instrument_id}")
                contribution = target.weight * allocation.capital_budget
                target_weights[instrument_id] = (
                    target_weights.get(instrument_id, 0.0) + contribution
                )
                asset_types[instrument_id] = allocation.asset_types[instrument_id]
                industries[instrument_id] = allocation.industries.get(instrument_id)
                sources.setdefault(instrument_id, []).append(allocation.strategy_id)
                reasons.setdefault(instrument_id, []).append(target.reason)
        for instrument_id in sorted(target_weights):
            limit = (
                self.config.maximum_stock_weight
                if asset_types[instrument_id] is AssetType.STOCK
                else self.config.maximum_etf_weight
            )
            if target_weights[instrument_id] > limit:
                warnings.append(f"{instrument_id}: target capped by instrument limit")
                target_weights[instrument_id] = limit
        industry_exposure: dict[str, float] = {}
        for instrument_id in sorted(target_weights):
            industry_id = industries[instrument_id]
            if industry_id is None:
                continue
            remaining = self.config.maximum_industry_weight - industry_exposure.get(
                industry_id, 0.0
            )
            if target_weights[instrument_id] > max(0.0, remaining):
                target_weights[instrument_id] = max(0.0, remaining)
                warnings.append(f"{instrument_id}: target capped by industry limit")
            industry_exposure[industry_id] = (
                industry_exposure.get(industry_id, 0.0) + target_weights[instrument_id]
            )
        equity = account.total_equity
        current_weights = (
            {item.instrument_id: item.market_value / equity for item in account.holdings}
            if equity
            else {}
        )
        all_ids = sorted(set(current_weights) | set(target_weights))
        holding_by_id = {item.instrument_id: item for item in account.holdings}
        lines = tuple(
            PortfolioLine(
                instrument_id=instrument_id,
                asset_type=asset_types[instrument_id]
                if instrument_id in asset_types
                else holding_by_id[instrument_id].asset_type,
                industry_id=industries[instrument_id]
                if instrument_id in industries
                else holding_by_id[instrument_id].industry_id,
                current_weight=current_weights.get(instrument_id, 0.0),
                target_weight=target_weights.get(instrument_id, 0.0),
                delta_weight=target_weights.get(instrument_id, 0.0)
                - current_weights.get(instrument_id, 0.0),
                strategy_sources=tuple(sorted(sources.get(instrument_id, []))),
                reason="; ".join(reasons.get(instrument_id, ["position exit"])),
            )
            for instrument_id in all_ids
        )
        invested = sum(target_weights.values())
        return PortfolioBuildResult(
            decision_id=decision_id,
            account_snapshot_id=account.snapshot_id,
            as_of=account.as_of,
            lines=lines,
            cash_target_weight=1.0 - invested,
            expected_turnover=sum(abs(item.delta_weight) for item in lines),
            industry_exposure=dict(sorted(industry_exposure.items())),
            strategy_budgets={item.strategy_id: item.capital_budget for item in allocations},
            data_version=data_version,
            builder_version=self.config.version,
            warnings=tuple(warnings),
        )
