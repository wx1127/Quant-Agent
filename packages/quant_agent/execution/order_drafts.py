"""Convert a risk-approved portfolio delta into deterministic, non-executable drafts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from quant_agent.backtest.contracts import AssetType, FeeSchedule, Side
from quant_agent.core.time import ensure_aware
from quant_agent.portfolio.builder import PortfolioBuildResult
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.risk.contracts import RiskDecision


@dataclass(frozen=True, slots=True)
class PriceQuote:
    instrument_id: str
    price: float
    as_of: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.price <= 0:
            raise ValueError("reference price must be positive")


@dataclass(frozen=True, slots=True)
class OrderDraft:
    draft_id: str
    account_id: str
    decision_id: str
    instrument_id: str
    asset_type: AssetType
    side: Side
    quantity: int
    reference_price: float
    estimated_commission: float
    estimated_tax: float
    estimated_slippage: float
    created_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.created_at)
        if self.quantity <= 0:
            raise ValueError("draft quantity must be positive")


@dataclass(frozen=True, slots=True)
class OrderDraftBatch:
    account_id: str
    decision_id: str
    account_snapshot_id: str
    drafts: tuple[OrderDraft, ...]
    expires_at: datetime
    risk_policy_version: str
    batch_version: str
    batch_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.expires_at)


class OrderDraftBuilder:
    batch_version = "order_draft_v1"

    def build(
        self,
        portfolio: PortfolioBuildResult,
        risk: RiskDecision,
        account: AccountSnapshot,
        quotes: list[PriceQuote],
        *,
        fees: FeeSchedule,
        created_at: datetime,
        expires_at: datetime,
    ) -> OrderDraftBatch:
        ensure_aware(created_at)
        ensure_aware(expires_at)
        if expires_at <= created_at:
            raise ValueError("draft expiry must be later than creation")
        if not risk.passed or risk.violations:
            raise ValueError("hard risk violation prevents order drafts")
        if portfolio.account_snapshot_id != account.snapshot_id:
            raise ValueError("portfolio and account snapshot mismatch")
        quote_by_id = {item.instrument_id: item for item in quotes}
        if any(item.as_of > created_at for item in quotes):
            raise ValueError("future price quote cannot create a draft")
        holding_by_id = {item.instrument_id: item for item in account.holdings}
        available_cash = account.available_cash
        drafts: list[OrderDraft] = []
        for line in sorted(portfolio.lines, key=lambda item: item.instrument_id):
            if abs(line.delta_weight) < 1e-10:
                continue
            quote = quote_by_id.get(line.instrument_id)
            if quote is None:
                raise ValueError(f"missing price quote for {line.instrument_id}")
            side = Side.BUY if line.delta_weight > 0 else Side.SELL
            desired_amount = abs(line.delta_weight) * account.total_equity
            lot = 100
            quantity = int(desired_amount / quote.price)
            quantity -= quantity % lot
            if side is Side.SELL:
                available = holding_by_id.get(line.instrument_id)
                available_quantity = available.available_quantity if available else 0
                quantity = min(quantity, available_quantity)
                quantity -= quantity % lot
            if quantity <= 0:
                continue
            gross = quantity * quote.price
            commission = max(fees.minimum_commission, gross * fees.commission_rate)
            tax = (
                gross * fees.stock_sell_tax_rate
                if line.asset_type is AssetType.STOCK and side is Side.SELL
                else 0.0
            )
            slippage = gross * fees.slippage_bps / 10_000
            if side is Side.BUY:
                needed = gross + commission + slippage
                if needed > available_cash:
                    affordable = int(
                        max(0.0, available_cash - fees.minimum_commission)
                        / (quote.price * (1 + fees.slippage_bps / 10_000))
                    )
                    quantity = affordable - affordable % lot
                    if quantity <= 0:
                        continue
                    gross = quantity * quote.price
                    commission = max(fees.minimum_commission, gross * fees.commission_rate)
                    slippage = gross * fees.slippage_bps / 10_000
                    needed = gross + commission + slippage
                available_cash -= needed
            draft_key = (
                f"{account.account_id}|{portfolio.decision_id}|{line.instrument_id}|"
                f"{side}|{quantity}|{self.batch_version}"
            )
            drafts.append(
                OrderDraft(
                    draft_id=hashlib.sha256(draft_key.encode()).hexdigest(),
                    account_id=account.account_id,
                    decision_id=portfolio.decision_id,
                    instrument_id=line.instrument_id,
                    asset_type=line.asset_type,
                    side=side,
                    quantity=quantity,
                    reference_price=quote.price,
                    estimated_commission=commission,
                    estimated_tax=tax,
                    estimated_slippage=slippage,
                    created_at=created_at,
                )
            )
        canonical = {
            "account_id": account.account_id,
            "decision_id": portfolio.decision_id,
            "account_snapshot_id": account.snapshot_id,
            "drafts": [asdict(item) for item in drafts],
            "expires_at": expires_at,
            "risk_policy_version": risk.checked_policy_version,
            "batch_version": self.batch_version,
        }
        batch_hash = hashlib.sha256(
            json.dumps(canonical, default=str, sort_keys=True).encode()
        ).hexdigest()
        return OrderDraftBatch(
            account_id=account.account_id,
            decision_id=portfolio.decision_id,
            account_snapshot_id=account.snapshot_id,
            drafts=tuple(drafts),
            expires_at=expires_at,
            risk_policy_version=risk.checked_policy_version,
            batch_version=self.batch_version,
            batch_hash=batch_hash,
        )
