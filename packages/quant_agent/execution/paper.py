"""Credential-free, idempotent paper broker using deterministic market rules."""

from dataclasses import dataclass
from datetime import date, datetime

from quant_agent.backtest.cn_market_rules import ChinaMarketRules
from quant_agent.backtest.contracts import (
    Fill,
    MarketBar,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from quant_agent.core.time import ensure_aware
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.portfolio.snapshots import AccountSnapshot, HoldingSnapshot


@dataclass(frozen=True, slots=True)
class PaperOrder:
    draft_id: str
    order: Order
    message: str


@dataclass(frozen=True, slots=True)
class PaperBatchResult:
    batch_hash: str
    orders: tuple[PaperOrder, ...]
    fills: tuple[Fill, ...]
    account_snapshot: AccountSnapshot


class PaperBroker:
    """Paper-only ledger; constructor has no credential or live endpoint input."""

    def __init__(self, account: AccountSnapshot, rules: ChinaMarketRules) -> None:
        self._account_id = account.account_id
        self._cash = account.available_cash
        self._frozen_cash = account.frozen_cash
        self._holdings = {item.instrument_id: item for item in account.holdings}
        self._rules = rules
        self._processed: dict[str, PaperBatchResult] = {}
        self._sequence = 0
        self._last_buy_dates: dict[str, date] = {}

    def _release_t_plus_one(self, current_date: date) -> None:
        for instrument_id, bought_on in list(self._last_buy_dates.items()):
            if bought_on >= current_date:
                continue
            holding = self._holdings.get(instrument_id)
            if holding is not None and holding.frozen_quantity:
                self._holdings[instrument_id] = HoldingSnapshot(
                    instrument_id=holding.instrument_id,
                    asset_type=holding.asset_type,
                    industry_id=holding.industry_id,
                    quantity=holding.quantity,
                    available_quantity=holding.quantity,
                    frozen_quantity=0,
                    average_cost=holding.average_cost,
                    last_price=holding.last_price,
                )
            del self._last_buy_dates[instrument_id]

    def submit(
        self,
        batch: OrderDraftBatch,
        *,
        bars: list[MarketBar],
        submitted_at: datetime,
    ) -> PaperBatchResult:
        ensure_aware(submitted_at)
        existing = self._processed.get(batch.batch_hash)
        if existing is not None:
            return existing
        if batch.account_id != self._account_id:
            raise ValueError("paper batch account mismatch")
        if submitted_at > batch.expires_at:
            raise ValueError("paper order batch has expired")
        self._release_t_plus_one(submitted_at.date())
        bars_by_id = {item.instrument_id: item for item in bars}
        paper_orders: list[PaperOrder] = []
        fills: list[Fill] = []
        for draft in batch.drafts:
            self._sequence += 1
            order = Order(
                order_id=f"PAPER-{self._sequence:08d}",
                signal_id=draft.draft_id,
                instrument_id=draft.instrument_id,
                asset_type=draft.asset_type,
                side=draft.side,
                quantity=draft.quantity,
                order_type=OrderType.MARKET,
                created_at=submitted_at,
            ).transition(OrderStatus.ACCEPTED)
            bar = bars_by_id.get(draft.instrument_id)
            holding = self._holdings.get(draft.instrument_id)
            available = holding.available_quantity if holding else 0
            if bar is None:
                paper_orders.append(PaperOrder(draft.draft_id, order, "waiting for market bar"))
                continue
            reason = self._rules.validate(order, bar, available_quantity=available)
            if reason is not None:
                rejected = order.transition(OrderStatus.REJECTED, rejection_reason=reason)
                paper_orders.append(PaperOrder(draft.draft_id, rejected, reason))
                continue
            fill = self._rules.match(order, bar, available_quantity=available)
            if fill is None:
                paper_orders.append(PaperOrder(draft.draft_id, order, "no executable quantity"))
                continue
            cash_cost = fill.gross_amount + fill.commission + fill.tax
            if fill.side is Side.BUY and cash_cost > self._cash:
                rejected = order.transition(
                    OrderStatus.REJECTED,
                    rejection_reason="insufficient paper cash",
                )
                paper_orders.append(PaperOrder(draft.draft_id, rejected, "insufficient paper cash"))
                continue
            if fill.side is Side.BUY:
                self._cash -= cash_cost
                old_quantity = holding.quantity if holding else 0
                old_cost = holding.average_cost * old_quantity if holding else 0.0
                quantity = old_quantity + fill.quantity
                stock_frozen = fill.quantity if draft.asset_type.value == "STOCK" else 0
                prior_frozen = holding.frozen_quantity if holding else 0
                self._holdings[draft.instrument_id] = HoldingSnapshot(
                    instrument_id=draft.instrument_id,
                    asset_type=draft.asset_type,
                    industry_id=holding.industry_id if holding else None,
                    quantity=quantity,
                    available_quantity=(holding.available_quantity if holding else 0)
                    + fill.quantity
                    - stock_frozen,
                    frozen_quantity=prior_frozen + stock_frozen,
                    average_cost=(old_cost + fill.gross_amount + fill.commission) / quantity,
                    last_price=fill.price,
                )
                if stock_frozen:
                    self._last_buy_dates[draft.instrument_id] = bar.trade_date
            else:
                assert holding is not None
                self._cash += fill.gross_amount - fill.commission - fill.tax
                quantity = holding.quantity - fill.quantity
                if quantity:
                    self._holdings[draft.instrument_id] = HoldingSnapshot(
                        instrument_id=holding.instrument_id,
                        asset_type=holding.asset_type,
                        industry_id=holding.industry_id,
                        quantity=quantity,
                        available_quantity=holding.available_quantity - fill.quantity,
                        frozen_quantity=holding.frozen_quantity,
                        average_cost=holding.average_cost,
                        last_price=fill.price,
                    )
                else:
                    del self._holdings[draft.instrument_id]
            fills.append(fill)
            total = fill.quantity
            status = OrderStatus.FILLED if total == order.quantity else OrderStatus.PARTIALLY_FILLED
            updated = order.transition(status, filled_quantity=total)
            paper_orders.append(PaperOrder(draft.draft_id, updated, "paper match completed"))
        snapshot_at = max(
            (fill.filled_at for fill in fills),
            default=submitted_at,
        )
        snapshot = AccountSnapshot(
            snapshot_id=f"paper-{batch.batch_hash[:16]}",
            account_id=self._account_id,
            as_of=snapshot_at,
            available_cash=self._cash,
            frozen_cash=self._frozen_cash,
            holdings=tuple(self._holdings[key] for key in sorted(self._holdings)),
            source="paper",
            version="paper_account_v1",
        )
        result = PaperBatchResult(
            batch_hash=batch.batch_hash,
            orders=tuple(paper_orders),
            fills=tuple(fills),
            account_snapshot=snapshot,
        )
        self._processed[batch.batch_hash] = result
        return result
