"""Effective-dated China stock and ETF trading constraints."""

from dataclasses import dataclass
from datetime import date
from uuid import NAMESPACE_URL, uuid5

from quant_agent.backtest.contracts import (
    AssetType,
    FeeSchedule,
    Fill,
    MarketBar,
    Order,
    OrderType,
    Side,
)


@dataclass(frozen=True, slots=True)
class ChinaMarketRuleConfig:
    version: str = "cn_market_rules_v1"
    effective_from: date = date(2020, 1, 1)
    stock_lot_size: int = 100
    etf_lot_size: int = 100
    maximum_volume_participation: float = 0.10

    def __post_init__(self) -> None:
        if self.stock_lot_size < 1 or self.etf_lot_size < 1:
            raise ValueError("lot sizes must be positive")
        if not 0 < self.maximum_volume_participation <= 1:
            raise ValueError("volume participation must be in (0, 1]")


class FeeScheduleRegistry:
    """Resolve the latest fee schedule effective on a trading date."""

    def __init__(self, schedules: list[FeeSchedule]) -> None:
        if not schedules:
            raise ValueError("at least one fee schedule is required")
        dates = [item.effective_from for item in schedules]
        if len(set(dates)) != len(dates):
            raise ValueError("fee schedule effective dates must be unique")
        self._schedules = sorted(schedules, key=lambda item: item.effective_from)

    def as_of(self, trade_date: date) -> FeeSchedule:
        eligible = [item for item in self._schedules if item.effective_from <= trade_date]
        if not eligible:
            raise ValueError("no fee schedule effective for trade date")
        return eligible[-1]


class ChinaMarketRules:
    def __init__(
        self,
        fees: FeeScheduleRegistry,
        config: ChinaMarketRuleConfig | None = None,
    ) -> None:
        self._fees = fees
        self.config = config or ChinaMarketRuleConfig()

    def validate(self, order: Order, bar: MarketBar, *, available_quantity: int) -> str | None:
        if order.instrument_id != bar.instrument_id or order.asset_type is not bar.asset_type:
            return "order and market bar instrument mismatch"
        if bar.trade_date < self.config.effective_from:
            return "market rule version is not yet effective"
        if bar.trade_date < order.created_at.date():
            return "market bar predates order creation"
        if bar.available_at <= order.created_at:
            return "market bar is not later than order creation"
        if bar.suspended or bar.volume == 0:
            return "instrument is suspended or has no volume"
        lot = (
            self.config.stock_lot_size
            if order.asset_type is AssetType.STOCK
            else self.config.etf_lot_size
        )
        if order.quantity % lot:
            return f"quantity must be a multiple of {lot}"
        if order.side is Side.SELL and order.remaining_quantity > available_quantity:
            return "sell quantity exceeds available position"
        if (
            order.side is Side.BUY
            and bar.limit_up_price is not None
            and bar.open >= bar.limit_up_price
        ):
            return "limit-up order is not buyable"
        if (
            order.side is Side.SELL
            and bar.limit_down_price is not None
            and bar.open <= bar.limit_down_price
        ):
            return "limit-down order is not sellable"
        if order.order_type is OrderType.LIMIT and order.limit_price is None:
            return "limit order requires limit price"
        if order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            if order.side is Side.BUY and order.limit_price < bar.open:
                return "buy limit price does not cross market"
            if order.side is Side.SELL and order.limit_price > bar.open:
                return "sell limit price does not cross market"
        return None

    def match(
        self,
        order: Order,
        bar: MarketBar,
        *,
        available_quantity: int,
    ) -> Fill | None:
        if self.validate(order, bar, available_quantity=available_quantity) is not None:
            return None
        lot = (
            self.config.stock_lot_size
            if order.asset_type is AssetType.STOCK
            else self.config.etf_lot_size
        )
        volume_cap = int(bar.volume * self.config.maximum_volume_participation)
        fill_quantity = min(order.remaining_quantity, volume_cap)
        fill_quantity -= fill_quantity % lot
        if fill_quantity <= 0:
            return None
        schedule = self._fees.as_of(bar.trade_date)
        direction = 1 if order.side is Side.BUY else -1
        price = bar.open * (1 + direction * schedule.slippage_bps / 10_000)
        gross = price * fill_quantity
        commission = max(schedule.minimum_commission, gross * schedule.commission_rate)
        tax = (
            gross * schedule.stock_sell_tax_rate
            if order.asset_type is AssetType.STOCK and order.side is Side.SELL
            else 0.0
        )
        slippage = abs(price - bar.open) * fill_quantity
        fill_key = f"{order.order_id}|{bar.trade_date}|{fill_quantity}|{order.filled_quantity}"
        return Fill(
            fill_id=str(uuid5(NAMESPACE_URL, fill_key)),
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            quantity=fill_quantity,
            price=price,
            gross_amount=gross,
            commission=commission,
            tax=tax,
            slippage=slippage,
            filled_at=bar.available_at,
            fee_version=schedule.version,
        )
