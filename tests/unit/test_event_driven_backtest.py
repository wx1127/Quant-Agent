from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.backtest.cn_market_rules import ChinaMarketRules, FeeScheduleRegistry
from quant_agent.backtest.contracts import (
    AssetType,
    FeeSchedule,
    MarketBar,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from quant_agent.backtest.event_driven import EventDrivenBacktest

START = datetime(2026, 1, 1, 15, tzinfo=UTC)


def rules() -> ChinaMarketRules:
    return ChinaMarketRules(
        FeeScheduleRegistry([FeeSchedule("v1", date(2020, 1, 1), 0.0003, 5, 0.0005, 0)])
    )


def market_bar(day: int, *, available_hour: int = 16) -> MarketBar:
    at = START + timedelta(days=day)
    available = at.replace(hour=available_hour)
    return MarketBar(
        "S1",
        AssetType.STOCK,
        available.date(),
        10,
        10,
        10,
        10,
        10_000,
        100_000,
        available,
    )


def make_order(order_id: str, side: Side, quantity: int, created: datetime) -> Order:
    return Order(
        order_id,
        f"signal-{order_id}",
        "S1",
        AssetType.STOCK,
        side,
        quantity,
        OrderType.MARKET,
        created,
    )


def test_event_backtest_cash_positions_t_plus_one_and_replay_snapshot() -> None:
    engine = EventDrivenBacktest(initial_cash=100_000, rules=rules())
    buy = engine.submit(make_order("buy", Side.BUY, 1_000, START))
    assert buy.status is OrderStatus.ACCEPTED
    filled = engine.process_bar("buy", market_bar(1))
    assert filled.status is OrderStatus.FILLED
    snapshot = engine.snapshot(as_of=market_bar(1).available_at)
    assert snapshot.positions[0].quantity == 1_000
    assert snapshot.positions[0].available_quantity == 0
    assert snapshot.cash == pytest.approx(89_995)

    same_day_sell = make_order("sell-same", Side.SELL, 100, market_bar(1).available_at)
    engine.submit(same_day_sell)
    same_day_later = market_bar(1, available_hour=17)
    rejected = engine.process_bar("sell-same", same_day_later)
    assert rejected.status is OrderStatus.REJECTED
    assert "available position" in (rejected.rejection_reason or "")

    next_sell = make_order("sell-next", Side.SELL, 100, market_bar(1).available_at)
    engine.submit(next_sell)
    sold = engine.process_bar("sell-next", market_bar(2))
    assert sold.status is OrderStatus.FILLED
    final = engine.snapshot(as_of=market_bar(2).available_at)
    assert final.positions[0].quantity == 900
    assert final.cash == pytest.approx(90_989.5)
    assert len(final.fills) == 2
    assert final.total_fees > 0


def test_event_backtest_rejects_duplicate_future_and_insufficient_cash() -> None:
    engine = EventDrivenBacktest(initial_cash=100, rules=rules())
    item = make_order("buy", Side.BUY, 100, START)
    engine.submit(item)
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit(item)
    future_rejected = engine.process_bar("buy", market_bar(1))
    assert future_rejected.status is OrderStatus.REJECTED
    assert future_rejected.rejection_reason == "insufficient cash"
    with pytest.raises(ValueError):
        EventDrivenBacktest(initial_cash=-1, rules=rules())
