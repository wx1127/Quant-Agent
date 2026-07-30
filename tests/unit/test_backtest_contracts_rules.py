from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.backtest.cn_market_rules import (
    ChinaMarketRuleConfig,
    ChinaMarketRules,
    FeeScheduleRegistry,
)
from quant_agent.backtest.contracts import (
    AssetType,
    FeeSchedule,
    MarketBar,
    Order,
    OrderStatus,
    OrderType,
    Side,
    SignalEvent,
)

NOW = datetime(2026, 1, 1, 15, tzinfo=UTC)


def fees() -> FeeScheduleRegistry:
    return FeeScheduleRegistry(
        [
            FeeSchedule("old", date(2020, 1, 1), 0.001, 5, 0.001, 0),
            FeeSchedule("new", date(2025, 1, 1), 0.0003, 5, 0.0005, 10),
        ]
    )


def order(
    side: Side = Side.BUY,
    *,
    asset_type: AssetType = AssetType.STOCK,
    quantity: int = 100,
    created_at: datetime = NOW,
) -> Order:
    return Order(
        "O1",
        "S1",
        "000001",
        asset_type,
        side,
        quantity,
        OrderType.MARKET,
        created_at,
    )


def bar(
    *,
    asset_type: AssetType = AssetType.STOCK,
    open_price: float = 10,
    volume: int = 10_000,
    available_at: datetime = NOW + timedelta(days=1),
    suspended: bool = False,
    limit_up: float | None = None,
    limit_down: float | None = None,
) -> MarketBar:
    return MarketBar(
        "000001",
        asset_type,
        available_at.date(),
        open_price,
        open_price,
        open_price,
        open_price,
        volume,
        open_price * volume,
        available_at,
        suspended,
        limit_up,
        limit_down,
    )


def test_events_serialize_and_order_transitions_are_replayable() -> None:
    signal = SignalEvent("S1", "strategy", "000001", AssetType.STOCK, NOW, 0.2, "d1", "s1", "p1")
    assert '"signal_id": "S1"' in signal.to_json()
    assert SignalEvent.from_json(signal.to_json()) == signal
    accepted = order().transition(OrderStatus.ACCEPTED)
    partial = accepted.transition(OrderStatus.PARTIALLY_FILLED, filled_quantity=50)
    filled = partial.transition(OrderStatus.FILLED, filled_quantity=100)
    assert filled.remaining_quantity == 0
    assert '"status": "FILLED"' in filled.to_json()
    assert Order.from_json(filled.to_json()) == filled
    with pytest.raises(ValueError, match="illegal"):
        filled.transition(OrderStatus.CANCELLED)
    with pytest.raises(ValueError, match="full quantity"):
        accepted.transition(OrderStatus.FILLED, filled_quantity=50)


def test_effective_fee_rules_minimum_fee_tax_slippage_and_partial_fill() -> None:
    registry = fees()
    assert registry.as_of(date(2024, 1, 1)).version == "old"
    assert registry.as_of(date(2026, 1, 1)).version == "new"
    rules = ChinaMarketRules(registry, ChinaMarketRuleConfig(maximum_volume_participation=0.05))
    buy = order(quantity=1_000).transition(OrderStatus.ACCEPTED)
    fill = rules.match(buy, bar(volume=5_000), available_quantity=0)
    assert fill is not None
    assert fill.quantity == 200
    assert fill.commission == 5
    assert fill.tax == 0
    assert fill.slippage > 0
    assert type(fill).from_json(fill.to_json()) == fill
    sell = order(Side.SELL).transition(OrderStatus.ACCEPTED)
    sell_fill = rules.match(sell, bar(), available_quantity=100)
    assert sell_fill is not None and sell_fill.tax > 0
    assert sell_fill.fee_version == "new"
    etf_sell = order(Side.SELL, asset_type=AssetType.ETF).transition(OrderStatus.ACCEPTED)
    etf_fill = rules.match(
        etf_sell,
        bar(asset_type=AssetType.ETF),
        available_quantity=100,
    )
    assert etf_fill is not None and etf_fill.tax == 0


@pytest.mark.parametrize(
    ("test_order", "test_bar", "available", "reason"),
    [
        (order(quantity=50), bar(), 0, "multiple"),
        (order(), bar(suspended=True), 0, "suspended"),
        (order(), bar(open_price=11, limit_up=11), 0, "limit-up"),
        (order(Side.SELL), bar(open_price=9, limit_down=9), 100, "limit-down"),
        (order(Side.SELL), bar(), 0, "available"),
    ],
)
def test_market_rule_rejections(
    test_order: Order, test_bar: MarketBar, available: int, reason: str
) -> None:
    message = ChinaMarketRules(fees()).validate(test_order, test_bar, available_quantity=available)
    assert message is not None and reason in message


def test_fee_and_rule_configuration_validation() -> None:
    with pytest.raises(ValueError):
        FeeScheduleRegistry([])
    with pytest.raises(ValueError):
        ChinaMarketRuleConfig(stock_lot_size=0)
    with pytest.raises(ValueError):
        FeeSchedule("x", date.today(), -1, 0, 0, 0)
