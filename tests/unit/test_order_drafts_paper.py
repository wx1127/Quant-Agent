from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.backtest.cn_market_rules import ChinaMarketRules, FeeScheduleRegistry
from quant_agent.backtest.contracts import (
    AssetType,
    FeeSchedule,
    MarketBar,
    OrderStatus,
    Side,
)
from quant_agent.execution.order_drafts import (
    OrderDraft,
    OrderDraftBatch,
    OrderDraftBuilder,
    PriceQuote,
)
from quant_agent.execution.paper import PaperBroker
from quant_agent.portfolio.builder import PortfolioBuildResult, PortfolioLine
from quant_agent.portfolio.snapshots import AccountSnapshot, HoldingSnapshot
from quant_agent.risk.contracts import RiskDecision

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)
FEE = FeeSchedule("fee-v1", date(2020, 1, 1), 0.0003, 5, 0.0005, 5)


def account() -> AccountSnapshot:
    return AccountSnapshot("s1", "paper-1", NOW, 100_000, 0, (), "paper", "v1")


def portfolio() -> PortfolioBuildResult:
    return PortfolioBuildResult(
        "d1",
        "s1",
        NOW,
        (
            PortfolioLine(
                "S1",
                AssetType.STOCK,
                "TECH",
                0,
                0.05,
                0.05,
                ("strategy",),
                "approved signal",
            ),
        ),
        0.95,
        0.05,
        {"TECH": 0.05},
        {"strategy": 0.05},
        "snap",
        "builder-v1",
        (),
    )


def risk(passed: bool = True) -> RiskDecision:
    return RiskDecision("r1", passed, (), (), "risk-v1", NOW)


def make_batch():
    return OrderDraftBuilder().build(
        portfolio(),
        risk(),
        account(),
        [PriceQuote("S1", 10, NOW)],
        fees=FEE,
        created_at=NOW,
        expires_at=NOW + timedelta(days=2),
    )


def test_order_draft_is_deterministic_non_executable_and_risk_gated() -> None:
    first = make_batch()
    second = make_batch()
    assert first.batch_hash == second.batch_hash
    assert first.drafts == second.drafts
    assert first.drafts[0].quantity == 500
    assert first.drafts[0].estimated_commission == 5
    assert not hasattr(first.drafts[0], "submit")
    rejected = RiskDecision.fail_closed(
        "r1", checked_at=NOW, policy_version="v1", reason="risk offline"
    )
    with pytest.raises(ValueError, match="risk"):
        OrderDraftBuilder().build(
            portfolio(),
            rejected,
            account(),
            [PriceQuote("S1", 10, NOW)],
            fees=FEE,
            created_at=NOW,
            expires_at=NOW + timedelta(days=1),
        )
    sell_account = AccountSnapshot(
        "sell-snapshot",
        "paper-1",
        NOW,
        90_000,
        0,
        (HoldingSnapshot("S1", AssetType.STOCK, "TECH", 1_000, 300, 700, 8, 10),),
        "paper",
        "v1",
    )
    sell_portfolio = replace(
        portfolio(),
        account_snapshot_id="sell-snapshot",
        lines=(
            PortfolioLine(
                "S1",
                AssetType.STOCK,
                "TECH",
                0.1,
                0,
                -0.1,
                (),
                "exit",
            ),
        ),
    )
    sell_batch = OrderDraftBuilder().build(
        sell_portfolio,
        risk(),
        sell_account,
        [PriceQuote("S1", 10, NOW)],
        fees=FEE,
        created_at=NOW,
        expires_at=NOW + timedelta(days=1),
    )
    assert sell_batch.drafts[0].side is Side.SELL
    assert sell_batch.drafts[0].quantity == 300


def test_paper_broker_is_idempotent_conserves_ledger_and_retains_t_plus_one() -> None:
    broker = PaperBroker(
        account(),
        ChinaMarketRules(FeeScheduleRegistry([FEE])),
    )
    batch = make_batch()
    bar_time = NOW + timedelta(days=1)
    bar = MarketBar(
        "S1",
        AssetType.STOCK,
        bar_time.date(),
        10,
        10,
        10,
        10,
        10_000,
        100_000,
        bar_time,
    )
    result = broker.submit(batch, bars=[bar], submitted_at=NOW + timedelta(hours=1))
    duplicate = broker.submit(batch, bars=[bar], submitted_at=NOW + timedelta(hours=2))
    assert duplicate is result
    assert len(result.fills) == 1
    holding = result.account_snapshot.holdings[0]
    assert holding.quantity == 500
    assert holding.available_quantity == 0
    assert holding.frozen_quantity == 500
    assert result.account_snapshot.available_cash < 95_000
    assert not hasattr(broker, "credentials")

    sell_draft = OrderDraft(
        "sell-draft",
        "paper-1",
        "d2",
        "S1",
        AssetType.STOCK,
        Side.SELL,
        100,
        10,
        5,
        0.5,
        0.5,
        NOW + timedelta(days=2),
    )
    sell_batch = OrderDraftBatch(
        "paper-1",
        "d2",
        result.account_snapshot.snapshot_id,
        (sell_draft,),
        NOW + timedelta(days=5),
        "risk-v1",
        "draft-v1",
        "sell-batch",
    )
    sell_bar_time = NOW + timedelta(days=3)
    sell_bar = replace(
        bar,
        trade_date=sell_bar_time.date(),
        available_at=sell_bar_time,
    )
    sold = broker.submit(
        sell_batch,
        bars=[sell_bar],
        submitted_at=NOW + timedelta(days=2),
    )
    assert sold.fills[0].quantity == 100
    assert sold.account_snapshot.holdings[0].available_quantity == 400


def test_paper_broker_retains_unmatched_order_without_live_fallback() -> None:
    broker = PaperBroker(account(), ChinaMarketRules(FeeScheduleRegistry([FEE])))
    result = broker.submit(
        make_batch(),
        bars=[],
        submitted_at=NOW + timedelta(hours=1),
    )
    assert result.fills == ()
    assert result.orders[0].order.status is OrderStatus.ACCEPTED
    assert result.orders[0].message == "waiting for market bar"
