from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_agent.backtest.contracts import (
    AssetType,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from quant_agent.execution.order_drafts import OrderDraft, OrderDraftBatch
from quant_agent.execution.paper import PaperBatchResult, PaperOrder
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.reconciliation.core import (
    ReconciliationEngine,
    ReconciliationSeverity,
)
from quant_agent.risk.kill_switch import KillSwitch, KillSwitchScope

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def account() -> AccountSnapshot:
    return AccountSnapshot("s1", "paper-1", NOW, 100_000, 0, (), "paper", "v1")


def make_batch() -> OrderDraftBatch:
    draft = OrderDraft(
        "D1",
        "paper-1",
        "d1",
        "S1",
        AssetType.STOCK,
        Side.BUY,
        500,
        10,
        5,
        0,
        0,
        NOW,
    )
    return OrderDraftBatch(
        "paper-1",
        "d1",
        "s1",
        (draft,),
        NOW + timedelta(days=1),
        "risk-v1",
        "draft-v1",
        "batch-hash",
    )


def paper_result() -> PaperBatchResult:
    batch = make_batch()
    draft = batch.drafts[0]
    order = Order(
        "P1",
        draft.draft_id,
        draft.instrument_id,
        AssetType.STOCK,
        Side.BUY,
        draft.quantity,
        OrderType.MARKET,
        NOW,
        status=OrderStatus.FILLED,
        filled_quantity=draft.quantity,
    )
    fill = Fill(
        "F1",
        "P1",
        "S1",
        Side.BUY,
        draft.quantity,
        10,
        5_000,
        5,
        0,
        0,
        NOW,
        "fee-v1",
    )
    return PaperBatchResult(
        batch.batch_hash,
        (PaperOrder(draft.draft_id, order, "filled"),),
        (fill,),
        account(),
    )


def test_reconciliation_handles_normal_partial_duplicate_and_cash_difference() -> None:
    batch = make_batch()
    normal_paper = paper_result()
    normal = ReconciliationEngine().reconcile(batch, normal_paper, expected_account=account())
    assert normal.matched and not normal.trigger_stop
    order = normal_paper.orders[0].order
    partial_order = replace(
        order,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=100,
    )
    partial_paper = replace(
        normal_paper,
        orders=(PaperOrder(batch.drafts[0].draft_id, partial_order, "partial"),),
        fills=(replace(normal_paper.fills[0], quantity=100, gross_amount=1_000),),
    )
    partial = ReconciliationEngine().reconcile(batch, partial_paper, expected_account=account())
    assert not partial.matched
    assert partial.differences[0].severity is ReconciliationSeverity.WARNING
    assert not partial.trigger_stop
    duplicate = replace(
        normal_paper,
        fills=(normal_paper.fills[0], normal_paper.fills[0]),
    )
    incident = ReconciliationEngine().reconcile(batch, duplicate, expected_account=account())
    assert incident.trigger_stop
    assert any(item.category == "DUPLICATE_FILL" for item in incident.differences)
    missing = ReconciliationEngine().reconcile(
        batch,
        replace(normal_paper, orders=()),
        expected_account=account(),
    )
    assert missing.trigger_stop
    assert missing.differences[0].category == "ORDER"


def test_kill_switch_auto_trigger_rejects_orders_and_agent_cannot_recover() -> None:
    switch = KillSwitch()
    assert switch.order_allowed("paper-1")
    record = switch.trigger_from_reconciliation(
        "paper-1",
        trigger_stop=True,
        occurred_at=NOW,
        incident_snapshot_hash="incident-hash",
    )
    assert record is not None
    assert not switch.order_allowed("paper-1")
    with pytest.raises(PermissionError):
        switch.recover(
            scope=KillSwitchScope.ACCOUNT,
            account_id="paper-1",
            reason="agent request",
            actor_id="agent",
            actor_role="AGENT",
            occurred_at=NOW,
            reviewed_snapshot_hash="review",
        )
    recovered = switch.recover(
        scope=KillSwitchScope.ACCOUNT,
        account_id="paper-1",
        reason="manual review complete",
        actor_id="risk-admin",
        actor_role="RISK_ADMIN",
        occurred_at=NOW,
        reviewed_snapshot_hash="review",
    )
    assert recovered.action == "RECOVER"
    assert switch.order_allowed("paper-1")
    assert [item.action for item in switch.audit_log] == ["TRIGGER", "RECOVER"]
