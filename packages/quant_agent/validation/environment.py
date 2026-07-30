"""Isolated, repeatable full-flow validation environment."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_agent.backtest.cn_market_rules import ChinaMarketRules, FeeScheduleRegistry
from quant_agent.backtest.contracts import AssetType, FeeSchedule, MarketBar, Side
from quant_agent.execution.order_drafts import OrderDraft, OrderDraftBatch
from quant_agent.execution.paper import PaperBroker
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.reconciliation.core import ReconciliationEngine


@dataclass(frozen=True, slots=True)
class E2EResult:
    decision_id: str
    data_version: str
    batch_hash: str
    order_ids: tuple[str, ...]
    fill_ids: tuple[str, ...]
    ending_cash: float
    ending_positions: tuple[tuple[str, int], ...]
    reconciled: bool

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class E2EEnvironment:
    """Fixed-data PAPER environment with no production endpoint or credentials."""

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path
        self._fixture: dict[str, Any] = {}
        self._broker: PaperBroker | None = None

    def reset(self) -> None:
        self._fixture = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        if self._fixture["runtime_mode"] != "PAPER":
            raise ValueError("E2E fixture must use PAPER mode")
        if self._fixture.get("broker") != "fake":
            raise ValueError("E2E fixture must use the fake broker")
        self._broker = PaperBroker(self._account(), self._rules())

    def cleanup(self) -> None:
        self._fixture = {}
        self._broker = None

    def run(self) -> E2EResult:
        self.reset()
        try:
            now = datetime.fromisoformat(self._fixture["as_of"])
            account = self._account()
            draft = OrderDraft(
                draft_id="draft-fixed-001",
                account_id=account.account_id,
                decision_id=self._fixture["decision_id"],
                instrument_id=self._fixture["instrument_id"],
                asset_type=AssetType.STOCK,
                side=Side.BUY,
                quantity=int(self._fixture["quantity"]),
                reference_price=float(self._fixture["reference_price"]),
                estimated_commission=5.0,
                estimated_tax=0.0,
                estimated_slippage=0.5,
                created_at=now,
            )
            batch = OrderDraftBatch(
                account_id=account.account_id,
                decision_id=self._fixture["decision_id"],
                account_snapshot_id=account.snapshot_id,
                drafts=(draft,),
                expires_at=now + timedelta(hours=4),
                risk_policy_version="risk-p8-v1",
                batch_version="e2e-fixed-v1",
                batch_hash=self._fixture["batch_hash"],
            )
            price = float(self._fixture["market_open"])
            bar = MarketBar(
                instrument_id=draft.instrument_id,
                asset_type=AssetType.STOCK,
                trade_date=now.date(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=100_000,
                turnover=1_000_000,
                available_at=now + timedelta(minutes=1),
            )
            assert self._broker is not None
            paper = self._broker.submit(batch, bars=[bar], submitted_at=now)
            duplicate = self._broker.submit(batch, bars=[bar], submitted_at=now)
            if duplicate is not paper:
                raise AssertionError("paper submission was not idempotent")
            reconciliation = ReconciliationEngine().reconcile(
                batch,
                paper,
                expected_account=paper.account_snapshot,
            )
            return E2EResult(
                decision_id=batch.decision_id,
                data_version=self._fixture["data_version"],
                batch_hash=batch.batch_hash,
                order_ids=tuple(item.order.order_id for item in paper.orders),
                fill_ids=tuple(item.fill_id for item in paper.fills),
                ending_cash=round(paper.account_snapshot.available_cash, 4),
                ending_positions=tuple(
                    (item.instrument_id, item.quantity) for item in paper.account_snapshot.holdings
                ),
                reconciled=reconciliation.matched,
            )
        finally:
            self.cleanup()

    def _account(self) -> AccountSnapshot:
        now = datetime.fromisoformat(self._fixture["as_of"])
        return AccountSnapshot(
            snapshot_id="account-fixed-001",
            account_id="paper-e2e",
            as_of=now,
            available_cash=100_000.0,
            frozen_cash=0.0,
            holdings=(),
            source="fixed-fixture",
            version="account-e2e-v1",
        )

    @staticmethod
    def _rules() -> ChinaMarketRules:
        fee = FeeSchedule("fee-e2e-v1", date(2020, 1, 1), 0.0003, 5.0, 0.0005, 1.0)
        return ChinaMarketRules(FeeScheduleRegistry([fee]))
