"""One-day PAPER validation using close signals and next-trading-day open fills."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from quant_agent.backtest.cn_market_rules import ChinaMarketRules, FeeScheduleRegistry
from quant_agent.backtest.contracts import AssetType, FeeSchedule, MarketBar, Side
from quant_agent.data.domain import DailyBar
from quant_agent.execution.order_drafts import OrderDraft, OrderDraftBatch
from quant_agent.execution.paper import PaperBroker
from quant_agent.portfolio.snapshots import AccountSnapshot, HoldingSnapshot
from quant_agent.shadow.historical import HistoricalShadowEngine

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_FEE = FeeSchedule("paper-fee-v1", date(2023, 8, 28), 0.0003, 5.0, 0.0005, 5.0)


class PaperDailyEngine:
    """Deterministic daily PAPER loop with no broker endpoint or credentials."""

    def run(
        self,
        *,
        trading_date: date,
        next_trading_date: date,
        observed_at: datetime,
        bars: tuple[DailyBar, ...],
        account: AccountSnapshot,
        pending: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current = tuple(item for item in bars if item.trade_date == trading_date)
        if not current:
            raise ValueError("current trading-day bars are required")
        if observed_at.date() != trading_date or observed_at.time() < time(16, 5):
            raise ValueError("paper day can run only after the trading-date close")
        if any(item.available_at > observed_at for item in bars):
            raise ValueError("future market data cannot enter paper validation")

        analysis = HistoricalShadowEngine().run_day(trading_date, bars)
        marked_account = _mark_account(account, current, observed_at)
        execution = self._execute_pending(pending, marked_account, current, trading_date)
        account_after = execution["account"]
        drafts = _build_next_day_drafts(
            analysis.details["candidates"],
            account_after,
            current,
            trading_date,
            next_trading_date,
        )
        reconciliation = _reconcile(account_after)
        return {
            "schema_version": "p8-paper-day-v1",
            "mode": "PAPER",
            "broker": "PaperBroker",
            "trading_date": trading_date.isoformat(),
            "observed_at": observed_at.isoformat(),
            "market_data_as_of": max(item.available_at for item in current).isoformat(),
            "input_snapshot_hash": _bars_hash(bars),
            "market": {
                "instrument_count": len(current),
                "regime": analysis.details["regime"],
                "mainlines": analysis.details["mainlines"],
                "candidates": analysis.details["candidates"],
            },
            "execution": {key: value for key, value in execution.items() if key != "account"},
            "next_day_order_draft": drafts,
            "account": json.loads(account_after.to_json()),
            "reconciliation": reconciliation,
            "alerts": [],
            "manual_interventions": [],
            "pipeline_succeeded": True,
            "live_connection_attempted": False,
            "notes": (
                "signals use close data and are eligible only for the next trading day; "
                "fills use next-day open plus configured slippage"
            ),
        }

    def _execute_pending(
        self,
        pending: dict[str, Any] | None,
        account: AccountSnapshot,
        current: tuple[DailyBar, ...],
        trading_date: date,
    ) -> dict[str, Any]:
        if pending is None:
            return {
                "signal_date": None,
                "orders": [],
                "fills": [],
                "fees": 0.0,
                "slippage": 0.0,
                "account": account,
            }
        if pending["execute_on"] != trading_date.isoformat():
            raise ValueError("pending draft is not eligible on this trading date")
        batch = _batch_from_mapping(pending["batch"])
        released = _release_prior_day_frozen(account, trading_date)
        bars_by_id = {item.instrument_id: item for item in current}
        requested = {item.instrument_id for item in batch.drafts}
        market_bars = [
            _execution_bar(bars_by_id[instrument_id], batch)
            for instrument_id in sorted(requested & bars_by_id.keys())
        ]
        submitted_at = datetime.combine(trading_date, time(9, 31), tzinfo=_SHANGHAI)
        result = PaperBroker(
            released,
            ChinaMarketRules(FeeScheduleRegistry([_FEE])),
        ).submit(batch, bars=market_bars, submitted_at=submitted_at)
        return {
            "signal_date": pending["signal_date"],
            "orders": [
                {
                    "draft_id": item.draft_id,
                    "status": item.order.status,
                    "message": item.message,
                }
                for item in result.orders
            ],
            "fills": [asdict(item) for item in result.fills],
            "fees": round(sum(item.commission + item.tax for item in result.fills), 6),
            "slippage": round(sum(item.slippage for item in result.fills), 6),
            "account": result.account_snapshot,
        }


def _build_next_day_drafts(
    candidate_rows: object,
    account: AccountSnapshot,
    current: tuple[DailyBar, ...],
    signal_date: date,
    execute_on: date,
) -> dict[str, Any]:
    candidates = list(candidate_rows) if isinstance(candidate_rows, list) else []
    selected = [str(item["instrument_id"]) for item in candidates[:5]]
    prices = {item.instrument_id: float(item.close) for item in current}
    holdings = {item.instrument_id: item for item in account.holdings}
    created_at = datetime.combine(signal_date, time(16, 5), tzinfo=_SHANGHAI)
    drafts: list[OrderDraft] = []
    for instrument_id, holding in sorted(holdings.items()):
        if instrument_id in selected or holding.available_quantity < 100:
            continue
        quantity = holding.available_quantity - holding.available_quantity % 100
        drafts.append(
            _draft(account, instrument_id, Side.SELL, quantity, prices[instrument_id], created_at)
        )
    target_value = account.total_equity * 0.15
    reserved_cash = 0.0
    for instrument_id in selected:
        if instrument_id in holdings or instrument_id not in prices:
            continue
        price = prices[instrument_id]
        quantity = int(target_value / price) // 100 * 100
        cost = quantity * price * 1.001
        if quantity < 100 or reserved_cash + cost > account.available_cash:
            continue
        reserved_cash += cost
        drafts.append(_draft(account, instrument_id, Side.BUY, quantity, price, created_at))
    decision_id = f"p8-paper-{signal_date}"
    expires_at = datetime.combine(execute_on, time(15, 0), tzinfo=_SHANGHAI)
    canonical = {
        "account_id": account.account_id,
        "decision_id": decision_id,
        "account_snapshot_id": account.snapshot_id,
        "drafts": [asdict(item) for item in drafts],
        "expires_at": expires_at.isoformat(),
    }
    batch_hash = hashlib.sha256(
        json.dumps(canonical, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    batch = OrderDraftBatch(
        account.account_id,
        decision_id,
        account.snapshot_id,
        tuple(drafts),
        expires_at,
        "p8-paper-risk-v1",
        "p8-paper-batch-v1",
        batch_hash,
    )
    return {
        "signal_date": signal_date.isoformat(),
        "execute_on": execute_on.isoformat(),
        "candidate_ids": selected,
        "batch": _batch_to_mapping(batch),
    }


def _draft(
    account: AccountSnapshot,
    instrument_id: str,
    side: Side,
    quantity: int,
    price: float,
    created_at: datetime,
) -> OrderDraft:
    key = f"{account.snapshot_id}|{instrument_id}|{side}|{quantity}|{created_at.isoformat()}"
    gross = quantity * price
    return OrderDraft(
        hashlib.sha256(key.encode()).hexdigest(),
        account.account_id,
        f"p8-paper-{created_at.date()}",
        instrument_id,
        AssetType.STOCK,
        side,
        quantity,
        price,
        max(_FEE.minimum_commission, gross * _FEE.commission_rate),
        gross * _FEE.stock_sell_tax_rate if side is Side.SELL else 0.0,
        gross * _FEE.slippage_bps / 10_000,
        created_at,
    )


def _execution_bar(bar: DailyBar, batch: OrderDraftBatch) -> MarketBar:
    required = sum(
        item.quantity for item in batch.drafts if item.instrument_id == bar.instrument_id
    )
    return MarketBar(
        bar.instrument_id,
        AssetType.STOCK,
        bar.trade_date,
        float(bar.open),
        float(bar.open),
        float(bar.open),
        float(bar.open),
        max(required * 10, 100),
        float(bar.open) * max(required * 10, 100),
        datetime.combine(bar.trade_date, time(9, 32), tzinfo=_SHANGHAI),
    )


def _release_prior_day_frozen(account: AccountSnapshot, trading_date: date) -> AccountSnapshot:
    return AccountSnapshot(
        account.snapshot_id,
        account.account_id,
        account.as_of,
        account.available_cash,
        account.frozen_cash,
        tuple(
            HoldingSnapshot(
                item.instrument_id,
                item.asset_type,
                item.industry_id,
                item.quantity,
                item.quantity if account.as_of.date() < trading_date else item.available_quantity,
                0 if account.as_of.date() < trading_date else item.frozen_quantity,
                item.average_cost,
                item.last_price,
            )
            for item in account.holdings
        ),
        account.source,
        account.version,
    )


def _mark_account(
    account: AccountSnapshot,
    current: tuple[DailyBar, ...],
    observed_at: datetime,
) -> AccountSnapshot:
    prices = {item.instrument_id: float(item.close) for item in current}
    return AccountSnapshot(
        f"paper-{observed_at.date()}",
        account.account_id,
        observed_at,
        account.available_cash,
        account.frozen_cash,
        tuple(
            HoldingSnapshot(
                item.instrument_id,
                item.asset_type,
                item.industry_id,
                item.quantity,
                item.available_quantity,
                item.frozen_quantity,
                item.average_cost,
                prices.get(item.instrument_id, item.last_price),
            )
            for item in account.holdings
        ),
        "paper",
        "p8-paper-account-v1",
    )


def _reconcile(account: AccountSnapshot) -> dict[str, Any]:
    holdings_value = sum(item.quantity * item.last_price for item in account.holdings)
    calculated = account.available_cash + account.frozen_cash + holdings_value
    return {
        "status": "RECONCILED" if abs(calculated - account.total_equity) < 0.01 else "MISMATCH",
        "cash": round(account.total_cash, 6),
        "holdings_value": round(holdings_value, 6),
        "total_equity": round(account.total_equity, 6),
        "difference": round(calculated - account.total_equity, 6),
    }


def _bars_hash(bars: tuple[DailyBar, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in bars]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _batch_to_mapping(batch: OrderDraftBatch) -> dict[str, Any]:
    return {
        "account_id": batch.account_id,
        "decision_id": batch.decision_id,
        "account_snapshot_id": batch.account_snapshot_id,
        "drafts": [asdict(item) for item in batch.drafts],
        "expires_at": batch.expires_at.isoformat(),
        "risk_policy_version": batch.risk_policy_version,
        "batch_version": batch.batch_version,
        "batch_hash": batch.batch_hash,
    }


def _batch_from_mapping(payload: dict[str, Any]) -> OrderDraftBatch:
    drafts = tuple(
        OrderDraft(
            **{
                **item,
                "asset_type": AssetType(item["asset_type"]),
                "side": Side(item["side"]),
                "created_at": (
                    datetime.fromisoformat(item["created_at"])
                    if isinstance(item["created_at"], str)
                    else item["created_at"]
                ),
            }
        )
        for item in payload["drafts"]
    )
    return OrderDraftBatch(
        payload["account_id"],
        payload["decision_id"],
        payload["account_snapshot_id"],
        drafts,
        (
            datetime.fromisoformat(payload["expires_at"])
            if isinstance(payload["expires_at"], str)
            else payload["expires_at"]
        ),
        payload["risk_policy_version"],
        payload["batch_version"],
        payload["batch_hash"],
    )
