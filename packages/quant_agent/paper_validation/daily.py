"""One-day PAPER validation using close signals and next-trading-day open fills."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import date, datetime, time
from decimal import Decimal
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
_EXCLUDED_BUY_RULE_VERSION = "p8-paper-buy-exclusions-v2"
_EXIT_RULE_VERSION = "p8-paper-exit-rules-v1"
_STOP_LOSS = -0.08
_TAKE_PROFIT = 0.20
_TRAIL_ACTIVATION = 0.10
_TRAIL_DRAWDOWN = -0.08
_GAP_DOWN = -0.05
_VOLUME_SELL_OFF = -0.07
_VOLUME_SPIKE = 1.5
_MAX_HOLDING_DAYS = 20


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
        instrument_names: dict[str, str] | None = None,
        instrument_industries: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        current = tuple(item for item in bars if item.trade_date == trading_date)
        if not current:
            raise ValueError("current trading-day bars are required")
        if observed_at.date() != trading_date or observed_at.time() < time(16, 5):
            raise ValueError("paper day can run only after the trading-date close")
        if any(item.available_at > observed_at for item in bars):
            raise ValueError("future market data cannot enter paper validation")

        industries = instrument_industries or {}
        analysis = HistoricalShadowEngine(
            candidate_exclusion=_candidate_exclusion,
            segment_resolver=(
                lambda instrument_id: industries.get(instrument_id)
            )
            if instrument_industries is not None
            else None,
        ).run_day(trading_date, bars)
        names = instrument_names or {}
        beginning_account = _mark_account(account, current, observed_at)
        execution = self._execute_pending(pending, beginning_account, current, trading_date)
        account_after = _mark_account(execution["account"], current, observed_at)
        position_state = _update_position_state(
            pending,
            account_after,
            current,
            execution["fills"],
            trading_date,
        )
        drafts = _build_next_day_drafts(
            analysis.details["candidates"],
            analysis.details["candidate_exclusions"],
            account_after,
            current,
            bars,
            analysis.evidence.mainline_ids,
            industries if instrument_industries is not None else None,
            position_state,
            trading_date,
            next_trading_date,
        )
        reconciliation = _reconcile(account_after)
        close_report = _close_report(beginning_account, account_after, current, names)
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
                "candidates": _named_rows(analysis.details["candidates"], names),
                "candidate_exclusions": _named_rows(
                    analysis.details["candidate_exclusions"], names
                ),
            },
            "execution": _named_execution(
                {key: value for key, value in execution.items() if key != "account"},
                names,
            ),
            "next_day_order_draft": _named_draft(drafts, names),
            "account": _named_account(account_after, names),
            "reconciliation": reconciliation,
            "close_report": close_report,
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
    exclusion_rows: object,
    account: AccountSnapshot,
    current: tuple[DailyBar, ...],
    bars: tuple[DailyBar, ...],
    mainline_ids: tuple[str, ...],
    instrument_industries: dict[str, str] | None,
    position_state: dict[str, dict[str, Any]],
    signal_date: date,
    execute_on: date,
) -> dict[str, Any]:
    candidates = list(candidate_rows) if isinstance(candidate_rows, list) else []
    excluded = list(exclusion_rows) if isinstance(exclusion_rows, list) else []
    selected = [
        str(item["instrument_id"])
        for item in candidates
        if isinstance(item, dict)
        and "instrument_id" in item
    ][:5]
    prices = {item.instrument_id: float(item.close) for item in current}
    holdings = {item.instrument_id: item for item in account.holdings}
    history = _history_by_instrument(bars, signal_date)
    exit_signals = _evaluate_exit_rules(
        holdings,
        history,
        set(mainline_ids),
        set(selected),
        position_state,
        instrument_industries,
    )
    created_at = datetime.combine(signal_date, time(16, 5), tzinfo=_SHANGHAI)
    drafts: list[OrderDraft] = []
    for instrument_id, holding in sorted(holdings.items()):
        signal = exit_signals.get(instrument_id)
        if (
            signal is None
            or holding.available_quantity < 100
            or instrument_id not in prices
        ):
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
        "excluded_buy_candidates": excluded,
        "buy_exclusion_rule_version": _EXCLUDED_BUY_RULE_VERSION,
        "exit_rule_version": _EXIT_RULE_VERSION,
        "exit_signals": list(exit_signals.values()),
        "position_state": position_state,
        "batch": _batch_to_mapping(batch),
    }


def _buy_exclusion_reason(
    instrument_id: str,
    *,
    daily_return: Decimal | None = None,
) -> str | None:
    try:
        country, exchange, symbol = instrument_id.split(".", maxsplit=2)
    except ValueError:
        return "invalid instrument identifier"
    if exchange == "HK":
        return "hong kong stock is excluded from PAPER buys"
    if country == "CN" and exchange == "BJ":
        return "daily price limit above 10% is excluded from PAPER buys"
    if country == "CN" and exchange == "SH" and symbol.startswith(("688", "689")):
        return "daily price limit above 10% is excluded from PAPER buys"
    if country == "CN" and exchange == "SZ" and symbol.startswith(("300", "301")):
        return "daily price limit above 10% is excluded from PAPER buys"
    if daily_return is not None and abs(daily_return) > Decimal("0.10"):
        return "absolute daily return above 10% is excluded before candidate ranking"
    return None


def _candidate_exclusion(item: DailyBar, series: list[DailyBar]) -> str | None:
    daily_return = item.close / series[-2].close - 1 if len(series) >= 2 else None
    return _buy_exclusion_reason(item.instrument_id, daily_return=daily_return)


def _history_by_instrument(
    bars: tuple[DailyBar, ...], signal_date: date
) -> dict[str, list[DailyBar]]:
    history: dict[str, list[DailyBar]] = {}
    for item in sorted(bars, key=lambda value: (value.trade_date, value.instrument_id)):
        if item.trade_date <= signal_date:
            history.setdefault(item.instrument_id, []).append(item)
    return history


def _update_position_state(
    pending: dict[str, Any] | None,
    account: AccountSnapshot,
    current: tuple[DailyBar, ...],
    fills: list[dict[str, Any]],
    trading_date: date,
) -> dict[str, dict[str, Any]]:
    prior = pending.get("position_state", {}) if isinstance(pending, dict) else {}
    prior = prior if isinstance(prior, dict) else {}
    bought = {
        str(item["instrument_id"])
        for item in fills
        if item.get("side") in {Side.BUY, Side.BUY.value}
    }
    closes = {item.instrument_id: float(item.close) for item in current}
    state: dict[str, dict[str, Any]] = {}
    for holding in account.holdings:
        previous = prior.get(holding.instrument_id)
        previous = previous if isinstance(previous, dict) else {}
        is_new = holding.instrument_id in bought or not previous
        prior_days = int(previous.get("holding_days", 0)) if previous else 0
        last_evaluated = str(previous.get("last_evaluated_date", ""))
        holding_days = 1 if is_new else prior_days + (last_evaluated != trading_date.isoformat())
        close = closes.get(holding.instrument_id, holding.last_price)
        previous_peak = float(previous.get("peak_close", close)) if previous else close
        state[holding.instrument_id] = {
            "entry_date": (
                trading_date.isoformat() if is_new else str(previous.get("entry_date"))
            ),
            "holding_days": holding_days,
            "peak_close": round(max(previous_peak, close), 6),
            "last_evaluated_date": trading_date.isoformat(),
            "inferred_entry": (
                bool(previous.get("inferred_entry", False))
                if previous
                else holding.instrument_id not in bought
            ),
        }
    return state


def _evaluate_exit_rules(
    holdings: dict[str, HoldingSnapshot],
    history: dict[str, list[DailyBar]],
    mainline_ids: set[str],
    selected: set[str],
    position_state: dict[str, dict[str, Any]],
    instrument_industries: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    for instrument_id, holding in sorted(holdings.items()):
        series = history.get(instrument_id, [])
        state = position_state.get(instrument_id, {})
        reasons: list[dict[str, Any]] = []
        if series:
            current = series[-1]
            close = float(current.close)
            pnl_return = close / holding.average_cost - 1 if holding.average_cost else 0.0
            peak = float(state.get("peak_close", close))
            peak_return = peak / holding.average_cost - 1 if holding.average_cost else 0.0
            trailing_drawdown = close / peak - 1 if peak else 0.0
            if pnl_return <= _STOP_LOSS:
                _add_exit_reason(reasons, "STOP_LOSS", 10, pnl_return, _STOP_LOSS)
            if pnl_return >= _TAKE_PROFIT:
                _add_exit_reason(reasons, "TAKE_PROFIT", 30, pnl_return, _TAKE_PROFIT)
            if peak_return >= _TRAIL_ACTIVATION and trailing_drawdown <= _TRAIL_DRAWDOWN:
                _add_exit_reason(
                    reasons,
                    "TRAILING_STOP",
                    20,
                    trailing_drawdown,
                    _TRAIL_DRAWDOWN,
                )
            _append_moving_average_exits(reasons, series)
            if len(series) >= 2:
                previous = series[-2]
                gap = float(current.open / previous.close - 1)
                daily_return = float(current.close / previous.close - 1)
                if gap <= _GAP_DOWN:
                    _add_exit_reason(reasons, "LARGE_GAP_DOWN", 12, gap, _GAP_DOWN)
                prior_volumes = [float(item.volume) for item in series[-6:-1]]
                average_volume = statistics.fmean(prior_volumes) if prior_volumes else 0.0
                volume_ratio = float(current.volume) / average_volume if average_volume else 0.0
                if daily_return <= _VOLUME_SELL_OFF and volume_ratio >= _VOLUME_SPIKE:
                    reasons.append(
                        {
                            "rule_id": "HIGH_VOLUME_SELL_OFF",
                            "priority": 11,
                            "observed": {
                                "daily_return": round(daily_return, 8),
                                "volume_ratio": round(volume_ratio, 6),
                            },
                            "threshold": {
                                "daily_return_lte": _VOLUME_SELL_OFF,
                                "volume_ratio_gte": _VOLUME_SPIKE,
                            },
                        }
                    )
        segment = (
            instrument_industries.get(instrument_id)
            if instrument_industries is not None
            else _paper_segment(instrument_id)
        )
        if segment not in mainline_ids:
            reasons.append(
                {
                    "rule_id": "MAINLINE_EXIT",
                    "priority": 50,
                    "observed": segment or "UNCLASSIFIED",
                    "threshold": "holding must remain in a confirmed mainline",
                }
            )
        holding_days = int(state.get("holding_days", 1))
        if holding_days >= _MAX_HOLDING_DAYS:
            _add_exit_reason(
                reasons,
                "MAX_HOLDING_DAYS",
                60,
                float(holding_days),
                float(_MAX_HOLDING_DAYS),
            )
        if instrument_id not in selected:
            reasons.append(
                {
                    "rule_id": "CANDIDATE_ROTATION",
                    "priority": 90,
                    "observed": "not selected",
                    "threshold": "remain in selected top candidates",
                }
            )
        if not reasons:
            continue
        reasons.sort(key=lambda item: (int(item["priority"]), str(item["rule_id"])))
        available = holding.available_quantity - holding.available_quantity % 100
        signals[instrument_id] = {
            "instrument_id": instrument_id,
            "action": Side.SELL.value,
            "status": "READY" if available >= 100 else "BLOCKED_T_PLUS_ONE",
            "quantity": available,
            "primary_rule": reasons[0]["rule_id"],
            "reasons": reasons,
        }
    return signals


def _append_moving_average_exits(
    reasons: list[dict[str, Any]], series: list[DailyBar]
) -> None:
    for window, priority in ((5, 40), (10, 41)):
        if len(series) < window + 1:
            continue
        current_close = float(series[-1].close)
        previous_close = float(series[-2].close)
        current_ma = statistics.fmean(float(item.close) for item in series[-window:])
        previous_ma = statistics.fmean(float(item.close) for item in series[-window - 1 : -1])
        if current_close < current_ma and previous_close >= previous_ma:
            reasons.append(
                {
                    "rule_id": f"CROSS_BELOW_MA{window}",
                    "priority": priority,
                    "observed": {
                        "close": round(current_close, 6),
                        "moving_average": round(current_ma, 6),
                    },
                    "threshold": f"close crosses below MA{window}",
                }
            )


def _add_exit_reason(
    reasons: list[dict[str, Any]],
    rule_id: str,
    priority: int,
    observed: float,
    threshold: float,
) -> None:
    reasons.append(
        {
            "rule_id": rule_id,
            "priority": priority,
            "observed": round(observed, 8),
            "threshold": threshold,
        }
    )


def _paper_segment(instrument_id: str) -> str:
    try:
        _country, exchange, symbol = instrument_id.split(".", maxsplit=2)
    except ValueError:
        return "BOARD.UNKNOWN"
    if exchange == "SH" and symbol.startswith(("688", "689")):
        return "BOARD.STAR"
    if exchange == "SZ" and symbol.startswith(("300", "301")):
        return "BOARD.CHINEXT"
    if exchange == "SH":
        return "BOARD.SH_MAIN"
    if exchange == "SZ":
        return "BOARD.SZ_MAIN"
    return f"BOARD.{exchange}"


def _name_for(instrument_id: str, names: dict[str, str]) -> str:
    return names.get(instrument_id, instrument_id)


def _named_rows(rows: object, names: dict[str, str]) -> object:
    if not isinstance(rows, list):
        return rows
    return [
        {
            **item,
            "name": _name_for(str(item["instrument_id"]), names),
        }
        if isinstance(item, dict) and "instrument_id" in item
        else item
        for item in rows
    ]


def _named_execution(execution: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    return {
        **execution,
        "fills": _named_rows(execution["fills"], names),
    }


def _named_draft(draft: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    batch = draft["batch"]
    return {
        **draft,
        "candidates": [
            {"instrument_id": instrument_id, "name": _name_for(instrument_id, names)}
            for instrument_id in draft["candidate_ids"]
        ],
        "excluded_buy_candidates": _named_rows(draft["excluded_buy_candidates"], names),
        "exit_signals": _named_rows(draft["exit_signals"], names),
        "batch": {
            **batch,
            "drafts": _named_rows(batch["drafts"], names),
        },
    }


def _named_account(account: AccountSnapshot, names: dict[str, str]) -> dict[str, Any]:
    payload = json.loads(account.to_json())
    if not isinstance(payload, dict):
        raise ValueError("account payload must be an object")
    payload["holdings"] = _named_rows(payload["holdings"], names)
    return payload


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


def _close_report(
    beginning: AccountSnapshot,
    ending: AccountSnapshot,
    current: tuple[DailyBar, ...],
    names: dict[str, str],
) -> dict[str, Any]:
    close_prices = {item.instrument_id: float(item.close) for item in current}
    beginning_equity = beginning.total_equity
    ending_equity = ending.total_equity
    daily_pnl = ending_equity - beginning_equity
    holdings = []
    for item in sorted(ending.holdings, key=lambda holding: names.get(holding.instrument_id, "")):
        close_price = close_prices.get(item.instrument_id, item.last_price)
        market_value = item.quantity * close_price
        pnl = market_value - item.quantity * item.average_cost
        holdings.append(
            {
                "instrument_id": item.instrument_id,
                "name": _name_for(item.instrument_id, names),
                "quantity": item.quantity,
                "average_cost": round(item.average_cost, 6),
                "close": round(close_price, 6),
                "market_value": round(market_value, 6),
                "unrealized_pnl": round(pnl, 6),
                "unrealized_return": round(pnl / (item.quantity * item.average_cost), 8)
                if item.quantity and item.average_cost
                else 0.0,
            }
        )
    return {
        "valuation_time": "close",
        "beginning_equity": round(beginning_equity, 6),
        "ending_equity": round(ending_equity, 6),
        "daily_pnl": round(daily_pnl, 6),
        "daily_return": round(daily_pnl / beginning_equity, 8) if beginning_equity else 0.0,
        "cash": round(ending.total_cash, 6),
        "holdings_value": round(ending.total_market_value, 6),
        "holdings": holdings,
    }


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
    display_keys = {"name"}
    drafts = tuple(
        OrderDraft(
            **{
                **{key: value for key, value in item.items() if key not in display_keys},
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
