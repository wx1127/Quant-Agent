from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from quant_agent.backtest.contracts import AssetType
from quant_agent.data.domain import DailyBar
from quant_agent.paper_validation.daily import (
    PaperDailyEngine,
    _buy_exclusion_reason,
    _evaluate_exit_rules,
)
from quant_agent.portfolio.snapshots import AccountSnapshot, HoldingSnapshot

TZ = ZoneInfo("Asia/Shanghai")


def _bars(start: date, count: int) -> tuple[DailyBar, ...]:
    rows = []
    for index in range(count):
        day = start + timedelta(days=index)
        for number, growth in (("CN.SH.600001", 0.02), ("CN.SZ.000001", 0.005)):
            close = Decimal("10") * Decimal(str((1 + growth) ** index))
            rows.append(
                DailyBar(
                    instrument_id=number,
                    trade_date=day,
                    open=close,
                    high=close * Decimal("1.01"),
                    low=close * Decimal("0.99"),
                    close=close,
                    volume=Decimal("1000000"),
                    turnover=close * Decimal("1000000"),
                    source="fixture",
                    available_at=datetime(day.year, day.month, day.day, 16, tzinfo=TZ),
                    version=day.isoformat(),
                )
            )
    return tuple(rows)


def _bars_with_excluded_buys(start: date, count: int) -> tuple[DailyBar, ...]:
    rows = []
    definitions = (
        ("CN.SH.600009", 0.11),
        ("CN.SZ.300001", 0.05),
        ("CN.SH.688001", 0.048),
        ("CN.BJ.920001", 0.047),
        ("CN.HK.00700", 0.045),
        ("CN.SH.600001", 0.03),
        ("CN.SZ.000001", 0.025),
    )
    for index in range(count):
        day = start + timedelta(days=index)
        for number, growth in definitions:
            close = Decimal("10") * Decimal(str((1 + growth) ** index))
            rows.append(
                DailyBar(
                    instrument_id=number,
                    trade_date=day,
                    open=close,
                    high=close * Decimal("1.01"),
                    low=close * Decimal("0.99"),
                    close=close,
                    volume=Decimal("1000000"),
                    turnover=close * Decimal("1000000"),
                    source="fixture",
                    available_at=datetime(day.year, day.month, day.day, 16, tzinfo=TZ),
                    version=day.isoformat(),
                )
            )
    return tuple(rows)


def test_daily_paper_uses_close_signal_only_on_next_day() -> None:
    start = date(2026, 7, 1)
    all_bars = _bars(start, 26)
    first_day = start + timedelta(days=24)
    first_observed = datetime(2026, 7, 25, 16, 30, tzinfo=TZ)
    account = AccountSnapshot(
        "initial",
        "p8-paper-1",
        first_observed - timedelta(days=1),
        1_000_000,
        0,
        (),
        "paper",
        "v1",
    )
    engine = PaperDailyEngine()
    first = engine.run(
        trading_date=first_day,
        next_trading_date=first_day + timedelta(days=1),
        observed_at=first_observed,
        bars=tuple(item for item in all_bars if item.trade_date <= first_day),
        account=account,
        pending=None,
        instrument_names={"CN.SH.600001": "强势股份", "CN.SZ.000001": "稳健银行"},
    )

    assert first["mode"] == "PAPER"
    assert first["execution"]["fills"] == []
    assert first["next_day_order_draft"]["execute_on"] == "2026-07-26"
    assert first["next_day_order_draft"]["batch"]["drafts"]
    assert first["next_day_order_draft"]["batch"]["drafts"][0]["name"] == "强势股份"

    second_day = first_day + timedelta(days=1)
    second = engine.run(
        trading_date=second_day,
        next_trading_date=second_day + timedelta(days=1),
        observed_at=datetime(2026, 7, 26, 16, 30, tzinfo=TZ),
        bars=all_bars,
        account=AccountSnapshot.from_json(
            __import__("json").dumps(first["account"], ensure_ascii=False)
        ),
        pending=first["next_day_order_draft"],
        instrument_names={"CN.SH.600001": "强势股份", "CN.SZ.000001": "稳健银行"},
    )

    assert second["execution"]["signal_date"] == "2026-07-25"
    assert second["execution"]["fills"]
    assert second["execution"]["fills"][0]["filled_at"] == datetime(2026, 7, 26, 9, 32, tzinfo=TZ)
    assert second["execution"]["fills"][0]["name"] == "强势股份"
    assert second["execution"]["fees"] > 0
    assert second["reconciliation"]["status"] == "RECONCILED"
    assert second["close_report"]["ending_equity"] == second["reconciliation"]["total_equity"]
    assert second["close_report"]["holdings"][0]["name"] == "强势股份"
    assert second["close_report"]["daily_pnl"] != -second["execution"]["fees"]
    assert second["live_connection_attempted"] is False


def test_daily_paper_excludes_chinext_and_hong_kong_buys() -> None:
    start = date(2026, 7, 1)
    trading_day = start + timedelta(days=24)
    bars = _bars_with_excluded_buys(start, 25)
    account = AccountSnapshot(
        "initial",
        "p8-paper-1",
        datetime(2026, 7, 24, 16, 30, tzinfo=TZ),
        1_000_000,
        0,
        (),
        "paper",
        "v1",
    )

    record = PaperDailyEngine().run(
        trading_date=trading_day,
        next_trading_date=trading_day + timedelta(days=1),
        observed_at=datetime(2026, 7, 25, 16, 30, tzinfo=TZ),
        bars=bars,
        account=account,
        pending=None,
        instrument_names={
            "CN.SH.600009": "大幅波动样本",
            "CN.SZ.300001": "创业样本",
            "CN.SH.688001": "科创样本",
            "CN.BJ.920001": "北交样本",
            "CN.HK.00700": "港股样本",
            "CN.SH.600001": "主板甲",
            "CN.SZ.000001": "主板乙",
        },
    )

    draft = record["next_day_order_draft"]
    ranked_ids = {item["instrument_id"] for item in record["market"]["candidates"]}
    assert "CN.SZ.300001" not in ranked_ids
    assert "CN.SH.688001" not in ranked_ids
    assert "CN.BJ.920001" not in ranked_ids
    assert "CN.HK.00700" not in ranked_ids
    assert "CN.SH.600009" not in ranked_ids
    assert "CN.SZ.300001" not in draft["candidate_ids"]
    assert "CN.SH.688001" not in draft["candidate_ids"]
    assert "CN.BJ.920001" not in draft["candidate_ids"]
    assert "CN.HK.00700" not in draft["candidate_ids"]
    excluded_ids = {
        "CN.SH.600009",
        "CN.SZ.300001",
        "CN.SH.688001",
        "CN.BJ.920001",
        "CN.HK.00700",
    }
    assert all(
        item["instrument_id"] not in excluded_ids
        for item in draft["batch"]["drafts"]
    )
    excluded = {item["name"]: item["reason"] for item in draft["excluded_buy_candidates"]}
    assert excluded["创业样本"] == "daily price limit above 10% is excluded from PAPER buys"
    assert (
        excluded["大幅波动样本"]
        == "absolute daily return above 10% is excluded before candidate ranking"
    )


def test_buy_exclusion_reason_blocks_all_price_limit_above_10_percent_boards() -> None:
    reason = "daily price limit above 10% is excluded from PAPER buys"

    assert _buy_exclusion_reason("CN.SZ.300001") == reason
    assert _buy_exclusion_reason("CN.SZ.301001") == reason
    assert _buy_exclusion_reason("CN.SH.688001") == reason
    assert _buy_exclusion_reason("CN.SH.689001") == reason
    assert _buy_exclusion_reason("CN.BJ.920001") == reason
    assert _buy_exclusion_reason("CN.HK.00700") == "hong kong stock is excluded from PAPER buys"
    assert _buy_exclusion_reason("CN.SH.600001") is None
    assert _buy_exclusion_reason("CN.SZ.000001") is None
    assert (
        _buy_exclusion_reason("CN.SH.600001", daily_return=Decimal("0.1001"))
        == "absolute daily return above 10% is excluded before candidate ranking"
    )
    assert _buy_exclusion_reason("CN.SH.600001", daily_return=Decimal("0.10")) is None


def _exit_series(
    instrument_id: str,
    closes: list[str],
    *,
    final_open: str | None = None,
    final_volume: str = "1000000",
) -> list[DailyBar]:
    start = date(2026, 7, 1)
    rows = []
    for index, close_text in enumerate(closes):
        day = start + timedelta(days=index)
        close = Decimal(close_text)
        open_price = (
            Decimal(final_open)
            if index == len(closes) - 1 and final_open is not None
            else close
        )
        high = max(open_price, close) * Decimal("1.01")
        low = min(open_price, close) * Decimal("0.99")
        volume = Decimal(final_volume) if index == len(closes) - 1 else Decimal("1000000")
        rows.append(
            DailyBar(
                instrument_id=instrument_id,
                trade_date=day,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                turnover=close * volume,
                source="fixture",
                available_at=datetime(day.year, day.month, day.day, 16, tzinfo=TZ),
                version=day.isoformat(),
            )
        )
    return rows


def _holding(
    instrument_id: str,
    *,
    cost: float,
    available: int = 1000,
) -> HoldingSnapshot:
    return HoldingSnapshot(
        instrument_id,
        AssetType.STOCK,
        None,
        1000,
        available,
        1000 - available,
        cost,
        cost,
    )


def test_exit_layer_covers_loss_profit_trailing_ma_market_and_time_rules() -> None:
    risk_id = "CN.SH.600101"
    profit_id = "CN.SH.600102"
    trail_id = "CN.SH.600103"
    off_mainline_id = "CN.SZ.000101"
    histories = {
        risk_id: _exit_series(
            risk_id,
            ["10", "10", "10", "10", "10", "10", "10", "10", "10", "10", "9"],
            final_open="9.4",
            final_volume="2000000",
        ),
        profit_id: _exit_series(profit_id, ["9", "9.2", "9.4", "9.6", "10"]),
        trail_id: _exit_series(
            trail_id,
            ["10", "11", "12", "11.5", "10.8"],
            final_open="11.5",
        ),
        off_mainline_id: _exit_series(off_mainline_id, ["10", "10.1"]),
    }
    holdings = {
        risk_id: _holding(risk_id, cost=10),
        profit_id: _holding(profit_id, cost=8),
        trail_id: _holding(trail_id, cost=10),
        off_mainline_id: _holding(off_mainline_id, cost=10),
    }
    state = {
        risk_id: {"holding_days": 20, "peak_close": 10},
        profit_id: {"holding_days": 2, "peak_close": 10},
        trail_id: {"holding_days": 4, "peak_close": 12},
        off_mainline_id: {"holding_days": 2, "peak_close": 10.1},
    }

    signals = _evaluate_exit_rules(
        holdings,
        histories,
        {"BOARD.SH_MAIN"},
        set(holdings),
        state,
    )

    risk_rules = {item["rule_id"] for item in signals[risk_id]["reasons"]}
    assert {
        "STOP_LOSS",
        "LARGE_GAP_DOWN",
        "HIGH_VOLUME_SELL_OFF",
        "CROSS_BELOW_MA5",
        "CROSS_BELOW_MA10",
        "MAX_HOLDING_DAYS",
    } <= risk_rules
    assert signals[risk_id]["primary_rule"] == "STOP_LOSS"
    assert signals[profit_id]["primary_rule"] == "TAKE_PROFIT"
    assert signals[trail_id]["primary_rule"] == "TRAILING_STOP"
    assert signals[off_mainline_id]["primary_rule"] == "MAINLINE_EXIT"


def test_exit_signal_is_recorded_but_not_sellable_when_t_plus_one_frozen() -> None:
    instrument_id = "CN.SH.600201"
    signals = _evaluate_exit_rules(
        {instrument_id: _holding(instrument_id, cost=10, available=0)},
        {instrument_id: _exit_series(instrument_id, ["10", "9"])},
        {"BOARD.SH_MAIN"},
        {instrument_id},
        {instrument_id: {"holding_days": 1, "peak_close": 10}},
    )

    assert signals[instrument_id]["primary_rule"] == "STOP_LOSS"
    assert signals[instrument_id]["status"] == "BLOCKED_T_PLUS_ONE"
    assert signals[instrument_id]["quantity"] == 0
