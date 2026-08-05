from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from quant_agent.data.domain import DailyBar
from quant_agent.paper_validation.daily import PaperDailyEngine
from quant_agent.portfolio.snapshots import AccountSnapshot

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
