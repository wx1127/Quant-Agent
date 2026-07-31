from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import DailyBar
from quant_agent.shadow.historical import HistoricalShadowEngine
from quant_agent.shadow.io import load_shadow_config
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.models import ShadowRunMode
from quant_agent.shadow.session import ShadowRunEvaluator

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("Asia/Shanghai")


def _bars() -> tuple[DailyBar, ...]:
    start = date(2026, 5, 25)
    days = [start + timedelta(days=index) for index in range(40)]
    trading_days = [item for item in days if item.weekday() < 5]
    instruments = ("CN.SH.600001", "CN.SH.688001", "CN.SZ.000001", "CN.SZ.300001")
    records: list[DailyBar] = []
    for day_index, trading_day in enumerate(trading_days):
        for item_index, instrument_id in enumerate(instruments):
            price = Decimal("10") + Decimal(day_index) / 10 + Decimal(item_index) / 20
            records.append(
                DailyBar(
                    instrument_id=instrument_id,
                    trade_date=trading_day,
                    open=price,
                    high=price + Decimal("0.2"),
                    low=price - Decimal("0.2"),
                    close=price + Decimal("0.1"),
                    volume=Decimal("100000"),
                    turnover=Decimal("1000000") * (item_index + 1),
                    source="tushare",
                    available_at=datetime.combine(
                        trading_day,
                        datetime.min.time().replace(hour=16),
                        tzinfo=TZ,
                    ),
                    version=trading_day.isoformat(),
                )
            )
    return tuple(records)


def test_historical_engine_uses_virtual_clock_and_excludes_future_rows() -> None:
    bars = _bars()
    target = sorted({item.trade_date for item in bars})[-2]
    result = HistoricalShadowEngine(minimum_daily_instruments=4).run_day(target, bars)
    evidence = result.evidence
    assert evidence.run_mode is ShadowRunMode.HISTORICAL_POINT_IN_TIME
    assert evidence.observed_at == datetime.combine(
        target,
        datetime.min.time().replace(hour=16, minute=5),
        tzinfo=TZ,
    )
    assert evidence.market_data_as_of <= evidence.observed_at
    assert evidence.future_data_violations == 0
    assert evidence.executable_orders_emitted == 0
    assert evidence.data_complete and evidence.pipeline_succeeded
    assert len(evidence.mainline_ids) == 2
    assert result.details["excluded_future_rows"] == 4


def test_historical_session_rejects_realtime_evidence(tmp_path: Path) -> None:
    config = load_shadow_config(ROOT / "configs" / "shadow" / "shadow_20260701_historical_v1.toml")
    assert config.run_mode is ShadowRunMode.HISTORICAL_POINT_IN_TIME
    bars = _bars()
    days = sorted({item.trade_date for item in bars})[-20:]
    config = replace(config, started_on=days[0])
    ledger = ShadowEvidenceLedger(tmp_path / "historical.jsonl")
    engine = HistoricalShadowEngine(minimum_daily_instruments=4)
    for day in days:
        ledger.append(engine.run_day(day, bars).evidence)
    acceptance = ShadowRunEvaluator().evaluate(
        config,
        ledger.read_all(),
        trading_calendar=tuple(days),
    )
    assert acceptance.consecutive_trading_days == 20
    assert acceptance.future_data_violations == 0

    realtime_config = load_shadow_config(ROOT / "configs" / "shadow" / "shadow_20260803_v1.toml")
    realtime_config = replace(realtime_config, started_on=days[0])
    with pytest.raises(ValueError, match="run mode"):
        ShadowRunEvaluator().evaluate(
            realtime_config,
            ledger.read_all(),
            trading_calendar=tuple(days),
        )
