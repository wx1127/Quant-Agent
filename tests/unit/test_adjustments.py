"""Tests for point-in-time price adjustment and corporate actions."""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from quant_agent.data.adjustments import AdjustmentMethod, AdjustmentService
from quant_agent.data.domain import AdjustmentFactor, CorporateAction, DailyBar
from quant_agent.data.ingestion import DailyBarIngestionService
from quant_agent.data.providers.base import ProviderBatch

TZ = ZoneInfo("Asia/Shanghai")


def _bar(trade_date: date, close: str) -> DailyBar:
    value = Decimal(close)
    return DailyBar(
        instrument_id="CN.SZ.000001",
        trade_date=trade_date,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Decimal("100"),
        turnover=Decimal("1000"),
        source="test",
        available_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=TZ),
        version="v1",
    )


def test_adjusted_views_preserve_raw_price_and_respect_as_of(
    seeded_session: Session,
) -> None:
    ingestion = DailyBarIngestionService(seeded_session)
    bars = (_bar(date(2026, 7, 29), "10"), _bar(date(2026, 7, 30), "6"))
    ingestion.ingest(
        ProviderBatch(
            provider="test",
            endpoint="daily",
            request_params={"range": "two-days"},
            raw_payload={"count": 2},
            available_at=datetime(2026, 7, 30, 16, tzinfo=TZ),
            records=bars,
        )
    )
    service = AdjustmentService(seeded_session)
    factors = [
        AdjustmentFactor(
            instrument_id="CN.SZ.000001",
            trade_date=date(2026, 7, 29),
            factor=Decimal("1"),
            source="test",
            available_at=datetime(2026, 7, 29, 16, tzinfo=TZ),
            version="v1",
        ),
        AdjustmentFactor(
            instrument_id="CN.SZ.000001",
            trade_date=date(2026, 7, 30),
            factor=Decimal("2"),
            source="test",
            available_at=datetime(2026, 7, 30, 16, tzinfo=TZ),
            version="v1",
        ),
    ]
    assert service.ingest_factors(factors) == (2, 0)
    assert service.ingest_factors(factors) == (0, 2)

    forward = service.adjusted_closes(
        "CN.SZ.000001",
        start=date(2026, 7, 29),
        end=date(2026, 7, 30),
        as_of=datetime(2026, 7, 30, 17, tzinfo=TZ),
        method=AdjustmentMethod.FORWARD,
    )
    backward = service.adjusted_closes(
        "CN.SZ.000001",
        start=date(2026, 7, 29),
        end=date(2026, 7, 30),
        as_of=datetime(2026, 7, 30, 17, tzinfo=TZ),
        method=AdjustmentMethod.BACKWARD,
    )

    assert [item.raw_close for item in forward] == [Decimal("10"), Decimal("6")]
    assert [item.adjusted_close for item in forward] == [Decimal("5"), Decimal("6")]
    assert [item.adjusted_close for item in backward] == [
        Decimal("10"),
        Decimal("12"),
    ]
    assert service.factor_jump_dates(
        "CN.SZ.000001",
        relative_threshold=Decimal("0.5"),
    ) == [date(2026, 7, 30)]

    with pytest.raises(ValueError, match="missing adjustment"):
        service.adjusted_closes(
            "CN.SZ.000001",
            start=date(2026, 7, 29),
            end=date(2026, 7, 30),
            as_of=datetime(2026, 7, 30, 10, tzinfo=TZ),
            method=AdjustmentMethod.FORWARD,
        )


def test_corporate_action_versions_are_retained(seeded_session: Session) -> None:
    service = AdjustmentService(seeded_session)
    action = CorporateAction(
        instrument_id="CN.SZ.000001",
        action_type="CASH_DIVIDEND",
        ex_date=date(2026, 8, 5),
        announced_at=datetime(2026, 7, 29, 16, tzinfo=TZ),
        available_at=datetime(2026, 7, 29, 16, tzinfo=TZ),
        cash_amount=Decimal("0.1"),
        source="test",
        version="v1",
    )

    assert service.ingest_actions([action]) == (1, 0)
    assert service.ingest_actions([action]) == (0, 1)
    assert service.ingest_actions([action.model_copy(update={"version": "v2"})]) == (
        1,
        0,
    )
