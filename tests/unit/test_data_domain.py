"""Validation tests for provider-neutral P1 records."""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_agent.data.domain import (
    AdjustmentFactor,
    CorporateAction,
    DailyBar,
    FundamentalPoint,
    IndexConstituentWeight,
    IndustryMembership,
    Instrument,
    InstrumentType,
)

TZ = ZoneInfo("Asia/Shanghai")


def test_instrument_rejects_invalid_lifecycle() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        Instrument(
            instrument_id="CN.SZ.000001",
            symbol="000001",
            exchange="SZ",
            instrument_type=InstrumentType.STOCK,
            name="test",
            listed_on=date(2020, 1, 1),
            delisted_on=date(2019, 1, 1),
            source="test",
            version="v1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("high", Decimal("9")),
        ("low", Decimal("11")),
        ("volume", Decimal("-1")),
    ],
)
def test_daily_bar_rejects_invalid_values(field: str, value: Decimal) -> None:
    payload = {
        "instrument_id": "CN.SZ.000001",
        "trade_date": date(2026, 7, 30),
        "open": Decimal("10"),
        "high": Decimal("11"),
        "low": Decimal("9"),
        "close": Decimal("10.5"),
        "volume": Decimal("100"),
        "turnover": Decimal("1000"),
        "source": "test",
        "available_at": datetime(2026, 7, 30, 16, tzinfo=TZ),
        "version": "v1",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        DailyBar.model_validate(payload)


def test_adjustment_factor_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="positive"):
        AdjustmentFactor(
            instrument_id="CN.SZ.000001",
            trade_date=date(2026, 7, 30),
            factor=Decimal("0"),
            source="test",
            available_at=datetime(2026, 7, 30, 16, tzinfo=TZ),
            version="v1",
        )


def test_point_in_time_records_reject_impossible_availability() -> None:
    announced = datetime(2026, 7, 30, 16, tzinfo=TZ)
    earlier = datetime(2026, 7, 29, 16, tzinfo=TZ)

    with pytest.raises(ValidationError, match="available_at"):
        FundamentalPoint(
            instrument_id="CN.SZ.000001",
            report_period=date(2026, 6, 30),
            metric_name="revenue",
            metric_value=Decimal("1"),
            announced_at=announced,
            available_at=earlier,
            provider_revision="r1",
            source="test",
        )
    with pytest.raises(ValidationError, match="available_at"):
        CorporateAction(
            instrument_id="CN.SZ.000001",
            action_type="CASH_DIVIDEND",
            ex_date=date(2026, 8, 1),
            announced_at=announced,
            available_at=earlier,
            source="test",
            version="v1",
        )


def test_industry_membership_rejects_reverse_range() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        IndustryMembership(
            instrument_id="CN.SZ.000001",
            industry_id="SW:L1:BANK:v1",
            effective_from=date(2026, 7, 30),
            effective_to=date(2026, 7, 29),
            source="test",
            version="v1",
        )


def test_index_constituent_weight_validates_range_identity_and_time() -> None:
    payload = {
        "index_instrument_id": "CN.CSI.000300",
        "constituent_instrument_id": "CN.SH.600000",
        "trade_date": date(2026, 7, 31),
        "weight_percent": Decimal("1.25"),
        "available_at": datetime(2026, 7, 31, 18, tzinfo=TZ),
        "source": "test",
        "version": "v1",
    }
    valid = IndexConstituentWeight.model_validate(payload)
    assert valid.weight_percent == Decimal("1.25")

    with pytest.raises(ValidationError, match="between 0 and 100"):
        IndexConstituentWeight.model_validate(payload | {"weight_percent": Decimal("101")})

    with pytest.raises(ValidationError, match="own constituent"):
        IndexConstituentWeight.model_validate(
            payload | {"constituent_instrument_id": payload["index_instrument_id"]}
        )

    with pytest.raises(ValidationError, match="timezone information"):
        IndexConstituentWeight.model_validate(payload | {"available_at": datetime(2026, 7, 31, 18)})
