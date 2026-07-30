"""Tests for data-quality rules and fail-closed outcomes."""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.data.models import DataQualityResultRow
from quant_agent.data.quality import (
    BarQualityRecord,
    FactorQualityRecord,
    QualityContext,
    QualityEngine,
    QualitySeverity,
)

TZ = ZoneInfo("Asia/Shanghai")


def _bar(
    instrument_id: str = "CN.SZ.000001",
    trade_date: date = date(2026, 7, 30),
) -> BarQualityRecord:
    return BarQualityRecord(
        instrument_id=instrument_id,
        trade_date=trade_date,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("100"),
        turnover=Decimal("1000"),
    )


def test_quality_engine_passes_valid_complete_dataset(
    db_session: Session,
) -> None:
    report = QualityEngine().run(
        "v1",
        QualityContext(
            bars=(_bar(),),
            factors=(
                FactorQualityRecord(
                    "CN.SZ.000001",
                    date(2026, 7, 30),
                    Decimal("1"),
                ),
            ),
            expected_instrument_ids=("CN.SZ.000001",),
            expected_open_dates=(date(2026, 7, 30),),
        ),
        session=db_session,
        observed_at=datetime(2026, 7, 30, 17, tzinfo=TZ),
    )

    assert report.qualified is True
    persisted = list(db_session.scalars(select(DataQualityResultRow)))
    assert len(persisted) == 4
    assert all(item.passed for item in persisted)


def test_quality_engine_detects_invalid_duplicate_missing_and_factor_jump() -> None:
    invalid = BarQualityRecord(
        instrument_id="CN.SZ.000001",
        trade_date=date(2026, 7, 30),
        open=Decimal("10"),
        high=Decimal("9"),
        low=Decimal("11"),
        close=Decimal("10"),
        volume=Decimal("-1"),
        turnover=Decimal("1"),
    )
    report = QualityEngine().run(
        "bad",
        QualityContext(
            bars=(invalid, invalid),
            factors=(
                FactorQualityRecord(
                    "CN.SZ.000001",
                    date(2026, 7, 29),
                    Decimal("1"),
                ),
                FactorQualityRecord(
                    "CN.SZ.000001",
                    date(2026, 7, 30),
                    Decimal("2"),
                ),
            ),
            expected_instrument_ids=("CN.SZ.000001", "CN.SZ.000002"),
            expected_open_dates=(date(2026, 7, 30),),
        ),
    )

    assert report.qualified is False
    assert {issue.rule_id for issue in report.issues} == {
        "OHLCV_VALIDITY",
        "DUPLICATE_DAILY_BAR",
        "CALENDAR_COMPLETENESS",
        "ADJUSTMENT_FACTOR_CONTINUITY",
    }
    assert any(issue.severity is QualitySeverity.WARNING for issue in report.issues)
