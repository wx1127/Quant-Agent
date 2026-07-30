"""Tests for point-in-time financial revisions."""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from quant_agent.data.domain import FundamentalPoint
from quant_agent.data.fundamentals import FundamentalService

TZ = ZoneInfo("Asia/Shanghai")


def test_fundamental_query_never_uses_future_revision(
    seeded_session: Session,
) -> None:
    service = FundamentalService(seeded_session)
    first = FundamentalPoint(
        instrument_id="CN.SZ.000001",
        report_period=date(2026, 6, 30),
        metric_name="revenue",
        metric_value=Decimal("100"),
        announced_at=datetime(2026, 7, 20, 18, tzinfo=TZ),
        available_at=datetime(2026, 7, 21, 9, tzinfo=TZ),
        provider_revision="r1",
        source="test",
    )
    revised = first.model_copy(
        update={
            "metric_value": Decimal("110"),
            "announced_at": datetime(2026, 8, 1, 18, tzinfo=TZ),
            "available_at": datetime(2026, 8, 2, 9, tzinfo=TZ),
            "provider_revision": "r2",
        }
    )

    assert service.ingest([first, revised]) == (2, 0)
    assert service.ingest([first]) == (0, 1)

    before = service.latest_as_of(
        "CN.SZ.000001",
        decision_time=datetime(2026, 7, 30, 15, tzinfo=TZ),
    )
    after = service.latest_as_of(
        "CN.SZ.000001",
        decision_time=datetime(2026, 8, 3, 15, tzinfo=TZ),
        report_period=date(2026, 6, 30),
    )

    assert before[(date(2026, 6, 30), "revenue")].metric_value == Decimal("100")
    assert after[(date(2026, 6, 30), "revenue")].metric_value == Decimal("110")
    assert len(service.revisions("CN.SZ.000001", date(2026, 6, 30), "revenue")) == 2
