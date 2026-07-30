"""Tests for historical industry classification."""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from quant_agent.data.domain import Industry, IndustryMembership
from quant_agent.data.industry import IndustryService


def _industry(industry_id: str, code: str, name: str) -> Industry:
    return Industry(
        industry_id=industry_id,
        classification="SW",
        code=code,
        name=name,
        level=1,
        version="v1",
    )


def test_historical_membership_does_not_backfill_current_classification(
    seeded_session: Session,
) -> None:
    service = IndustryService(seeded_session)
    bank = _industry("SW:L1:BANK:v1", "BANK", "银行")
    finance = _industry("SW:L1:FIN:v1", "FIN", "非银金融")
    assert service.upsert_industries([bank, finance]) == (2, 0)
    assert service.upsert_industries([bank]) == (0, 1)

    first = IndustryMembership(
        instrument_id="CN.SZ.000001",
        industry_id=bank.industry_id,
        effective_from=date(2000, 1, 1),
        effective_to=date(2020, 12, 31),
        source="test",
        version="v1",
    )
    second = IndustryMembership(
        instrument_id="CN.SZ.000001",
        industry_id=finance.industry_id,
        effective_from=date(2021, 1, 1),
        source="test",
        version="v1",
    )
    assert service.add_memberships([first, second]) == (2, 0)
    assert service.add_memberships([first]) == (0, 1)

    assert service.constituents(bank.industry_id, date(2010, 1, 1)) == ["CN.SZ.000001"]
    assert service.constituents(bank.industry_id, date(2026, 1, 1)) == []
    current = service.classification_for(
        "CN.SZ.000001",
        "SW",
        1,
        date(2026, 1, 1),
    )
    assert current is not None
    assert current.industry_id == finance.industry_id


def test_same_classification_membership_overlap_is_rejected(
    seeded_session: Session,
) -> None:
    service = IndustryService(seeded_session)
    bank = _industry("SW:L1:BANK:v1", "BANK", "银行")
    finance = _industry("SW:L1:FIN:v1", "FIN", "非银金融")
    service.upsert_industries([bank, finance])
    service.add_memberships(
        [
            IndustryMembership(
                instrument_id="CN.SZ.000001",
                industry_id=bank.industry_id,
                effective_from=date(2020, 1, 1),
                source="test",
                version="v1",
            )
        ]
    )

    with pytest.raises(ValueError, match="overlaps"):
        service.add_memberships(
            [
                IndustryMembership(
                    instrument_id="CN.SZ.000001",
                    industry_id=finance.industry_id,
                    effective_from=date(2026, 1, 1),
                    source="test",
                    version="v1",
                )
            ]
        )
