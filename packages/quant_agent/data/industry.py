"""Versioned industry classification and historical membership services."""

from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from quant_agent.data.domain import Industry, IndustryMembership
from quant_agent.data.models import (
    IndustryMembershipRow,
    IndustryRow,
    InstrumentRow,
)


class IndustryService:
    """Maintain non-overlapping historical memberships."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_industries(self, industries: list[Industry]) -> tuple[int, int]:
        """Insert immutable industry versions and skip exact IDs."""

        inserted = 0
        skipped = 0
        for industry in industries:
            existing = self._session.get(IndustryRow, industry.industry_id)
            if existing is not None:
                if (
                    existing.classification != industry.classification
                    or existing.code != industry.code
                    or existing.version != industry.version
                ):
                    raise ValueError(
                        f"industry_id {industry.industry_id} conflicts with existing node"
                    )
                skipped += 1
                continue
            self._session.add(
                IndustryRow(
                    industry_id=industry.industry_id,
                    classification=industry.classification,
                    code=industry.code,
                    name=industry.name,
                    level=industry.level,
                    parent_id=industry.parent_id,
                    version=industry.version,
                )
            )
            inserted += 1
        self._session.flush()
        return inserted, skipped

    def add_memberships(
        self,
        memberships: list[IndustryMembership],
    ) -> tuple[int, int]:
        """Insert memberships while rejecting same-classification overlaps."""

        inserted = 0
        skipped = 0
        for membership in memberships:
            industry = self._session.get(IndustryRow, membership.industry_id)
            if industry is None:
                raise ValueError(f"unknown industry_id: {membership.industry_id}")
            if self._session.get(InstrumentRow, membership.instrument_id) is None:
                raise ValueError(f"unknown instrument_id: {membership.instrument_id}")
            exact = self._session.scalar(
                select(IndustryMembershipRow.id).where(
                    IndustryMembershipRow.instrument_id == membership.instrument_id,
                    IndustryMembershipRow.industry_id == membership.industry_id,
                    IndustryMembershipRow.effective_from == membership.effective_from,
                    IndustryMembershipRow.version == membership.version,
                )
            )
            if exact is not None:
                skipped += 1
                continue
            overlap = self._session.scalar(
                select(IndustryMembershipRow.id)
                .join(
                    IndustryRow,
                    IndustryRow.industry_id == IndustryMembershipRow.industry_id,
                )
                .where(
                    IndustryMembershipRow.instrument_id == membership.instrument_id,
                    IndustryRow.classification == industry.classification,
                    IndustryRow.level == industry.level,
                    IndustryMembershipRow.effective_from <= (membership.effective_to or date.max),
                    or_(
                        IndustryMembershipRow.effective_to.is_(None),
                        IndustryMembershipRow.effective_to >= membership.effective_from,
                    ),
                )
            )
            if overlap is not None:
                raise ValueError(
                    "industry membership overlaps an existing range for the same "
                    "classification and level"
                )
            self._session.add(
                IndustryMembershipRow(
                    instrument_id=membership.instrument_id,
                    industry_id=membership.industry_id,
                    effective_from=membership.effective_from,
                    effective_to=membership.effective_to,
                    source=membership.source,
                    version=membership.version,
                )
            )
            inserted += 1
        self._session.flush()
        return inserted, skipped

    def constituents(self, industry_id: str, as_of: date) -> list[str]:
        """Return stable instrument IDs in an industry at a date."""

        statement = (
            select(IndustryMembershipRow.instrument_id)
            .where(
                IndustryMembershipRow.industry_id == industry_id,
                IndustryMembershipRow.effective_from <= as_of,
                or_(
                    IndustryMembershipRow.effective_to.is_(None),
                    IndustryMembershipRow.effective_to >= as_of,
                ),
            )
            .order_by(IndustryMembershipRow.instrument_id)
        )
        return list(self._session.scalars(statement))

    def classification_for(
        self,
        instrument_id: str,
        classification: str,
        level: int,
        as_of: date,
    ) -> IndustryRow | None:
        """Return an instrument's industry using historical membership."""

        statement = (
            select(IndustryRow)
            .join(
                IndustryMembershipRow,
                IndustryMembershipRow.industry_id == IndustryRow.industry_id,
            )
            .where(
                IndustryMembershipRow.instrument_id == instrument_id,
                IndustryRow.classification == classification,
                IndustryRow.level == level,
                IndustryMembershipRow.effective_from <= as_of,
                or_(
                    IndustryMembershipRow.effective_to.is_(None),
                    IndustryMembershipRow.effective_to >= as_of,
                ),
            )
        )
        return self._session.scalar(statement)
