"""Point-in-time fundamental ingestion and queries."""

from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.data.domain import FundamentalPoint
from quant_agent.data.models import FundamentalPointRow


class FundamentalService:
    """Retain every provider revision and expose only data known at a decision time."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ingest(self, points: list[FundamentalPoint]) -> tuple[int, int]:
        """Insert revisions idempotently without replacing history."""

        inserted = 0
        skipped = 0
        for point in points:
            existing = self._session.scalar(
                select(FundamentalPointRow.id).where(
                    FundamentalPointRow.instrument_id == point.instrument_id,
                    FundamentalPointRow.report_period == point.report_period,
                    FundamentalPointRow.metric_name == point.metric_name,
                    FundamentalPointRow.source == point.source,
                    FundamentalPointRow.provider_revision == point.provider_revision,
                )
            )
            if existing is not None:
                skipped += 1
                continue
            self._session.add(
                FundamentalPointRow(
                    instrument_id=point.instrument_id,
                    report_period=point.report_period,
                    metric_name=point.metric_name,
                    metric_value=point.metric_value,
                    announced_at=point.announced_at,
                    available_at=point.available_at,
                    provider_revision=point.provider_revision,
                    source=point.source,
                )
            )
            inserted += 1
        self._session.flush()
        return inserted, skipped

    def latest_as_of(
        self,
        instrument_id: str,
        *,
        decision_time: datetime,
        report_period: date | None = None,
    ) -> dict[tuple[date, str], FundamentalPointRow]:
        """Return the latest known revision per report period and metric."""

        statement = (
            select(FundamentalPointRow)
            .where(
                FundamentalPointRow.instrument_id == instrument_id,
                FundamentalPointRow.available_at <= decision_time,
            )
            .order_by(
                FundamentalPointRow.report_period,
                FundamentalPointRow.metric_name,
                FundamentalPointRow.available_at.desc(),
                FundamentalPointRow.id.desc(),
            )
        )
        if report_period is not None:
            statement = statement.where(FundamentalPointRow.report_period == report_period)
        result: dict[tuple[date, str], FundamentalPointRow] = {}
        for row in self._session.scalars(statement):
            key = (row.report_period, row.metric_name)
            result.setdefault(key, row)
        return result

    def revisions(
        self,
        instrument_id: str,
        report_period: date,
        metric_name: str,
    ) -> list[FundamentalPointRow]:
        """Return all retained revisions in availability order."""

        statement = (
            select(FundamentalPointRow)
            .where(
                FundamentalPointRow.instrument_id == instrument_id,
                FundamentalPointRow.report_period == report_period,
                FundamentalPointRow.metric_name == metric_name,
            )
            .order_by(FundamentalPointRow.available_at)
        )
        return list(self._session.scalars(statement))
