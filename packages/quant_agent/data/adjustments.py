"""Adjustment-factor and corporate-action services."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.data.domain import AdjustmentFactor, CorporateAction
from quant_agent.data.models import (
    AdjustmentFactorRow,
    CorporateActionRow,
    DailyBarRow,
)


class AdjustmentMethod(StrEnum):
    """Supported research price views."""

    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"


@dataclass(frozen=True, slots=True)
class AdjustedPricePoint:
    """Derived adjusted close that never overwrites the raw bar."""

    trade_date: date
    raw_close: Decimal
    factor: Decimal
    adjusted_close: Decimal


class AdjustmentService:
    """Persist factors/actions and derive point-in-time adjusted prices."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ingest_factors(self, factors: list[AdjustmentFactor]) -> tuple[int, int]:
        """Insert factors idempotently by instrument/date/source/version."""

        inserted = 0
        skipped = 0
        for factor in factors:
            existing = self._session.scalar(
                select(AdjustmentFactorRow.id).where(
                    AdjustmentFactorRow.instrument_id == factor.instrument_id,
                    AdjustmentFactorRow.trade_date == factor.trade_date,
                    AdjustmentFactorRow.source == factor.source,
                    AdjustmentFactorRow.version == factor.version,
                )
            )
            if existing is not None:
                skipped += 1
                continue
            self._session.add(
                AdjustmentFactorRow(
                    instrument_id=factor.instrument_id,
                    trade_date=factor.trade_date,
                    factor=factor.factor,
                    source=factor.source,
                    available_at=factor.available_at,
                    version=factor.version,
                )
            )
            inserted += 1
        self._session.flush()
        return inserted, skipped

    def ingest_actions(self, actions: list[CorporateAction]) -> tuple[int, int]:
        """Insert corporate actions without replacing earlier versions."""

        inserted = 0
        skipped = 0
        for action in actions:
            existing = self._session.scalar(
                select(CorporateActionRow.id).where(
                    CorporateActionRow.instrument_id == action.instrument_id,
                    CorporateActionRow.action_type == action.action_type,
                    CorporateActionRow.ex_date == action.ex_date,
                    CorporateActionRow.source == action.source,
                    CorporateActionRow.version == action.version,
                )
            )
            if existing is not None:
                skipped += 1
                continue
            self._session.add(
                CorporateActionRow(
                    instrument_id=action.instrument_id,
                    action_type=action.action_type,
                    ex_date=action.ex_date,
                    announced_at=action.announced_at,
                    available_at=action.available_at,
                    cash_amount=action.cash_amount,
                    share_ratio=action.share_ratio,
                    source=action.source,
                    version=action.version,
                )
            )
            inserted += 1
        self._session.flush()
        return inserted, skipped

    def adjusted_closes(
        self,
        instrument_id: str,
        *,
        start: date,
        end: date,
        as_of: datetime,
        method: AdjustmentMethod,
    ) -> list[AdjustedPricePoint]:
        """Build an adjusted view using only factors available at ``as_of``."""

        bar_rows = list(
            self._session.scalars(
                select(DailyBarRow)
                .where(
                    DailyBarRow.instrument_id == instrument_id,
                    DailyBarRow.trade_date >= start,
                    DailyBarRow.trade_date <= end,
                    DailyBarRow.available_at <= as_of,
                )
                .order_by(DailyBarRow.trade_date)
            )
        )
        factor_rows = list(
            self._session.scalars(
                select(AdjustmentFactorRow)
                .where(
                    AdjustmentFactorRow.instrument_id == instrument_id,
                    AdjustmentFactorRow.trade_date >= start,
                    AdjustmentFactorRow.trade_date <= end,
                    AdjustmentFactorRow.available_at <= as_of,
                )
                .order_by(AdjustmentFactorRow.trade_date)
            )
        )
        factors = {row.trade_date: Decimal(row.factor) for row in factor_rows}
        if not bar_rows:
            return []
        missing = [row.trade_date for row in bar_rows if row.trade_date not in factors]
        if missing:
            raise ValueError(f"missing adjustment factors for dates: {missing}")
        anchor = factors[bar_rows[-1].trade_date]
        result = []
        for row in bar_rows:
            raw_close = Decimal(row.close)
            factor = factors[row.trade_date]
            adjusted = (
                raw_close * factor / anchor
                if method is AdjustmentMethod.FORWARD
                else raw_close * factor
            )
            result.append(
                AdjustedPricePoint(
                    trade_date=row.trade_date,
                    raw_close=raw_close,
                    factor=factor,
                    adjusted_close=adjusted,
                )
            )
        return result

    def factor_jump_dates(
        self,
        instrument_id: str,
        *,
        relative_threshold: Decimal = Decimal("0.5"),
    ) -> list[date]:
        """Return dates with unusually large relative factor changes."""

        rows = list(
            self._session.scalars(
                select(AdjustmentFactorRow)
                .where(AdjustmentFactorRow.instrument_id == instrument_id)
                .order_by(AdjustmentFactorRow.trade_date)
            )
        )
        jumps: list[date] = []
        previous: Decimal | None = None
        for row in rows:
            current = Decimal(row.factor)
            if previous is not None and abs(current / previous - Decimal(1)) > relative_threshold:
                jumps.append(row.trade_date)
            previous = current
        return jumps
