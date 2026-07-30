"""Trading calendar and point-in-time instrument master services."""

from datetime import date, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from quant_agent.data.domain import Instrument, TradingDay
from quant_agent.data.models import (
    InstrumentAliasRow,
    InstrumentRow,
    InstrumentStatusRow,
    TradingDayRow,
)


class TradingCalendarService:
    """Persist and query a market trading calendar."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, days: list[TradingDay]) -> tuple[int, int]:
        """Insert new dates and update changed flags without duplication."""

        inserted = 0
        updated = 0
        for day in days:
            row = self._session.get(TradingDayRow, (day.market, day.trade_date))
            if row is None:
                self._session.add(
                    TradingDayRow(
                        market=day.market,
                        trade_date=day.trade_date,
                        is_open=day.is_open,
                        source=day.source,
                        version=day.version,
                    )
                )
                inserted += 1
            elif (
                row.is_open != day.is_open or row.source != day.source or row.version != day.version
            ):
                row.is_open = day.is_open
                row.source = day.source
                row.version = day.version
                updated += 1
        self._session.flush()
        return inserted, updated

    def is_trading_day(self, market: str, value: date) -> bool:
        """Return whether the persisted calendar marks a date as open."""

        row = self._session.get(TradingDayRow, (market, value))
        return bool(row is not None and row.is_open)

    def open_days(self, market: str, start: date, end: date) -> list[date]:
        """Return ascending open dates in an inclusive interval."""

        statement = (
            select(TradingDayRow.trade_date)
            .where(
                TradingDayRow.market == market,
                TradingDayRow.trade_date >= start,
                TradingDayRow.trade_date <= end,
                TradingDayRow.is_open.is_(True),
            )
            .order_by(TradingDayRow.trade_date)
        )
        return list(self._session.scalars(statement))

    def previous_open_day(self, market: str, value: date) -> date | None:
        """Return the latest open day strictly before a date."""

        statement = (
            select(TradingDayRow.trade_date)
            .where(
                TradingDayRow.market == market,
                TradingDayRow.trade_date < value,
                TradingDayRow.is_open.is_(True),
            )
            .order_by(TradingDayRow.trade_date.desc())
            .limit(1)
        )
        return self._session.scalar(statement)


class InstrumentMasterService:
    """Maintain stable security IDs and historical exchange-code aliases."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, instruments: list[Instrument]) -> tuple[int, int]:
        """Insert or update current master records and initial aliases."""

        inserted = 0
        updated = 0
        for instrument in instruments:
            row = self._session.get(InstrumentRow, instrument.instrument_id)
            if row is None:
                row = InstrumentRow(
                    instrument_id=instrument.instrument_id,
                    symbol=instrument.symbol,
                    exchange=instrument.exchange,
                    instrument_type=instrument.instrument_type.value,
                    name=instrument.name,
                    listed_on=instrument.listed_on,
                    delisted_on=instrument.delisted_on,
                    status=instrument.status.value,
                    source=instrument.source,
                    version=instrument.version,
                )
                self._session.add(row)
                self._session.add(
                    InstrumentAliasRow(
                        instrument_id=instrument.instrument_id,
                        symbol=instrument.symbol,
                        exchange=instrument.exchange,
                        effective_from=instrument.listed_on,
                        effective_to=instrument.delisted_on,
                        source=instrument.source,
                    )
                )
                if instrument.delisted_on is None:
                    self._session.add(
                        InstrumentStatusRow(
                            instrument_id=instrument.instrument_id,
                            status=instrument.status.value,
                            effective_from=instrument.listed_on,
                            effective_to=None,
                            source=instrument.source,
                            version=instrument.version,
                        )
                    )
                else:
                    listed_end = instrument.delisted_on - timedelta(days=1)
                    if listed_end >= instrument.listed_on:
                        self._session.add(
                            InstrumentStatusRow(
                                instrument_id=instrument.instrument_id,
                                status="LISTED",
                                effective_from=instrument.listed_on,
                                effective_to=listed_end,
                                source=instrument.source,
                                version=instrument.version,
                            )
                        )
                    self._session.add(
                        InstrumentStatusRow(
                            instrument_id=instrument.instrument_id,
                            status="DELISTED",
                            effective_from=instrument.delisted_on,
                            effective_to=None,
                            source=instrument.source,
                            version=instrument.version,
                        )
                    )
                inserted += 1
            else:
                changed = (
                    row.name != instrument.name
                    or row.delisted_on != instrument.delisted_on
                    or row.status != instrument.status.value
                    or row.version != instrument.version
                )
                if changed:
                    row.name = instrument.name
                    row.delisted_on = instrument.delisted_on
                    row.status = instrument.status.value
                    row.version = instrument.version
                    updated += 1
        self._session.flush()
        return inserted, updated

    def add_alias(
        self,
        *,
        instrument_id: str,
        symbol: str,
        exchange: str,
        effective_from: date,
        effective_to: date | None,
        source: str,
    ) -> None:
        """Add a non-overlapping historical code alias."""

        instrument = self._session.get(InstrumentRow, instrument_id)
        if instrument is None:
            raise ValueError(f"unknown instrument_id: {instrument_id}")
        overlaps = self._session.scalar(
            select(InstrumentAliasRow.id).where(
                InstrumentAliasRow.symbol == symbol,
                InstrumentAliasRow.exchange == exchange,
                InstrumentAliasRow.effective_from <= (effective_to or date.max),
                or_(
                    InstrumentAliasRow.effective_to.is_(None),
                    InstrumentAliasRow.effective_to >= effective_from,
                ),
            )
        )
        if overlaps is not None:
            raise ValueError("instrument alias effective range overlaps an existing alias")
        self._session.add(
            InstrumentAliasRow(
                instrument_id=instrument_id,
                symbol=symbol,
                exchange=exchange,
                effective_from=effective_from,
                effective_to=effective_to,
                source=source,
            )
        )
        self._session.flush()

    def resolve(self, symbol: str, exchange: str, as_of: date) -> InstrumentRow | None:
        """Resolve an exchange code to its stable instrument at a date."""

        statement = (
            select(InstrumentRow)
            .join(
                InstrumentAliasRow,
                InstrumentAliasRow.instrument_id == InstrumentRow.instrument_id,
            )
            .where(
                InstrumentAliasRow.symbol == symbol,
                InstrumentAliasRow.exchange == exchange,
                InstrumentAliasRow.effective_from <= as_of,
                or_(
                    InstrumentAliasRow.effective_to.is_(None),
                    InstrumentAliasRow.effective_to >= as_of,
                ),
                InstrumentRow.listed_on <= as_of,
                or_(
                    InstrumentRow.delisted_on.is_(None),
                    InstrumentRow.delisted_on >= as_of,
                ),
            )
        )
        return self._session.scalar(statement)

    def is_active(self, instrument_id: str, as_of: date) -> bool:
        """Return whether a stable instrument exists and is active at a date."""

        statement = select(InstrumentRow.instrument_id).where(
            InstrumentRow.instrument_id == instrument_id,
            InstrumentRow.listed_on <= as_of,
            or_(
                InstrumentRow.delisted_on.is_(None),
                InstrumentRow.delisted_on > as_of,
            ),
        )
        return self._session.scalar(statement) is not None

    def status_on(self, instrument_id: str, as_of: date) -> str | None:
        """Return the known lifecycle status at a date."""

        statement = select(InstrumentStatusRow.status).where(
            InstrumentStatusRow.instrument_id == instrument_id,
            InstrumentStatusRow.effective_from <= as_of,
            or_(
                InstrumentStatusRow.effective_to.is_(None),
                InstrumentStatusRow.effective_to >= as_of,
            ),
        )
        return self._session.scalar(statement)
