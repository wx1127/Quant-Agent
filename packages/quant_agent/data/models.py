"""Versioned relational models for raw, curated and snapshot data."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from quant_agent.core.time import shanghai_now
from quant_agent.data.database import Base


class InstrumentRow(Base):
    """Point-in-time security master record."""

    __tablename__ = "instrument"

    instrument_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(16), nullable=False)
    instrument_type: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    listed_on: Mapped[date] = mapped_column(Date, nullable=False)
    delisted_on: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="LISTED")
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )

    __table_args__ = (Index("ix_instrument_active_dates", "listed_on", "delisted_on"),)


class InstrumentAliasRow(Base):
    """Historical exchange code mapped to a stable instrument identifier."""

    __tablename__ = "instrument_alias"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(16), nullable=False)
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "exchange",
            "effective_from",
            name="uq_instrument_alias_effective",
        ),
        Index(
            "ix_instrument_alias_lookup",
            "symbol",
            "exchange",
            "effective_from",
            "effective_to",
        ),
    )


class InstrumentStatusRow(Base):
    """Historical security status interval."""

    __tablename__ = "instrument_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "status",
            "effective_from",
            "version",
            name="uq_instrument_status_version",
        ),
        Index(
            "ix_instrument_status_dates",
            "instrument_id",
            "effective_from",
            "effective_to",
        ),
    )


class TradingDayRow(Base):
    """Trading calendar record."""

    __tablename__ = "trading_calendar"

    market: Mapped[str] = mapped_column(String(16), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )


class RawPayloadRow(Base):
    """Immutable raw provider response."""

    __tablename__ = "raw_payload"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )


class DailyBarRow(Base):
    """Unadjusted daily OHLCV bar."""

    __tablename__ = "market_bar_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(28, 4), nullable=False)
    turnover: Mapped[Decimal] = mapped_column(Numeric(28, 4), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "trade_date",
            "source",
            "version",
            name="uq_daily_bar_version",
        ),
        Index("ix_daily_bar_instrument_date", "instrument_id", "trade_date"),
    )


class AdjustmentFactorRow(Base):
    """Adjustment factor stored separately from raw prices."""

    __tablename__ = "adjustment_factor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    factor: Mapped[Decimal] = mapped_column(Numeric(28, 10), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "trade_date",
            "source",
            "version",
            name="uq_adjustment_factor_version",
        ),
    )


class CorporateActionRow(Base):
    """Versioned corporate action without mutating historical prices."""

    __tablename__ = "corporate_action"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    announced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cash_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    share_ratio: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "action_type",
            "ex_date",
            "source",
            "version",
            name="uq_corporate_action_version",
        ),
    )


class FundamentalPointRow(Base):
    """Point-in-time fundamental metric with revision history."""

    __tablename__ = "financial_statement_point_in_time"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    report_period: Mapped[date] = mapped_column(Date, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(96), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    announced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    provider_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "report_period",
            "metric_name",
            "source",
            "provider_revision",
            name="uq_fundamental_revision",
        ),
        Index(
            "ix_fundamental_point_in_time",
            "instrument_id",
            "metric_name",
            "available_at",
        ),
    )


class IndustryRow(Base):
    """Versioned industry classification node."""

    __tablename__ = "industry_classification"

    industry_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("industry_classification.industry_id"))
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "classification",
            "code",
            "version",
            name="uq_industry_classification_version",
        ),
    )


class IndustryMembershipRow(Base):
    """Historical instrument membership in an industry."""

    __tablename__ = "industry_membership_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("instrument.instrument_id"),
        nullable=False,
    )
    industry_id: Mapped[str] = mapped_column(
        ForeignKey("industry_classification.industry_id"),
        nullable=False,
    )
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "industry_id",
            "effective_from",
            "version",
            name="uq_industry_membership_version",
        ),
        Index(
            "ix_industry_membership_dates",
            "instrument_id",
            "effective_from",
            "effective_to",
        ),
    )


class DataQualityResultRow(Base):
    """Persisted result from one data-quality rule."""

    __tablename__ = "data_quality_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_id: Mapped[str] = mapped_column(String(96), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    entity_key: Mapped[str | None] = mapped_column(String(192))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_quality_version_passed", "data_version", "passed"),)


class DatasetVersionRow(Base):
    """Immutable manifest reference for a released dataset snapshot."""

    __tablename__ = "dataset_version"

    data_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=shanghai_now,
    )
