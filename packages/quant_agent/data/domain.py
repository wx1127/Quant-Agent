"""Validated provider and service data records."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from quant_agent.core.time import ensure_aware


class InstrumentType(StrEnum):
    """Supported P1 instrument types."""

    STOCK = "STOCK"
    ETF = "ETF"
    INDEX = "INDEX"


class InstrumentStatus(StrEnum):
    """Security lifecycle state."""

    LISTED = "LISTED"
    SUSPENDED = "SUSPENDED"
    DELISTED = "DELISTED"


class Instrument(BaseModel):
    """Provider-neutral security master record."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    symbol: str
    exchange: str
    instrument_type: InstrumentType
    name: str
    listed_on: date
    delisted_on: date | None = None
    status: InstrumentStatus = InstrumentStatus.LISTED
    source: str
    version: str

    @model_validator(mode="after")
    def validate_dates(self) -> "Instrument":
        """Reject impossible lifecycle ranges."""

        if self.delisted_on is not None and self.delisted_on < self.listed_on:
            raise ValueError("delisted_on cannot precede listed_on")
        return self


class TradingDay(BaseModel):
    """Provider-neutral trading calendar record."""

    model_config = ConfigDict(frozen=True)

    market: str
    trade_date: date
    is_open: bool
    source: str
    version: str


class DailyBar(BaseModel):
    """Unadjusted daily market bar."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal
    source: str
    available_at: datetime
    version: str

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_ohlcv(self) -> "DailyBar":
        """Reject structurally invalid market bars."""

        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least the other OHLC prices")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be at most the other OHLC prices")
        if self.volume < 0 or self.turnover < 0:
            raise ValueError("volume and turnover must be non-negative")
        return self


class AdjustmentFactor(BaseModel):
    """Daily adjustment factor."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    trade_date: date
    factor: Decimal
    source: str
    available_at: datetime
    version: str

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @field_validator("factor")
    @classmethod
    def validate_factor(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("adjustment factor must be positive")
        return value


class CorporateAction(BaseModel):
    """Point-in-time corporate action."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    action_type: str
    ex_date: date
    announced_at: datetime
    available_at: datetime
    cash_amount: Decimal | None = None
    share_ratio: Decimal | None = None
    source: str
    version: str

    @field_validator("announced_at", "available_at")
    @classmethod
    def validate_times(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_availability(self) -> "CorporateAction":
        if self.available_at < self.announced_at:
            raise ValueError("available_at cannot precede announced_at")
        return self


class FundamentalPoint(BaseModel):
    """One point-in-time fundamental metric revision."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    report_period: date
    metric_name: str
    metric_value: Decimal
    announced_at: datetime
    available_at: datetime
    provider_revision: str
    source: str

    @field_validator("announced_at", "available_at")
    @classmethod
    def validate_times(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_availability(self) -> "FundamentalPoint":
        if self.available_at < self.announced_at:
            raise ValueError("available_at cannot precede announced_at")
        return self


class Industry(BaseModel):
    """Versioned industry node."""

    model_config = ConfigDict(frozen=True)

    industry_id: str
    classification: str
    code: str
    name: str
    level: int
    parent_id: str | None = None
    version: str


class IndustryMembership(BaseModel):
    """Historical industry membership."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    industry_id: str
    effective_from: date
    effective_to: date | None = None
    source: str
    version: str

    @model_validator(mode="after")
    def validate_dates(self) -> "IndustryMembership":
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        return self
