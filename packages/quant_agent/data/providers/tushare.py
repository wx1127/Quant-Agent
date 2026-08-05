"""Minimal Tushare Pro HTTP adapter for instrument, calendar and daily bars."""

import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from datetime import time as datetime_time
from decimal import Decimal
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndustryMembership,
    Instrument,
    InstrumentStatus,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.providers.base import (
    ProviderBatch,
    ProviderError,
    ProviderRequestPolicy,
)
from quant_agent.research.current import StockProfile

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_VOLUME_LOT_TO_SHARES = Decimal("100")
_AMOUNT_THOUSANDS_TO_CNY = Decimal("1000")


def instrument_id_from_ts_code(ts_code: str) -> str:
    """Map a Tushare code to a stable Quant Agent identifier."""

    try:
        symbol, exchange = ts_code.split(".", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"invalid Tushare code: {ts_code}") from error
    return f"CN.{exchange}.{symbol}"


class TushareHttpProvider:
    """Language-neutral Tushare Pro REST adapter with bounded retries."""

    provider_name = "tushare"

    def __init__(
        self,
        token: str,
        *,
        client: httpx.Client | None = None,
        policy: ProviderRequestPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        base_url: str = "https://api.tushare.pro",
    ) -> None:
        if not token:
            raise ValueError("Tushare token is required")
        self._token = token
        self._policy = policy or ProviderRequestPolicy()
        self._client = client or httpx.Client(timeout=self._policy.timeout_seconds)
        self._owns_client = client is None
        self._sleeper = sleeper
        self._base_url = base_url

    def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "TushareHttpProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _call(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: Sequence[str],
    ) -> dict[str, Any]:
        request = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
            "fields": ",".join(fields),
        }
        last_error: Exception | None = None
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                response = self._client.post(self._base_url, json=request)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ProviderError(
                        self.provider_name,
                        api_name,
                        "provider response must be an object",
                        retryable=False,
                    )
                if payload.get("code") != 0:
                    raise ProviderError(
                        self.provider_name,
                        api_name,
                        str(payload.get("msg") or "provider returned an error"),
                        retryable=False,
                    )
                if not isinstance(payload.get("data"), dict):
                    raise ProviderError(
                        self.provider_name,
                        api_name,
                        "provider response is missing data",
                        retryable=False,
                    )
                return cast(dict[str, Any], payload)
            except ProviderError:
                raise
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt < self._policy.max_attempts:
                    delay = self._policy.initial_backoff_seconds * (2 ** (attempt - 1))
                    self._sleeper(delay)
        raise ProviderError(
            self.provider_name,
            api_name,
            f"request failed after {self._policy.max_attempts} attempts: {last_error}",
            retryable=True,
        )

    @staticmethod
    def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload["data"]
        fields = data.get("fields")
        items = data.get("items")
        if not isinstance(fields, list) or not isinstance(items, list):
            raise ProviderError(
                "tushare",
                "response",
                "provider fields/items are invalid",
                retryable=False,
            )
        return [dict(zip(fields, values, strict=True)) for values in items]

    @staticmethod
    def _available_at(trade_date: date) -> datetime:
        return datetime.combine(trade_date, datetime_time(16, 0), tzinfo=_SHANGHAI)

    def fetch_instruments(self, as_of: date) -> ProviderBatch[Instrument]:
        fields = (
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "list_date",
            "delist_date",
            "list_status",
        )
        payload = self._call("stock_basic", {"list_status": ""}, fields)
        records: list[Instrument] = []
        for row in self._rows(payload):
            ts_code = str(row["ts_code"])
            _, exchange = ts_code.split(".", maxsplit=1)
            status = (
                InstrumentStatus.DELISTED
                if row.get("list_status") == "D"
                else InstrumentStatus.LISTED
            )
            delist_date = str(row.get("delist_date") or "")
            records.append(
                Instrument(
                    instrument_id=instrument_id_from_ts_code(ts_code),
                    symbol=str(row["symbol"]),
                    exchange=exchange,
                    instrument_type=InstrumentType.STOCK,
                    name=str(row["name"]),
                    listed_on=datetime.strptime(str(row["list_date"]), "%Y%m%d").date(),
                    industry=str(row.get("industry") or "").strip() or None,
                    delisted_on=(
                        datetime.strptime(delist_date, "%Y%m%d").date() if delist_date else None
                    ),
                    status=status,
                    source=self.provider_name,
                    version=as_of.isoformat(),
                )
            )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="stock_basic",
            request_params={"as_of": as_of.isoformat()},
            raw_payload=payload,
            available_at=self._available_at(as_of),
            records=tuple(records),
        )

    def fetch_stock_profiles(self, as_of: datetime) -> ProviderBatch[StockProfile]:
        """Fetch current listed-stock names and industries for current research only."""

        if as_of.utcoffset() is None:
            raise ValueError("stock profile as_of must be timezone-aware")
        fields = (
            "ts_code",
            "name",
            "industry",
            "list_date",
            "list_status",
        )
        payload = self._call("stock_basic", {"list_status": "L"}, fields)
        records = []
        for row in self._rows(payload):
            industry = str(row.get("industry") or "").strip()
            if not industry or str(row.get("list_status")) != "L":
                continue
            records.append(
                StockProfile(
                    instrument_id=instrument_id_from_ts_code(str(row["ts_code"])),
                    name=str(row["name"]),
                    industry=industry,
                    listed_on=datetime.strptime(str(row["list_date"]), "%Y%m%d").date(),
                    source=self.provider_name,
                    available_at=as_of,
                    version=as_of.isoformat(),
                )
            )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="stock_basic",
            request_params={"list_status": "L", "as_of": as_of.isoformat()},
            raw_payload=payload,
            available_at=as_of,
            records=tuple(records),
        )

    def fetch_trading_calendar(
        self,
        market: str,
        start: date,
        end: date,
    ) -> ProviderBatch[TradingDay]:
        fields = ("exchange", "cal_date", "is_open", "pretrade_date")
        payload = self._call(
            "trade_cal",
            {
                "exchange": market,
                "start_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
            },
            fields,
        )
        records = tuple(
            TradingDay(
                market=market,
                trade_date=datetime.strptime(str(row["cal_date"]), "%Y%m%d").date(),
                is_open=str(row["is_open"]) == "1",
                source=self.provider_name,
                version=end.isoformat(),
            )
            for row in self._rows(payload)
        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="trade_cal",
            request_params={"market": market, "start": str(start), "end": str(end)},
            raw_payload=payload,
            available_at=self._available_at(end),
            records=records,
        )

    def fetch_daily_bars(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[DailyBar]:
        fields = (
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        )
        params: dict[str, Any] = {"trade_date": trade_date.strftime("%Y%m%d")}
        if instrument_ids is not None and len(instrument_ids) == 1:
            _, exchange, symbol = instrument_ids[0].split(".", maxsplit=2)
            params["ts_code"] = f"{symbol}.{exchange}"
        payload = self._call("daily", params, fields)
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records = []
        for row in self._rows(payload):
            instrument_id = instrument_id_from_ts_code(str(row["ts_code"]))
            if allowed is not None and instrument_id not in allowed:
                continue
            records.append(
                DailyBar(
                    instrument_id=instrument_id,
                    trade_date=datetime.strptime(
                        str(row["trade_date"]),
                        "%Y%m%d",
                    ).date(),
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["vol"])) * _VOLUME_LOT_TO_SHARES,
                    turnover=Decimal(str(row["amount"])) * _AMOUNT_THOUSANDS_TO_CNY,
                    source=self.provider_name,
                    available_at=self._available_at(trade_date),
                    version=trade_date.isoformat(),
                )
            )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="daily",
            request_params=params,
            raw_payload=payload,
            available_at=self._available_at(trade_date),
            records=tuple(records),
        )

    def fetch_adjustment_factors(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[AdjustmentFactor]:
        raise ProviderError(
            self.provider_name,
            "adj_factor",
            "P1 adapter only implements daily bars; use a dedicated factor adapter",
            retryable=False,
        )

    def fetch_fundamentals(
        self,
        instrument_ids: Sequence[str],
        as_of: datetime,
    ) -> ProviderBatch[FundamentalPoint]:
        raise ProviderError(
            self.provider_name,
            "fundamentals",
            "P1 adapter only implements daily bars; use a dedicated fundamentals adapter",
            retryable=False,
        )

    def fetch_industry_memberships(
        self,
        as_of: date,
    ) -> ProviderBatch[IndustryMembership]:
        raise ProviderError(
            self.provider_name,
            "industry_memberships",
            "P1 adapter only implements daily bars; use a dedicated classification adapter",
            retryable=False,
        )
