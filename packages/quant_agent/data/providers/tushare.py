"""Tushare Pro HTTP adapter for stock, ETF, and index market data."""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndexConstituentWeight,
    Industry,
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

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_VOLUME_LOT_TO_SHARES = Decimal("100")
_AMOUNT_THOUSANDS_TO_CNY = Decimal("1000")
_STOCK_BASIC_LIMIT = 6000
_DAILY_LIMIT = 6000
_ETF_BASIC_LIMIT = 5000
_FUND_DAILY_LIMIT = 5000
_INDEX_BASIC_LIMIT = 8000
_ANNOUNCEMENT_LIMIT = 2000
_STOCK_CLOSE_CUTOFF = datetime_time(16, 0)
_POST_CLOSE_CUTOFF = datetime_time(17, 0)
_MASTER_DATA_CUTOFF = datetime_time(18, 0)
_ADJUSTMENT_FACTOR_CUTOFF = datetime_time(9, 30)
_FUNDAMENTAL_AVAILABLE_CUTOFF = datetime_time(9, 30)
_FINA_INDICATOR_LIMIT = 100
_INDEX_MEMBER_ALL_LIMIT = 2000
_FUNDAMENTAL_METRICS = (
    "eps",
    "dt_eps",
    "revenue_ps",
    "profit_dedt",
    "grossprofit_margin",
    "roe",
    "roe_waa",
    "roa",
    "roic",
    "debt_to_assets",
    "ocfps",
    "basic_eps_yoy",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "tr_yoy",
    "or_yoy",
    "rd_exp",
)
_ENDPOINT_FIELDS: dict[str, tuple[str, ...]] = {
    "stock_basic": (
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "market",
        "list_date",
        "delist_date",
        "list_status",
    ),
    "etf_basic": (
        "ts_code",
        "csname",
        "extname",
        "list_date",
        "list_status",
        "exchange",
    ),
    "index_basic": ("ts_code", "name", "market", "list_date", "exp_date"),
    "trade_cal": ("exchange", "cal_date", "is_open", "pretrade_date"),
    "daily": ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"),
    "fund_daily": (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ),
    "index_daily": (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ),
    "adj_factor": ("ts_code", "trade_date", "adj_factor"),
    "anns_d": ("ann_date", "ts_code", "name", "title", "url", "rec_time"),
}
_ENDPOINT_ROW_LIMITS: dict[str, int] = {
    "stock_basic": _STOCK_BASIC_LIMIT,
    "etf_basic": _ETF_BASIC_LIMIT,
    "index_basic": _INDEX_BASIC_LIMIT,
    "daily": _DAILY_LIMIT,
    "fund_daily": _FUND_DAILY_LIMIT,
    "anns_d": _ANNOUNCEMENT_LIMIT,
}
_ENDPOINT_CUTOFFS: dict[str, datetime_time] = {
    "stock_basic": _MASTER_DATA_CUTOFF,
    "etf_basic": _MASTER_DATA_CUTOFF,
    "index_basic": _MASTER_DATA_CUTOFF,
    "trade_cal": _STOCK_CLOSE_CUTOFF,
    "daily": _STOCK_CLOSE_CUTOFF,
    "fund_daily": _POST_CLOSE_CUTOFF,
    "index_daily": _POST_CLOSE_CUTOFF,
    "adj_factor": _ADJUSTMENT_FACTOR_CUTOFF,
}


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

    def fetch_raw_json(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch one untouched successful Tushare JSON envelope.

        The token is added only at the HTTP boundary. Callers can therefore persist
        ``params`` as request metadata without ever handling the credential.
        """

        selected_fields = tuple(fields) if fields is not None else _ENDPOINT_FIELDS.get(api_name)
        if selected_fields is None:
            raise ValueError(f"fields are required for unknown Tushare endpoint: {api_name}")
        return self._call(api_name, params, selected_fields)

    @staticmethod
    def raw_record_count(payload: dict[str, Any], *, endpoint: str) -> int:
        """Return the raw item count without performing domain decoding."""

        data = payload.get("data")
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ProviderError(
                "tushare",
                endpoint,
                "provider response is missing data items",
                retryable=False,
            )
        return len(items)

    @classmethod
    def decode_raw_rows(
        cls,
        payload: dict[str, Any],
        *,
        endpoint: str,
    ) -> tuple[dict[str, Any], ...]:
        """Decode a documented endpoint and enforce its known row limit.

        Raw-first sources that require record-level JSON pointers use this
        boundary instead of duplicating envelope checks or reaching into the
        private decoder.
        """

        return tuple(
            cls._rows(
                payload,
                endpoint=endpoint,
                row_limit=_ENDPOINT_ROW_LIMITS.get(endpoint),
            )
        )

    @staticmethod
    def available_at_for_endpoint(endpoint: str, value_date: date) -> datetime:
        """Return the conservative point-in-time availability for an endpoint."""

        cutoff = _ENDPOINT_CUTOFFS.get(endpoint)
        if cutoff is None:
            raise ValueError(f"unknown Tushare availability endpoint: {endpoint}")
        return TushareHttpProvider._available_at(value_date, cutoff)

    @staticmethod
    def _rows(
        payload: dict[str, Any],
        *,
        endpoint: str,
        row_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ProviderError(
                "tushare",
                endpoint,
                "provider response is missing data",
                retryable=False,
            )
        fields = data.get("fields")
        items = data.get("items")
        if (
            not isinstance(fields, list)
            or not all(isinstance(field, str) for field in fields)
            or not isinstance(items, list)
        ):
            raise ProviderError(
                "tushare",
                endpoint,
                "provider fields/items are invalid",
                retryable=False,
            )
        if row_limit is not None and len(items) >= row_limit:
            raise ProviderError(
                "tushare",
                endpoint,
                (
                    f"provider returned {len(items)} rows, reaching the documented "
                    f"{row_limit}-row limit; narrow the request to prevent truncation"
                ),
                retryable=False,
            )
        field_names = cast(list[str], fields)
        rows: list[dict[str, Any]] = []
        for values in items:
            if not isinstance(values, list):
                raise ProviderError(
                    "tushare",
                    endpoint,
                    "provider row is not an array",
                    retryable=False,
                )
            try:
                rows.append(dict(zip(field_names, values, strict=True)))
            except ValueError as error:
                raise ProviderError(
                    "tushare",
                    endpoint,
                    "provider row length does not match fields",
                    retryable=False,
                ) from error
        return rows

    @staticmethod
    def _available_at(value_date: date, cutoff: datetime_time = _STOCK_CLOSE_CUTOFF) -> datetime:
        return datetime.combine(value_date, cutoff, tzinfo=_SHANGHAI)

    @staticmethod
    def ts_code_from_instrument_id(instrument_id: str) -> str:
        """Map a stable Quant Agent instrument ID back to a Tushare code."""

        parts = instrument_id.split(".", maxsplit=2)
        if len(parts) != 3 or parts[0] != "CN" or not parts[1] or not parts[2]:
            raise ValueError(f"invalid Quant Agent instrument ID: {instrument_id}")
        _, exchange, symbol = parts
        return f"{symbol}.{exchange}"

    @staticmethod
    def _parse_date(value: object, *, endpoint: str, field: str) -> date:
        text = str(value or "")
        if not text:
            raise ProviderError(
                "tushare",
                endpoint,
                f"provider row is missing required {field}",
                retryable=False,
            )
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError as error:
            raise ProviderError(
                "tushare",
                endpoint,
                f"provider returned invalid {field}: {text}",
                retryable=False,
            ) from error

    @staticmethod
    def instrument_endpoint(instrument_type: InstrumentType) -> str:
        """Map a supported instrument type to its Tushare master endpoint."""

        endpoints = {
            InstrumentType.STOCK: "stock_basic",
            InstrumentType.ETF: "etf_basic",
            InstrumentType.INDEX: "index_basic",
        }
        try:
            return endpoints[instrument_type]
        except KeyError as error:
            raise ProviderError(
                "tushare",
                "instruments",
                f"unsupported instrument type: {instrument_type}",
                retryable=False,
            ) from error

    @staticmethod
    def daily_endpoint(instrument_type: InstrumentType) -> str:
        """Map a supported instrument type to its Tushare daily endpoint."""

        endpoints = {
            InstrumentType.STOCK: "daily",
            InstrumentType.ETF: "fund_daily",
            InstrumentType.INDEX: "index_daily",
        }
        try:
            return endpoints[instrument_type]
        except KeyError as error:
            raise ProviderError(
                "tushare",
                "daily_bars",
                f"unsupported instrument type: {instrument_type}",
                retryable=False,
            ) from error

    def decode_instruments_payload(
        self,
        payload: dict[str, Any],
        *,
        as_of: date,
        instrument_type: InstrumentType,
    ) -> tuple[Instrument, ...]:
        """Decode a persisted master-data envelope without any network access."""

        endpoint = self.instrument_endpoint(instrument_type)
        rows = self._rows(
            payload,
            endpoint=endpoint,
            row_limit=_ENDPOINT_ROW_LIMITS.get(endpoint),
        )
        records: list[Instrument] = []
        for row in rows:
            if instrument_type is InstrumentType.STOCK:
                if not str(row.get("list_date") or ""):
                    continue
                ts_code = str(row["ts_code"])
                symbol, exchange = ts_code.split(".", maxsplit=1)
                status_code = str(row.get("list_status") or "L")
                status = {
                    "D": InstrumentStatus.DELISTED,
                    "P": InstrumentStatus.SUSPENDED,
                }.get(status_code, InstrumentStatus.LISTED)
                raw_delisted = str(row.get("delist_date") or "")
                delisted_on = (
                    self._parse_date(
                        raw_delisted,
                        endpoint=endpoint,
                        field="delist_date",
                    )
                    if raw_delisted
                    else None
                )
                records.append(
                    Instrument(
                        instrument_id=instrument_id_from_ts_code(ts_code),
                        symbol=symbol,
                        exchange=exchange,
                        instrument_type=instrument_type,
                        name=str(row["name"]),
                        listed_on=self._parse_date(
                            row.get("list_date"),
                            endpoint=endpoint,
                            field="list_date",
                        ),
                        delisted_on=delisted_on,
                        status=status,
                        source=self.provider_name,
                        version=as_of.isoformat(),
                    )
                )
                continue

            if instrument_type is InstrumentType.ETF:
                list_date = str(row.get("list_date") or "")
                if str(row.get("list_status") or "") != "L" or not list_date:
                    continue
                ts_code = str(row["ts_code"])
                symbol, suffix = ts_code.split(".", maxsplit=1)
                records.append(
                    Instrument(
                        instrument_id=instrument_id_from_ts_code(ts_code),
                        symbol=symbol,
                        exchange=str(row.get("exchange") or suffix),
                        instrument_type=instrument_type,
                        name=str(row.get("extname") or row.get("csname") or symbol),
                        listed_on=self._parse_date(
                            list_date,
                            endpoint=endpoint,
                            field="list_date",
                        ),
                        status=InstrumentStatus.LISTED,
                        source=self.provider_name,
                        version=as_of.isoformat(),
                    )
                )
                continue

            ts_code = str(row["ts_code"])
            symbol, exchange = ts_code.split(".", maxsplit=1)
            raw_expiration = str(row.get("exp_date") or "")
            delisted_on = (
                self._parse_date(
                    raw_expiration,
                    endpoint=endpoint,
                    field="exp_date",
                )
                if raw_expiration
                else None
            )
            records.append(
                Instrument(
                    instrument_id=instrument_id_from_ts_code(ts_code),
                    symbol=symbol,
                    exchange=exchange,
                    instrument_type=instrument_type,
                    name=str(row["name"]),
                    listed_on=self._parse_date(
                        row.get("list_date"),
                        endpoint=endpoint,
                        field="list_date",
                    ),
                    delisted_on=delisted_on,
                    status=(
                        InstrumentStatus.DELISTED
                        if delisted_on is not None and delisted_on <= as_of
                        else InstrumentStatus.LISTED
                    ),
                    source=self.provider_name,
                    version=as_of.isoformat(),
                )
            )
        return tuple(records)

    def decode_trading_calendar_payload(
        self,
        payload: dict[str, Any],
        *,
        market: str,
        start: date,
        end: date,
    ) -> tuple[TradingDay, ...]:
        """Decode a persisted calendar envelope for one explicit interval."""

        if start > end:
            raise ValueError("start cannot be after end")
        records: list[TradingDay] = []
        for row in self._rows(payload, endpoint="trade_cal"):
            trade_date = self._parse_date(
                row.get("cal_date"),
                endpoint="trade_cal",
                field="cal_date",
            )
            if trade_date < start or trade_date > end:
                raise ProviderError(
                    self.provider_name,
                    "trade_cal",
                    "provider returned a date outside the requested partition",
                    retryable=False,
                )
            records.append(
                TradingDay(
                    market=market,
                    trade_date=trade_date,
                    is_open=str(row["is_open"]) == "1",
                    source=self.provider_name,
                    version=end.isoformat(),
                )
            )
        return tuple(records)

    def decode_daily_bars_payload(
        self,
        payload: dict[str, Any],
        *,
        trade_date: date,
        instrument_type: InstrumentType,
        instrument_ids: Sequence[str] | None = None,
    ) -> tuple[DailyBar, ...]:
        """Decode persisted unadjusted OHLCV rows using stable unit rules."""

        endpoint = self.daily_endpoint(instrument_type)
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records: list[DailyBar] = []
        for row in self._rows(
            payload,
            endpoint=endpoint,
            row_limit=_ENDPOINT_ROW_LIMITS.get(endpoint),
        ):
            instrument_id = instrument_id_from_ts_code(str(row["ts_code"]))
            if allowed is not None and instrument_id not in allowed:
                continue
            row_trade_date = self._parse_date(
                row.get("trade_date"),
                endpoint=endpoint,
                field="trade_date",
            )
            if row_trade_date != trade_date:
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    "provider returned a date outside the requested partition",
                    retryable=False,
                )
            records.append(
                DailyBar(
                    instrument_id=instrument_id,
                    trade_date=row_trade_date,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["vol"])) * _VOLUME_LOT_TO_SHARES,
                    turnover=Decimal(str(row["amount"])) * _AMOUNT_THOUSANDS_TO_CNY,
                    source=self.provider_name,
                    available_at=self.available_at_for_endpoint(endpoint, row_trade_date),
                    version=trade_date.isoformat(),
                )
            )
        return tuple(records)

    def decode_adjustment_factors_payload(
        self,
        payload: dict[str, Any],
        *,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> tuple[AdjustmentFactor, ...]:
        """Decode persisted stock adjustment-factor rows."""

        endpoint = "adj_factor"
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records: list[AdjustmentFactor] = []
        for row in self._rows(payload, endpoint=endpoint):
            instrument_id = instrument_id_from_ts_code(str(row["ts_code"]))
            if allowed is not None and instrument_id not in allowed:
                continue
            row_trade_date = self._parse_date(
                row.get("trade_date"),
                endpoint=endpoint,
                field="trade_date",
            )
            if row_trade_date != trade_date:
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    "provider returned a date outside the requested partition",
                    retryable=False,
                )
            records.append(
                AdjustmentFactor(
                    instrument_id=instrument_id,
                    trade_date=row_trade_date,
                    factor=Decimal(str(row["adj_factor"])),
                    source=self.provider_name,
                    available_at=self.available_at_for_endpoint(endpoint, row_trade_date),
                    version=trade_date.isoformat(),
                )
            )
        return tuple(records)

    def fetch_instruments(
        self,
        as_of: date,
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
        market: str | None = None,
    ) -> ProviderBatch[Instrument]:
        if instrument_type is InstrumentType.STOCK:
            return self._fetch_stock_instruments(as_of, market=market)
        if instrument_type is InstrumentType.ETF:
            return self._fetch_etf_instruments(as_of, market=market)
        if instrument_type is InstrumentType.INDEX:
            return self._fetch_index_instruments(as_of, market=market)
        raise ProviderError(
            self.provider_name,
            "instruments",
            f"unsupported instrument type: {instrument_type}",
            retryable=False,
        )

    def _fetch_stock_instruments(
        self,
        as_of: date,
        *,
        market: str | None,
    ) -> ProviderBatch[Instrument]:
        payloads: list[dict[str, Any]] = []
        records_by_code: dict[str, Instrument] = {}
        skipped_missing_list_date = 0
        for list_status in ("L", "D", "P"):
            params: dict[str, Any] = {"list_status": list_status}
            if market is not None:
                params["market"] = market
            payload = self.fetch_raw_json("stock_basic", params)
            payloads.append({"params": params, "payload": payload})
            decoded = self.decode_instruments_payload(
                payload,
                as_of=as_of,
                instrument_type=InstrumentType.STOCK,
            )
            skipped_missing_list_date += self.raw_record_count(
                payload,
                endpoint="stock_basic",
            ) - len(decoded)
            records_by_code.update((record.instrument_id, record) for record in decoded)
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="stock_basic",
            request_params={
                "as_of": as_of.isoformat(),
                "instrument_type": InstrumentType.STOCK.value,
                "list_statuses": ["L", "D", "P"],
                "market": market,
            },
            raw_payload={
                "responses": payloads,
                "metadata": {"skipped_missing_list_date": skipped_missing_list_date},
            },
            available_at=self.available_at_for_endpoint("stock_basic", as_of),
            records=tuple(records_by_code.values()),
        )

    def _fetch_etf_instruments(
        self,
        as_of: date,
        *,
        market: str | None,
    ) -> ProviderBatch[Instrument]:
        params: dict[str, Any] = {"list_status": "L"}
        if market is not None:
            params["exchange"] = {"SSE": "SH", "SZSE": "SZ"}.get(market, market)
        payload = self.fetch_raw_json("etf_basic", params)
        records = self.decode_instruments_payload(
            payload,
            as_of=as_of,
            instrument_type=InstrumentType.ETF,
        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="etf_basic",
            request_params={
                "as_of": as_of.isoformat(),
                "instrument_type": InstrumentType.ETF.value,
                **params,
            },
            raw_payload=payload,
            available_at=self._available_at(as_of, _MASTER_DATA_CUTOFF),
            records=records,
        )

    def _fetch_index_instruments(
        self,
        as_of: date,
        *,
        market: str | None,
    ) -> ProviderBatch[Instrument]:
        if not market:
            raise ProviderError(
                self.provider_name,
                "index_basic",
                "index instrument master requires an explicit market",
                retryable=False,
            )
        params = {"market": market}
        payload = self.fetch_raw_json("index_basic", params)
        records = self.decode_instruments_payload(
            payload,
            as_of=as_of,
            instrument_type=InstrumentType.INDEX,
        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint="index_basic",
            request_params={
                "as_of": as_of.isoformat(),
                "instrument_type": InstrumentType.INDEX.value,
                **params,
            },
            raw_payload=payload,
            available_at=self._available_at(as_of, _MASTER_DATA_CUTOFF),
            records=records,
        )

    def fetch_trading_calendar(
        self,
        market: str,
        start: date,
        end: date,
    ) -> ProviderBatch[TradingDay]:
        params = {
            "exchange": market,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        payload = self.fetch_raw_json(
            "trade_cal",
            params,
        )
        records = self.decode_trading_calendar_payload(
            payload,
            market=market,
            start=start,
            end=end,
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
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
    ) -> ProviderBatch[DailyBar]:
        endpoint = self.daily_endpoint(instrument_type)
        if instrument_type is InstrumentType.INDEX and (
            instrument_ids is None or len(instrument_ids) != 1
        ):
            raise ProviderError(
                self.provider_name,
                endpoint,
                "index daily bars require exactly one instrument_id",
                retryable=False,
            )
        params: dict[str, Any] = {"trade_date": trade_date.strftime("%Y%m%d")}
        if instrument_ids is not None and len(instrument_ids) == 1:
            params["ts_code"] = self.ts_code_from_instrument_id(instrument_ids[0])
        payload = self.fetch_raw_json(endpoint, params)
        records = self.decode_daily_bars_payload(
            payload,
            trade_date=trade_date,
            instrument_type=instrument_type,
            instrument_ids=instrument_ids,
        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params=params,
            raw_payload=payload,
            available_at=self.available_at_for_endpoint(endpoint, trade_date),
            records=records,
        )

    def fetch_adjustment_factors(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
    ) -> ProviderBatch[AdjustmentFactor]:
        endpoint = "adj_factor"
        if instrument_type is not InstrumentType.STOCK:
            raise ProviderError(
                self.provider_name,
                endpoint,
                "Tushare adj_factor supports STOCK instruments only",
                retryable=False,
            )
        params: dict[str, Any] = {"trade_date": trade_date.strftime("%Y%m%d")}
        if instrument_ids is not None and len(instrument_ids) == 1:
            params["ts_code"] = self.ts_code_from_instrument_id(instrument_ids[0])
        payload = self.fetch_raw_json(endpoint, params)
        records = self.decode_adjustment_factors_payload(
            payload,
            trade_date=trade_date,
            instrument_ids=instrument_ids,
        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params=params,
            raw_payload=payload,
            available_at=self._available_at(
                trade_date,
                _ADJUSTMENT_FACTOR_CUTOFF,
            ),
            records=records,
        )

    def fetch_index_constituent_weights(
        self,
        index_id: str,
        start: date,
        end: date,
    ) -> ProviderBatch[IndexConstituentWeight]:
        if start > end:
            raise ValueError("start cannot be after end")
        endpoint = "index_weight"
        fields = ("index_code", "con_code", "trade_date", "weight")
        params = {
            "index_code": self.ts_code_from_instrument_id(index_id),
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        payload = self._call(endpoint, params, fields)
        records: list[IndexConstituentWeight] = []
        for row in self._rows(payload, endpoint=endpoint):
            trade_date = self._parse_date(
                row.get("trade_date"),
                endpoint=endpoint,
                field="trade_date",
            )
            records.append(
                IndexConstituentWeight(
                    index_instrument_id=instrument_id_from_ts_code(str(row["index_code"])),
                    constituent_instrument_id=instrument_id_from_ts_code(str(row["con_code"])),
                    trade_date=trade_date,
                    weight_percent=Decimal(str(row["weight"])),
                    available_at=self._available_at(trade_date, _MASTER_DATA_CUTOFF),
                    source=self.provider_name,
                    version=trade_date.isoformat(),
                )
            )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params=params,
            raw_payload=payload,
            available_at=self._available_at(end, _MASTER_DATA_CUTOFF),
            records=tuple(records),
        )

    @staticmethod
    def _fundamental_revision(row: dict[str, Any]) -> str:
        content = {
            field: None if row.get(field) is None else str(row[field])
            for field in ("ts_code", "ann_date", "end_date", *_FUNDAMENTAL_METRICS)
        }
        digest = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        update_flag = row.get("update_flag") or ""
        return f"{update_flag!s}:{digest}"

    def fetch_fundamentals(
        self,
        instrument_ids: Sequence[str],
        as_of: datetime,
        *,
        start_date: date,
    ) -> ProviderBatch[FundamentalPoint]:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must include timezone information")
        if start_date > as_of.date():
            raise ValueError("start_date cannot be after as_of")
        endpoint = "fina_indicator"
        fields = ("ts_code", "ann_date", "end_date", *_FUNDAMENTAL_METRICS, "update_flag")
        responses: list[dict[str, Any]] = []
        records_by_key: dict[tuple[str, date, str, str], FundamentalPoint] = {}
        for instrument_id in dict.fromkeys(instrument_ids):
            params = {
                "ts_code": self.ts_code_from_instrument_id(instrument_id),
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": as_of.date().strftime("%Y%m%d"),
            }
            payload = self._call(endpoint, params, fields)
            responses.append({"params": params, "payload": payload})
            for row in self._rows(
                payload,
                endpoint=endpoint,
                row_limit=_FINA_INDICATOR_LIMIT,
            ):
                row_instrument_id = instrument_id_from_ts_code(str(row["ts_code"]))
                report_period = self._parse_date(
                    row.get("end_date"),
                    endpoint=endpoint,
                    field="end_date",
                )
                if report_period < start_date or report_period > as_of.date():
                    continue
                announcement_date = self._parse_date(
                    row.get("ann_date"),
                    endpoint=endpoint,
                    field="ann_date",
                )
                announced_at = self._available_at(announcement_date, datetime_time(0, 0))
                available_at = self._available_at(
                    announcement_date + timedelta(days=1),
                    _FUNDAMENTAL_AVAILABLE_CUTOFF,
                )
                if available_at > as_of:
                    continue
                revision = self._fundamental_revision(row)
                for metric_name in _FUNDAMENTAL_METRICS:
                    raw_value = row.get(metric_name)
                    if raw_value is None or raw_value == "":
                        continue
                    try:
                        metric_value = Decimal(str(raw_value))
                    except (InvalidOperation, ValueError) as error:
                        raise ProviderError(
                            self.provider_name,
                            endpoint,
                            f"provider returned invalid {metric_name}: {raw_value}",
                            retryable=False,
                        ) from error
                    if not metric_value.is_finite():
                        raise ProviderError(
                            self.provider_name,
                            endpoint,
                            f"provider returned non-finite {metric_name}: {raw_value}",
                            retryable=False,
                        )
                    point = FundamentalPoint(
                        instrument_id=row_instrument_id,
                        report_period=report_period,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        announced_at=announced_at,
                        available_at=available_at,
                        provider_revision=revision,
                        source=self.provider_name,
                    )
                    records_by_key[(row_instrument_id, report_period, metric_name, revision)] = (
                        point
                    )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params={
                "instrument_ids": list(dict.fromkeys(instrument_ids)),
                "start_date": start_date.isoformat(),
                "as_of": as_of.isoformat(),
            },
            raw_payload={"responses": responses},
            available_at=as_of,
            records=tuple(records_by_key.values()),
        )

    def fetch_industries(
        self,
        as_of: date,
        classification: str,
        level: int | None = None,
    ) -> ProviderBatch[Industry]:
        if not classification:
            raise ValueError("classification must be non-empty")
        if level is not None and level not in (1, 2, 3):
            raise ValueError("level must be 1, 2, 3, or None")
        endpoint = "index_classify"
        fields = (
            "index_code",
            "industry_name",
            "parent_code",
            "level",
            "industry_code",
            "is_pub",
            "src",
        )
        params = {"src": classification}
        payload = self._call(endpoint, params, fields)
        rows = self._rows(payload, endpoint=endpoint)
        index_code_by_industry_code: dict[str, str] = {}
        for row in rows:
            row_source = str(row.get("src") or "")
            if row_source != classification:
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    f"provider returned unexpected classification source: {row_source}",
                    retryable=False,
                )
            industry_code = str(row.get("industry_code") or "")
            index_code = str(row.get("index_code") or "")
            if not industry_code or not index_code:
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    "provider row is missing industry_code or index_code",
                    retryable=False,
                )
            existing = index_code_by_industry_code.get(industry_code)
            if existing is not None and existing != index_code:
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    f"industry_code {industry_code} maps to multiple index codes",
                    retryable=False,
                )
            index_code_by_industry_code[industry_code] = index_code

        records: list[Industry] = []
        for row in rows:
            level_text = str(row.get("level") or "")
            if not level_text.startswith("L") or not level_text[1:].isdigit():
                raise ProviderError(
                    self.provider_name,
                    endpoint,
                    f"provider returned invalid industry level: {level_text}",
                    retryable=False,
                )
            row_level = int(level_text[1:])
            parent_code = str(row.get("parent_code") or "")
            parent_id: str | None = None
            if parent_code not in ("", "0"):
                parent_index_code = index_code_by_industry_code.get(parent_code)
                if parent_index_code is None:
                    raise ProviderError(
                        self.provider_name,
                        endpoint,
                        f"cannot resolve parent_code {parent_code}",
                        retryable=False,
                    )
                parent_id = f"{classification}:{parent_index_code}"
            if level is not None and row_level != level:
                continue
            index_code = str(row["index_code"])
            records.append(
                Industry(
                    industry_id=f"{classification}:{index_code}",
                    classification=classification,
                    code=str(row["industry_code"]),
                    name=str(row["industry_name"]),
                    level=row_level,
                    parent_id=parent_id,
                    version=classification,
                )
            )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params={
                "as_of": as_of.isoformat(),
                "classification": classification,
                "level": level,
            },
            raw_payload=payload,
            available_at=self._available_at(as_of, _MASTER_DATA_CUTOFF),
            records=tuple(records),
        )

    def fetch_industry_memberships(
        self,
        as_of: date,
        *,
        classification: str,
        l3_codes: Sequence[str],
    ) -> ProviderBatch[IndustryMembership]:
        if not classification:
            raise ValueError("classification must be non-empty")
        normalized_l3_codes = tuple(
            dict.fromkeys(code.removeprefix(f"{classification}:") for code in l3_codes if code)
        )
        if not normalized_l3_codes:
            raise ValueError("l3_codes must contain at least one code")
        endpoint = "index_member_all"
        fields = (
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "l3_code",
            "l3_name",
            "ts_code",
            "name",
            "in_date",
            "out_date",
            "is_new",
        )
        responses: list[dict[str, Any]] = []
        records_by_key: dict[tuple[str, str, date, date | None], IndustryMembership] = {}
        for l3_code in normalized_l3_codes:
            for is_new in ("Y", "N"):
                params = {"l3_code": l3_code, "is_new": is_new}
                payload = self._call(endpoint, params, fields)
                responses.append({"params": params, "payload": payload})
                for row in self._rows(
                    payload,
                    endpoint=endpoint,
                    row_limit=_INDEX_MEMBER_ALL_LIMIT,
                ):
                    effective_from = self._parse_date(
                        row.get("in_date"),
                        endpoint=endpoint,
                        field="in_date",
                    )
                    if effective_from > as_of:
                        continue
                    out_date = str(row.get("out_date") or "")
                    effective_to = (
                        self._parse_date(
                            out_date,
                            endpoint=endpoint,
                            field="out_date",
                        )
                        if out_date
                        else None
                    )
                    if effective_to is not None and effective_to > as_of:
                        effective_to = None
                    instrument_id = instrument_id_from_ts_code(str(row["ts_code"]))
                    for level_number in (1, 2, 3):
                        code = str(row.get(f"l{level_number}_code") or "")
                        if not code:
                            raise ProviderError(
                                self.provider_name,
                                endpoint,
                                f"provider row is missing l{level_number}_code",
                                retryable=False,
                            )
                        industry_id = f"{classification}:{code}"
                        key = (
                            instrument_id,
                            industry_id,
                            effective_from,
                            effective_to,
                        )
                        records_by_key[key] = IndustryMembership(
                            instrument_id=instrument_id,
                            industry_id=industry_id,
                            effective_from=effective_from,
                            effective_to=effective_to,
                            source=self.provider_name,
                            version=classification,
                        )
        return ProviderBatch(
            provider=self.provider_name,
            endpoint=endpoint,
            request_params={
                "as_of": as_of.isoformat(),
                "classification": classification,
                "l3_codes": list(normalized_l3_codes),
            },
            raw_payload={"responses": responses},
            available_at=self._available_at(as_of, _MASTER_DATA_CUTOFF),
            records=tuple(records_by_key.values()),
        )
