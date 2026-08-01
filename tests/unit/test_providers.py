"""Tests for provider contracts and the Tushare HTTP adapter."""

import json
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pytest

from quant_agent.data.domain import DailyBar
from quant_agent.data.providers import (
    FakeMarketDataProvider,
    ProviderError,
    ProviderRequestPolicy,
    TushareHttpProvider,
)
from quant_agent.data.providers.tushare import instrument_id_from_ts_code

TZ = ZoneInfo("Asia/Shanghai")


def _daily_payload() -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {
            "fields": [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
            "items": [
                ["000001.SZ", "20260730", 10, 11, 9, 10.5, 12, 34],
            ],
        },
    }


def test_tushare_daily_mapping_normalizes_units() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "daily"
        assert body["token"] == "secret"
        return httpx.Response(200, json=_daily_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TushareHttpProvider("secret", client=client)

    batch = provider.fetch_daily_bars(date(2026, 7, 30))

    assert batch.endpoint == "daily"
    assert batch.records[0].instrument_id == "CN.SZ.000001"
    assert batch.records[0].volume == Decimal("1200")
    assert batch.records[0].turnover == Decimal("34000")
    assert batch.records[0].available_at.tzinfo is not None


def test_tushare_retries_transport_errors_then_succeeds() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary")
        return httpx.Response(200, json=_daily_payload())

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        policy=ProviderRequestPolicy(max_attempts=3, initial_backoff_seconds=0.1),
        sleeper=delays.append,
    )

    assert len(provider.fetch_daily_bars(date(2026, 7, 30)).records) == 1
    assert delays == [0.1, 0.2]


def test_tushare_maps_instruments_and_trading_calendar() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        api_name = json.loads(request.content)["api_name"]
        if api_name == "stock_basic":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": None,
                    "data": {
                        "fields": [
                            "ts_code",
                            "symbol",
                            "name",
                            "area",
                            "industry",
                            "market",
                            "list_date",
                            "delist_date",
                            "list_status",
                        ],
                        "items": [
                            [
                                "000001.SZ",
                                "000001",
                                "平安银行",
                                "深圳",
                                "银行",
                                "主板",
                                "19910403",
                                None,
                                "L",
                            ]
                        ],
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "exchange",
                        "cal_date",
                        "is_open",
                        "pretrade_date",
                    ],
                    "items": [["SSE", "20260730", "1", "20260729"]],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    instruments = provider.fetch_instruments(date(2026, 7, 30))
    profiles = provider.fetch_stock_profiles(datetime(2026, 7, 30, 18, tzinfo=TZ))
    calendar = provider.fetch_trading_calendar(
        "SSE",
        date(2026, 7, 30),
        date(2026, 7, 30),
    )

    assert instruments.records[0].instrument_id == "CN.SZ.000001"
    assert instruments.records[0].listed_on == date(1991, 4, 3)
    assert profiles.records[0].name == "平安银行"
    assert profiles.records[0].industry == "银行"
    assert calendar.records[0].is_open is True


def test_tushare_provider_error_is_not_empty_data() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"code": -1, "msg": "bad token", "data": None},
            )
        )
    )
    provider = TushareHttpProvider("secret", client=client)

    with pytest.raises(ProviderError, match="bad token") as captured:
        provider.fetch_daily_bars(date(2026, 7, 30))

    assert captured.value.retryable is False


def test_fake_provider_filters_by_date_and_instrument() -> None:
    bar = DailyBar(
        instrument_id="CN.SZ.000001",
        trade_date=date(2026, 7, 30),
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("100"),
        turnover=Decimal("1000"),
        source="fake",
        available_at=datetime(2026, 7, 30, 16, tzinfo=TZ),
        version="v1",
    )
    provider = FakeMarketDataProvider(
        available_at=datetime(2026, 7, 30, 16, tzinfo=TZ),
        daily_bars=[bar],
    )

    assert len(provider.fetch_daily_bars(date(2026, 7, 30)).records) == 1
    assert (
        provider.fetch_daily_bars(
            date(2026, 7, 30),
            ["CN.SZ.999999"],
        ).records
        == ()
    )


def test_tushare_code_validation_and_policy_validation() -> None:
    with pytest.raises(ValueError, match="invalid"):
        instrument_id_from_ts_code("000001")
    with pytest.raises(ValueError, match="positive"):
        ProviderRequestPolicy(timeout_seconds=0)
    with pytest.raises(ValueError, match="at least one"):
        ProviderRequestPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        ProviderRequestPolicy(initial_backoff_seconds=-1)
