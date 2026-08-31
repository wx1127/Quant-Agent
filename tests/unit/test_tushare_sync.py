"""Unit tests for raw-first Tushare dataset sources."""

import json
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pytest

from quant_agent.data.domain import InstrumentType
from quant_agent.data.providers import ProviderError, TushareHttpProvider
from quant_agent.data.tushare_sync import (
    TushareAdjustmentFactorSource,
    TushareDailyBarSource,
    TushareInstrumentSource,
    TushareTradingCalendarSource,
)

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 30, 18, 1, tzinfo=TZ)


def _payload(fields: list[str], items: list[list[object]]) -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {"fields": fields, "items": items},
    }


@pytest.mark.parametrize(
    ("instrument_type", "instrument_id", "endpoint", "available_hour"),
    [
        (InstrumentType.STOCK, "CN.SZ.000001", "daily", 16),
        (InstrumentType.ETF, "CN.SH.510300", "fund_daily", 17),
        (InstrumentType.INDEX, "CN.CSI.000300", "index_daily", 17),
    ],
)
def test_daily_source_fetches_tokenless_raw_and_replays_all_asset_types(
    instrument_type: InstrumentType,
    instrument_id: str,
    endpoint: str,
    available_hour: int,
) -> None:
    ts_code = ".".join(reversed(instrument_id.split(".")[1:]))
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        assert body["api_name"] == endpoint
        assert body["token"] == "secret"
        assert body["params"] == {"trade_date": "20260730", "ts_code": ts_code}
        return httpx.Response(
            200,
            json=_payload(
                [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                ],
                [[ts_code, "20260730", 10, 11, 9, 10.5, 12, 34]],
            ),
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    source = TushareDailyBarSource(
        provider,
        instrument_type=instrument_type,
        instrument_ids=[instrument_id],
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"trade_date": "2026-07-29"},
        {"trade_date": "2026-07-30"},
    )[0]

    request = source.page_request(partition, None)
    raw = source.fetch_page(partition, None)
    decoded = source.decode(raw)

    assert "token" not in request.request_params
    assert "token" not in raw.request_params
    assert raw.cursor_out is None and raw.has_more is False
    assert decoded.records[0].instrument_id == instrument_id
    assert decoded.records[0].volume == Decimal("1200")
    assert decoded.records[0].turnover == Decimal("34000")
    assert decoded.records[0].available_at.hour == available_hour
    assert requests == 1

    offline = TushareDailyBarSource(
        TushareHttpProvider(
            "rotated-secret",
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: pytest.fail("raw replay must not use HTTP")
                )
            ),
        ),
        instrument_type=instrument_type,
        instrument_ids=[instrument_id],
    )
    assert offline.decode(raw).records == decoded.records


def test_calendar_master_and_adjustment_sources_have_explicit_partitions() -> None:
    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("partition planning must not use HTTP")
            )
        ),
    )
    calendar = TushareTradingCalendarSource(provider, market="SSE")
    assert calendar.partitions(
        {"cal_date": "2026-07-28"},
        {"cal_date": "2026-07-30"},
    ) == (
        {"cal_date": "2026-07-29", "market": "SSE"},
        {"cal_date": "2026-07-30", "market": "SSE"},
    )

    stock = TushareInstrumentSource(
        provider,
        instrument_type=InstrumentType.STOCK,
        market="主板",
    )
    partitions = stock.partitions(
        {"as_of": "2026-07-29"},
        {"as_of": "2026-07-30"},
    )
    assert [partition["list_status"] for partition in partitions] == ["L", "D", "P"]
    assert stock.partition_watermark(partitions[0]) == {
        "as_of": "2026-07-30",
        "list_status": "L",
    }
    assert stock.partition_watermark(partitions[-1]) == {"as_of": "2026-07-30"}
    assert stock.page_request(partitions[0], None).request_params == {
        "list_status": "L",
        "market": "主板",
    }

    etf = TushareInstrumentSource(
        provider,
        instrument_type=InstrumentType.ETF,
        market="SSE",
    )
    etf_partition = etf.partitions(
        {"as_of": "2026-07-29"},
        {"as_of": "2026-07-30"},
    )[0]
    assert etf.page_request(etf_partition, None).endpoint == "etf_basic"
    assert etf.page_request(etf_partition, None).request_params["exchange"] == "SH"

    index = TushareInstrumentSource(
        provider,
        instrument_type=InstrumentType.INDEX,
        market="CSI",
    )
    index_partition = index.partitions(
        {"as_of": "2026-07-29"},
        {"as_of": "2026-07-30"},
    )[0]
    assert index.page_request(index_partition, None).endpoint == "index_basic"
    assert index.page_request(index_partition, None).request_params == {"market": "CSI"}

    adjustment = TushareAdjustmentFactorSource(
        provider,
        instrument_ids=["CN.SZ.000001"],
    )
    adjustment_partition = adjustment.partitions(
        {"trade_date": "2026-07-29"},
        {"trade_date": "2026-07-30"},
    )[0]
    assert adjustment.page_request(adjustment_partition, None).request_params == {
        "trade_date": "20260730",
        "ts_code": "000001.SZ",
    }
    with pytest.raises(ValueError, match="pagination"):
        adjustment.page_request(adjustment_partition, {"offset": 1})


@pytest.mark.parametrize(
    ("instrument_type", "market", "endpoint", "fields", "item", "instrument_id"),
    [
        (
            InstrumentType.ETF,
            "SSE",
            "etf_basic",
            ["ts_code", "csname", "extname", "list_date", "list_status", "exchange"],
            ["510300.SH", "沪深300ETF", "300ETF", "20120528", "L", "SH"],
            "CN.SH.510300",
        ),
        (
            InstrumentType.INDEX,
            "CSI",
            "index_basic",
            ["ts_code", "name", "market", "list_date", "exp_date"],
            ["000300.CSI", "沪深300", "CSI", "20050408", None],
            "CN.CSI.000300",
        ),
    ],
)
def test_master_source_fetches_and_replays_etf_and_index_snapshots(
    instrument_type: InstrumentType,
    market: str,
    endpoint: str,
    fields: list[str],
    item: list[object],
    instrument_id: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == endpoint
        assert body["token"] == "secret"
        return httpx.Response(200, json=_payload(fields, [item]))

    source = TushareInstrumentSource(
        TushareHttpProvider(
            "secret",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        instrument_type=instrument_type,
        market=market,
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"as_of": "2026-07-29"},
        {"as_of": "2026-07-30"},
    )[0]
    raw = source.fetch_page(partition, None)
    decoded = source.decode(raw)

    assert raw.available_at.hour == 18
    assert decoded.records[0].instrument_id == instrument_id
    assert decoded.records[0].instrument_type is instrument_type


def test_adjustment_source_preserves_conservative_point_in_time_cutoff() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_payload(
                ["ts_code", "trade_date", "adj_factor"],
                [["000001.SZ", "20260730", 123.456]],
            ),
        )

    source = TushareAdjustmentFactorSource(
        TushareHttpProvider(
            "secret",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        instrument_ids=["CN.SZ.000001"],
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"trade_date": "2026-07-29"},
        {"trade_date": "2026-07-30"},
    )[0]
    decoded = source.decode(source.fetch_page(partition, None))

    assert decoded.records[0].factor == Decimal("123.456")
    assert decoded.records[0].available_at == datetime(2026, 7, 30, 9, 30, tzinfo=TZ)


def test_documented_limit_fails_during_decode_without_fake_pagination() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_payload([], [[] for _ in range(6000)]),
        )

    source = TushareDailyBarSource(
        TushareHttpProvider(
            "secret",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        instrument_type=InstrumentType.STOCK,
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"trade_date": "2026-07-29"},
        {"trade_date": "2026-07-30"},
    )[0]

    raw = source.fetch_page(partition, None)

    assert raw.record_count == 6000
    assert raw.has_more is False
    with pytest.raises(ProviderError, match="6000-row limit"):
        source.decode(raw)


def test_source_validates_watermarks_and_index_asset_scope() -> None:
    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: pytest.fail("no HTTP expected"))
        ),
    )
    with pytest.raises(ValueError, match="explicit market"):
        TushareInstrumentSource(provider, instrument_type=InstrumentType.INDEX)
    with pytest.raises(ValueError, match="exactly one"):
        TushareDailyBarSource(provider, instrument_type=InstrumentType.INDEX)
    source = TushareTradingCalendarSource(provider, market="SSE")
    with pytest.raises(ValueError, match="after target"):
        source.partitions(
            {"cal_date": "2026-07-31"},
            {"cal_date": "2026-07-30"},
        )
