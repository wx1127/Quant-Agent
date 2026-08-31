"""Tests for provider contracts and the Tushare HTTP adapter."""

import json
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pytest

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndexConstituentWeight,
    Industry,
    IndustryMembership,
    InstrumentStatus,
    InstrumentType,
)
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
    stock_statuses: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        api_name = body["api_name"]
        if api_name == "stock_basic":
            list_status = body["params"]["list_status"]
            stock_statuses.append(list_status)
            items = {
                "L": [
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
                "D": [
                    [
                        "000002.SZ",
                        "000002",
                        "退市样本",
                        "深圳",
                        "地产",
                        "主板",
                        "19910129",
                        "20200101",
                        "D",
                    ]
                ],
                "P": [
                    [
                        "000003.SZ",
                        "000003",
                        "暂停样本",
                        "深圳",
                        "综合",
                        "主板",
                        "19910703",
                        None,
                        "P",
                    ],
                    [
                        "000004.SZ",
                        "000004",
                        "缺日期样本",
                        "深圳",
                        "综合",
                        "主板",
                        None,
                        None,
                        "P",
                    ],
                ],
            }[list_status]
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
                        "items": items,
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
    calendar = provider.fetch_trading_calendar(
        "SSE",
        date(2026, 7, 30),
        date(2026, 7, 30),
    )

    assert instruments.records[0].instrument_id == "CN.SZ.000001"
    assert instruments.records[0].listed_on == date(1991, 4, 3)
    assert stock_statuses == ["L", "D", "P"]
    assert [item.status for item in instruments.records] == [
        InstrumentStatus.LISTED,
        InstrumentStatus.DELISTED,
        InstrumentStatus.SUSPENDED,
    ]
    assert instruments.raw_payload["metadata"] == {"skipped_missing_list_date": 1}
    assert calendar.records[0].is_open is True


def test_tushare_etf_master_uses_etf_basic_and_filters_non_listed_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "etf_basic"
        assert body["params"] == {"list_status": "L", "exchange": "SH"}
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "ts_code",
                        "csname",
                        "extname",
                        "list_date",
                        "list_status",
                        "exchange",
                    ],
                    "items": [
                        ["510300.SH", "沪深300ETF", "300ETF", "20120528", "L", "SH"],
                        ["510301.SH", "已退市基金", "退市基金", "20120528", "D", "SH"],
                        ["510302.SH", "待上市基金", "待上市", None, "L", "SH"],
                    ],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_instruments(
        date(2026, 7, 30),
        instrument_type=InstrumentType.ETF,
        market="SSE",
    )

    assert batch.endpoint == "etf_basic"
    assert len(batch.records) == 1
    assert batch.records[0].instrument_id == "CN.SH.510300"
    assert batch.records[0].instrument_type is InstrumentType.ETF
    assert batch.records[0].name == "300ETF"
    assert batch.records[0].listed_on == date(2012, 5, 28)
    assert batch.available_at.hour == 18


def test_tushare_index_master_requires_and_passes_market() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        assert body["api_name"] == "index_basic"
        assert body["params"] == {"market": "CSI"}
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": ["ts_code", "name", "market", "list_date", "exp_date"],
                    "items": [["000300.CSI", "沪深300", "CSI", "20050408", None]],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="explicit market"):
        provider.fetch_instruments(
            date(2026, 7, 30),
            instrument_type=InstrumentType.INDEX,
        )

    batch = provider.fetch_instruments(
        date(2026, 7, 30),
        instrument_type=InstrumentType.INDEX,
        market="CSI",
    )

    assert requests == 1
    assert batch.records[0].instrument_id == "CN.CSI.000300"
    assert batch.records[0].instrument_type is InstrumentType.INDEX
    assert batch.records[0].listed_on == date(2005, 4, 8)
    assert batch.available_at.hour == 18


@pytest.mark.parametrize(
    ("instrument_type", "endpoint", "instrument_id", "ts_code"),
    [
        (InstrumentType.ETF, "fund_daily", "CN.SH.510300", "510300.SH"),
        (InstrumentType.INDEX, "index_daily", "CN.CSI.000300", "000300.CSI"),
    ],
)
def test_tushare_multi_asset_daily_mapping_normalizes_units(
    instrument_type: InstrumentType,
    endpoint: str,
    instrument_id: str,
    ts_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == endpoint
        assert body["params"] == {"trade_date": "20260730", "ts_code": ts_code}
        return httpx.Response(
            200,
            json={
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
                    "items": [[ts_code, "20260730", 10, 11, 9, 10.5, 12, 34]],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_daily_bars(
        date(2026, 7, 30),
        [instrument_id],
        instrument_type=instrument_type,
    )

    assert batch.endpoint == endpoint
    assert batch.records[0].instrument_id == instrument_id
    assert batch.records[0].volume == Decimal("1200")
    assert batch.records[0].turnover == Decimal("34000")
    assert batch.records[0].available_at.hour == 17


def test_tushare_index_daily_requires_exactly_one_instrument() -> None:
    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("index_daily must fail before HTTP")
            )
        ),
    )

    with pytest.raises(ProviderError, match="exactly one"):
        provider.fetch_daily_bars(
            date(2026, 7, 30),
            instrument_type=InstrumentType.INDEX,
        )
    with pytest.raises(ProviderError, match="exactly one"):
        provider.fetch_daily_bars(
            date(2026, 7, 30),
            ["CN.CSI.000300", "CN.SH.000001"],
            instrument_type=InstrumentType.INDEX,
        )


def test_tushare_adjustment_factor_mapping_and_asset_type_guard() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        assert body["api_name"] == "adj_factor"
        assert body["params"] == {
            "trade_date": "20260730",
            "ts_code": "000001.SZ",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": ["ts_code", "trade_date", "adj_factor"],
                    "items": [["000001.SZ", "20260730", 123.456]],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_adjustment_factors(
        date(2026, 7, 30),
        ["CN.SZ.000001"],
    )

    assert batch.records[0].factor == Decimal("123.456")
    assert batch.records[0].available_at.hour == 9
    assert batch.records[0].available_at.minute == 30
    with pytest.raises(ProviderError, match="STOCK instruments only"):
        provider.fetch_adjustment_factors(
            date(2026, 7, 30),
            instrument_type=InstrumentType.ETF,
        )
    assert requests == 1


def test_tushare_index_weight_mapping_keeps_percentage_units() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "index_weight"
        assert body["params"] == {
            "index_code": "000300.SH",
            "start_date": "20180901",
            "end_date": "20180930",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": ["index_code", "con_code", "trade_date", "weight"],
                    "items": [["000300.SH", "000001.SZ", "20180903", 0.8656]],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_index_constituent_weights(
        "CN.SH.000300",
        date(2018, 9, 1),
        date(2018, 9, 30),
    )

    assert batch.endpoint == "index_weight"
    assert batch.records[0].index_instrument_id == "CN.SH.000300"
    assert batch.records[0].constituent_instrument_id == "CN.SZ.000001"
    assert batch.records[0].trade_date == date(2018, 9, 3)
    assert batch.records[0].weight_percent == Decimal("0.8656")
    assert batch.records[0].available_at.hour == 18


def test_tushare_fundamentals_are_point_in_time_and_skip_nulls() -> None:
    requested_codes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "fina_indicator"
        assert body["params"]["start_date"] == "20250101"
        assert body["params"]["end_date"] == "20260730"
        ts_code = body["params"]["ts_code"]
        requested_codes.append(ts_code)
        fields = body["fields"].split(",")
        first_values = {
            "ts_code": ts_code,
            "ann_date": "20260729",
            "end_date": "20260630",
            "eps": 1.25 if ts_code == "000001.SZ" else None,
            "roe": None if ts_code == "000001.SZ" else 12.5,
            "update_flag": "1",
        }
        items = [[first_values.get(field) for field in fields]]
        if ts_code == "000001.SZ":
            future_values = {
                "ts_code": ts_code,
                "ann_date": "20260730",
                "end_date": "20260630",
                "eps": 9.99,
                "update_flag": "1",
            }
            items.append([future_values.get(field) for field in fields])
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {"fields": fields, "items": items},
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    as_of = datetime(2026, 7, 30, 10, tzinfo=TZ)

    batch = provider.fetch_fundamentals(
        ["CN.SZ.000001", "CN.SH.600000"],
        as_of,
        start_date=date(2025, 1, 1),
    )

    assert requested_codes == ["000001.SZ", "600000.SH"]
    assert {(item.instrument_id, item.metric_name) for item in batch.records} == {
        ("CN.SZ.000001", "eps"),
        ("CN.SH.600000", "roe"),
    }
    eps = next(item for item in batch.records if item.metric_name == "eps")
    assert eps.metric_value == Decimal("1.25")
    assert eps.announced_at.date() == date(2026, 7, 29)
    assert eps.available_at == datetime(2026, 7, 30, 9, 30, tzinfo=TZ)
    assert eps.provider_revision.startswith("1:")
    assert len(eps.provider_revision) == 66


def test_tushare_fundamentals_fail_at_documented_row_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fields = json.loads(request.content)["fields"].split(",")
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": fields,
                    "items": [[None for _ in fields] for _ in range(100)],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="100-row limit"):
        provider.fetch_fundamentals(
            ["CN.SZ.000001"],
            datetime(2026, 7, 30, 10, tzinfo=TZ),
            start_date=date(2025, 1, 1),
        )


def test_tushare_industry_hierarchy_resolves_parent_by_industry_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "index_classify"
        assert body["params"] == {"src": "SW2021"}
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "index_code",
                        "industry_name",
                        "parent_code",
                        "level",
                        "industry_code",
                        "is_pub",
                        "src",
                    ],
                    "items": [
                        ["801050.SI", "有色金属", "0", "L1", "240000", "1", "SW2021"],
                        ["801053.SI", "贵金属", "240000", "L2", "240400", "1", "SW2021"],
                        ["850531.SI", "黄金", "240400", "L3", "240401", "1", "SW2021"],
                    ],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_industries(date(2026, 7, 30), "SW2021", level=2)

    assert len(batch.records) == 1
    industry = batch.records[0]
    assert industry.industry_id == "SW2021:801053.SI"
    assert industry.classification == "SW2021"
    assert industry.code == "240400"
    assert industry.parent_id == "SW2021:801050.SI"
    assert industry.level == 2


def test_tushare_industry_hierarchy_fails_on_unknown_parent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "index_code",
                        "industry_name",
                        "parent_code",
                        "level",
                        "industry_code",
                        "is_pub",
                        "src",
                    ],
                    "items": [
                        [
                            "801053.SI",
                            "贵金属",
                            "240000",
                            "L2",
                            "240400",
                            "1",
                            "SW2021",
                        ]
                    ],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="cannot resolve parent_code"):
        provider.fetch_industries(date(2026, 7, 30), "SW2021")


def test_tushare_industry_memberships_fetch_history_and_deduplicate() -> None:
    requests: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "index_member_all"
        requests.append(body["params"])
        fields = body["fields"].split(",")

        def item(in_date: str, out_date: str | None, is_new: str) -> list[object]:
            values = {
                "l1_code": "801050.SI",
                "l1_name": "有色金属",
                "l2_code": "801053.SI",
                "l2_name": "贵金属",
                "l3_code": "850531.SI",
                "l3_name": "黄金",
                "ts_code": "600988.SH",
                "name": "赤峰黄金",
                "in_date": in_date,
                "out_date": out_date,
                "is_new": is_new,
            }
            return [values.get(field) for field in fields]

        current = item("20220729", None, "Y")
        items = (
            [current]
            if body["params"]["is_new"] == "Y"
            else [current, item("20100101", "20211212", "N")]
        )
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {"fields": fields, "items": items},
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    batch = provider.fetch_industry_memberships(
        date(2026, 7, 30),
        classification="SW2021",
        l3_codes=["SW2021:850531.SI"],
    )

    assert requests == [
        {"l3_code": "850531.SI", "is_new": "Y"},
        {"l3_code": "850531.SI", "is_new": "N"},
    ]
    assert len(batch.records) == 6
    assert {item.industry_id for item in batch.records} == {
        "SW2021:801050.SI",
        "SW2021:801053.SI",
        "SW2021:850531.SI",
    }
    assert {item.effective_to for item in batch.records} == {None, date(2021, 12, 12)}


def test_tushare_industry_memberships_fail_at_documented_fragment_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fields = json.loads(request.content)["fields"].split(",")
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": fields,
                    "items": [[None for _ in fields] for _ in range(2000)],
                },
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="2000-row limit"):
        provider.fetch_industry_memberships(
            date(2026, 7, 30),
            classification="SW2021",
            l3_codes=["850531.SI"],
        )


@pytest.mark.parametrize(
    ("endpoint", "row_limit", "instrument_type", "is_master"),
    [
        ("stock_basic", 6000, InstrumentType.STOCK, True),
        ("etf_basic", 5000, InstrumentType.ETF, True),
        ("index_basic", 8000, InstrumentType.INDEX, True),
        ("daily", 6000, InstrumentType.STOCK, False),
        ("fund_daily", 5000, InstrumentType.ETF, False),
    ],
)
def test_tushare_rejects_responses_at_documented_row_limit(
    endpoint: str,
    row_limit: int,
    instrument_type: InstrumentType,
    is_master: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["api_name"] == endpoint
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {"fields": [], "items": [[] for _ in range(row_limit)]},
            },
        )

    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="reaching the documented"):
        if is_master:
            provider.fetch_instruments(
                date(2026, 7, 30),
                instrument_type=instrument_type,
                market="CSI" if instrument_type is InstrumentType.INDEX else None,
            )
        else:
            provider.fetch_daily_bars(
                date(2026, 7, 30),
                instrument_type=instrument_type,
            )


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


def test_fake_provider_supports_second_phase_contracts() -> None:
    available_at = datetime(2026, 7, 30, 18, tzinfo=TZ)
    adjustment_factor = AdjustmentFactor(
        instrument_id="CN.SZ.000001",
        trade_date=date(2026, 7, 30),
        factor=Decimal("123.456"),
        source="fake",
        available_at=datetime(2026, 7, 30, 9, 30, tzinfo=TZ),
        version="v1",
    )
    index_weight = IndexConstituentWeight(
        index_instrument_id="CN.SH.000300",
        constituent_instrument_id="CN.SZ.000001",
        trade_date=date(2026, 7, 1),
        weight_percent=Decimal("1.5"),
        available_at=available_at,
        source="fake",
        version="v1",
    )
    fundamental = FundamentalPoint(
        instrument_id="CN.SZ.000001",
        report_period=date(2026, 6, 30),
        metric_name="roe",
        metric_value=Decimal("12.5"),
        announced_at=datetime(2026, 7, 29, tzinfo=TZ),
        available_at=datetime(2026, 7, 30, 9, 30, tzinfo=TZ),
        provider_revision="v1",
        source="fake",
    )
    industries = [
        Industry(
            industry_id="SW2021:801050.SI",
            classification="SW2021",
            code="240000",
            name="有色金属",
            level=1,
            version="SW2021",
        ),
        Industry(
            industry_id="SW2021:801053.SI",
            classification="SW2021",
            code="240400",
            name="贵金属",
            level=2,
            parent_id="SW2021:801050.SI",
            version="SW2021",
        ),
        Industry(
            industry_id="SW2021:850531.SI",
            classification="SW2021",
            code="240401",
            name="黄金",
            level=3,
            parent_id="SW2021:801053.SI",
            version="SW2021",
        ),
    ]
    memberships = [
        IndustryMembership(
            instrument_id="CN.SH.600988",
            industry_id=item.industry_id,
            effective_from=date(2022, 7, 29),
            source="fake",
            version="SW2021",
        )
        for item in industries
    ]
    provider = FakeMarketDataProvider(
        available_at=available_at,
        adjustment_factors=[adjustment_factor],
        index_constituent_weights=[index_weight],
        fundamentals=[fundamental],
        industries=industries,
        industry_memberships=memberships,
    )

    assert provider.fetch_index_constituent_weights(
        "CN.SH.000300", date(2026, 7, 1), date(2026, 7, 31)
    ).records == (index_weight,)
    assert provider.fetch_adjustment_factors(date(2026, 7, 30), ["CN.SZ.000001"]).records == (
        adjustment_factor,
    )
    with pytest.raises(ProviderError, match="STOCK instruments only"):
        provider.fetch_adjustment_factors(
            date(2026, 7, 30),
            instrument_type=InstrumentType.ETF,
        )
    assert provider.fetch_fundamentals(
        ["CN.SZ.000001"],
        available_at,
        start_date=date(2026, 1, 1),
    ).records == (fundamental,)
    assert provider.fetch_industries(date(2026, 7, 30), "SW2021", level=2).records == (
        industries[1],
    )
    assert (
        len(
            provider.fetch_industry_memberships(
                date(2026, 7, 30),
                classification="SW2021",
                l3_codes=["850531.SI"],
            ).records
        )
        == 3
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
