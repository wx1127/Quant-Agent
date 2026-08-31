"""Raw-first Tushare announcement source and PIT event decoding tests."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from quant_agent.data.events import EventTrustLevel, EventType
from quant_agent.data.providers import ProviderError, TushareHttpProvider
from quant_agent.data.tushare_sync import TushareAnnouncementSource

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 30, 18, 1, tzinfo=TZ)
FIELDS = ["ann_date", "ts_code", "name", "title", "url", "rec_time"]


def _payload(items: list[list[object]]) -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {"fields": FIELDS, "items": items},
    }


def _provider(handler: httpx.MockTransport) -> TushareHttpProvider:
    return TushareHttpProvider("secret", client=httpx.Client(transport=handler))


def test_filtered_partitions_resume_inside_one_date_without_skipping_instruments() -> None:
    provider = _provider(
        httpx.MockTransport(lambda _request: pytest.fail("partition planning must stay offline"))
    )
    source = TushareAnnouncementSource(
        provider,
        instrument_ids=("CN.SZ.000001", "CN.SH.600000"),
    )

    partitions = source.partitions(
        {"ann_date": "2026-07-29"},
        {"ann_date": "2026-07-30"},
    )

    assert partitions == (
        {"ann_date": "2026-07-30", "instrument_id": "CN.SH.600000"},
        {"ann_date": "2026-07-30", "instrument_id": "CN.SZ.000001"},
    )
    assert source.partition_watermark(partitions[0]) == {
        "ann_date": "2026-07-30",
        "instrument_id": "CN.SH.600000",
    }
    assert source.partition_watermark(partitions[1]) == {"ann_date": "2026-07-30"}
    assert source.partitions(
        {
            "ann_date": "2026-07-30",
            "instrument_id": "CN.SH.600000",
        },
        {"ann_date": "2026-07-30"},
    ) == (partitions[1],)
    assert source.page_request(partitions[0], None).request_params == {
        "ann_date": "20260730",
        "ts_code": "600000.SH",
    }


def test_announcement_fetch_decodes_exact_raw_row_and_replays_offline() -> None:
    item = [
        "20260730",
        "000001.SZ",
        "平安银行",
        "关于收到监管问询函的公告",
        "https://example.test/notice.pdf",
        "2026-07-30 09:31:00",
    ]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body == {
            "api_name": "anns_d",
            "fields": ",".join(FIELDS),
            "params": {"ann_date": "20260730", "ts_code": "000001.SZ"},
            "token": "secret",
        }
        return httpx.Response(200, json=_payload([item]))

    source = TushareAnnouncementSource(
        _provider(httpx.MockTransport(handler)),
        instrument_ids=("CN.SZ.000001",),
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"ann_date": "2026-07-29"},
        {"ann_date": "2026-07-30"},
    )[0]
    request = source.page_request(partition, None)
    raw = source.fetch_page(partition, None)
    decoded = source.decode(raw)

    assert calls == 1
    assert "token" not in request.request_params
    assert "token" not in raw.request_params
    assert len(decoded.records) == 1
    event = decoded.records[0]
    assert event.event_type is EventType.REGULATORY
    assert event.instrument_ids == ("CN.SZ.000001",)
    assert event.published_at == datetime(2026, 7, 30, 9, 31, tzinfo=TZ)
    assert event.observed_at == NOW
    assert event.available_at == NOW
    assert event.source_path == "/data/items/0"
    assert event.source_url == "https://example.test/notice.pdf"
    assert event.trust_level is EventTrustLevel.UNTRUSTED_SOURCE
    assert len(event.event_id) == len(event.content_hash) == len(event.version) == 64

    offline = TushareAnnouncementSource(
        _provider(
            httpx.MockTransport(
                lambda _request: pytest.fail("archived announcement replay must not use HTTP")
            )
        ),
        instrument_ids=("CN.SZ.000001",),
    )
    assert offline.decode(raw) == decoded


def test_full_date_partition_and_documented_limit_fail_closed() -> None:
    source = TushareAnnouncementSource(
        _provider(
            httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json=_payload(
                        [
                            [
                                "20260730",
                                "000001.SZ",
                                "平安银行",
                                "公告",
                                "https://example.test/notice.pdf",
                                "2026-07-30 09:31:00",
                            ]
                            for _ in range(2000)
                        ]
                    ),
                )
            )
        ),
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"ann_date": "2026-07-29"},
        {"ann_date": "2026-07-30"},
    )[0]
    assert source.page_request(partition, None).request_params == {"ann_date": "20260730"}
    raw = source.fetch_page(partition, None)

    with pytest.raises(ProviderError, match="2000-row limit"):
        source.decode(raw)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            [
                "20260729",
                "000001.SZ",
                "平安银行",
                "公告",
                "https://example.test/a.pdf",
                "2026-07-29 10:00:00",
            ],
            "outside its date partition",
        ),
        (
            [
                "20260730",
                "000001.SZ",
                "平安银行",
                "公告",
                "https://example.test/a.pdf",
                "not-a-time",
            ],
            "invalid anns_d rec_time",
        ),
        (
            [
                "20260730",
                "600000.SH",
                "浦发银行",
                "公告",
                "https://example.test/a.pdf",
                "2026-07-30 10:00:00",
            ],
            "outside its partition scope",
        ),
    ],
)
def test_decode_rejects_partition_and_time_mismatches(
    item: list[object],
    message: str,
) -> None:
    source = TushareAnnouncementSource(
        _provider(httpx.MockTransport(lambda _request: httpx.Response(200, json=_payload([item])))),
        instrument_ids=("CN.SZ.000001",),
        clock=lambda: NOW,
    )
    partition = source.partitions(
        {"ann_date": "2026-07-29"},
        {"ann_date": "2026-07-30"},
    )[0]

    with pytest.raises(ValueError, match=message):
        source.decode(source.fetch_page(partition, None))
