"""Boundary tests for the deterministic paginated synchronization source."""

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.sync import (
    FakeDatasetSource,
    FakePageSpec,
    FakePartitionSpec,
)

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 28, 18, tzinfo=TZ)
PARTITION = {"trade_date": "2026-08-28"}
WATERMARK = {"trade_date": "2026-08-28"}


def _page() -> FakePageSpec:
    return FakePageSpec(
        page_ordinal=0,
        cursor_in=None,
        cursor_out=None,
        records=({"id": "one"},),
    )


def _source() -> FakeDatasetSource:
    return FakeDatasetSource(
        dataset="records",
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=WATERMARK,
                pages=(_page(),),
            ),
        ),
        available_at=NOW,
        clock=lambda: NOW,
    )


def test_fake_specifications_reject_broken_page_chains() -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(_page(), page_ordinal=-1)
    with pytest.raises(ValueError, match="at least one"):
        FakePartitionSpec(partition=PARTITION, watermark=WATERMARK, pages=())
    with pytest.raises(ValueError, match="contiguous"):
        FakePartitionSpec(
            partition=PARTITION,
            watermark=WATERMARK,
            pages=(replace(_page(), page_ordinal=1),),
        )
    with pytest.raises(ValueError, match="final"):
        FakePartitionSpec(
            partition=PARTITION,
            watermark=WATERMARK,
            pages=(replace(_page(), cursor_out={"offset": 1}),),
        )
    with pytest.raises(ValueError, match="dataset"):
        FakeDatasetSource(
            dataset="",
            partitions=(),
            available_at=NOW,
        )


def test_fake_partition_planning_handles_completed_and_missing_targets() -> None:
    source = _source()

    assert source.partitions(WATERMARK, WATERMARK) == ()
    assert source.partitions({"trade_date": "2026-08-27"}, WATERMARK) == (PARTITION,)
    with pytest.raises(ValueError, match="target watermark"):
        source.partitions({"trade_date": "2026-08-27"}, {"trade_date": "2026-08-29"})


def test_fake_source_rejects_unknown_partition_cursor_and_malformed_payload() -> None:
    source = _source()
    raw = source.fetch_page(PARTITION, None)

    with pytest.raises(ValueError, match="partition"):
        source.partition_watermark({"trade_date": "1900-01-01"})
    with pytest.raises(ValueError, match="cursor"):
        source.page_request(PARTITION, {"offset": 999})
    with pytest.raises(ValueError, match="raw page cursor"):
        source.decode(replace(raw, cursor_in={"offset": 999}))
    with pytest.raises(ValueError, match="records must be objects"):
        source.decode(replace(raw, payload={"records": ["invalid"]}))
