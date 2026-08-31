"""Tests for the operational Tushare synchronization boundary."""

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from quant_agent.data.database import Database
from quant_agent.data.domain import InstrumentType
from quant_agent.data.providers import TushareHttpProvider
from quant_agent.data.sync import SyncMode, SyncRunState
from quant_agent.data.tushare_sync import (
    TushareAdjustmentFactorSource,
    TushareAnnouncementSource,
    TushareDailyBarSource,
    TushareInstrumentSource,
)
from quant_agent.pipelines.tushare_sync import (
    TushareDataset,
    TushareSyncSpec,
    _components,
    run_tushare_sync,
    sync_result_dict,
)


def _payload(fields: list[str], items: list[list[object]]) -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {"fields": fields, "items": items},
    }


def test_sync_spec_normalizes_identity_without_credentials() -> None:
    spec = TushareSyncSpec(
        dataset=TushareDataset.DAILY,
        start=date(2026, 8, 27),
        end=date(2026, 8, 28),
        instrument_ids=(" CN.SZ.000002 ", "CN.SZ.000001", "CN.SZ.000002"),
        market=" SSE ",
        mode=SyncMode.BACKFILL,
        idempotency_key=" stable-key ",
    )

    assert spec.market == "SSE"
    assert spec.instrument_ids == ("CN.SZ.000001", "CN.SZ.000002")
    assert spec.idempotency_key == "stable-key"
    assert spec.identity_payload() == {
        "dataset": "daily",
        "end": "2026-08-28",
        "instrument_ids": ["CN.SZ.000001", "CN.SZ.000002"],
        "instrument_type": "STOCK",
        "market": "SSE",
        "mode": "BACKFILL",
        "start": "2026-08-27",
    }


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.CALENDAR,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
            ),
            "requires --market",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.DAILY,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_type=InstrumentType.INDEX,
            ),
            "exactly one instrument_id",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.ADJUSTMENT,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_type=InstrumentType.ETF,
            ),
            "supports STOCK only",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.ANNOUNCEMENT,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_type=InstrumentType.ETF,
            ),
            "announcement sync supports STOCK only",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.ANNOUNCEMENT,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                market="SSE",
            ),
            "does not accept --market",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.DAILY,
                start=date(2026, 8, 29),
                end=date(2026, 8, 28),
            ),
            "start cannot be after end",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.INSTRUMENT,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_type=InstrumentType.INDEX,
            ),
            "index instrument sync requires --market",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.DAILY,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                market=" ",
            ),
            "market cannot be empty",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.DAILY,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_ids=("",),
            ),
            "instrument_ids cannot contain empty",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.DAILY,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                idempotency_key=" ",
            ),
            "idempotency_key must contain",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.CALENDAR,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                market="SSE",
                instrument_ids=("CN.SZ.000001",),
            ),
            "calendar sync does not accept instrument_ids",
        ),
        (
            lambda: TushareSyncSpec(
                dataset=TushareDataset.INSTRUMENT,
                start=date(2026, 8, 28),
                end=date(2026, 8, 28),
                instrument_ids=("CN.SZ.000001",),
            ),
            "instrument sync does not support instrument_id filtering",
        ),
    ],
)
def test_sync_spec_rejects_ambiguous_or_invalid_requests(
    spec: Callable[[], TushareSyncSpec],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        spec()


def test_component_selection_preserves_dataset_scope() -> None:
    provider = TushareHttpProvider(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("component selection must not perform I/O")
            )
        ),
    )
    instrument = _components(
        TushareSyncSpec(
            dataset=TushareDataset.INSTRUMENT,
            start=date(2026, 8, 28),
            end=date(2026, 8, 28),
        ),
        provider,
    )
    daily = _components(
        TushareSyncSpec(
            dataset=TushareDataset.DAILY,
            start=date(2026, 8, 28),
            end=date(2026, 8, 28),
            instrument_ids=("CN.SZ.000001",),
        ),
        provider,
    )
    adjustment = _components(
        TushareSyncSpec(
            dataset=TushareDataset.ADJUSTMENT,
            start=date(2026, 8, 28),
            end=date(2026, 8, 28),
            instrument_ids=("CN.SZ.000001",),
        ),
        provider,
    )
    announcement = _components(
        TushareSyncSpec(
            dataset=TushareDataset.ANNOUNCEMENT,
            start=date(2026, 8, 28),
            end=date(2026, 8, 28),
            instrument_ids=("CN.SZ.000001",),
        ),
        provider,
    )

    assert isinstance(instrument[0], TushareInstrumentSource)
    assert instrument[2:] == (
        "as_of",
        {"instrument_type": "STOCK", "market": None},
        {},
    )
    assert isinstance(daily[0], TushareDailyBarSource)
    assert daily[2] == "trade_date"
    assert daily[3]["market"] == "SSE"
    assert isinstance(adjustment[0], TushareAdjustmentFactorSource)
    assert adjustment[3]["instrument_type"] == "STOCK"
    assert isinstance(announcement[0], TushareAnnouncementSource)
    assert announcement[2] == "ann_date"
    assert announcement[3] == {
        "instrument_ids": ["CN.SZ.000001"],
        "instrument_type": "STOCK",
    }
    assert announcement[4]["classifier_version"] == "announcement-title-rules-v1"


@pytest.mark.parametrize("token", [None, ""])
def test_sync_requires_token_when_provider_is_not_injected(token: str | None) -> None:
    spec = TushareSyncSpec(
        dataset=TushareDataset.CALENDAR,
        start=date(2026, 8, 28),
        end=date(2026, 8, 28),
        market="SSE",
    )

    with pytest.raises(ValueError, match="market data token is required"):
        run_tushare_sync(
            spec=spec,
            database_url="sqlite:///:memory:",
            market_data_token=None if token is None else SecretStr(token),
        )


def test_calendar_pipeline_runs_raw_first_and_reuses_idempotent_result(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'calendar.db').as_posix()}"
    Database(database_url).create_schema()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["api_name"] == "trade_cal"
        assert body["token"] == "secret"
        assert body["params"] == {
            "exchange": "SSE",
            "start_date": "20260828",
            "end_date": "20260828",
        }
        return httpx.Response(
            200,
            json=_payload(
                ["exchange", "cal_date", "is_open", "pretrade_date"],
                [["SSE", "20260828", "1", "20260827"]],
            ),
        )

    spec = TushareSyncSpec(
        dataset=TushareDataset.CALENDAR,
        start=date(2026, 8, 28),
        end=date(2026, 8, 28),
        market="SSE",
    )
    with TushareHttpProvider(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as provider:
        first = run_tushare_sync(
            spec=spec,
            database_url=database_url,
            provider=provider,
        )
        second = run_tushare_sync(
            spec=spec,
            database_url=database_url,
            provider=provider,
        )

    assert first.state is SyncRunState.SUCCEEDED
    assert first.inserted == 1
    assert second.run_id == first.run_id
    assert second.fetched_pages == 0
    assert calls == 1
    assert "secret" not in json.dumps(sync_result_dict(first), sort_keys=True)
