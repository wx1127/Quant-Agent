"""Operational orchestration for raw-first Tushare synchronization."""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

from pydantic import SecretStr

from quant_agent import __version__
from quant_agent.data.database import Database
from quant_agent.data.domain import InstrumentType
from quant_agent.data.providers import TushareHttpProvider
from quant_agent.data.sync import DatasetIngestor, DatasetSource, SyncMode
from quant_agent.data.sync.hashing import JsonObject, canonical_hash
from quant_agent.data.sync.runner import IncrementalSyncRunner, SyncExecutionResult
from quant_agent.data.tushare_sync import (
    ANNOUNCEMENT_CLASSIFIER_VERSION,
    TushareAdjustmentFactorIngestor,
    TushareAdjustmentFactorSource,
    TushareAnnouncementIngestor,
    TushareAnnouncementSource,
    TushareDailyBarIngestor,
    TushareDailyBarSource,
    TushareInstrumentIngestor,
    TushareInstrumentSource,
    TushareTradingCalendarIngestor,
    TushareTradingCalendarSource,
)


class TushareDataset(StrEnum):
    """Raw-first datasets exposed by the first operational sync command."""

    CALENDAR = "calendar"
    INSTRUMENT = "instrument"
    DAILY = "daily"
    ADJUSTMENT = "adjustment"
    ANNOUNCEMENT = "announcement"


@dataclass(frozen=True, slots=True)
class TushareSyncSpec:
    """Credential-free, versionable synchronization request."""

    dataset: TushareDataset
    start: date
    end: date
    instrument_type: InstrumentType = InstrumentType.STOCK
    market: str | None = None
    instrument_ids: tuple[str, ...] = ()
    mode: SyncMode = SyncMode.INCREMENTAL
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("sync start cannot be after end")
        normalized_market = self.market.strip() if self.market is not None else None
        if normalized_market == "":
            raise ValueError("sync market cannot be empty")
        object.__setattr__(self, "market", normalized_market)
        normalized_ids = tuple(value.strip() for value in self.instrument_ids)
        if any(not value for value in normalized_ids):
            raise ValueError("instrument_ids cannot contain empty values")
        object.__setattr__(
            self,
            "instrument_ids",
            tuple(sorted(set(normalized_ids))),
        )
        if self.dataset is TushareDataset.CALENDAR and self.market is None:
            raise ValueError("calendar sync requires --market")
        if self.dataset is TushareDataset.CALENDAR and self.instrument_ids:
            raise ValueError("calendar sync does not accept instrument_ids")
        if self.dataset is TushareDataset.INSTRUMENT and self.instrument_ids:
            raise ValueError("instrument sync does not support instrument_id filtering")
        if (
            self.dataset is TushareDataset.INSTRUMENT
            and self.instrument_type is InstrumentType.INDEX
            and self.market is None
        ):
            raise ValueError("index instrument sync requires --market")
        if (
            self.dataset is TushareDataset.DAILY
            and self.instrument_type is InstrumentType.INDEX
            and len(self.instrument_ids) != 1
        ):
            raise ValueError("index daily sync requires exactly one instrument_id")
        if (
            self.dataset is TushareDataset.ADJUSTMENT
            and self.instrument_type is not InstrumentType.STOCK
        ):
            raise ValueError("adjustment sync supports STOCK only")
        if (
            self.dataset is TushareDataset.ANNOUNCEMENT
            and self.instrument_type is not InstrumentType.STOCK
        ):
            raise ValueError("announcement sync supports STOCK only")
        if self.dataset is TushareDataset.ANNOUNCEMENT and self.market is not None:
            raise ValueError("announcement sync does not accept --market")
        if self.idempotency_key is not None and (
            not self.idempotency_key.strip() or len(self.idempotency_key) > 64
        ):
            raise ValueError("idempotency_key must contain 1 to 64 characters")
        if self.idempotency_key is not None:
            object.__setattr__(self, "idempotency_key", self.idempotency_key.strip())

    @property
    def effective_market(self) -> str:
        """Return the calendar market used by bar and adjustment validation."""

        return self.market or "SSE"

    def identity_payload(self) -> JsonObject:
        return {
            "dataset": self.dataset.value,
            "end": self.end.isoformat(),
            "instrument_ids": list(self.instrument_ids),
            "instrument_type": self.instrument_type.value,
            "market": self.market,
            "mode": self.mode.value,
            "start": self.start.isoformat(),
        }


def _components(
    spec: TushareSyncSpec,
    provider: TushareHttpProvider,
) -> tuple[DatasetSource[Any], DatasetIngestor[Any], str, JsonObject, JsonObject]:
    if spec.dataset is TushareDataset.CALENDAR:
        calendar_source = TushareTradingCalendarSource(
            provider,
            market=spec.effective_market,
        )
        return (
            calendar_source,
            TushareTradingCalendarIngestor(),
            calendar_source.watermark_key,
            {"market": spec.effective_market},
            {},
        )
    if spec.dataset is TushareDataset.INSTRUMENT:
        instrument_source = TushareInstrumentSource(
            provider,
            instrument_type=spec.instrument_type,
            market=spec.market,
        )
        return (
            instrument_source,
            TushareInstrumentIngestor(),
            instrument_source.watermark_key,
            {
                "instrument_type": spec.instrument_type.value,
                "market": spec.market,
            },
            {},
        )
    if spec.dataset is TushareDataset.DAILY:
        daily_source = TushareDailyBarSource(
            provider,
            instrument_type=spec.instrument_type,
            instrument_ids=spec.instrument_ids or None,
        )
        return (
            daily_source,
            TushareDailyBarIngestor(calendar_market=spec.effective_market),
            daily_source.watermark_key,
            {
                "instrument_ids": list(spec.instrument_ids),
                "instrument_type": spec.instrument_type.value,
                "market": spec.effective_market,
            },
            {"instrument_ids": list(spec.instrument_ids)},
        )
    if spec.dataset is TushareDataset.ANNOUNCEMENT:
        announcement_source = TushareAnnouncementSource(
            provider,
            instrument_ids=spec.instrument_ids or None,
        )
        return (
            announcement_source,
            TushareAnnouncementIngestor(),
            announcement_source.watermark_key,
            {
                "instrument_ids": list(spec.instrument_ids),
                "instrument_type": InstrumentType.STOCK.value,
            },
            {
                "classifier_version": ANNOUNCEMENT_CLASSIFIER_VERSION,
                "instrument_ids": list(spec.instrument_ids),
            },
        )
    adjustment_source = TushareAdjustmentFactorSource(
        provider,
        instrument_ids=spec.instrument_ids or None,
    )
    return (
        adjustment_source,
        TushareAdjustmentFactorIngestor(calendar_market=spec.effective_market),
        adjustment_source.watermark_key,
        {
            "instrument_ids": list(spec.instrument_ids),
            "instrument_type": InstrumentType.STOCK.value,
            "market": spec.effective_market,
        },
        {"instrument_ids": list(spec.instrument_ids)},
    )


def _run_with_provider(
    *,
    spec: TushareSyncSpec,
    database_url: str,
    provider: TushareHttpProvider,
) -> SyncExecutionResult:
    source, ingestor, watermark_key, scope, source_config = _components(spec, provider)
    identity = spec.identity_payload()
    idempotency_key = spec.idempotency_key or canonical_hash(identity)
    config_hash = canonical_hash(
        {
            "code_version": __version__,
            "source_config": source_config,
            "spec": identity,
        }
    )
    database = Database(database_url)
    runner: IncrementalSyncRunner[Any] = IncrementalSyncRunner(
        session_factory=database.session,
        source=source,
        ingestor=ingestor,
        lease_owner="quant-agent-cli",
    )
    return runner.run(
        scope=scope,
        initial_watermark={watermark_key: (spec.start - timedelta(days=1)).isoformat()},
        target_watermark={watermark_key: spec.end.isoformat()},
        idempotency_key=idempotency_key,
        mode=spec.mode,
        requested_from={watermark_key: spec.start.isoformat()},
        config_hash=config_hash,
        code_version=__version__,
    )


def run_tushare_sync(
    *,
    spec: TushareSyncSpec,
    database_url: str,
    market_data_token: SecretStr | None = None,
    provider: TushareHttpProvider | None = None,
) -> SyncExecutionResult:
    """Run one dataset sync, constructing the HTTP provider only at the secret boundary."""

    if provider is not None:
        return _run_with_provider(spec=spec, database_url=database_url, provider=provider)
    if market_data_token is None or not market_data_token.get_secret_value():
        raise ValueError("market data token is required")
    with TushareHttpProvider(market_data_token.get_secret_value()) as owned_provider:
        return _run_with_provider(
            spec=spec,
            database_url=database_url,
            provider=owned_provider,
        )


def sync_result_dict(result: SyncExecutionResult) -> dict[str, object]:
    """Return a credential-free stable JSON representation."""

    return {
        "checkpoint_revision": result.checkpoint_revision,
        "committed_watermark": result.committed_watermark,
        "fetched_pages": result.fetched_pages,
        "inserted": result.inserted,
        "reused_raw_pages": result.reused_raw_pages,
        "run_id": result.run_id,
        "skipped": result.skipped,
        "state": result.state.value,
        "updated": result.updated,
    }
