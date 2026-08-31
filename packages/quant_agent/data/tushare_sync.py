"""Raw-first Tushare dataset sources and lineage-aware curated ingestors."""

from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware, shanghai_now
from quant_agent.data.adjustments import AdjustmentService
from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    Instrument,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.events import (
    EventEvidence,
    EventEvidenceService,
    EventType,
    build_event_evidence,
)
from quant_agent.data.master import InstrumentMasterService
from quant_agent.data.models import (
    CuratedRecordLineageRow,
    DailyBarRow,
    InstrumentRow,
    TradingDayRow,
)
from quant_agent.data.providers import TushareHttpProvider
from quant_agent.data.providers.tushare import instrument_id_from_ts_code
from quant_agent.data.sync.contracts import (
    DecodedPage,
    IngestionSession,
    PageIngestionResult,
    PageRequest,
    RawPage,
)
from quant_agent.data.sync.hashing import (
    JsonObject,
    JsonValue,
    canonical_hash,
    canonical_json,
    clone_json_object,
)

_SCHEMA_VERSION = "tushare-json-v1"
_STOCK_STATUSES = ("L", "D", "P")
ANNOUNCEMENT_CLASSIFIER_VERSION = "announcement-title-rules-v1"
type Clock = Callable[[], datetime]


def _watermark_date(value: JsonObject, key: str) -> date:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise ValueError(f"watermark must contain string {key}")
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"watermark contains invalid {key}: {raw}") from error


def _partition_date(value: JsonObject, key: str) -> date:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise ValueError(f"partition must contain string {key}")
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"partition contains invalid {key}: {raw}") from error


def _date_partitions(
    committed_watermark: JsonObject,
    target_watermark: JsonObject,
    *,
    key: str,
    metadata: JsonObject,
) -> tuple[JsonObject, ...]:
    committed = _watermark_date(committed_watermark, key)
    target = _watermark_date(target_watermark, key)
    if committed > target:
        raise ValueError("committed watermark cannot be after target watermark")
    partitions: list[JsonObject] = []
    current = committed + timedelta(days=1)
    while current <= target:
        partitions.append({key: current.isoformat(), **clone_json_object(metadata)})
        current += timedelta(days=1)
    return tuple(partitions)


def _provider_params(request: PageRequest) -> dict[str, Any]:
    return cast(dict[str, Any], clone_json_object(request.request_params))


class _TushareSourceBase[T]:
    provider = "tushare"
    dataset: str

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        clock: Clock = shanghai_now,
    ) -> None:
        self._http = provider
        self._clock = clock

    def page_request(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> PageRequest:
        if cursor is not None:
            raise ValueError("Tushare source does not expose undocumented pagination")
        endpoint, params = self._request(partition)
        return PageRequest(
            provider=self.provider,
            endpoint=endpoint,
            request_params=params,
            partition=clone_json_object(partition),
            cursor_in=None,
        )

    def fetch_page(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> RawPage:
        request = self.page_request(partition, cursor)
        payload = self._http.fetch_raw_json(
            request.endpoint,
            _provider_params(request),
        )
        return RawPage(
            provider=self.provider,
            endpoint=request.endpoint,
            request_params=clone_json_object(request.request_params),
            payload=cast(JsonObject, payload),
            partition=clone_json_object(partition),
            cursor_in=None,
            cursor_out=None,
            has_more=False,
            record_count=self._http.raw_record_count(payload, endpoint=request.endpoint),
            fetched_at=ensure_aware(self._clock()),
            available_at=self._available_at(partition, request.endpoint),
            schema_version=f"{_SCHEMA_VERSION}:{request.endpoint}",
        )

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        raise NotImplementedError

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        raise NotImplementedError


class TushareTradingCalendarSource(_TushareSourceBase[TradingDay]):
    """One raw ``trade_cal`` response per explicit market/date partition."""

    dataset = "trading_calendar"
    watermark_key = "cal_date"

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        market: str,
        clock: Clock = shanghai_now,
    ) -> None:
        if not market.strip():
            raise ValueError("market must be non-empty")
        super().__init__(provider, clock=clock)
        self.market = market.strip()

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        return _date_partitions(
            committed_watermark,
            target_watermark,
            key=self.watermark_key,
            metadata={"market": self.market},
        )

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        value = _partition_date(partition, self.watermark_key)
        return {self.watermark_key: value.isoformat()}

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        value = _partition_date(partition, self.watermark_key)
        if partition.get("market") != self.market:
            raise ValueError("calendar partition market does not match source")
        encoded = value.strftime("%Y%m%d")
        return (
            "trade_cal",
            {
                "exchange": self.market,
                "start_date": encoded,
                "end_date": encoded,
            },
        )

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        return self._http.available_at_for_endpoint(
            endpoint,
            _partition_date(partition, self.watermark_key),
        )

    def decode(self, raw_page: RawPage) -> DecodedPage[TradingDay]:
        value = _partition_date(raw_page.partition, self.watermark_key)
        records = self._http.decode_trading_calendar_payload(
            cast(dict[str, Any], raw_page.payload),
            market=self.market,
            start=value,
            end=value,
        )
        return DecodedPage(
            raw_page=raw_page,
            records=records,
            rejected_count=raw_page.record_count - len(records),
        )


class TushareInstrumentSource(_TushareSourceBase[Instrument]):
    """Snapshot master source for STOCK, ETF, or INDEX instruments."""

    dataset = "instrument"
    watermark_key = "as_of"

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        instrument_type: InstrumentType,
        market: str | None = None,
        clock: Clock = shanghai_now,
    ) -> None:
        if instrument_type is InstrumentType.INDEX and not market:
            raise ValueError("index instrument source requires an explicit market")
        super().__init__(provider, clock=clock)
        self.instrument_type = instrument_type
        self.market = market

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        target = _watermark_date(target_watermark, self.watermark_key)
        if committed_watermark == target_watermark:
            return ()
        base: JsonObject = {
            self.watermark_key: target.isoformat(),
            "instrument_type": self.instrument_type.value,
        }
        if self.market is not None:
            base["market"] = self.market
        if self.instrument_type is not InstrumentType.STOCK:
            return (base,)

        statuses = list(_STOCK_STATUSES)
        committed_as_of = committed_watermark.get(self.watermark_key)
        committed_status = committed_watermark.get("list_status")
        start = 0
        if committed_as_of == target.isoformat() and isinstance(committed_status, str):
            try:
                start = statuses.index(committed_status) + 1
            except ValueError as error:
                raise ValueError("unknown stock master status watermark") from error
        return tuple(
            {**clone_json_object(base), "list_status": status} for status in statuses[start:]
        )

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        value = _partition_date(partition, self.watermark_key)
        if self.instrument_type is InstrumentType.STOCK:
            status = partition.get("list_status")
            if status not in _STOCK_STATUSES:
                raise ValueError("stock master partition has invalid list_status")
            if status != _STOCK_STATUSES[-1]:
                return {self.watermark_key: value.isoformat(), "list_status": status}
        return {self.watermark_key: value.isoformat()}

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        _partition_date(partition, self.watermark_key)
        if partition.get("instrument_type") != self.instrument_type.value:
            raise ValueError("instrument partition type does not match source")
        endpoint = self._http.instrument_endpoint(self.instrument_type)
        if self.instrument_type is InstrumentType.STOCK:
            status = partition.get("list_status")
            if status not in _STOCK_STATUSES:
                raise ValueError("stock master partition has invalid list_status")
            params: JsonObject = {"list_status": cast(str, status)}
            if self.market is not None:
                params["market"] = self.market
            return endpoint, params
        if self.instrument_type is InstrumentType.ETF:
            params = {"list_status": "L"}
            if self.market is not None:
                params["exchange"] = {"SSE": "SH", "SZSE": "SZ"}.get(
                    self.market,
                    self.market,
                )
            return endpoint, params
        if self.market is None:  # pragma: no cover - constructor invariant
            raise ValueError("index instrument source requires a market")
        return endpoint, {"market": self.market}

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        return self._http.available_at_for_endpoint(
            endpoint,
            _partition_date(partition, self.watermark_key),
        )

    def decode(self, raw_page: RawPage) -> DecodedPage[Instrument]:
        as_of = _partition_date(raw_page.partition, self.watermark_key)
        records = self._http.decode_instruments_payload(
            cast(dict[str, Any], raw_page.payload),
            as_of=as_of,
            instrument_type=self.instrument_type,
        )
        return DecodedPage(
            raw_page=raw_page,
            records=records,
            rejected_count=raw_page.record_count - len(records),
        )


class TushareDailyBarSource(_TushareSourceBase[DailyBar]):
    """Unadjusted daily bars partitioned by date and explicit asset universe."""

    dataset = "daily_bar"
    watermark_key = "trade_date"

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        instrument_type: InstrumentType,
        instrument_ids: Sequence[str] | None = None,
        clock: Clock = shanghai_now,
    ) -> None:
        normalized = tuple(dict.fromkeys(instrument_ids)) if instrument_ids is not None else None
        if normalized == ():
            raise ValueError("instrument_ids cannot be empty")
        if instrument_type is InstrumentType.INDEX and (normalized is None or len(normalized) != 1):
            raise ValueError("index daily source requires exactly one instrument_id")
        super().__init__(provider, clock=clock)
        self.instrument_type = instrument_type
        self.instrument_ids = normalized

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        metadata: JsonObject = {"instrument_type": self.instrument_type.value}
        if self.instrument_ids is not None:
            metadata["instrument_ids"] = list(self.instrument_ids)
        return _date_partitions(
            committed_watermark,
            target_watermark,
            key=self.watermark_key,
            metadata=metadata,
        )

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        value = _partition_date(partition, self.watermark_key)
        return {self.watermark_key: value.isoformat()}

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        value = _partition_date(partition, self.watermark_key)
        if partition.get("instrument_type") != self.instrument_type.value:
            raise ValueError("daily partition type does not match source")
        endpoint = self._http.daily_endpoint(self.instrument_type)
        params: JsonObject = {"trade_date": value.strftime("%Y%m%d")}
        if self.instrument_ids is not None and len(self.instrument_ids) == 1:
            params["ts_code"] = self._http.ts_code_from_instrument_id(self.instrument_ids[0])
        return endpoint, params

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        return self._http.available_at_for_endpoint(
            endpoint,
            _partition_date(partition, self.watermark_key),
        )

    def decode(self, raw_page: RawPage) -> DecodedPage[DailyBar]:
        value = _partition_date(raw_page.partition, self.watermark_key)
        records = self._http.decode_daily_bars_payload(
            cast(dict[str, Any], raw_page.payload),
            trade_date=value,
            instrument_type=self.instrument_type,
            instrument_ids=self.instrument_ids,
        )
        return DecodedPage(
            raw_page=raw_page,
            records=records,
            rejected_count=raw_page.record_count - len(records),
        )


class TushareAdjustmentFactorSource(_TushareSourceBase[AdjustmentFactor]):
    """Stock adjustment factors partitioned by date and optional asset universe."""

    dataset = "adjustment_factor"
    watermark_key = "trade_date"

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        instrument_ids: Sequence[str] | None = None,
        clock: Clock = shanghai_now,
    ) -> None:
        normalized = tuple(dict.fromkeys(instrument_ids)) if instrument_ids is not None else None
        if normalized == ():
            raise ValueError("instrument_ids cannot be empty")
        super().__init__(provider, clock=clock)
        self.instrument_ids = normalized

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        metadata: JsonObject = {"instrument_type": InstrumentType.STOCK.value}
        if self.instrument_ids is not None:
            metadata["instrument_ids"] = list(self.instrument_ids)
        return _date_partitions(
            committed_watermark,
            target_watermark,
            key=self.watermark_key,
            metadata=metadata,
        )

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        value = _partition_date(partition, self.watermark_key)
        return {self.watermark_key: value.isoformat()}

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        value = _partition_date(partition, self.watermark_key)
        if partition.get("instrument_type") != InstrumentType.STOCK.value:
            raise ValueError("adjustment partition must contain STOCK instrument_type")
        params: JsonObject = {"trade_date": value.strftime("%Y%m%d")}
        if self.instrument_ids is not None and len(self.instrument_ids) == 1:
            params["ts_code"] = self._http.ts_code_from_instrument_id(self.instrument_ids[0])
        return "adj_factor", params

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        return self._http.available_at_for_endpoint(
            endpoint,
            _partition_date(partition, self.watermark_key),
        )

    def decode(self, raw_page: RawPage) -> DecodedPage[AdjustmentFactor]:
        value = _partition_date(raw_page.partition, self.watermark_key)
        records = self._http.decode_adjustment_factors_payload(
            cast(dict[str, Any], raw_page.payload),
            trade_date=value,
            instrument_ids=self.instrument_ids,
        )
        return DecodedPage(
            raw_page=raw_page,
            records=records,
            rejected_count=raw_page.record_count - len(records),
        )


def _announcement_event_type(title: str) -> EventType:
    """Classify a title with deterministic, versioned keyword rules."""

    groups = (
        (
            EventType.REGULATORY,
            ("监管", "问询", "处罚", "立案", "风险警示", "退市", "纪律处分"),
        ),
        (
            EventType.MANAGEMENT,
            ("董事", "监事", "高管", "总经理", "董事会秘书", "管理层"),
        ),
        (
            EventType.CORPORATE_ACTION,
            ("分红", "派息", "回购", "增持", "减持", "并购", "重组", "股权激励"),
        ),
        (
            EventType.EARNINGS,
            ("业绩", "年报", "半年报", "季报", "财务报告", "盈利", "亏损"),
        ),
        (EventType.PRODUCT, ("产品", "项目", "中标", "合同", "订单")),
    )
    for event_type, keywords in groups:
        if any(keyword in title for keyword in keywords):
            return event_type
    return EventType.OTHER


def _announcement_time(value: object, *, announcement_date: date) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.combine(announcement_date, datetime.min.time(), tzinfo=SHANGHAI_TZ)
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M%S",
        "%Y%m%d %H:%M:%S",
    )
    for value_format in formats:
        try:
            return datetime.strptime(text, value_format).replace(tzinfo=SHANGHAI_TZ)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"invalid anns_d rec_time: {text}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=SHANGHAI_TZ)
    return ensure_aware(parsed)


def _announcement_date(value: object) -> date:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(f"invalid anns_d ann_date: {text}") from error


class TushareAnnouncementSource(_TushareSourceBase[EventEvidence]):
    """Raw-first listed-company announcements partitioned by publication date."""

    dataset = "event_evidence"
    watermark_key = "ann_date"

    def __init__(
        self,
        provider: TushareHttpProvider,
        *,
        instrument_ids: Sequence[str] | None = None,
        clock: Clock = shanghai_now,
    ) -> None:
        normalized = tuple(sorted(set(instrument_ids))) if instrument_ids is not None else None
        if normalized == ():
            raise ValueError("instrument_ids cannot be empty")
        super().__init__(provider, clock=clock)
        self.instrument_ids = normalized

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        committed = _watermark_date(committed_watermark, self.watermark_key)
        target = _watermark_date(target_watermark, self.watermark_key)
        if committed > target:
            raise ValueError("committed watermark cannot be after target watermark")
        if self.instrument_ids is None:
            return _date_partitions(
                committed_watermark,
                target_watermark,
                key=self.watermark_key,
                metadata={},
            )

        committed_instrument = committed_watermark.get("instrument_id")
        start_date = committed + timedelta(days=1)
        start_index = 0
        if committed_instrument is not None:
            if not isinstance(committed_instrument, str):
                raise ValueError("announcement watermark instrument_id must be a string")
            try:
                start_index = self.instrument_ids.index(committed_instrument) + 1
            except ValueError as error:
                raise ValueError("unknown announcement instrument watermark") from error
            start_date = committed

        result: list[JsonObject] = []
        current = start_date
        while current <= target:
            ids = (
                self.instrument_ids[start_index:] if current == start_date else self.instrument_ids
            )
            result.extend(
                {
                    self.watermark_key: current.isoformat(),
                    "instrument_id": instrument_id,
                }
                for instrument_id in ids
            )
            start_index = 0
            current += timedelta(days=1)
        return tuple(result)

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        value = _partition_date(partition, self.watermark_key)
        if self.instrument_ids is None:
            return {self.watermark_key: value.isoformat()}
        instrument_id = partition.get("instrument_id")
        if instrument_id not in self.instrument_ids:
            raise ValueError("announcement partition instrument does not match source")
        if instrument_id == self.instrument_ids[-1]:
            return {self.watermark_key: value.isoformat()}
        return {
            self.watermark_key: value.isoformat(),
            "instrument_id": cast(str, instrument_id),
        }

    def _request(self, partition: JsonObject) -> tuple[str, JsonObject]:
        value = _partition_date(partition, self.watermark_key)
        params: JsonObject = {"ann_date": value.strftime("%Y%m%d")}
        instrument_id = partition.get("instrument_id")
        if self.instrument_ids is not None:
            if instrument_id not in self.instrument_ids:
                raise ValueError("announcement partition instrument does not match source")
            params["ts_code"] = self._http.ts_code_from_instrument_id(cast(str, instrument_id))
        elif instrument_id is not None:
            raise ValueError("unfiltered announcement partition cannot contain instrument_id")
        return "anns_d", params

    def _available_at(self, partition: JsonObject, endpoint: str) -> datetime:
        del partition, endpoint
        return ensure_aware(self._clock())

    def decode(self, raw_page: RawPage) -> DecodedPage[EventEvidence]:
        announcement_date = _partition_date(raw_page.partition, self.watermark_key)
        payload = cast(dict[str, Any], raw_page.payload)
        rows = self._http.decode_raw_rows(payload, endpoint="anns_d")
        raw_data = payload.get("data")
        raw_items = raw_data.get("items") if isinstance(raw_data, dict) else None
        if not isinstance(raw_items, list) or len(raw_items) != len(rows):
            raise ValueError("announcement raw rows do not align with raw items")

        records: list[EventEvidence] = []
        for index, (row, raw_item) in enumerate(zip(rows, raw_items, strict=True)):
            row_date = _announcement_date(row.get("ann_date"))
            if row_date != announcement_date:
                raise ValueError("anns_d returned a row outside its date partition")
            ts_code = str(row.get("ts_code") or "").strip()
            title = str(row.get("title") or "").strip()
            if not ts_code or not title:
                raise ValueError("anns_d row is missing ts_code or title")
            instrument_id = instrument_id_from_ts_code(ts_code)
            if self.instrument_ids is not None and instrument_id not in self.instrument_ids:
                raise ValueError("anns_d returned an instrument outside its partition scope")
            source_url = str(row.get("url") or "").strip() or None
            published_at = _announcement_time(
                row.get("rec_time"),
                announcement_date=announcement_date,
            )
            raw_value = cast(JsonValue, raw_item)
            source_record_id = canonical_hash(
                {
                    "ann_date": announcement_date.isoformat(),
                    "instrument_id": instrument_id,
                    "source_url": source_url,
                    "title": title if source_url is None else None,
                }
            )
            version = canonical_hash(
                {
                    "classifier_version": ANNOUNCEMENT_CLASSIFIER_VERSION,
                    "raw_item": raw_value,
                }
            )
            observed_at = ensure_aware(raw_page.fetched_at)
            available_at = max(observed_at, ensure_aware(raw_page.available_at))
            records.append(
                build_event_evidence(
                    event_type=_announcement_event_type(title),
                    headline=title,
                    published_at=published_at,
                    observed_at=observed_at,
                    available_at=available_at,
                    source=self.provider,
                    source_record_id=source_record_id,
                    source_url=source_url,
                    source_path=f"/data/items/{index}",
                    source_content=raw_value,
                    version=version,
                    instrument_ids=(instrument_id,),
                )
            )
        return DecodedPage(
            raw_page=raw_page,
            records=tuple(records),
            rejected_count=raw_page.record_count - len(records),
        )


def _record_payload(record: BaseModel) -> JsonObject:
    value = record.model_dump(mode="json")
    if not isinstance(value, dict):  # pragma: no cover - pydantic model invariant
        raise TypeError("domain record must serialize to an object")
    return cast(JsonObject, value)


def _lineage(
    session: IngestionSession,
    *,
    raw_payload_id: int,
    dataset: str,
    entity_type: str,
    entity_key: JsonObject,
    record: BaseModel,
) -> None:
    key = canonical_json(entity_key)
    content_hash = canonical_hash(_record_payload(record))
    existing = session.scalar(
        select(CuratedRecordLineageRow.id).where(
            CuratedRecordLineageRow.raw_payload_id == raw_payload_id,
            CuratedRecordLineageRow.dataset == dataset,
            CuratedRecordLineageRow.entity_type == entity_type,
            CuratedRecordLineageRow.entity_key == key,
            CuratedRecordLineageRow.content_hash == content_hash,
        )
    )
    if existing is None:
        session.add(
            CuratedRecordLineageRow(
                raw_payload_id=raw_payload_id,
                dataset=dataset,
                entity_type=entity_type,
                entity_key=key,
                content_hash=content_hash,
            )
        )


class TushareTradingCalendarIngestor:
    """Idempotently upsert trading days and retain record-level raw lineage."""

    dataset = "trading_calendar"

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[TradingDay],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        inserted = updated = skipped = 0
        for day in page.records:
            row = cast(
                TradingDayRow | None,
                session.get(TradingDayRow, (day.market, day.trade_date)),
            )
            if row is None:
                session.add(
                    TradingDayRow(
                        market=day.market,
                        trade_date=day.trade_date,
                        is_open=day.is_open,
                        source=day.source,
                        version=day.version,
                    )
                )
                inserted += 1
            elif (
                row.is_open != day.is_open or row.source != day.source or row.version != day.version
            ):
                row.is_open = day.is_open
                row.source = day.source
                row.version = day.version
                updated += 1
            else:
                skipped += 1
            _lineage(
                session,
                raw_payload_id=raw_payload_id,
                dataset=self.dataset,
                entity_type="TradingDay",
                entity_key={"market": day.market, "trade_date": day.trade_date.isoformat()},
                record=day,
            )
        session.flush()
        return PageIngestionResult(inserted=inserted, updated=updated, skipped=skipped)


class TushareInstrumentIngestor:
    """Upsert master rows through the historical master service with lineage."""

    dataset = "instrument"

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[Instrument],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        inserted, updated = InstrumentMasterService(cast(Session, session)).upsert(
            list(page.records)
        )
        for instrument in page.records:
            _lineage(
                session,
                raw_payload_id=raw_payload_id,
                dataset=self.dataset,
                entity_type="Instrument",
                entity_key={"instrument_id": instrument.instrument_id},
                record=instrument,
            )
        session.flush()
        return PageIngestionResult(
            inserted=inserted,
            updated=updated,
            skipped=len(page.records) - inserted - updated,
        )


def _validate_historical_instrument(
    session: IngestionSession,
    instrument_id: str,
    value_date: date,
) -> None:
    active = session.scalar(
        select(InstrumentRow.instrument_id).where(
            InstrumentRow.instrument_id == instrument_id,
            InstrumentRow.listed_on <= value_date,
            or_(InstrumentRow.delisted_on.is_(None), InstrumentRow.delisted_on > value_date),
        )
    )
    if active is None:
        raise ValueError(f"{instrument_id} is not active on {value_date}")


def _validate_trading_day(session: IngestionSession, market: str, value_date: date) -> None:
    calendar = cast(
        TradingDayRow | None,
        session.get(TradingDayRow, (market, value_date)),
    )
    if calendar is None or not calendar.is_open:
        raise ValueError(f"{value_date} is not an open {market} day")


class TushareDailyBarIngestor:
    """Validate PIT master/calendar state and persist immutable raw-price versions."""

    dataset = "daily_bar"

    def __init__(self, *, calendar_market: str = "SSE") -> None:
        if not calendar_market.strip():
            raise ValueError("calendar_market must be non-empty")
        self.calendar_market = calendar_market.strip()

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[DailyBar],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        inserted = skipped = 0
        for bar in page.records:
            _validate_trading_day(session, self.calendar_market, bar.trade_date)
            _validate_historical_instrument(session, bar.instrument_id, bar.trade_date)
            existing = session.scalar(
                select(DailyBarRow.id).where(
                    DailyBarRow.instrument_id == bar.instrument_id,
                    DailyBarRow.trade_date == bar.trade_date,
                    DailyBarRow.source == bar.source,
                    DailyBarRow.version == bar.version,
                )
            )
            if existing is None:
                session.add(
                    DailyBarRow(
                        instrument_id=bar.instrument_id,
                        trade_date=bar.trade_date,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        turnover=bar.turnover,
                        source=bar.source,
                        available_at=bar.available_at,
                        version=bar.version,
                    )
                )
                inserted += 1
            else:
                skipped += 1
            _lineage(
                session,
                raw_payload_id=raw_payload_id,
                dataset=self.dataset,
                entity_type="DailyBar",
                entity_key={
                    "instrument_id": bar.instrument_id,
                    "source": bar.source,
                    "trade_date": bar.trade_date.isoformat(),
                    "version": bar.version,
                },
                record=bar,
            )
        session.flush()
        return PageIngestionResult(inserted=inserted, skipped=skipped)


class TushareAdjustmentFactorIngestor:
    """Persist stock adjustment-factor versions with raw-record lineage."""

    dataset = "adjustment_factor"

    def __init__(self, *, calendar_market: str = "SSE") -> None:
        if not calendar_market.strip():
            raise ValueError("calendar_market must be non-empty")
        self.calendar_market = calendar_market.strip()

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[AdjustmentFactor],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        for factor in page.records:
            _validate_trading_day(session, self.calendar_market, factor.trade_date)
            _validate_historical_instrument(session, factor.instrument_id, factor.trade_date)
        inserted, skipped = AdjustmentService(cast(Session, session)).ingest_factors(
            list(page.records)
        )
        for factor in page.records:
            _lineage(
                session,
                raw_payload_id=raw_payload_id,
                dataset=self.dataset,
                entity_type="AdjustmentFactor",
                entity_key={
                    "instrument_id": factor.instrument_id,
                    "source": factor.source,
                    "trade_date": factor.trade_date.isoformat(),
                    "version": factor.version,
                },
                record=factor,
            )
        session.flush()
        return PageIngestionResult(inserted=inserted, skipped=skipped)


class TushareAnnouncementIngestor:
    """Persist immutable event revisions and their exact raw-row lineage."""

    dataset = "event_evidence"

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[EventEvidence],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        result = EventEvidenceService(session).ingest(
            page.records,
            raw_payload_id=raw_payload_id,
        )
        return PageIngestionResult(
            inserted=result.inserted,
            skipped=result.skipped,
        )


__all__ = [
    "ANNOUNCEMENT_CLASSIFIER_VERSION",
    "TushareAdjustmentFactorIngestor",
    "TushareAdjustmentFactorSource",
    "TushareAnnouncementIngestor",
    "TushareAnnouncementSource",
    "TushareDailyBarIngestor",
    "TushareDailyBarSource",
    "TushareInstrumentIngestor",
    "TushareInstrumentSource",
    "TushareTradingCalendarIngestor",
    "TushareTradingCalendarSource",
]
