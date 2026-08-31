"""Deterministic paginated source for local recovery and replay verification."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.data.sync.contracts import DecodedPage, PageRequest, RawPage
from quant_agent.data.sync.hashing import JsonObject, JsonValue, clone_json_object


class FakeSourceError(RuntimeError):
    """Deterministic failure injected by a fake page specification."""


@dataclass(frozen=True, slots=True)
class FakePageSpec:
    """One cursor-addressed fake provider page."""

    page_ordinal: int
    cursor_in: JsonObject | None
    cursor_out: JsonObject | None
    records: tuple[JsonObject, ...]
    fail_fetch_once: bool = False
    fail_decode_once: bool = False
    request_params: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_ordinal < 0:
            raise ValueError("page_ordinal cannot be negative")


@dataclass(frozen=True, slots=True)
class FakePartitionSpec:
    """Ordered fake partition and the watermark committed after its last page."""

    partition: JsonObject
    watermark: JsonObject
    pages: tuple[FakePageSpec, ...]

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("fake partition requires at least one page")
        expected_cursor: JsonObject | None = None
        for index, page in enumerate(self.pages):
            if page.page_ordinal != index or page.cursor_in != expected_cursor:
                raise ValueError("fake pages must have contiguous ordinals and cursors")
            expected_cursor = page.cursor_out
        if expected_cursor is not None:
            raise ValueError("the final fake page must not expose a next cursor")


class FakeDatasetSource:
    """JSON-record DatasetSource with deterministic one-shot failures."""

    provider = "fake"

    def __init__(
        self,
        *,
        dataset: str,
        partitions: tuple[FakePartitionSpec, ...],
        available_at: datetime,
        clock: Callable[[], datetime] = shanghai_now,
    ) -> None:
        if not dataset:
            raise ValueError("dataset must be non-empty")
        ensure_aware(available_at)
        self.dataset = dataset
        self._partitions = partitions
        self._available_at = available_at
        self._clock = clock
        self._fetch_attempts: dict[tuple[int, int], int] = {}
        self._decode_attempts: dict[tuple[int, int], int] = {}

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        if committed_watermark == target_watermark:
            return ()
        start = 0
        for index, partition in enumerate(self._partitions):
            if partition.watermark == committed_watermark:
                start = index + 1
        planned: list[JsonObject] = []
        for partition in self._partitions[start:]:
            planned.append(clone_json_object(partition.partition))
            if partition.watermark == target_watermark:
                return tuple(planned)
        raise ValueError("target watermark is absent from fake partition specifications")

    def partition_watermark(self, partition: JsonObject) -> JsonObject:
        partition_index, specification = self._partition(partition)
        del partition_index
        return clone_json_object(specification.watermark)

    def page_request(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> PageRequest:
        _, _, page = self._page(partition, cursor)
        params = clone_json_object(page.request_params)
        if cursor is not None:
            params["cursor"] = clone_json_object(cursor)
        return PageRequest(
            provider=self.provider,
            endpoint=self.dataset,
            request_params=params,
            partition=clone_json_object(partition),
            cursor_in=clone_json_object(cursor) if cursor is not None else None,
        )

    def fetch_page(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> RawPage:
        partition_index, _, page = self._page(partition, cursor)
        key = (partition_index, page.page_ordinal)
        attempts = self._fetch_attempts.get(key, 0) + 1
        self._fetch_attempts[key] = attempts
        if page.fail_fetch_once and attempts == 1:
            raise FakeSourceError(f"injected fetch failure at page {page.page_ordinal}")
        request = self.page_request(partition, cursor)
        payload: JsonObject = {
            "page_ordinal": page.page_ordinal,
            "records": [clone_json_object(record) for record in page.records],
        }
        return RawPage(
            provider=self.provider,
            endpoint=self.dataset,
            request_params=request.request_params,
            payload=payload,
            partition=clone_json_object(partition),
            cursor_in=clone_json_object(cursor) if cursor is not None else None,
            cursor_out=(
                clone_json_object(page.cursor_out) if page.cursor_out is not None else None
            ),
            has_more=page.cursor_out is not None,
            record_count=len(page.records),
            fetched_at=ensure_aware(self._clock()),
            available_at=self._available_at,
            schema_version="fake-v1",
        )

    def decode(self, raw_page: RawPage) -> DecodedPage[JsonObject]:
        partition_index, specification = self._partition(raw_page.partition)
        page = next(
            (item for item in specification.pages if item.cursor_in == raw_page.cursor_in),
            None,
        )
        if page is None:
            raise ValueError("raw page cursor is absent from fake specifications")
        key = (partition_index, page.page_ordinal)
        attempts = self._decode_attempts.get(key, 0) + 1
        self._decode_attempts[key] = attempts
        if page.fail_decode_once and attempts == 1:
            raise FakeSourceError(f"injected decode failure at page {page.page_ordinal}")
        raw_records = raw_page.payload.get("records")
        if not isinstance(raw_records, list) or not all(
            isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("fake raw payload records must be objects")
        records = tuple(
            clone_json_object(cast(dict[str, JsonValue], record)) for record in raw_records
        )
        return DecodedPage(raw_page=raw_page, records=records)

    def fetch_attempts(self, page_ordinal: int, *, partition_index: int = 0) -> int:
        """Return provider call count for a page."""

        return self._fetch_attempts.get((partition_index, page_ordinal), 0)

    def decode_attempts(self, page_ordinal: int, *, partition_index: int = 0) -> int:
        """Return decoder call count for a page."""

        return self._decode_attempts.get((partition_index, page_ordinal), 0)

    def _partition(self, value: JsonObject) -> tuple[int, FakePartitionSpec]:
        for index, partition in enumerate(self._partitions):
            if partition.partition == value:
                return index, partition
        raise ValueError("partition is absent from fake source specifications")

    def _page(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> tuple[int, FakePartitionSpec, FakePageSpec]:
        partition_index, specification = self._partition(partition)
        page = next((item for item in specification.pages if item.cursor_in == cursor), None)
        if page is None:
            raise ValueError("cursor is absent from fake page specifications")
        return partition_index, specification, page
