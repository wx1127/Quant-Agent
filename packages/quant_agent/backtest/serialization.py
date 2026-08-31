"""Canonical event serialization, hashing, and replay-stream validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import TypeAdapter

from quant_agent.backtest.contracts import (
    BacktestEvent,
    FillEvent,
    FillStatus,
    OrderEvent,
    OrderStatus,
)

_EVENT_ADAPTER: TypeAdapter[BacktestEvent] = TypeAdapter(BacktestEvent)
_EVENT_LIST_ADAPTER: TypeAdapter[list[BacktestEvent]] = TypeAdapter(list[BacktestEvent])


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("cannot serialize a non-finite Decimal")
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cannot serialize a naive datetime")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def serialize_event(event: BacktestEvent) -> str:
    """Return one deterministic JSON representation suitable for hashing and storage."""

    return _json(event.model_dump(mode="python"))


def deserialize_event(payload: str | bytes) -> BacktestEvent:
    """Restore and validate a serialized event using its event_type discriminator."""

    return _EVENT_ADAPTER.validate_json(payload)


def event_hash(event: BacktestEvent) -> str:
    """Bind an event to its canonical serialized content."""

    return hashlib.sha256(serialize_event(event).encode("utf-8")).hexdigest()


def validate_replay_stream(events: Iterable[BacktestEvent]) -> tuple[BacktestEvent, ...]:
    """Require deterministic ordering and contiguous order/fill lifecycle histories."""

    replay = tuple(events)
    if not replay:
        return replay
    run_id = replay[0].run_id
    previous_sequence = -1
    previous_time: datetime | None = None
    event_ids: set[str] = set()
    order_states: dict[str, OrderStatus] = {}
    fill_states: dict[str, FillStatus] = {}
    for event in replay:
        if event.run_id != run_id:
            raise ValueError("a replay stream cannot mix run_id values")
        if event.sequence <= previous_sequence:
            raise ValueError("replay event sequences must be strictly increasing")
        if previous_time is not None and event.event_time < previous_time:
            raise ValueError("replay event_time values must be monotonic")
        if event.event_id in event_ids:
            raise ValueError("replay event_id values must be unique")
        event_ids.add(event.event_id)
        if isinstance(event, OrderEvent):
            recorded_order_status = order_states.get(event.order_id)
            if event.previous_status is not recorded_order_status:
                raise ValueError("order previous_status does not match replayed state")
            order_states[event.order_id] = event.status
        elif isinstance(event, FillEvent):
            recorded_fill_status = fill_states.get(event.fill_id)
            if event.previous_status is not recorded_fill_status:
                raise ValueError("fill previous_status does not match replayed state")
            fill_states[event.fill_id] = event.status
        previous_sequence = event.sequence
        previous_time = event.event_time
    return replay


def serialize_event_stream(events: Iterable[BacktestEvent]) -> str:
    """Validate and serialize a replay stream while preserving its event order."""

    replay = validate_replay_stream(events)
    return _json([event.model_dump(mode="python") for event in replay])


def deserialize_event_stream(payload: str | bytes) -> tuple[BacktestEvent, ...]:
    """Restore a complete event stream and re-check replay ordering invariants."""

    events = _EVENT_LIST_ADAPTER.validate_json(payload)
    return validate_replay_stream(events)
