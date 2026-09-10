"""Hash-chained Harness audit records and deterministic replay comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str

    @classmethod
    def build(
        cls, sequence: int, event_type: str, payload: dict[str, Any], previous_hash: str
    ) -> AuditEvent:
        body = json.dumps(
            {
                "sequence": sequence,
                "event_type": event_type,
                "payload": payload,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event_hash = hashlib.sha256(body.encode()).hexdigest()
        return cls(sequence, event_type, payload, previous_hash, event_hash)


class AuditChain:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append(self, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        previous = self._events[-1].event_hash if self._events else "GENESIS"
        event = AuditEvent.build(len(self._events), event_type, payload, previous)
        self._events.append(event)
        return event

    def verify(self) -> bool:
        previous = "GENESIS"
        for index, event in enumerate(self._events):
            if event.sequence != index or event.previous_hash != previous:
                return False
            rebuilt = AuditEvent.build(
                event.sequence, event.event_type, event.payload, event.previous_hash
            )
            if rebuilt.event_hash != event.event_hash:
                return False
            previous = event.event_hash
        return True

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)


def compare_replay(
    expected: tuple[AuditEvent, ...], actual: tuple[AuditEvent, ...]
) -> tuple[str, ...]:
    differences: list[str] = []
    if len(expected) != len(actual):
        differences.append("event_count")
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left.event_type != right.event_type:
            differences.append(f"event_type:{index}")
        if left.payload != right.payload:
            differences.append(f"payload:{index}")
    return tuple(differences)
