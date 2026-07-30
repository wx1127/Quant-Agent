"""Hash-verified, redacted Agent decision audit and replay."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from quant_agent.agent.runtime import StateTransition
from quant_agent.agent.snapshots import DecisionSnapshot, DecisionSnapshotStore
from quant_agent.core.time import ensure_aware
from quant_agent.observability.redaction import redact


class ReplayDifferenceKind(StrEnum):
    MISSING = "MISSING"
    HASH_MISMATCH = "HASH_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"


@dataclass(frozen=True, slots=True)
class ToolAuditRecord:
    decision_id: str
    sequence: int
    tool_name: str
    occurred_at: datetime
    arguments: dict[str, Any]
    response: dict[str, Any]
    argument_hash: str
    response_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.occurred_at)
        if self.sequence < 1:
            raise ValueError("audit sequence must be positive")

    @classmethod
    def create(
        cls,
        *,
        decision_id: str,
        sequence: int,
        tool_name: str,
        occurred_at: datetime,
        arguments: dict[str, Any],
        response: dict[str, Any],
    ) -> "ToolAuditRecord":
        safe_arguments = redact(arguments)
        safe_response = redact(response)
        return cls(
            decision_id=decision_id,
            sequence=sequence,
            tool_name=tool_name,
            occurred_at=occurred_at,
            arguments=safe_arguments,
            response=safe_response,
            argument_hash=_hash(safe_arguments),
            response_hash=_hash(safe_response),
        )

    def verify(self) -> None:
        if self.argument_hash != _hash(self.arguments):
            raise ValueError("tool argument audit hash mismatch")
        if self.response_hash != _hash(self.response):
            raise ValueError("tool response audit hash mismatch")


@dataclass(frozen=True, slots=True)
class ReplayDifference:
    kind: ReplayDifferenceKind
    path: str
    expected: str
    actual: str | None


@dataclass(frozen=True, slots=True)
class DecisionReplay:
    snapshot: DecisionSnapshot
    tool_records: tuple[ToolAuditRecord, ...]
    state_trace: tuple[StateTransition, ...]
    final_response_hash: str


class InMemoryHarnessAuditStore:
    """Append-only records keyed by decision ID."""

    def __init__(self) -> None:
        self._tools: dict[str, list[ToolAuditRecord]] = {}
        self._states: dict[str, tuple[StateTransition, ...]] = {}
        self._final: dict[str, str] = {}

    def append_tool_record(self, record: ToolAuditRecord) -> None:
        records = self._tools.setdefault(record.decision_id, [])
        expected_sequence = len(records) + 1
        if record.sequence != expected_sequence:
            raise ValueError("tool audit sequence is not contiguous")
        record.verify()
        records.append(record)

    def save_state_trace(self, decision_id: str, trace: tuple[StateTransition, ...]) -> None:
        if decision_id in self._states and self._states[decision_id] != trace:
            raise ValueError("state trace is immutable")
        self._states[decision_id] = trace

    def save_final_response(self, decision_id: str, response: dict[str, Any]) -> str:
        response_hash = _hash(redact(response))
        existing = self._final.get(decision_id)
        if existing is not None and existing != response_hash:
            raise ValueError("final response is immutable")
        self._final[decision_id] = response_hash
        return response_hash

    def tool_records(self, decision_id: str) -> tuple[ToolAuditRecord, ...]:
        return tuple(self._tools.get(decision_id, ()))

    def state_trace(self, decision_id: str) -> tuple[StateTransition, ...] | None:
        return self._states.get(decision_id)

    def final_response_hash(self, decision_id: str) -> str | None:
        return self._final.get(decision_id)


class HarnessReplayService:
    def __init__(
        self,
        snapshots: DecisionSnapshotStore,
        audit: InMemoryHarnessAuditStore,
    ) -> None:
        self._snapshots = snapshots
        self._audit = audit

    def replay(self, decision_id: str) -> DecisionReplay:
        snapshot = self._snapshots.get(decision_id)
        records = self._audit.tool_records(decision_id)
        states = self._audit.state_trace(decision_id)
        final_hash = self._audit.final_response_hash(decision_id)
        missing = []
        if not records:
            missing.append("tool_records")
        if states is None:
            missing.append("state_trace")
        if final_hash is None:
            missing.append("final_response_hash")
        if missing:
            raise ValueError(f"critical audit fields missing: {', '.join(missing)}")
        assert states is not None
        assert final_hash is not None
        for record in records:
            record.verify()
        return DecisionReplay(snapshot, records, states, final_hash)

    def compare(
        self, decision_id: str, actual_tool_responses: list[dict[str, Any]]
    ) -> tuple[ReplayDifference, ...]:
        replay = self.replay(decision_id)
        differences: list[ReplayDifference] = []
        for index, record in enumerate(replay.tool_records):
            if index >= len(actual_tool_responses):
                differences.append(
                    ReplayDifference(
                        ReplayDifferenceKind.MISSING,
                        f"tool_records/{index}",
                        record.response_hash,
                        None,
                    )
                )
                continue
            actual_hash = _hash(redact(actual_tool_responses[index]))
            if actual_hash != record.response_hash:
                differences.append(
                    ReplayDifference(
                        ReplayDifferenceKind.HASH_MISMATCH,
                        f"tool_records/{index}/response",
                        record.response_hash,
                        actual_hash,
                    )
                )
        if len(actual_tool_responses) > len(replay.tool_records):
            differences.append(
                ReplayDifference(
                    ReplayDifferenceKind.VALUE_MISMATCH,
                    "tool_records/length",
                    str(len(replay.tool_records)),
                    str(len(actual_tool_responses)),
                )
            )
        return tuple(differences)


def _hash(value: Any) -> str:
    canonical = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
