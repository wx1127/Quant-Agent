"""Thread-safe, compare-and-swap storage for explicit Agent runs."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from .contracts import AgentRunSnapshot
from .replay import AgentRunReplayError, replay_agent_run


class AgentRunRepositoryError(RuntimeError):
    """Base class for safe runtime repository failures."""


class AgentRunNotFound(AgentRunRepositoryError):
    """Raised when a requested run identity is absent."""


class AgentRunConflict(AgentRunRepositoryError):
    """Raised when create or compare-and-swap observes conflicting state."""


class AgentRunRepository(Protocol):
    """Minimal persistence boundary required by the Agent runtime."""

    def create(self, snapshot: AgentRunSnapshot) -> AgentRunSnapshot:
        """Create one run, allowing only an exact retry of an existing create."""

    def get(self, run_id: str) -> AgentRunSnapshot:
        """Return a detached immutable copy of one run."""

    def save(
        self,
        snapshot: AgentRunSnapshot,
        *,
        expected_revision: int,
    ) -> AgentRunSnapshot:
        """Append one or more revisions if the stored revision matches the CAS token."""


def _validated_run_id(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 256
        or not value.isprintable()
    ):
        raise ValueError("run_id must be an exact, trimmed, printable identifier")
    return value


def _detached(snapshot: AgentRunSnapshot) -> AgentRunSnapshot:
    if type(snapshot) is not AgentRunSnapshot:
        raise TypeError("snapshot must be an exact AgentRunSnapshot")
    return AgentRunSnapshot.from_json(snapshot.to_json())


def _replay_verified(snapshot: AgentRunSnapshot) -> AgentRunSnapshot:
    """Detach and prove every derived field from the immutable event chain."""

    candidate = _detached(snapshot)
    try:
        replayed = replay_agent_run(candidate.events)
    except AgentRunReplayError as error:
        raise AgentRunConflict("Agent run event chain is not replayable") from error
    if replayed != candidate:
        raise AgentRunConflict("Agent run snapshot differs from its replayed event chain")
    return candidate


def _immutable_identity(snapshot: AgentRunSnapshot) -> tuple[object, ...]:
    return (
        snapshot.schema_version,
        snapshot.runtime_version,
        snapshot.run_id,
        snapshot.decision_id,
        snapshot.decision_snapshot_hash,
        snapshot.runtime_mode,
        snapshot.goal,
        snapshot.created_at,
        snapshot.deadline,
        snapshot.limits,
    )


class InMemoryAgentRunRepository:
    """Process-local repository with defensive JSON copies and atomic CAS writes."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, str] = {}

    def create(self, snapshot: AgentRunSnapshot) -> AgentRunSnapshot:
        candidate = _replay_verified(snapshot)
        payload = candidate.to_json()
        with self._lock:
            existing = self._snapshots.get(candidate.run_id)
            if existing is None:
                self._snapshots[candidate.run_id] = payload
                stored = payload
            elif existing == payload:
                stored = existing
            else:
                raise AgentRunConflict("run_id already exists with different state")
        return AgentRunSnapshot.from_json(stored)

    def get(self, run_id: str) -> AgentRunSnapshot:
        exact_run_id = _validated_run_id(run_id)
        with self._lock:
            payload = self._snapshots.get(exact_run_id)
        if payload is None:
            raise AgentRunNotFound("Agent run was not found")
        return AgentRunSnapshot.from_json(payload)

    def save(
        self,
        snapshot: AgentRunSnapshot,
        *,
        expected_revision: int,
    ) -> AgentRunSnapshot:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be an exact non-negative integer")
        candidate = _replay_verified(snapshot)
        candidate_payload = candidate.to_json()
        with self._lock:
            current_payload = self._snapshots.get(candidate.run_id)
            if current_payload is None:
                raise AgentRunNotFound("Agent run was not found")
            current = AgentRunSnapshot.from_json(current_payload)
            if current.revision != expected_revision:
                if current_payload == candidate_payload:
                    return AgentRunSnapshot.from_json(current_payload)
                raise AgentRunConflict("Agent run revision changed before save")
            if candidate.revision <= expected_revision:
                raise AgentRunConflict("save must append one or more Agent run revisions")
            if _immutable_identity(candidate) != _immutable_identity(current):
                raise AgentRunConflict("immutable Agent run identity cannot change")
            if candidate.events[: expected_revision + 1] != current.events:
                raise AgentRunConflict("save must preserve the complete prior event chain")
            self._snapshots[candidate.run_id] = candidate_payload
        return AgentRunSnapshot.from_json(candidate_payload)


__all__ = [
    "AgentRunConflict",
    "AgentRunNotFound",
    "AgentRunRepository",
    "AgentRunRepositoryError",
    "InMemoryAgentRunRepository",
]
