"""Atomic storage boundary for non-executable order-draft artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import ClassVar, Protocol

from pydantic import ConfigDict, TypeAdapter, ValidationError

from quant_agent.execution.order_drafts import OrderDraftBatch


class DraftArtifactStoreError(RuntimeError):
    """Base class for draft artifact persistence failures."""


class DraftArtifactNotFound(DraftArtifactStoreError):
    """No draft exists under the requested account-scoped identity."""


class DraftArtifactConflict(DraftArtifactStoreError):
    """An idempotency key or batch identity is already bound differently."""


@dataclass(frozen=True, slots=True)
class DraftArtifact:
    """Stored identity envelope around one immutable, non-executable batch."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    account_id: str
    decision_id: str
    idempotency_key: str
    draft: OrderDraftBatch


class DraftArtifactStore(Protocol):
    """Atomic account-scoped storage required by order-draft tools."""

    def by_idempotency_key(
        self,
        account_id: str,
        decision_id: str,
        idempotency_key: str,
    ) -> DraftArtifact | None:
        """Return the exact artifact previously written for a delivery key."""

        ...

    def put(self, artifact: DraftArtifact) -> DraftArtifact:
        """Atomically create or replay one exact draft artifact."""

        ...

    def get(self, account_id: str, batch_hash: str) -> DraftArtifact:
        """Return one account-scoped artifact or raise ``DraftArtifactNotFound``."""

        ...


def _exact_text(value: object, label: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > maximum:
        raise DraftArtifactStoreError(f"{label} must be an exact bounded string")
    if not value.isprintable():
        raise DraftArtifactStoreError(f"{label} must be printable")
    return value


def _detach_artifact(value: object) -> DraftArtifact:
    if type(value) is not DraftArtifact:
        raise DraftArtifactStoreError("draft artifact must have the exact trusted type")
    try:
        adapter = TypeAdapter(DraftArtifact)
        detached = adapter.validate_json(adapter.dump_json(value, warnings="error"), strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise DraftArtifactStoreError("draft artifact failed strict revalidation") from error
    if type(detached) is not DraftArtifact:
        raise DraftArtifactStoreError("draft artifact failed exact-type revalidation")
    return detached


class InMemoryDraftArtifactStore:
    """Thread-safe local/test store with atomic idempotency and batch indexes.

    Production composition should replace this process-local implementation with
    a durable transactional repository.  It is intentionally never constructed
    implicitly by the toolset.
    """

    __slots__ = ("_by_batch", "_by_key", "_lock")

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_key: dict[tuple[str, str, str], DraftArtifact] = {}
        self._by_batch: dict[tuple[str, str], DraftArtifact] = {}

    def by_idempotency_key(
        self,
        account_id: str,
        decision_id: str,
        idempotency_key: str,
    ) -> DraftArtifact | None:
        key = (
            _exact_text(account_id, "account_id"),
            _exact_text(decision_id, "decision_id"),
            _exact_text(idempotency_key, "idempotency_key", maximum=512),
        )
        with self._lock:
            value = self._by_key.get(key)
            return _detach_artifact(value) if value is not None else None

    def put(self, artifact: DraftArtifact) -> DraftArtifact:
        detached = _detach_artifact(artifact)
        account_id = _exact_text(detached.account_id, "account_id")
        decision_id = _exact_text(detached.decision_id, "decision_id")
        idempotency_key = _exact_text(
            detached.idempotency_key,
            "idempotency_key",
            maximum=512,
        )
        if detached.draft.decision_id != decision_id:
            raise DraftArtifactStoreError("artifact decision does not match its draft")
        key = (account_id, decision_id, idempotency_key)
        batch_key = (account_id, detached.draft.batch_hash)
        with self._lock:
            existing_key = self._by_key.get(key)
            if existing_key is not None:
                if existing_key.draft != detached.draft:
                    raise DraftArtifactConflict(
                        "draft idempotency key belongs to different artifact content"
                    )
                return _detach_artifact(existing_key)
            existing_batch = self._by_batch.get(batch_key)
            if existing_batch is not None and existing_batch.draft != detached.draft:
                raise DraftArtifactConflict("draft batch hash belongs to different content")
            # Preserve the delivery-key envelope for exact lookup semantics while
            # keeping one canonical content-addressed batch entry.
            self._by_key[key] = detached
            self._by_batch.setdefault(batch_key, detached)
            return _detach_artifact(detached)

    def get(self, account_id: str, batch_hash: str) -> DraftArtifact:
        key = (
            _exact_text(account_id, "account_id"),
            _exact_text(batch_hash, "batch_hash", maximum=64),
        )
        with self._lock:
            value = self._by_batch.get(key)
            if value is None:
                raise DraftArtifactNotFound("unknown account-scoped order draft")
            return _detach_artifact(value)


__all__ = [
    "DraftArtifact",
    "DraftArtifactConflict",
    "DraftArtifactNotFound",
    "DraftArtifactStore",
    "DraftArtifactStoreError",
    "InMemoryDraftArtifactStore",
]
