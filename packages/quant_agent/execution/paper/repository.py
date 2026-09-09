"""Thread-safe repository boundary for deterministic paper execution."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from quant_agent.execution.paper.contracts import (
    PaperAccountState,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperIdempotencyConflict,
)
from quant_agent.execution.paper.transition import (
    replay_account_refresh,
    replay_execution_transition,
)
from quant_agent.portfolio import AccountSnapshot


class PaperRepositoryError(RuntimeError):
    """Base class for paper-account repository failures."""


class PaperAccountNotFound(PaperRepositoryError):
    """Raised when a requested paper account has not been opened."""


class PaperAccountConflict(PaperRepositoryError):
    """Raised when immutable paper-account identities disagree."""


class PaperConcurrentUpdate(PaperAccountConflict):
    """Raised when the current paper-account state no longer matches a CAS token."""


class PaperRepository(Protocol):
    """Atomic persistence required by a paper-execution engine."""

    def open_account(self, state: PaperAccountState) -> PaperAccountState:
        """Create an account or return its exact existing state idempotently."""

    def get_account(self, account_id: str) -> PaperAccountState:
        """Return the current state for an opened paper account."""

    def refresh_account(
        self,
        expected_state_hash: str,
        snapshot: AccountSnapshot,
        state_after: PaperAccountState,
    ) -> PaperAccountState:
        """CAS-commit a canonical account refresh and settlement transition."""

    def account_by_refresh_snapshot_hash(
        self,
        account_id: str,
        snapshot_hash: str,
    ) -> PaperAccountState | None:
        """Return the state originally committed for a refresh snapshot."""

    def receipt_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> PaperExecutionReceipt | None:
        """Return the receipt committed under an account-scoped delivery key."""

    def receipt_by_batch_hash(
        self,
        account_id: str,
        batch_hash: str,
    ) -> PaperExecutionReceipt | None:
        """Return the receipt that already consumed an account-scoped draft batch."""

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        """Commit one state transition and its idempotency indexes atomically."""


class InMemoryPaperRepository:
    """Process-local, credential-free paper repository with optimistic concurrency."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._accounts: dict[str, PaperAccountState] = {}
        self._receipts_by_key: dict[
            tuple[str, str],
            tuple[str, PaperExecutionReceipt],
        ] = {}
        self._receipts_by_batch: dict[tuple[str, str], PaperExecutionReceipt] = {}
        self._states_by_refresh: dict[tuple[str, str], PaperAccountState] = {}

    def open_account(self, state: PaperAccountState) -> PaperAccountState:
        """Create an account or return its exact existing state idempotently."""

        with self._lock:
            existing = self._accounts.get(state.account_id)
            if existing is None:
                self._accounts[state.account_id] = state
                return state
            if existing.state_hash == state.state_hash and existing == state:
                return existing
            raise PaperAccountConflict(
                f"paper account already exists with different state: {state.account_id}"
            )

    def get_account(self, account_id: str) -> PaperAccountState:
        """Return the current immutable state for an opened account."""

        with self._lock:
            state = self._accounts.get(account_id)
            if state is None:
                raise PaperAccountNotFound(f"unknown paper account: {account_id}")
            return state

    def refresh_account(
        self,
        expected_state_hash: str,
        snapshot: AccountSnapshot,
        state_after: PaperAccountState,
    ) -> PaperAccountState:
        """CAS-commit an exact snapshot-linked refresh transition."""

        try:
            snapshot = AccountSnapshot.from_json(snapshot.to_json())
        except (AttributeError, TypeError, ValueError) as error:
            raise PaperAccountConflict("paper refresh snapshot is invalid") from error
        with self._lock:
            refresh_identity = (snapshot.account_id, snapshot.content_hash)
            existing_refresh = self._states_by_refresh.get(refresh_identity)
            if existing_refresh is not None:
                if existing_refresh == state_after:
                    return existing_refresh
                raise PaperAccountConflict(
                    "paper refresh snapshot is already committed with a different state"
                )
            current = self._accounts.get(snapshot.account_id)
            if current is None:
                raise PaperAccountNotFound(f"unknown paper account: {snapshot.account_id}")
            if current.state_hash != expected_state_hash:
                raise PaperConcurrentUpdate(
                    f"paper account changed before refresh commit: {snapshot.account_id}"
                )
            try:
                expected = replay_account_refresh(
                    account_before=current,
                    snapshot=snapshot,
                )
            except (PaperExecutionInputError, TypeError, ValueError) as error:
                raise PaperAccountConflict("paper account refresh cannot be replayed") from error
            if state_after != expected:
                raise PaperAccountConflict(
                    "paper account state does not match canonical refresh replay"
                )
            self._accounts[snapshot.account_id] = state_after
            self._states_by_refresh[refresh_identity] = state_after
            return state_after

    def account_by_refresh_snapshot_hash(
        self,
        account_id: str,
        snapshot_hash: str,
    ) -> PaperAccountState | None:
        """Return the state originally committed for a refresh snapshot."""

        with self._lock:
            return self._states_by_refresh.get((account_id, snapshot_hash))

    def receipt_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> PaperExecutionReceipt | None:
        """Return a committed receipt by its account-scoped delivery identity."""

        with self._lock:
            indexed = self._receipts_by_key.get((account_id, idempotency_key))
            return indexed[1] if indexed is not None else None

    def receipt_by_batch_hash(
        self,
        account_id: str,
        batch_hash: str,
    ) -> PaperExecutionReceipt | None:
        """Return the receipt that already consumed a draft batch, if any."""

        with self._lock:
            return self._receipts_by_batch.get((account_id, batch_hash))

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        """CAS-commit one execution while preventing key and batch replay."""

        account_id = request.account_id
        key_identity = (account_id, request.idempotency_key)
        batch_identity = (account_id, request.batch_hash)
        with self._lock:
            indexed = self._receipts_by_key.get(key_identity)
            if indexed is not None:
                committed_request_hash, committed_receipt = indexed
                if committed_request_hash == request.request_hash:
                    return committed_receipt
                raise PaperIdempotencyConflict(
                    "paper idempotency key belongs to a different execution request"
                )

            duplicate_batch = self._receipts_by_batch.get(batch_identity)
            if duplicate_batch is not None:
                if duplicate_batch.request_hash == request.request_hash:
                    return duplicate_batch
                raise PaperIdempotencyConflict(
                    "paper draft batch belongs to a different execution request"
                )

            current = self._accounts.get(account_id)
            if current is None:
                raise PaperAccountNotFound(f"unknown paper account: {account_id}")
            if current.state_hash != expected_state_hash:
                raise PaperConcurrentUpdate(
                    f"paper account changed before execution commit: {account_id}"
                )

            self._validate_transition(
                expected_state_hash=expected_state_hash,
                current=current,
                request=request,
                state_after=state_after,
                receipt=receipt,
            )

            self._accounts[account_id] = state_after
            self._receipts_by_key[key_identity] = (request.request_hash, receipt)
            self._receipts_by_batch[batch_identity] = receipt
            return receipt

    @staticmethod
    def _validate_transition(
        *,
        expected_state_hash: str,
        current: PaperAccountState,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> None:
        if request.expected_account_state_hash != expected_state_hash:
            raise PaperAccountConflict("repository CAS token does not match the execution request")
        if request.submitted_at < current.as_of:
            raise PaperAccountConflict("execution request cannot precede the paper account state")
        if request.expected_account_snapshot_hash != current.source_snapshot_hash:
            raise PaperAccountConflict(
                "execution request does not bind to the paper account source snapshot"
            )
        if state_after.account_id != current.account_id:
            raise PaperAccountConflict("paper account identity cannot change during commit")
        if state_after.previous_state_hash != expected_state_hash:
            raise PaperAccountConflict("paper account state does not extend the CAS state")
        if state_after.as_of < current.as_of:
            raise PaperAccountConflict("paper account state time cannot move backwards")
        if (
            state_after.runtime_mode is not current.runtime_mode
            or state_after.currency != current.currency
            or state_after.source_snapshot_id != current.source_snapshot_id
            or state_after.source_snapshot_hash != current.source_snapshot_hash
            or state_after.source_snapshot_as_of != current.source_snapshot_as_of
        ):
            raise PaperAccountConflict(
                "paper account runtime, currency, and source identities must remain stable"
            )
        if (
            receipt.request_id != request.request_id
            or receipt.request_hash != request.request_hash
            or receipt.idempotency_key != request.idempotency_key
            or receipt.batch_hash != request.batch_hash
            or receipt.request_submitted_at != request.submitted_at
        ):
            raise PaperAccountConflict("paper receipt does not match the execution request")
        if (
            receipt.account_before_hash != expected_state_hash
            or receipt.event_log_before_hash != current.event_log_hash
        ):
            raise PaperAccountConflict("paper receipt does not extend the current account state")
        try:
            expected_state_after = replay_execution_transition(
                account_before=current,
                batch_hash=request.batch_hash,
                processed_at=receipt.processed_at,
                event_log_hash=receipt.event_log_hash,
                orders=receipt.orders,
                fills=receipt.fills,
            )
        except (PaperExecutionInputError, TypeError, ValueError) as error:
            raise PaperAccountConflict(
                "paper receipt cannot be replayed from the current account state"
            ) from error
        if state_after != expected_state_after:
            raise PaperAccountConflict(
                "paper account state does not match canonical execution replay"
            )
        if (
            receipt.account_after_hash != state_after.state_hash
            or receipt.account_after != state_after
        ):
            raise PaperAccountConflict("paper receipt account_after does not match committed state")


__all__ = [
    "InMemoryPaperRepository",
    "PaperAccountConflict",
    "PaperAccountNotFound",
    "PaperConcurrentUpdate",
    "PaperRepository",
    "PaperRepositoryError",
]
