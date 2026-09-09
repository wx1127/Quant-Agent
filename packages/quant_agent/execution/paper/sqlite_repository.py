"""Durable SQLite repository for paper accounts and idempotent receipts."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from quant_agent.portfolio import AccountSnapshot

from .contracts import (
    PaperAccountState,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperIdempotencyConflict,
)
from .repository import (
    InMemoryPaperRepository,
    PaperAccountConflict,
    PaperAccountNotFound,
    PaperConcurrentUpdate,
    PaperRepositoryError,
)
from .transition import replay_account_refresh

_STATE_ADAPTER = TypeAdapter(PaperAccountState)
_RECEIPT_ADAPTER = TypeAdapter(PaperExecutionReceipt)


class SQLitePaperRepository:
    """File-backed paper repository with transactional CAS and durable deduplication."""

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str) and not path.strip():
            raise ValueError("paper repository path must be non-empty")
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if schema_version not in {0, 1}:
                raise PaperRepositoryError(
                    f"unsupported paper repository schema version: {schema_version}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS paper_accounts (
                    account_id TEXT PRIMARY KEY,
                    state_hash TEXT NOT NULL,
                    state_json BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_receipts (
                    account_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    batch_hash TEXT NOT NULL,
                    receipt_json BLOB NOT NULL,
                    PRIMARY KEY (account_id, idempotency_key),
                    UNIQUE (account_id, batch_hash),
                    FOREIGN KEY (account_id) REFERENCES paper_accounts(account_id)
                );
                CREATE TABLE IF NOT EXISTS paper_refreshes (
                    account_id TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    account_before_hash TEXT NOT NULL,
                    account_after_hash TEXT NOT NULL,
                    event_log_before_hash TEXT NOT NULL,
                    event_log_after_hash TEXT NOT NULL,
                    snapshot_json BLOB NOT NULL,
                    state_before_json BLOB NOT NULL,
                    state_after_json BLOB NOT NULL,
                    PRIMARY KEY (account_id, snapshot_hash),
                    FOREIGN KEY (account_id) REFERENCES paper_accounts(account_id)
                );
                """
            )
            refresh_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(paper_refreshes)").fetchall()
            }
            if "state_before_json" not in refresh_columns:
                connection.execute("ALTER TABLE paper_refreshes ADD COLUMN state_before_json BLOB")
            if schema_version == 0:
                connection.execute("PRAGMA user_version = 1")

    @property
    def path(self) -> Path:
        return self._path

    def open_account(self, state: PaperAccountState) -> PaperAccountState:
        """Create an account or return its exact durable state idempotently."""

        state = _round_trip_state(state)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT account_id, state_hash, state_json "
                "FROM paper_accounts WHERE account_id = ?",
                (state.account_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO paper_accounts(account_id, state_hash, state_json) "
                    "VALUES (?, ?, ?)",
                    (state.account_id, state.state_hash, _dump_state(state)),
                )
                return state
            existing = _load_bound_state(row)
            if existing.state_hash == state.state_hash and existing == state:
                return existing
            raise PaperAccountConflict(
                f"paper account already exists with different state: {state.account_id}"
            )

    def get_account(self, account_id: str) -> PaperAccountState:
        """Return the current durable state for an opened account."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT account_id, state_hash, state_json "
                "FROM paper_accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if row is None:
            raise PaperAccountNotFound(f"unknown paper account: {account_id}")
        return _load_bound_state(row)

    def refresh_account(
        self,
        expected_state_hash: str,
        snapshot: AccountSnapshot,
        state_after: PaperAccountState,
    ) -> PaperAccountState:
        """CAS-commit a canonical refresh in one durable transaction."""

        try:
            snapshot = AccountSnapshot.from_json(snapshot.to_json())
        except (AttributeError, TypeError, ValueError) as error:
            raise PaperAccountConflict("paper refresh snapshot is invalid") from error
        state_after = _round_trip_state(state_after)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_refresh = self._refresh_for(
                connection,
                account_id=snapshot.account_id,
                snapshot_hash=snapshot.content_hash,
            )
            if existing_refresh is not None:
                if existing_refresh == state_after:
                    return existing_refresh
                raise PaperAccountConflict(
                    "paper refresh snapshot is already committed with a different state"
                )
            current = self._current_for_update(connection, snapshot.account_id)
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
            self._update_account(connection, state_after)
            connection.execute(
                "INSERT INTO paper_refreshes("
                "account_id, snapshot_hash, account_before_hash, account_after_hash, "
                "event_log_before_hash, event_log_after_hash, snapshot_json, "
                "state_before_json, state_after_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.account_id,
                    snapshot.content_hash,
                    current.state_hash,
                    state_after.state_hash,
                    current.event_log_hash,
                    state_after.event_log_hash,
                    snapshot.to_json().encode("utf-8"),
                    _dump_state(current),
                    _dump_state(state_after),
                ),
            )
            return state_after

    def account_by_refresh_snapshot_hash(
        self,
        account_id: str,
        snapshot_hash: str,
    ) -> PaperAccountState | None:
        """Return the durable state originally committed for a refresh snapshot."""

        with self._connection() as connection:
            return self._refresh_for(
                connection,
                account_id=account_id,
                snapshot_hash=snapshot_hash,
            )

    def receipt_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> PaperExecutionReceipt | None:
        """Return a durable receipt by its account-scoped delivery key."""

        return self._receipt_for(
            "idempotency_key",
            account_id=account_id,
            identity=idempotency_key,
        )

    def _refresh_for(
        self,
        connection: sqlite3.Connection,
        *,
        account_id: str,
        snapshot_hash: str,
    ) -> PaperAccountState | None:
        row = connection.execute(
            "SELECT account_id, snapshot_hash, account_before_hash, account_after_hash, "
            "event_log_before_hash, event_log_after_hash, snapshot_json, state_before_json, "
            "state_after_json "
            "FROM paper_refreshes WHERE account_id = ? AND snapshot_hash = ?",
            (account_id, snapshot_hash),
        ).fetchone()
        return None if row is None else _load_bound_refresh(row)

    def receipt_by_batch_hash(
        self,
        account_id: str,
        batch_hash: str,
    ) -> PaperExecutionReceipt | None:
        """Return the durable receipt that consumed a draft batch."""

        return self._receipt_for(
            "batch_hash",
            account_id=account_id,
            identity=batch_hash,
        )

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        """Atomically persist account state, receipt, key, and batch identities."""

        request = replace(request)
        state_after = _round_trip_state(state_after)
        receipt = _round_trip_receipt(receipt)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            indexed = connection.execute(
                "SELECT account_id, idempotency_key, request_hash, batch_hash, receipt_json "
                "FROM paper_receipts "
                "WHERE account_id = ? AND idempotency_key = ?",
                (request.account_id, request.idempotency_key),
            ).fetchone()
            if indexed is not None:
                committed_receipt = _load_bound_receipt(indexed)
                if committed_receipt.request_hash == request.request_hash:
                    return committed_receipt
                raise PaperIdempotencyConflict(
                    "paper idempotency key belongs to a different execution request"
                )
            duplicate_batch = connection.execute(
                "SELECT account_id, idempotency_key, request_hash, batch_hash, receipt_json "
                "FROM paper_receipts "
                "WHERE account_id = ? AND batch_hash = ?",
                (request.account_id, request.batch_hash),
            ).fetchone()
            if duplicate_batch is not None:
                committed_receipt = _load_bound_receipt(duplicate_batch)
                if committed_receipt.request_hash == request.request_hash:
                    return committed_receipt
                raise PaperIdempotencyConflict(
                    "paper draft batch belongs to a different execution request"
                )
            current = self._current_for_update(connection, request.account_id)
            if current.state_hash != expected_state_hash:
                raise PaperConcurrentUpdate(
                    f"paper account changed before execution commit: {request.account_id}"
                )
            InMemoryPaperRepository._validate_transition(
                expected_state_hash=expected_state_hash,
                current=current,
                request=request,
                state_after=state_after,
                receipt=receipt,
            )
            self._update_account(connection, state_after)
            connection.execute(
                "INSERT INTO paper_receipts("
                "account_id, idempotency_key, request_hash, batch_hash, receipt_json"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    request.account_id,
                    request.idempotency_key,
                    request.request_hash,
                    request.batch_hash,
                    _dump_receipt(receipt),
                ),
            )
            return receipt

    def _receipt_for(
        self,
        column: str,
        *,
        account_id: str,
        identity: str,
    ) -> PaperExecutionReceipt | None:
        if column not in {"idempotency_key", "batch_hash"}:
            raise ValueError("unsupported paper receipt identity")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT account_id, idempotency_key, request_hash, batch_hash, receipt_json "
                f"FROM paper_receipts WHERE account_id = ? AND {column} = ?",
                (account_id, identity),
            ).fetchone()
        return None if row is None else _load_bound_receipt(row)

    def _current_for_update(
        self,
        connection: sqlite3.Connection,
        account_id: str,
    ) -> PaperAccountState:
        row = connection.execute(
            "SELECT account_id, state_hash, state_json FROM paper_accounts WHERE account_id = ?",
            (account_id,),
        ).fetchone()
        if row is None:
            raise PaperAccountNotFound(f"unknown paper account: {account_id}")
        return _load_bound_state(row)

    @staticmethod
    def _update_account(
        connection: sqlite3.Connection,
        state: PaperAccountState,
    ) -> None:
        connection.execute(
            "UPDATE paper_accounts SET state_hash = ?, state_json = ? WHERE account_id = ?",
            (state.state_hash, _dump_state(state), state.account_id),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            with connection:
                yield connection
        except sqlite3.Error as error:
            raise PaperRepositoryError("SQLite paper repository operation failed") from error
        finally:
            if connection is not None:
                connection.close()


def _dump_state(state: PaperAccountState) -> bytes:
    return _STATE_ADAPTER.dump_json(state)


def _load_state(value: bytes | str) -> PaperAccountState:
    try:
        return _STATE_ADAPTER.validate_json(value)
    except (ValidationError, TypeError, ValueError) as error:
        raise PaperRepositoryError("stored paper account state is invalid") from error


def _load_bound_state(row: tuple[object, ...]) -> PaperAccountState:
    if (
        len(row) != 3
        or not isinstance(row[0], str)
        or not isinstance(row[1], str)
        or not isinstance(row[2], bytes | str)
    ):
        raise PaperRepositoryError("stored paper account row identity is invalid")
    state = _load_state(row[2])
    if state.account_id != row[0] or state.state_hash != row[1]:
        raise PaperRepositoryError("stored paper account row identity does not match its state")
    return state


def _round_trip_state(state: PaperAccountState) -> PaperAccountState:
    return _load_state(_dump_state(state))


def _load_snapshot(value: bytes | str) -> AccountSnapshot:
    try:
        rendered = value.decode("utf-8") if isinstance(value, bytes) else value
        return AccountSnapshot.from_json(rendered)
    except (UnicodeDecodeError, TypeError, ValueError) as error:
        raise PaperRepositoryError("stored paper refresh snapshot is invalid") from error


def _load_bound_refresh(row: tuple[object, ...]) -> PaperAccountState:
    if (
        len(row) != 9
        or any(not isinstance(item, str) for item in row[:6])
        or not isinstance(row[6], bytes | str)
        or not isinstance(row[7], bytes | str)
        or not isinstance(row[8], bytes | str)
    ):
        raise PaperRepositoryError("stored paper refresh row identity is invalid")
    account_id = row[0]
    snapshot_hash = row[1]
    account_before_hash = row[2]
    account_after_hash = row[3]
    event_log_before_hash = row[4]
    event_log_after_hash = row[5]
    assert isinstance(account_id, str)
    assert isinstance(snapshot_hash, str)
    assert isinstance(account_before_hash, str)
    assert isinstance(account_after_hash, str)
    assert isinstance(event_log_before_hash, str)
    assert isinstance(event_log_after_hash, str)
    assert isinstance(row[6], bytes | str)
    assert isinstance(row[7], bytes | str)
    assert isinstance(row[8], bytes | str)
    snapshot = _load_snapshot(row[6])
    state_before = _load_state(row[7])
    state_after = _load_state(row[8])
    if (
        snapshot.account_id != account_id
        or snapshot.content_hash != snapshot_hash
        or snapshot.source_event_log_hash != event_log_before_hash
        or state_before.account_id != account_id
        or state_before.state_hash != account_before_hash
        or state_before.event_log_hash != event_log_before_hash
        or state_after.account_id != account_id
        or state_after.source_snapshot_hash != snapshot_hash
        or state_after.source_snapshot_id != snapshot.snapshot_id
        or state_after.source_snapshot_as_of != snapshot.as_of
        or state_after.as_of != snapshot.as_of
        or state_after.data_version != snapshot.data_version
        or state_after.currency != snapshot.currency
        or state_after.previous_state_hash != account_before_hash
        or state_after.state_hash != account_after_hash
        or state_after.event_log_hash != event_log_after_hash
    ):
        raise PaperRepositoryError("stored paper refresh row identity does not match its state")
    try:
        expected_state_after = replay_account_refresh(
            account_before=state_before,
            snapshot=snapshot,
        )
    except (PaperExecutionInputError, TypeError, ValueError) as error:
        raise PaperRepositoryError("stored paper refresh cannot be replayed") from error
    if state_after != expected_state_after:
        raise PaperRepositoryError("stored paper refresh state does not match canonical replay")
    return state_after


def _dump_receipt(receipt: PaperExecutionReceipt) -> bytes:
    return _RECEIPT_ADAPTER.dump_json(receipt)


def _load_receipt(value: bytes | str) -> PaperExecutionReceipt:
    try:
        return _RECEIPT_ADAPTER.validate_json(value)
    except (ValidationError, TypeError, ValueError) as error:
        raise PaperRepositoryError("stored paper execution receipt is invalid") from error


def _load_bound_receipt(row: tuple[object, ...]) -> PaperExecutionReceipt:
    if (
        len(row) != 5
        or not isinstance(row[0], str)
        or not isinstance(row[1], str)
        or not isinstance(row[2], str)
        or not isinstance(row[3], str)
        or not isinstance(row[4], bytes | str)
    ):
        raise PaperRepositoryError("stored paper receipt row identity is invalid")
    receipt = _load_receipt(row[4])
    if (
        receipt.account_after.account_id != row[0]
        or receipt.idempotency_key != row[1]
        or receipt.request_hash != row[2]
        or receipt.batch_hash != row[3]
    ):
        raise PaperRepositoryError("stored paper receipt row identity does not match its receipt")
    return receipt


def _round_trip_receipt(receipt: PaperExecutionReceipt) -> PaperExecutionReceipt:
    return _load_receipt(_dump_receipt(receipt))


__all__ = ["SQLitePaperRepository"]
