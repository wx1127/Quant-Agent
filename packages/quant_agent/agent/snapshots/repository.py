"""Write-once repositories for immutable decision snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Protocol

from .contracts import DecisionSnapshot

_SCHEMA_VERSION = 1
_TABLE_NAME = "decision_snapshots"
_DDL = """
CREATE TABLE IF NOT EXISTS decision_snapshots (
    decision_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    snapshot_json BLOB NOT NULL
) WITHOUT ROWID
"""
_NO_UPDATE_TRIGGER = "decision_snapshots_no_update"
_NO_DELETE_TRIGGER = "decision_snapshots_no_delete"
_NO_REPLACE_TRIGGER = "decision_snapshots_no_replace"
_TRIGGER_NAMES = (
    _NO_UPDATE_TRIGGER,
    _NO_DELETE_TRIGGER,
    _NO_REPLACE_TRIGGER,
)
_TRIGGER_DDLS = (
    f"""
CREATE TRIGGER {_NO_UPDATE_TRIGGER}
BEFORE UPDATE ON decision_snapshots
BEGIN
    SELECT RAISE(ABORT, 'decision snapshots are immutable');
END
""",
    f"""
CREATE TRIGGER {_NO_DELETE_TRIGGER}
BEFORE DELETE ON decision_snapshots
BEGIN
    SELECT RAISE(ABORT, 'decision snapshots are immutable');
END
""",
    f"""
CREATE TRIGGER {_NO_REPLACE_TRIGGER}
BEFORE INSERT ON decision_snapshots
WHEN EXISTS (
    SELECT 1 FROM decision_snapshots WHERE decision_id = NEW.decision_id
)
BEGIN
    SELECT RAISE(ABORT, 'decision snapshots are immutable');
END
""",
)
_EXPECTED_COLUMNS = (
    ("decision_id", "TEXT", 1, 1),
    ("content_hash", "TEXT", 1, 0),
    ("snapshot_json", "BLOB", 1, 0),
)
_EXPECTED_SQL = "".join(_DDL.lower().split()).replace(
    "createtableifnotexists",
    "createtable",
    1,
)
_EXPECTED_TRIGGERS = {
    name: "".join(ddl.lower().split())
    for name, ddl in zip(
        _TRIGGER_NAMES,
        _TRIGGER_DDLS,
        strict=True,
    )
}


class DecisionSnapshotRepositoryError(RuntimeError):
    """Base class for authoritative decision-snapshot storage failures."""


class DecisionSnapshotNotFound(DecisionSnapshotRepositoryError):
    """Raised when a decision identity has never been stored."""


class DecisionSnapshotConflict(DecisionSnapshotRepositoryError):
    """Raised when a write-once decision identity is reused with new content."""


class DecisionSnapshotRepository(Protocol):
    """Minimal write-once persistence boundary for decision snapshots."""

    def create(self, snapshot: DecisionSnapshot) -> DecisionSnapshot:
        """Store a new snapshot or replay its exact previously stored value."""

    def get(self, decision_id: str) -> DecisionSnapshot:
        """Return the exact immutable snapshot for one decision identity."""


class InMemoryDecisionSnapshotRepository:
    """Thread-safe process-local repository retaining only canonical JSON."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, str] = {}

    def create(self, snapshot: DecisionSnapshot) -> DecisionSnapshot:
        canonical, payload = _canonical_snapshot(snapshot)
        with self._lock:
            stored_payload = self._snapshots.get(canonical.decision_id)
            if stored_payload is None:
                self._snapshots[canonical.decision_id] = payload
                return _load_snapshot(payload)
            stored = _load_snapshot(stored_payload)
            if stored_payload == payload and stored == canonical:
                return stored
            raise DecisionSnapshotConflict(
                "decision snapshot identity is already stored with different content"
            )

    def get(self, decision_id: str) -> DecisionSnapshot:
        normalized_id = _non_empty(decision_id, "decision_id")
        with self._lock:
            payload = self._snapshots.get(normalized_id)
            if payload is None:
                raise DecisionSnapshotNotFound(f"unknown decision snapshot: {normalized_id}")
            snapshot = _load_snapshot(payload)
            if snapshot.decision_id != normalized_id:
                raise DecisionSnapshotRepositoryError(
                    "stored decision snapshot identity does not match its key"
                )
            return snapshot


class SQLiteDecisionSnapshotRepository:
    """File-backed write-once repository with cross-process atomic creation."""

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str) and not path.strip():
            raise ValueError("decision snapshot repository path must be non-empty")
        self._path = Path(path).expanduser().resolve()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DecisionSnapshotRepositoryError(
                "decision snapshot repository directory cannot be created"
            ) from error
        with self._transaction(immediate=True) as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            if (
                row is None
                or len(row) != 1
                or not isinstance(row[0], int)
                or isinstance(row[0], bool)
            ):
                raise DecisionSnapshotRepositoryError(
                    "stored decision snapshot schema version is invalid"
                )
            schema_version = row[0]
            if schema_version not in {0, _SCHEMA_VERSION}:
                raise DecisionSnapshotRepositoryError(
                    f"unsupported decision snapshot repository schema version: {schema_version}"
                )
            if schema_version == 0:
                connection.execute(_DDL)
                for ddl in _TRIGGER_DDLS:
                    connection.execute(ddl)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._verify_schema(connection)

    @property
    def path(self) -> Path:
        return self._path

    def create(self, snapshot: DecisionSnapshot) -> DecisionSnapshot:
        canonical, payload = _canonical_snapshot(snapshot)
        with self._transaction(immediate=True) as connection:
            self._verify_schema(connection)
            row = connection.execute(
                "SELECT decision_id, content_hash, snapshot_json "
                "FROM decision_snapshots WHERE decision_id = ?",
                (canonical.decision_id,),
            ).fetchone()
            if row is not None:
                stored, stored_payload = _load_bound_row(row)
                if stored_payload == payload and stored == canonical:
                    return stored
                raise DecisionSnapshotConflict(
                    "decision snapshot identity is already stored with different content"
                )
            connection.execute(
                "INSERT INTO decision_snapshots("
                "decision_id, content_hash, snapshot_json"
                ") VALUES (?, ?, ?)",
                (
                    canonical.decision_id,
                    canonical.content_hash,
                    payload.encode("utf-8"),
                ),
            )
            return canonical

    def get(self, decision_id: str) -> DecisionSnapshot:
        normalized_id = _non_empty(decision_id, "decision_id")
        with self._transaction() as connection:
            self._verify_schema(connection)
            row = connection.execute(
                "SELECT decision_id, content_hash, snapshot_json "
                "FROM decision_snapshots WHERE decision_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise DecisionSnapshotNotFound(f"unknown decision snapshot: {normalized_id}")
            snapshot, _ = _load_bound_row(row)
            return snapshot

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        if version_row != (_SCHEMA_VERSION,):
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot schema version is invalid"
            )
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot database failed SQLite quick_check"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot database has foreign-key violations"
            )
        rows = connection.execute('PRAGMA table_info("decision_snapshots")').fetchall()
        actual: list[tuple[str, str, int, int]] = []
        for row in rows:
            if (
                len(row) < 6
                or not isinstance(row[1], str)
                or not isinstance(row[2], str)
                or not isinstance(row[3], int)
                or isinstance(row[3], bool)
                or not isinstance(row[5], int)
                or isinstance(row[5], bool)
            ):
                raise DecisionSnapshotRepositoryError(
                    "stored decision snapshot table metadata is invalid"
                )
            actual.append((row[1], row[2].upper(), row[3], row[5]))
        if tuple(actual) != _EXPECTED_COLUMNS:
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot table schema is invalid"
            )
        sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_TABLE_NAME,),
        ).fetchone()
        if (
            sql_row is None
            or len(sql_row) != 1
            or not isinstance(sql_row[0], str)
            or "".join(sql_row[0].lower().split()) != _EXPECTED_SQL
        ):
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot table definition is invalid"
            )
        trigger_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = ? ORDER BY name",
            (_TABLE_NAME,),
        ).fetchall()
        if any(
            len(row) != 2 or not isinstance(row[0], str) or not isinstance(row[1], str)
            for row in trigger_rows
        ):
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot trigger metadata is invalid"
            )
        actual_triggers = {row[0]: "".join(row[1].lower().split()) for row in trigger_rows}
        if actual_triggers != _EXPECTED_TRIGGERS:
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot immutability triggers are invalid"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(
        self,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except sqlite3.Error as error:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise DecisionSnapshotRepositoryError(
                "SQLite decision snapshot repository operation failed"
            ) from error
        except BaseException:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()


def _canonical_snapshot(snapshot: DecisionSnapshot) -> tuple[DecisionSnapshot, str]:
    if not isinstance(snapshot, DecisionSnapshot):
        raise DecisionSnapshotRepositoryError("snapshot must be a DecisionSnapshot")
    try:
        payload = snapshot.to_json()
        canonical = DecisionSnapshot.from_json(payload)
        canonical_payload = canonical.to_json()
    except (TypeError, ValueError) as error:
        raise DecisionSnapshotRepositoryError(
            "decision snapshot cannot be canonically serialized"
        ) from error
    if payload != canonical_payload:
        raise DecisionSnapshotRepositoryError("decision snapshot serialization is not canonical")
    return canonical, canonical_payload


def _load_snapshot(value: object) -> DecisionSnapshot:
    if isinstance(value, bytes):
        try:
            rendered = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DecisionSnapshotRepositoryError(
                "stored decision snapshot is not valid UTF-8"
            ) from error
    elif isinstance(value, str):
        rendered = value
    else:
        raise DecisionSnapshotRepositoryError("stored decision snapshot JSON has an invalid type")
    try:
        snapshot = DecisionSnapshot.from_json(rendered)
        canonical = snapshot.to_json()
    except (TypeError, ValueError) as error:
        raise DecisionSnapshotRepositoryError("stored decision snapshot JSON is invalid") from error
    if rendered != canonical:
        raise DecisionSnapshotRepositoryError("stored decision snapshot JSON is not canonical")
    return snapshot


def _load_bound_row(
    row: tuple[object, ...],
) -> tuple[DecisionSnapshot, str]:
    if (
        len(row) != 3
        or not isinstance(row[0], str)
        or not isinstance(row[1], str)
        or not isinstance(row[2], bytes | str)
    ):
        raise DecisionSnapshotRepositoryError("stored decision snapshot row has invalid types")
    snapshot = _load_snapshot(row[2])
    if snapshot.decision_id != row[0] or snapshot.content_hash != row[1]:
        raise DecisionSnapshotRepositoryError("stored decision snapshot row does not bind its JSON")
    return snapshot, snapshot.to_json()


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty without surrounding whitespace")
    return value


__all__ = [
    "DecisionSnapshotConflict",
    "DecisionSnapshotNotFound",
    "DecisionSnapshotRepository",
    "DecisionSnapshotRepositoryError",
    "InMemoryDecisionSnapshotRepository",
    "SQLiteDecisionSnapshotRepository",
]
