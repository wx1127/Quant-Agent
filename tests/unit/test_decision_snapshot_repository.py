"""Write-once and durability tests for decision-snapshot repositories."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Protocol

import pytest

import quant_agent.agent.snapshots.repository as snapshot_repository_module
from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.snapshots.repository import (
    DecisionSnapshotConflict,
    DecisionSnapshotNotFound,
    DecisionSnapshotRepository,
    DecisionSnapshotRepositoryError,
    InMemoryDecisionSnapshotRepository,
    SQLiteDecisionSnapshotRepository,
)
from quant_agent.config import RuntimeMode
from quant_agent.regime.contracts import stable_hash

AS_OF = datetime(2026, 9, 7, 7, 0, tzinfo=UTC)


class _RepositoryFactory(Protocol):
    def __call__(self) -> DecisionSnapshotRepository: ...


def _digest(label: str) -> str:
    return stable_hash({"decision-snapshot-repository-fixture": label})


def _strategy_ref(label: str = "v1") -> StrategySnapshotRef:
    return StrategySnapshotRef(
        strategy_name="etf-rotation",
        strategy_version=f"etf-rotation-{label}",
        config_hash=_digest(f"strategy-config:{label}"),
        parameter_version=f"parameters-{label}",
        parameter_hash=_digest(f"parameters:{label}"),
        registered_at=AS_OF - timedelta(days=1),
    )


def _snapshot(
    decision_id: str = "decision-001",
    *,
    data_version: str = "snapshot-v1",
    data_label: str = "v1",
) -> DecisionSnapshot:
    return DecisionSnapshot.build(
        decision_id=decision_id,
        mode=RuntimeMode.PAPER,
        market="CN_A",
        as_of=AS_OF,
        data_version=data_version,
        data_content_hash=_digest(f"data:{data_label}"),
        strategy_refs=(_strategy_ref(),),
        risk_policy_version="portfolio-risk-v1",
        risk_policy_hash=_digest("risk-policy:v1"),
        account_id="paper-account-a",
        account_snapshot_id="account-snapshot-001",
        account_snapshot_hash=_digest("account-snapshot:v1"),
        code_commit="a" * 40,
        code_artifact_hash=_digest("code-artifact:v1"),
        agent_version="quant-agent-v1",
        model_version="decision-model-v1",
    )


@pytest.fixture(
    params=("memory", "sqlite"),
    ids=("memory", "sqlite"),
)
def repository_factory(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _RepositoryFactory:
    if request.param == "memory":
        return InMemoryDecisionSnapshotRepository
    path = tmp_path / "decision-snapshots.db"
    return lambda: SQLiteDecisionSnapshotRepository(path)


def test_create_is_write_once_and_exact_same_content_is_idempotent(
    repository_factory: _RepositoryFactory,
) -> None:
    repository = repository_factory()
    snapshot = _snapshot()

    created = repository.create(snapshot)
    replayed = repository.create(DecisionSnapshot.from_json(snapshot.to_json()))

    assert created == snapshot
    assert replayed == created
    assert repository.get(snapshot.decision_id) == created
    assert repository.get(snapshot.decision_id).to_json() == snapshot.to_json()


@pytest.mark.parametrize(
    "conflicting",
    (
        _snapshot(data_version="snapshot-v2", data_label="v1"),
        _snapshot(data_version="snapshot-v1", data_label="changed-content"),
    ),
    ids=("data-version", "data-content-hash"),
)
def test_same_decision_id_with_changed_input_version_or_hash_conflicts(
    repository_factory: _RepositoryFactory,
    conflicting: DecisionSnapshot,
) -> None:
    repository = repository_factory()
    original = repository.create(_snapshot())

    with pytest.raises(DecisionSnapshotConflict):
        repository.create(conflicting)

    assert repository.get(original.decision_id) == original


def test_new_decision_identity_can_record_a_new_input_version(
    repository_factory: _RepositoryFactory,
) -> None:
    repository = repository_factory()
    first = _snapshot("decision-001")
    second = _snapshot(
        "decision-002",
        data_version="snapshot-v2",
        data_label="v2",
    )

    assert repository.create(first) == first
    assert repository.create(second) == second
    assert repository.get(first.decision_id) == first
    assert repository.get(second.decision_id) == second


@pytest.mark.parametrize("decision_id", ("missing-decision", " ", " decision-001 "))
def test_get_rejects_missing_or_invalid_identity(
    repository_factory: _RepositoryFactory,
    decision_id: str,
) -> None:
    repository = repository_factory()

    expected = DecisionSnapshotNotFound if decision_id == decision_id.strip() else ValueError
    if not decision_id.strip():
        expected = ValueError
    with pytest.raises(expected):
        repository.get(decision_id)


def test_sqlite_reopen_preserves_snapshot_and_idempotent_replay(tmp_path: Path) -> None:
    path = tmp_path / "durable-decisions.db"
    snapshot = _snapshot()
    first = SQLiteDecisionSnapshotRepository(path)
    created = first.create(snapshot)

    reopened = SQLiteDecisionSnapshotRepository(path)

    assert reopened.path == path.resolve()
    assert reopened.get(snapshot.decision_id) == created
    assert reopened.create(snapshot) == created


@pytest.mark.parametrize(
    "schema_kind",
    (
        "future-version",
        "missing-column",
        "weakened-definition",
        "rowid-table",
        "unexpected-trigger",
    ),
)
def test_sqlite_rejects_unknown_or_modified_schema(
    tmp_path: Path,
    schema_kind: str,
) -> None:
    path = tmp_path / f"bad-schema-{schema_kind}.db"
    if schema_kind == "future-version":
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA user_version = 99")
    elif schema_kind == "missing-column":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE decision_snapshots (decision_id TEXT PRIMARY KEY)")
            connection.execute("PRAGMA user_version = 1")
    elif schema_kind == "weakened-definition":
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE decision_snapshots ("
                "decision_id TEXT PRIMARY KEY, content_hash TEXT, snapshot_json BLOB NOT NULL)"
            )
            connection.execute("PRAGMA user_version = 1")
    elif schema_kind == "rowid-table":
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE decision_snapshots ("
                "decision_id TEXT PRIMARY KEY, "
                "content_hash TEXT NOT NULL, "
                "snapshot_json BLOB NOT NULL)"
            )
            connection.execute("PRAGMA user_version = 1")
    else:
        SQLiteDecisionSnapshotRepository(path)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TRIGGER mutate_decision AFTER INSERT ON decision_snapshots "
                "BEGIN UPDATE decision_snapshots SET content_hash = content_hash; END"
            )

    with pytest.raises(DecisionSnapshotRepositoryError):
        SQLiteDecisionSnapshotRepository(path)


@pytest.mark.parametrize(
    ("column", "replacement", "lookup_id"),
    (
        ("snapshot_json", b"{", "decision-001"),
        ("content_hash", "0" * 64, "decision-001"),
        ("decision_id", "decision-row-alias", "decision-row-alias"),
    ),
)
def test_sqlite_corrupt_json_hash_or_row_identity_is_rejected(
    tmp_path: Path,
    column: str,
    replacement: bytes | str,
    lookup_id: str,
) -> None:
    path = tmp_path / f"corrupt-{column}.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    repository.create(_snapshot())
    with sqlite3.connect(path) as connection:
        for trigger_name in snapshot_repository_module._TRIGGER_NAMES:
            connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(
            f"UPDATE decision_snapshots SET {column} = ? WHERE decision_id = ?",
            (replacement, "decision-001"),
        )
        for ddl in snapshot_repository_module._TRIGGER_DDLS:
            connection.execute(ddl)

    with pytest.raises(DecisionSnapshotRepositoryError):
        reopened = SQLiteDecisionSnapshotRepository(path)
        reopened.get(lookup_id)


def test_sqlite_triggers_reject_coherent_replacement_and_delete(tmp_path: Path) -> None:
    path = tmp_path / "database-level-immutability.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    original = repository.create(_snapshot())
    replacement = _snapshot(data_version="snapshot-v2", data_label="v2")

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE decision_snapshots SET content_hash = ?, snapshot_json = ? "
                "WHERE decision_id = ?",
                (
                    replacement.content_hash,
                    replacement.to_json().encode("utf-8"),
                    original.decision_id,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM decision_snapshots WHERE decision_id = ?",
                (original.decision_id,),
            )

    assert repository.get(original.decision_id) == original


@pytest.mark.parametrize(
    "statement",
    (
        "INSERT OR REPLACE INTO decision_snapshots("
        "decision_id, content_hash, snapshot_json) VALUES (?, ?, ?)",
        "REPLACE INTO decision_snapshots("
        "decision_id, content_hash, snapshot_json) VALUES (?, ?, ?)",
    ),
    ids=("insert-or-replace", "replace-into"),
)
def test_sqlite_trigger_rejects_replace_without_recursive_triggers(
    tmp_path: Path,
    statement: str,
) -> None:
    path = tmp_path / "database-level-no-replace.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    original = repository.create(_snapshot())
    replacement = _snapshot(data_version="snapshot-v2", data_label="v2")

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA recursive_triggers").fetchone() == (0,)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                statement,
                (
                    replacement.decision_id,
                    replacement.content_hash,
                    replacement.to_json().encode("utf-8"),
                ),
            )

    assert repository.get(original.decision_id) == original


def test_sqlite_without_rowid_rejects_replace_through_hidden_rowid(tmp_path: Path) -> None:
    path = tmp_path / "database-level-no-rowid-replace.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    original = repository.create(_snapshot())
    replacement = _snapshot("decision-002", data_version="snapshot-v2", data_label="v2")

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA recursive_triggers").fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match="rowid"):
            connection.execute(
                "INSERT OR REPLACE INTO decision_snapshots("
                "rowid, decision_id, content_hash, snapshot_json) VALUES (?, ?, ?, ?)",
                (
                    1,
                    replacement.decision_id,
                    replacement.content_hash,
                    replacement.to_json().encode("utf-8"),
                ),
            )

    assert repository.get(original.decision_id) == original
    with pytest.raises(DecisionSnapshotNotFound):
        repository.get(replacement.decision_id)


@pytest.mark.parametrize("operation", ("get", "create"))
@pytest.mark.parametrize(
    "tamper",
    ("missing-update", "missing-no-replace", "modified-no-replace", "extra"),
)
def test_existing_repository_rechecks_exact_immutability_triggers_on_every_operation(
    tmp_path: Path,
    operation: str,
    tamper: str,
) -> None:
    path = tmp_path / f"tampered-trigger-{tamper}-{operation}.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    repository.create(_snapshot())
    with sqlite3.connect(path) as connection:
        if tamper == "missing-update":
            connection.execute(f"DROP TRIGGER {snapshot_repository_module._NO_UPDATE_TRIGGER}")
        elif tamper == "missing-no-replace":
            connection.execute(f"DROP TRIGGER {snapshot_repository_module._NO_REPLACE_TRIGGER}")
        elif tamper == "modified-no-replace":
            connection.execute(f"DROP TRIGGER {snapshot_repository_module._NO_REPLACE_TRIGGER}")
            connection.execute(
                "CREATE TRIGGER decision_snapshots_no_replace "
                "BEFORE INSERT ON decision_snapshots BEGIN SELECT 1; END"
            )
        else:
            connection.execute(
                "CREATE TRIGGER decision_snapshots_unexpected "
                "AFTER INSERT ON decision_snapshots BEGIN SELECT 1; END"
            )

    with pytest.raises(DecisionSnapshotRepositoryError, match="immutability triggers"):
        if operation == "get":
            repository.get("decision-001")
        else:
            repository.create(_snapshot())


@pytest.mark.parametrize("operation", ("get", "create"))
def test_existing_repository_rechecks_schema_version_on_every_operation(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / f"changed-schema-version-{operation}.db"
    repository = SQLiteDecisionSnapshotRepository(path)
    repository.create(_snapshot())
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(DecisionSnapshotRepositoryError, match="schema version"):
        if operation == "get":
            repository.get("decision-001")
        else:
            repository.create(_snapshot())


def test_two_sqlite_instances_atomically_replay_concurrent_identical_create(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent-identical.db"
    repositories = (
        SQLiteDecisionSnapshotRepository(path),
        SQLiteDecisionSnapshotRepository(path),
    )
    snapshot = _snapshot()
    barrier = Barrier(2)

    def create(repository: SQLiteDecisionSnapshotRepository) -> DecisionSnapshot:
        barrier.wait(timeout=5)
        return repository.create(snapshot)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(create, repository) for repository in repositories)
        results = tuple(future.result(timeout=5) for future in futures)

    assert results == (snapshot, snapshot)
    assert SQLiteDecisionSnapshotRepository(path).get(snapshot.decision_id) == snapshot
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM decision_snapshots").fetchone() == (1,)


def test_two_sqlite_instances_atomically_choose_one_concurrent_conflicting_create(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent-conflict.db"
    repositories = (
        SQLiteDecisionSnapshotRepository(path),
        SQLiteDecisionSnapshotRepository(path),
    )
    candidates = (
        _snapshot(data_version="snapshot-v1", data_label="v1"),
        _snapshot(data_version="snapshot-v2", data_label="v2"),
    )
    barrier = Barrier(2)

    def create(
        repository: SQLiteDecisionSnapshotRepository,
        snapshot: DecisionSnapshot,
    ) -> tuple[str, DecisionSnapshot | None]:
        barrier.wait(timeout=5)
        try:
            return ("created", repository.create(snapshot))
        except DecisionSnapshotConflict:
            return ("conflict", None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(create, repository, snapshot)
            for repository, snapshot in zip(repositories, candidates, strict=True)
        )
        results = tuple(future.result(timeout=5) for future in futures)

    created = tuple(snapshot for status, snapshot in results if status == "created")
    conflicts = tuple(status for status, _ in results if status == "conflict")
    assert len(created) == 1
    assert conflicts == ("conflict",)
    assert created[0] is not None
    assert SQLiteDecisionSnapshotRepository(path).get("decision-001") == created[0]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM decision_snapshots").fetchone() == (1,)
