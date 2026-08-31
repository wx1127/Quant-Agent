"""Tests for the read-only synchronization status pipeline and CLI."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_agent.cli import main
from quant_agent.data.database import Database
from quant_agent.data.sync import SyncMode, SyncRepository, SyncRunState
from quant_agent.pipelines.sync_status import query_sync_status

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 28, 18, 0, tzinfo=TZ)


def _seed(database_url: str) -> str:
    database = Database(database_url)
    database.create_schema()
    with database.session() as session:
        repository = SyncRepository(session)
        checkpoint = repository.acquire_checkpoint(
            provider="fake",
            dataset="daily",
            scope={"market": "CN"},
            initial_watermark={"trade_date": "2026-08-27"},
            lease_owner="seed-worker",
            lease_ttl=timedelta(minutes=5),
            now=NOW,
        )
        run = repository.create_run(
            provider="fake",
            dataset="daily",
            scope={"market": "CN"},
            target_watermark={"trade_date": "2026-08-28"},
            checkpoint_before={"trade_date": "2026-08-27"},
            mode=SyncMode.INCREMENTAL,
            idempotency_key="status-seed".ljust(64, "0"),
            code_version="test",
            created_at=NOW,
        )
        repository.transition_run(run.run_id, SyncRunState.RUNNING, at=NOW)
        repository.release_checkpoint(
            checkpoint.id,
            lease_owner="seed-worker",
            expected_revision=checkpoint.revision,
            now=NOW,
        )
        return run.run_id


def test_query_sync_status_returns_safe_operational_fields(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'status.db').as_posix()}"
    run_id = _seed(database_url)

    result = query_sync_status(
        database_url=database_url,
        provider="fake",
        dataset="daily",
        limit=5,
    )

    assert result["runs"][0]["run_id"] == run_id
    assert result["runs"][0]["state"] == "RUNNING"
    assert result["checkpoints"][0]["is_leased"] is False
    assert "lease_owner" not in result["checkpoints"][0]


def test_data_status_cli_outputs_stable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'status-cli.db').as_posix()}"
    _seed(database_url)

    assert main(["data", "status", "--database-url", database_url, "--limit", "1"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["ok"] is True
    assert len(payload["runs"]) == 1


def test_sync_status_rejects_unbounded_limits(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    Database(database_url).create_schema()

    for value in (0, 201):
        with pytest.raises(ValueError, match="between 1 and 200"):
            query_sync_status(database_url=database_url, limit=value)
