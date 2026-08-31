"""Tests for the local command-line boundary."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from quant_agent.cli import main
from quant_agent.data.sync import SyncRunState
from quant_agent.data.sync.runner import SyncExecutionResult


def test_demo_command_emits_json_and_can_be_repeated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'database' / 'demo.db').as_posix()}"
    storage_path = tmp_path / "snapshots"
    arguments = [
        "demo",
        "--database-url",
        database_url,
        "--storage-path",
        str(storage_path),
    ]

    assert main(arguments) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert main(arguments) == 0
    second_output = json.loads(capsys.readouterr().out)

    assert first_output["ok"] is True
    assert first_output["snapshot_reused"] is False
    assert second_output["snapshot_reused"] is True
    assert second_output["content_hash"] == first_output["content_hash"]
    assert database_url not in json.dumps(first_output)


def test_demo_command_returns_structured_error_for_unsafe_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "demo",
            "--database-url",
            f"sqlite:///{(tmp_path / 'demo.db').as_posix()}",
            "--storage-path",
            str(tmp_path / "snapshots"),
            "--data-version",
            "../unsafe",
        ]
    )
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 1
    assert output["ok"] is False
    assert output["error"] == "ValueError"


def test_demo_command_fails_closed_for_tampered_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'demo.db').as_posix()}"
    storage_path = tmp_path / "snapshots"
    arguments = [
        "demo",
        "--database-url",
        database_url,
        "--storage-path",
        str(storage_path),
    ]
    assert main(arguments) == 0
    capsys.readouterr()
    manifest_path = storage_path / "demo_20260730_v1" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(arguments) == 1
    output = json.loads(capsys.readouterr().err)

    assert output["ok"] is False
    assert output["error"] == "ValueError"


def test_data_sync_uses_environment_secret_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> SyncExecutionResult:
        captured.update(kwargs)
        return SyncExecutionResult(
            run_id="sync-run",
            state=SyncRunState.SUCCEEDED,
            committed_watermark={"cal_date": "2026-08-28"},
            checkpoint_revision=1,
            fetched_pages=1,
            reused_raw_pages=0,
            inserted=1,
            updated=0,
            skipped=0,
        )

    monkeypatch.setenv("TEST_TUSHARE_TOKEN", "highly-secret-token")
    monkeypatch.setattr("quant_agent.cli.run_tushare_sync", fake_run)

    exit_code = main(
        [
            "data",
            "sync",
            "--dataset",
            "calendar",
            "--start",
            "2026-08-28",
            "--end",
            "2026-08-28",
            "--market",
            "SSE",
            "--token-env",
            "TEST_TUSHARE_TOKEN",
        ]
    )
    streams = capsys.readouterr()
    payload = json.loads(streams.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["dataset"] == "calendar"
    assert payload["run_id"] == "sync-run"
    assert captured["spec"].start == date(2026, 8, 28)
    assert captured["market_data_token"].get_secret_value() == "highly-secret-token"
    assert "highly-secret-token" not in streams.out + streams.err


def test_data_sync_fails_safely_when_token_environment_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("MISSING_TUSHARE_TOKEN", raising=False)

    exit_code = main(
        [
            "data",
            "sync",
            "--dataset",
            "calendar",
            "--start",
            "2026-08-28",
            "--end",
            "2026-08-28",
            "--market",
            "SSE",
            "--token-env",
            "MISSING_TUSHARE_TOKEN",
        ]
    )
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert payload == {
        "error": "INVALID_INPUT",
        "message": "market data token environment variable 'MISSING_TUSHARE_TOKEN' is not set",
        "ok": False,
    }


def test_data_sync_reports_safe_dataset_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MARKET_DATA_TOKEN", "hidden-token")

    exit_code = main(
        [
            "data",
            "sync",
            "--dataset",
            "calendar",
            "--start",
            "2026-08-28",
            "--end",
            "2026-08-28",
        ]
    )
    streams = capsys.readouterr()
    payload = json.loads(streams.err)

    assert exit_code == 2
    assert payload == {
        "error": "INVALID_INPUT",
        "message": "calendar sync requires --market",
        "ok": False,
    }
    assert "hidden-token" not in streams.out + streams.err
