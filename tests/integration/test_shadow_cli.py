from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from quant_agent.shadow.io import shadow_day_to_mapping
from quant_agent.shadow.models import ShadowDayEvidence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "shadow" / "shadow_20260803_v1.toml"
CALENDAR = ROOT / "configs" / "shadow" / "calendar_20260803_20d.json"


def _day(day: date, *, complete: bool) -> ShadowDayEvidence:
    observed = datetime.fromisoformat(f"{day}T15:45:00+08:00")
    return ShadowDayEvidence(
        trading_date=day,
        observed_at=observed,
        market_data_as_of=observed - timedelta(minutes=15),
        input_snapshot_hash="1" * 64,
        virtual_account_snapshot_hash="2" * 64,
        data_version=f"market-{day}",
        regime="RANGE",
        mainline_ids=("theme-ai",),
        candidate_ids=("CN.SSE.600000",),
        data_complete=complete,
        report_generated=complete,
        pipeline_succeeded=complete,
        reconciliation_matched=True,
        future_data_violations=0,
        executable_orders_emitted=0,
    )


def test_shadow_record_and_status_cli_rebuild_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "shadow.jsonl"
    reports = tmp_path / "reports"
    alerts = tmp_path / "alerts.jsonl"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "packages"), str(ROOT)]),
    }
    first_input = tmp_path / "first.json"
    first_input.write_text(
        json.dumps(shadow_day_to_mapping(_day(date(2026, 8, 3), complete=True))),
        encoding="utf-8",
    )
    record_command = [
        sys.executable,
        str(ROOT / "scripts" / "shadow" / "record_day.py"),
        "--config",
        str(CONFIG),
        "--calendar",
        str(CALENDAR),
        "--evidence",
        str(first_input),
        "--ledger",
        str(ledger),
        "--report-dir",
        str(reports),
        "--alert-log",
        str(alerts),
    ]
    first = subprocess.run(
        record_command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["acceptance"]["consecutive_trading_days"] == 1

    second_input = tmp_path / "second.json"
    second_input.write_text(
        json.dumps(shadow_day_to_mapping(_day(date(2026, 8, 4), complete=False))),
        encoding="utf-8",
    )
    second_command = record_command.copy()
    second_command[second_command.index(str(first_input))] = str(second_input)
    second = subprocess.run(
        second_command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["alerts_delivered"] == 2
    assert len(alerts.read_text(encoding="utf-8").splitlines()) == 2

    status = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "shadow" / "status.py"),
            "--config",
            str(CONFIG),
            "--calendar",
            str(CALENDAR),
            "--ledger",
            str(ledger),
            "--report-dir",
            str(reports),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["observed_trading_days"] == 2
    assert (reports / "shadow-status.md").exists()
