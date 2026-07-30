import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_test_environment_process_starts_and_passes_health_check(tmp_path) -> None:
    port = _available_port()
    environment = {
        **os.environ,
        "QUANT_AGENT_APP_ENV": "test",
        "QUANT_AGENT_RUNTIME_MODE": "RESEARCH",
        "QUANT_AGENT_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        payload = None
        for _ in range(50):
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health/ready",
                    timeout=1,
                ) as response:
                    payload = json.load(response)
                    break
            except OSError:
                time.sleep(0.1)
        if payload is None:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            error = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"deployed process did not become ready: {error}")
        assert payload["status"] == "ready"
        assert payload["environment"] == "test"
        assert payload["kill_switch_active"] is False
        health_environment = {
            **environment,
            "QUANT_AGENT_HEALTH_URL": f"http://127.0.0.1:{port}/health/ready",
        }
        healthcheck = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "deploy" / "healthcheck.py")],
            cwd=ROOT,
            env=health_environment,
            check=False,
        )
        assert healthcheck.returncode == 0
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_container_and_rollback_files_enforce_release_safety() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    production = (ROOT / "deploy" / "compose.production.yml").read_text(encoding="utf-8")
    rollback = (ROOT / "scripts" / "deploy" / "rollback.ps1").read_text(encoding="utf-8")
    acceptance = (ROOT / "scripts" / "deploy" / "acceptance.ps1").read_text(encoding="utf-8")
    assert "FROM python:3.12.13-slim-bookworm@sha256:" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "mkdir -p /var/lib/quant-agent/data /var/lib/quant-agent/audit" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "read_only: true" in production
    assert "no-new-privileges:true" in production
    assert "cap_drop:" in production
    assert "quant-agent-production-audit" in production
    assert "command.downgrade" not in rollback.casefold()
    assert "audit volume retained" in rollback
    assert "Rollback image did not become healthy." in rollback
    assert "deployment_acceptance_marker" in acceptance
    assert "at least one persisted audit record" in acceptance
