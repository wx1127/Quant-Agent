import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from apps.api.app import create_app
from apps.api.core.services import (
    ApiServices,
    BacktestTaskStore,
    DecisionStore,
    OrderApprovalService,
    PortfolioService,
    ResearchCatalog,
)
from fastapi.testclient import TestClient

from quant_agent.agent.responses import AgentAnswer
from quant_agent.config.models import AppEnvironment, RuntimeMode, RuntimeSettings
from quant_agent.deployment.migrations import MigrationGuard
from quant_agent.deployment.models import ReleaseManifest, ReleasePlanner
from quant_agent.observability.audit import AuditEvent, JsonLinesAuditSink
from quant_agent.risk.kill_switch import KillSwitch

NOW = datetime(2026, 7, 30, 16, tzinfo=UTC)
ROOT = Path(__file__).parents[2]
DIGEST_A = "registry.example/quant-agent@sha256:" + "a" * 64
DIGEST_B = "registry.example/quant-agent@sha256:" + "b" * 64


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


def manifest(
    release_id: str,
    image_ref: str,
    *,
    schema_revision: str = "0001",
    audit_store_id: str = "audit-production",
) -> ReleaseManifest:
    return ReleaseManifest(
        release_id,
        image_ref,
        "abcdef1234567890",
        schema_revision,
        "production-v1",
        audit_store_id,
        True,
        NOW,
    )


def test_production_boot_requires_audit_and_activates_global_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeSettings(
        app_env=AppEnvironment.PRODUCTION,
        mode=RuntimeMode.LIVE_ASSISTED,
    )
    monkeypatch.delenv("QUANT_AGENT_AUDIT_LOG_PATH", raising=False)
    with pytest.raises(ValueError, match="AUDIT"):
        create_app(runtime_settings=runtime)

    kill_switch = KillSwitch()
    audit = MemoryAuditSink()
    with TestClient(
        create_app(runtime_settings=runtime, kill_switch=kill_switch, audit_sink=audit)
    ) as client:
        readiness = client.get("/health/ready")
        assert readiness.status_code == 200
        assert readiness.json() == {
            "status": "ready",
            "version": "0.9.0",
            "environment": "production",
            "mode": "LIVE_ASSISTED",
            "kill_switch_active": True,
        }
    assert not kill_switch.order_allowed("paper-1")
    assert audit.events[0].action == "PRODUCTION_BOOT_KILL_SWITCH"
    assert audit.events[0].metadata["version"] == "0.9.0"


def test_production_rejects_services_with_a_different_kill_switch() -> None:
    runtime = RuntimeSettings(
        app_env=AppEnvironment.PRODUCTION,
        mode=RuntimeMode.LIVE_ASSISTED,
    )
    service_switch = KillSwitch()
    services = ApiServices(
        research=ResearchCatalog(),
        backtests=BacktestTaskStore(set(), lambda *_: {}),
        portfolios=PortfolioService(),
        orders=OrderApprovalService(b"x" * 32, service_switch, lambda _: {}),
        decisions=DecisionStore(),
        agent_handler=lambda *_: AgentAnswer.insufficient(decision_id=None, as_of=None),
        reconcile_handler=lambda _: {},
        risk_handler=lambda *_: {},
    )
    try:
        with pytest.raises(ValueError, match="share"):
            create_app(
                runtime_settings=runtime,
                services=services,
                kill_switch=KillSwitch(),
                audit_sink=MemoryAuditSink(),
            )
    finally:
        services.backtests.shutdown()


def test_forward_migration_reaches_single_head(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'deployment.db'}"
    guard = MigrationGuard(ROOT)
    before = guard.inspect(database_url)
    assert before.current_revisions == ()
    assert before.target_revisions == ("0001",)
    assert before.upgrade_required
    assert before.compatible

    after = guard.upgrade(database_url)
    assert after.current_revisions == ("0001",)
    assert not after.upgrade_required
    assert after.compatible
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "deploy" / "migrate.py"), "--check"],
        cwd=ROOT,
        env={**os.environ, "QUANT_AGENT_DATABASE_URL": database_url},
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0
    assert json.loads(check.stdout)["current_revisions"] == ["0001"]


def test_release_and_rollback_preserve_schema_audit_and_kill_switch(tmp_path) -> None:
    previous = manifest("release-a", DIGEST_A)
    candidate = manifest("release-b", DIGEST_B)
    planner = ReleasePlanner()
    deploy = planner.deploy(previous, candidate)
    rollback = planner.rollback(candidate, previous)
    assert deploy.database_downgrade is False
    assert rollback.database_downgrade is False
    assert rollback.audit_store_id == previous.audit_store_id
    assert rollback.production_kill_switch_active

    audit_path = tmp_path / "audit" / "events.jsonl"
    sink = JsonLinesAuditSink(audit_path)
    first = AuditEvent(
        event_type="deployment",
        actor_id="operator",
        action="DEPLOY",
        result="failed",
        request_id="deploy-1",
        occurred_at=NOW,
    )
    sink.append(first)
    original_line = audit_path.read_text(encoding="utf-8").splitlines()[0]
    sink.append(
        first.model_copy(
            update={"action": "ROLLBACK", "result": "succeeded", "request_id": "rollback-1"}
        )
    )
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == original_line
    assert [json.loads(line)["action"] for line in lines] == ["DEPLOY", "ROLLBACK"]


def test_release_contract_rejects_mutable_images_schema_downgrade_and_audit_replacement() -> None:
    with pytest.raises(ValueError, match="immutable"):
        manifest("mutable", "registry.example/quant-agent:latest")
    current = manifest("release-b", DIGEST_B, schema_revision="0002")
    old_schema = manifest("release-a", DIGEST_A, schema_revision="0001")
    with pytest.raises(ValueError, match="schema-compatible"):
        ReleasePlanner().rollback(current, old_schema)
    other_audit = manifest("release-c", DIGEST_A, audit_store_id="another-audit")
    with pytest.raises(ValueError, match="audit"):
        ReleasePlanner().rollback(old_schema, other_audit)
