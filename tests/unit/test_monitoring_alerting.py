from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from apps.api.app import create_app
from fastapi.testclient import TestClient

from quant_agent.observability.alerting import (
    AlertDispatcher,
    AlertPolicyConfig,
    AlertPolicyEngine,
    AlertSeverity,
    InMemoryAlertSink,
    JsonLinesAlertSink,
)
from quant_agent.observability.monitoring import MonitoringRegistry

NOW = datetime.fromisoformat("2026-07-31T09:30:00+08:00")
ROOT = Path(__file__).resolve().parents[2]


def _incident_registry() -> MonitoringRegistry:
    registry = MonitoringRegistry()
    registry.record_data_completeness("daily-bars", 0.75)
    for index in range(10):
        registry.record_service(
            "api",
            "/v1/risk/checks",
            success=index == 0,
            latency_seconds=3.0 if index == 0 else 0.1,
        )
    for _ in range(5):
        registry.record_risk_check(available=True, passed=False)
    registry.record_risk_check(available=False, passed=False)
    registry.record_order_submission(duplicate_risk=True)
    registry.record_reconciliation(("CRITICAL", "WARNING"))
    registry.set_kill_switch(True)
    return registry


def test_registry_covers_required_metrics_and_prometheus_contract() -> None:
    registry = _incident_registry()
    snapshot = registry.snapshot(generated_at=NOW)
    assert snapshot.data_completeness == (("daily-bars", 0.75),)
    assert snapshot.services[0].requests == 10
    assert snapshot.services[0].success_rate == pytest.approx(0.1)
    assert snapshot.risk_service_available is False
    assert snapshot.risk_rejections_total == 5
    assert snapshot.duplicate_order_risks_total == 1
    assert snapshot.reconciliation_count("CRITICAL") == 1
    assert snapshot.kill_switch_active

    rendered = registry.render_prometheus()
    for metric in (
        "quant_agent_data_completeness_ratio",
        "quant_agent_service_requests_total",
        "quant_agent_service_success_total",
        "quant_agent_service_latency_seconds_max",
        "quant_agent_risk_rejections_total",
        "quant_agent_duplicate_order_risks_total",
        "quant_agent_reconciliation_differences_total",
        "quant_agent_kill_switch_active",
    ):
        assert metric in rendered
    assert "account_id" not in rendered
    assert "token" not in rendered.casefold()

    with pytest.raises(ValueError, match="between zero and one"):
        registry.record_data_completeness("daily-bars", 1.1)
    with pytest.raises(ValueError, match="unsupported"):
        registry.record_reconciliation(("UNKNOWN",))
    with pytest.raises(ValueError, match="unsupported characters"):
        registry.record_service("api", "/accounts/real id", success=True, latency_seconds=0)


def test_p0_to_p3_policies_route_deliver_deduplicate_and_redact(
    tmp_path: Path,
) -> None:
    config = AlertPolicyConfig.from_toml(ROOT / "configs" / "observability" / "alerts_v1.toml")
    events = AlertPolicyEngine(config).evaluate(_incident_registry().snapshot(generated_at=NOW))
    by_policy = {event.policy_id: event for event in events}
    assert by_policy["duplicate-order-risk"].severity is AlertSeverity.P0
    assert by_policy["critical-reconciliation-difference"].severity is AlertSeverity.P0
    assert by_policy["risk-service-unavailable"].severity is AlertSeverity.P1
    assert by_policy["data-incomplete:daily-bars"].severity is AlertSeverity.P2
    assert by_policy["elevated-risk-rejections"].severity is AlertSeverity.P3
    assert all(event.runbook.startswith("docs/07-") for event in events)

    memory = InMemoryAlertSink()
    dispatcher = AlertDispatcher(memory)
    first = dispatcher.dispatch(events, delivered_at=NOW)
    duplicate = dispatcher.dispatch(events, delivered_at=NOW + timedelta(minutes=1))
    repeated = dispatcher.dispatch(events, delivered_at=NOW + timedelta(minutes=16))
    assert len(first) == len(events)
    assert duplicate == ()
    assert len(repeated) == len(events)
    assert set(first[0].targets) == {"incident-primary", "risk-lead"}

    output = tmp_path / "alerts.jsonl"
    JsonLinesAlertSink(output).send(first[0])
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload).casefold()
    assert "authorization" not in serialized
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert payload["event"]["source"]
    assert payload["event"]["runbook"]
    assert asdict(first[0].event)["diagnostics"]


def test_internal_metrics_endpoint_records_template_and_kill_switch() -> None:
    registry = MonitoringRegistry()
    with TestClient(create_app(monitoring=registry)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.get("/internal/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert 'operation="/health/ready"' in response.text
    assert "quant_agent_kill_switch_active 0" in response.text
