from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from quant_agent.core.time import shanghai_now
from quant_agent.risk.kill_switch import KillSwitchScope
from tests.integration.test_api_research_portfolio import (
    TOKENS,
    auth_header,
)


def approve(client: TestClient) -> str:
    response = client.post(
        "/v1/order-drafts/draft-1/approve",
        headers=auth_header("approver"),
        json={},
    )
    assert response.status_code == 200
    expires_at = datetime.fromisoformat(response.json()["data"]["expires_at"])
    assert expires_at <= shanghai_now() + timedelta(minutes=5, seconds=1)
    return response.json()["data"]["approval_token"]


def test_approval_is_hash_bound_one_time_and_submission_is_idempotent(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    token = approve(client)
    tampered = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-tampered"},
        json={"approval_token": f"{token}x"},
    )
    assert tampered.status_code == 403
    missing_key = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers=auth_header("trader"),
        json={"approval_token": token},
    )
    assert missing_key.status_code == 422
    malformed = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-malformed"},
        json={"approval_token": "raw-secret"},
    )
    assert malformed.status_code == 422
    assert "raw-secret" not in malformed.text
    service = api_fixture["services"].orders
    original = service.get_draft("draft-1")
    service.put_draft(replace(original, batch_hash="modified-batch"))
    invalidated = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-modified"},
        json={"approval_token": token},
    )
    assert invalidated.status_code == 403
    assert invalidated.json()["error"]["code"] == "APPROVAL_REQUIRED"

    service.put_draft(original)
    token = approve(client)
    first = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-success"},
        json={"approval_token": token},
    )
    duplicate = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-success"},
        json={"approval_token": token},
    )
    assert first.status_code == duplicate.status_code == 200
    assert first.json()["data"] == duplicate.json()["data"]
    assert api_fixture["submissions"] == ["batch-v1"]
    assert [event.action for event in api_fixture["audit_events"]] == [
        "APPROVE_ORDER_DRAFT",
        "APPROVE_ORDER_DRAFT",
        "SUBMIT_APPROVED_PAPER_ORDERS",
    ]
    assert all("approval_token" not in event.metadata for event in api_fixture["audit_events"])
    second = replace(original, draft_id="draft-2", batch_hash="batch-v2")
    service.put_draft(second)
    second_approval = client.post(
        "/v1/order-drafts/draft-2/approve",
        headers=auth_header("approver"),
        json={},
    ).json()["data"]["approval_token"]
    key_rebound = client.post(
        "/v1/order-drafts/draft-2/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-success"},
        json={"approval_token": second_approval},
    )
    assert key_rebound.status_code == 409
    reused = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-other"},
        json={"approval_token": token},
    )
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "CONFLICT"


def test_kill_switch_and_expired_drafts_block_submission(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    token = approve(client)
    api_fixture["kill_switch"].trigger(
        scope=KillSwitchScope.ACCOUNT,
        account_id="paper-1",
        reason="test incident",
        actor_id="risk",
        actor_role="RISK_ADMIN",
        occurred_at=api_fixture["now"],
        incident_snapshot_hash="incident-hash",
    )
    blocked = client.post(
        "/v1/order-drafts/draft-1/submit",
        headers={**auth_header("trader"), "Idempotency-Key": "idem-kill"},
        json={"approval_token": token},
    )
    assert blocked.status_code == 423
    assert blocked.json()["error"]["code"] == "KILL_SWITCH_ACTIVE"

    service = api_fixture["services"].orders
    old = service.get_draft("draft-1")
    service.put_draft(replace(old, expires_at=api_fixture["now"] - timedelta(seconds=1)))
    expired = client.post(
        "/v1/order-drafts/draft-1/approve",
        headers=auth_header("approver"),
        json={},
    )
    assert expired.status_code == 410


def test_reconciliation_uses_service_and_web_pages_expose_safety_controls(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    reconciliation = client.post(
        "/v1/accounts/paper-1/reconcile",
        headers=auth_header("trader"),
    )
    assert reconciliation.json()["data"]["status"] == "RECONCILED"

    research = client.get("/web/research")
    trading = client.get("/web/trading")
    chat = client.get("/web/chat")
    evidence = client.get("/web/evidence")
    assert (
        research.status_code
        == trading.status_code
        == chat.status_code
        == evidence.status_code
        == 200
    )
    assert "阶段分类，不是上涨概率" in research.text
    assert "风险警告（始终显示）" in trading.text
    assert "审批不能由 Agent 代替" in trading.text
    assert "回测中心" in trading.text and "组合中心" in trading.text
    assert "不能审批、跳过风控或切换运行模式" in chat.text
    assert "支持证据、反对证据、风险和失效条件" in evidence.text
    web_script = client.get("/web/static/app.js").text
    assert "localStorage" not in web_script
    assert "sessionStorage" in web_script
    assert "button.dataset.idempotencyKey" in web_script
    assert TOKENS["approver"] not in trading.text
