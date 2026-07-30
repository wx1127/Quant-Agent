from datetime import timedelta
from threading import Event
from time import monotonic, sleep
from typing import Any

import pytest
from apps.api.app import create_app
from apps.api.core.auth import AuthService, Principal, Role
from apps.api.core.services import (
    ApiServices,
    BacktestTaskStore,
    DecisionStore,
    OrderApprovalService,
    OrderDraftRecord,
    PortfolioService,
    ResearchCatalog,
    ResearchRecord,
)
from fastapi.testclient import TestClient

from quant_agent.agent.responses import AgentAnswer
from quant_agent.agent.security import AgentSecurityGuard
from quant_agent.config.models import RuntimeMode
from quant_agent.core.time import shanghai_now
from quant_agent.observability.audit import AuditEvent
from quant_agent.risk.kill_switch import KillSwitch

TOKENS = {
    "viewer": "viewer-token-123456789",
    "researcher": "researcher-token-123456",
    "trader": "trader-token-123456789",
    "approver": "approver-token-123456",
    "risk": "risk-admin-token-12345",
}


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


def auth_header(name: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[name]}"}


@pytest.fixture
def api_fixture() -> Any:
    now = shanghai_now()
    auth = AuthService()
    auth.register_token(
        TOKENS["viewer"],
        Principal("viewer", frozenset({Role.VIEWER}), frozenset({"paper-1"})),
    )
    auth.register_token(
        TOKENS["researcher"],
        Principal("researcher", frozenset({Role.RESEARCHER}), frozenset({"paper-1"})),
    )
    auth.register_token(
        TOKENS["trader"],
        Principal("trader", frozenset({Role.TRADER}), frozenset({"paper-1"})),
    )
    auth.register_token(
        TOKENS["approver"],
        Principal("approver", frozenset({Role.APPROVER}), frozenset({"paper-1"})),
    )
    auth.register_token(
        TOKENS["risk"],
        Principal("risk", frozenset({Role.RISK_ADMIN}), frozenset({"paper-1"})),
    )

    research = ResearchCatalog()
    research.publish_regime(
        ResearchRecord(
            "CN_A",
            now,
            "market_v1",
            {
                "regime": "RANGE_STRONG",
                "score": 68.2,
                "support_evidence": ["市场宽度高于中性"],
                "counter_evidence": ["成交额尚未扩张"],
            },
        )
    )
    research.publish(
        "themes",
        [
            ResearchRecord(
                "theme-a",
                now,
                "market_v1",
                {"name": "先进制造", "state": "CONFIRMED", "score": 76.0},
            ),
            ResearchRecord(
                "theme-b",
                now,
                "market_v1",
                {"name": "新能源", "state": "FADING", "score": 52.0},
            ),
        ],
    )
    research.publish(
        "leaders",
        [
            ResearchRecord(
                "000001.SZ",
                now,
                "market_v1",
                {"theme_id": "theme-a", "name": "示例股份", "score": 73.0},
            )
        ],
    )
    research.publish(
        "candidates",
        [
            ResearchRecord(
                "000001.SZ",
                now,
                "market_v1",
                {
                    "theme_id": "theme-a",
                    "name": "示例股份",
                    "tier": "A",
                    "score": 78.0,
                    "tradable": True,
                },
            )
        ],
    )
    research.publish(
        "evidence",
        [
            ResearchRecord(
                "000001.SZ",
                now,
                "market_v1",
                {
                    "support_evidence": ["相对强度保持"],
                    "counter_evidence": ["估值不低"],
                    "invalidations": ["主线退潮"],
                },
            )
        ],
    )

    runner_release = Event()

    def runner(strategy: str, parameters: str, data: str) -> dict[str, Any]:
        runner_release.wait(timeout=2)
        return {
            "strategy_id": strategy,
            "parameter_version": parameters,
            "data_version": data,
            "annualized_return": "fixture-only",
        }

    backtests = BacktestTaskStore({("etf_rotation", "etf_rotation_v1")}, runner)
    portfolios = PortfolioService()
    portfolios.publish_snapshot(
        "paper-1",
        {
            "account_id": "paper-1",
            "as_of": now.isoformat(),
            "data_version": "account_v1",
            "total_equity": 1_000_000,
        },
    )
    kill_switch = KillSwitch()
    submissions: list[str] = []
    audit_sink = MemoryAuditSink()

    def submitter(draft: OrderDraftRecord) -> dict[str, Any]:
        submissions.append(draft.batch_hash)
        return {"status": "PAPER_SUBMITTED", "batch_hash": draft.batch_hash}

    order_service = OrderApprovalService(b"x" * 32, kill_switch, submitter, audit_sink)
    order_service.put_draft(
        OrderDraftRecord(
            "draft-1",
            "paper-1",
            "dec_20260730_api",
            "batch-v1",
            now + timedelta(hours=1),
            (
                {
                    "instrument_id": "000001.SZ",
                    "side": "BUY",
                    "quantity": 100,
                    "reference_price": 10.0,
                },
            ),
            5.2,
            ("价格变化可能使审批失效",),
        )
    )
    agent_messages: list[str] = []
    risk_state = {"passed": False}

    def agent_handler(message: str, _user_id: str) -> AgentAnswer:
        agent_messages.append(message)
        return AgentAnswer.insufficient(
            decision_id="dec_20260730_agent",
            as_of=now,
            reason="证据不足, 当前仅返回可审计状态。",
        )

    services = ApiServices(
        research,
        backtests,
        portfolios,
        order_service,
        DecisionStore(),
        agent_handler,
        lambda account_id: {
            "account_id": account_id,
            "status": "RECONCILED",
            "as_of": now.isoformat(),
            "differences": [],
        },
        lambda account_id, decision_id, target: {
            "passed": risk_state["passed"],
            "account_id": account_id,
            "decision_id": decision_id,
            "target_hash": str(sorted(target.items())),
            "policy_version": "risk_v1",
            "violations": [] if risk_state["passed"] else ["risk fixture rejected"],
        },
    )
    app = create_app(
        services=services,
        auth=auth,
        agent_security=AgentSecurityGuard(
            configured_mode=RuntimeMode.PAPER,
            allowed_instruments={"000001.SZ"},
        ),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield {
            "client": client,
            "services": services,
            "release": runner_release,
            "kill_switch": kill_switch,
            "submissions": submissions,
            "agent_messages": agent_messages,
            "audit_events": audit_sink.events,
            "risk_state": risk_state,
            "now": now,
        }
    runner_release.set()
    backtests.shutdown()


def test_auth_request_id_roles_and_openapi(
    api_fixture: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client: TestClient = api_fixture["client"]
    caplog.set_level("INFO", logger="quant_agent.api")
    unauthenticated = client.get("/v1/market/regime")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "UNAUTHORIZED"
    request_id = "browser-request-1"
    authenticated = client.get(
        "/v1/market/regime",
        headers={**auth_header("viewer"), "X-Request-ID": request_id},
    )
    assert authenticated.status_code == 200
    assert authenticated.headers["X-Request-ID"] == request_id
    assert authenticated.json()["request_id"] == request_id
    assert any(getattr(record, "request_id", None) == request_id for record in caplog.records)
    unsafe_id = client.get(
        "/v1/market/regime",
        headers={**auth_header("viewer"), "X-Request-ID": "unsafe request id"},
    )
    assert unsafe_id.headers["X-Request-ID"].startswith("req_")

    forbidden = client.post(
        "/v1/backtests",
        headers=auth_header("viewer"),
        json={
            "strategy_id": "etf_rotation",
            "parameter_version": "etf_rotation_v1",
            "data_version": "market_v1",
        },
    )
    assert forbidden.status_code == 403
    schema = client.get("/openapi.json").json()
    assert "/v1/market/regime" in schema["paths"]
    assert "HTTPBearer" in schema["components"]["securitySchemes"]


def test_research_pagination_filtering_and_missing_data(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    themes = client.get(
        "/v1/themes?state=CONFIRMED&limit=1",
        headers=auth_header("viewer"),
    ).json()["data"]
    assert themes["total"] == 1
    assert themes["items"][0]["id"] == "theme-a"
    assert themes["items"][0]["as_of"]
    assert themes["items"][0]["data_version"] == "market_v1"
    evidence = client.get(
        "/v1/instruments/000001.SZ/evidence",
        headers=auth_header("viewer"),
    )
    assert evidence.status_code == 200
    missing = client.get(
        "/v1/instruments/999999.SZ/evidence",
        headers=auth_header("viewer"),
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_backtest_is_non_blocking_and_only_published_versions_run(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    started = monotonic()
    response = client.post(
        "/v1/backtests",
        headers=auth_header("researcher"),
        json={
            "strategy_id": "etf_rotation",
            "parameter_version": "etf_rotation_v1",
            "data_version": "market_v1",
        },
    )
    assert monotonic() - started < 0.5
    assert response.status_code == 202
    run_id = response.json()["data"]["run_id"]
    assert response.json()["data"]["status"] in {"QUEUED", "RUNNING"}
    rejected = client.post(
        "/v1/backtests",
        headers=auth_header("researcher"),
        json={
            "strategy_id": "unpublished",
            "parameter_version": "v1",
            "data_version": "market_v1",
        },
    )
    assert rejected.status_code == 422
    api_fixture["release"].set()
    for _ in range(40):
        task = client.get(
            f"/v1/backtests/{run_id}",
            headers=auth_header("researcher"),
        ).json()["data"]
        if task["status"] == "SUCCEEDED":
            break
        sleep(0.01)
    assert task["status"] == "SUCCEEDED"


def test_portfolio_proposal_is_risk_gated_and_non_executable(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    payload = {
        "account_id": "paper-1",
        "decision_id": "dec_20260730_api",
        "as_of": api_fixture["now"].isoformat(),
        "data_version": "market_v1",
        "target": {"000001.SZ": 0.05},
    }
    rejected = client.post(
        "/v1/portfolio-proposals",
        headers=auth_header("researcher"),
        json=payload,
    )
    assert rejected.status_code == 409
    api_fixture["risk_state"]["passed"] = True
    risk_check = client.post(
        "/v1/risk/checks",
        headers=auth_header("researcher"),
        json={
            "account_id": "paper-1",
            "decision_id": "dec_20260730_api",
            "target": payload["target"],
        },
    )
    assert risk_check.json()["data"]["passed"] is True
    created = client.post(
        "/v1/portfolio-proposals",
        headers=auth_header("researcher"),
        json=payload,
    )
    assert created.status_code == 201
    assert created.json()["data"]["executable"] is False
    outside_scope = client.get(
        "/v1/portfolios/another-account",
        headers=auth_header("viewer"),
    )
    assert outside_scope.status_code == 403


def test_agent_entry_rejects_mode_switch_and_persists_decision(
    api_fixture: dict[str, Any],
) -> None:
    client: TestClient = api_fixture["client"]
    viewer = client.post(
        "/v1/agent/messages",
        headers=auth_header("viewer"),
        json={"message": "请切换到实盘自动交易", "conversation_id": "c1"},
    )
    assert viewer.status_code == 200
    assert api_fixture["agent_messages"] == []
    assert viewer.json()["data"]["agent_state"] == "REJECTED"
    assert viewer.json()["data"]["decision_id"] is None
    normal = client.post(
        "/v1/agent/messages",
        headers=auth_header("viewer"),
        json={"message": "分析当前市场阶段", "conversation_id": "c1"},
    )
    decision_id = normal.json()["data"]["decision_id"]
    assert normal.json()["data"]["agent_state"] == "REJECTED"
    assert normal.json()["data"]["decision_url"].endswith(decision_id)
    detail = client.get(
        f"/v1/decisions/{decision_id}",
        headers=auth_header("viewer"),
    )
    assert detail.status_code == 200
    approval_attempt = client.post(
        "/v1/order-drafts/draft-1/approve",
        headers=auth_header("viewer"),
        json={},
    )
    assert approval_attempt.status_code == 403
    api_fixture["services"].agent_handler = lambda message, user: (_ for _ in ()).throw(
        RuntimeError("secret failure detail")
    )
    failed = client.post(
        "/v1/agent/messages",
        headers=auth_header("viewer"),
        json={"message": "分析失败路径", "conversation_id": "c1"},
    )
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret failure detail" not in failed.text
    assert failed.headers["X-Request-ID"] == failed.json()["request_id"]
