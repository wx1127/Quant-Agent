from datetime import UTC, datetime

from core.app import create_app
from fastapi.testclient import TestClient
from routes.backtest_portfolio import BacktestPortfolioService
from routes.orders import OrderDraft, OrderService
from routes.reports import DailyReport, InMemoryReportProvider
from routes.research import InMemoryResearchProvider, ResearchResult


def _client() -> TestClient:
    now = datetime.now(UTC)
    research = InMemoryResearchProvider(
        (
            ResearchResult(
                kind="market",
                as_of=now,
                data_version="d1",
                model_version="m1",
                items=({"state": "neutral"},),
            ),
            ResearchResult(
                kind="leaders",
                as_of=now,
                data_version="d1",
                model_version="m1",
                items=({"symbol": "000001", "score": 0.9},),
            ),
        )
    )
    reports = InMemoryReportProvider(
        (
            DailyReport(
                report_date="2026-09-10",
                as_of=now,
                data_version="d1",
                model_version="m1",
                market_summary="neutral",
                evidence_ids=("e1",),
            ),
        )
    )
    return TestClient(
        create_app(
            research_provider=research,
            report_provider=reports,
            backtest_service=BacktestPortfolioService(frozenset({("s1", "v1")})),
            orders_service=OrderService((OrderDraft(draft_id="draft-1", decision_id="dec-1"),)),
        )
    )


def test_api_golden_smoke_flow() -> None:
    client = _client()
    research_headers = {"Authorization": "Bearer analyst:research"}
    approver_headers = {"Authorization": "Bearer reviewer:approver"}

    assert client.get("/v1/health").status_code == 401
    market = client.get("/v1/research/market", headers=research_headers).json()
    assert market["items"][0]["state"] == "neutral"
    assert client.get("/v1/reports/daily/2026-09-10", headers=research_headers).status_code == 200

    backtest = client.post(
        "/v1/backtest-portfolio/backtests",
        headers=research_headers,
        json={
            "strategy_id": "s1",
            "strategy_version": "v1",
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-02-01T00:00:00Z",
            "initial_capital": 100000,
        },
    )
    assert backtest.status_code == 202
    assert client.post(
        "/v1/backtest-portfolio/backtests",
        headers=research_headers,
        json={**backtest.json(), "strategy_id": "unpublished"},
    ).status_code in {400, 422}

    denied = client.post(
        "/v1/orders/drafts/draft-1/approve",
        headers=research_headers,
        json={"idempotency_key": "golden-key-1"},
    )
    assert denied.status_code == 403
    approved = client.post(
        "/v1/orders/drafts/draft-1/approve",
        headers=approver_headers,
        json={"idempotency_key": "golden-key-1"},
    )
    assert approved.status_code == 200
    submitted = client.post(
        "/v1/orders/drafts/draft-1/submit-paper",
        headers=approver_headers,
        json={"approval_token": approved.json()["approval_token"]},
    )
    assert submitted.json()["state"] == "submitted"

    chat = client.post(
        "/v1/agent/chat",
        headers=research_headers,
        json={"message": "请解释当前市场"},
    )
    assert chat.status_code == 200
    assert chat.json()["requires_human_approval"] is True


def test_tushare_status_reflects_adapter_readiness(monkeypatch) -> None:
    monkeypatch.delenv("MARKET_DATA_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    empty = TestClient(create_app())
    headers = {"Authorization": "Bearer analyst:research"}
    unavailable = empty.get("/v1/data/tushare-status", headers=headers)
    assert unavailable.status_code == 200
    assert unavailable.json()["configured"] is False
    assert unavailable.json()["research_provider_connected"] is False
    assert unavailable.json()["research_cache_ttl_seconds"] == 30.0

    configured = TestClient(create_app(research_provider=InMemoryResearchProvider()))
    ready = configured.get("/v1/data/tushare-status", headers=headers)
    assert ready.status_code == 200
    assert ready.json()["research_provider_connected"] is True

    monkeypatch.setenv("RESEARCH_CACHE_TTL_SECONDS", "999")
    capped = TestClient(create_app()).get("/v1/data/tushare-status", headers=headers)
    assert capped.json()["research_cache_ttl_seconds"] == 300.0
    monkeypatch.setenv("RESEARCH_CACHE_TTL_SECONDS", "invalid")
    fallback = TestClient(create_app()).get("/v1/data/tushare-status", headers=headers)
    assert fallback.json()["research_cache_ttl_seconds"] == 30.0
