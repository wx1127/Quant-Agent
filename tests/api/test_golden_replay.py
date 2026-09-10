"""Golden scenario replay: stable semantics, fresh runtime identifiers."""

from test_api_smoke import _client


def _run_scenario() -> dict[str, object]:
    client = _client()
    research = {"Authorization": "Bearer analyst:research"}
    approver = {"Authorization": "Bearer reviewer:approver"}
    market = client.get("/v1/research/market", headers=research)
    report = client.get("/v1/reports/daily/2026-09-10", headers=research)
    backtest = client.post(
        "/v1/backtest-portfolio/backtests",
        headers=research,
        json={
            "strategy_id": "s1",
            "strategy_version": "v1",
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-02-01T00:00:00Z",
            "initial_capital": 100000,
        },
    )
    approval = client.post(
        "/v1/orders/drafts/draft-1/approve",
        headers=approver,
        json={"idempotency_key": "golden-replay-key"},
    )
    submit = client.post(
        "/v1/orders/drafts/draft-1/submit-paper",
        headers=approver,
        json={"approval_token": approval.json()["approval_token"]},
    )
    chat = client.post("/v1/agent/chat", headers=research, json={"message": "请解释当前市场"})
    return {
        "market": (market.status_code, market.json()["data_version"]),
        "report": (report.status_code, report.json()["model_version"]),
        "backtest": (backtest.status_code, backtest.json()["status"]),
        "approval": (approval.status_code, approval.json()["state"]),
        "submit": (submit.status_code, submit.json()["state"]),
        "chat": (chat.status_code, chat.json()["requires_human_approval"]),
    }


def test_golden_scenario_replays_stable_semantics() -> None:
    assert _run_scenario() == _run_scenario()
