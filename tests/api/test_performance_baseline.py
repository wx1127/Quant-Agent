"""Small, repeatable local capacity baselines for the API surface."""

from time import perf_counter

from test_api_smoke import _client


def test_read_and_chat_baseline_under_local_budget() -> None:
    client = _client()
    headers = {"Authorization": "Bearer analyst:research"}
    started = perf_counter()
    for _ in range(25):
        assert client.get("/v1/research/market", headers=headers).status_code == 200
        assert client.post(
            "/v1/agent/chat",
            headers=headers,
            json={"message": "当前市场状态"},
        ).status_code == 200
    elapsed = perf_counter() - started
    assert elapsed < 5.0


def test_backtest_submission_baseline_under_local_budget() -> None:
    client = _client()
    headers = {"Authorization": "Bearer analyst:research"}
    payload = {
        "strategy_id": "s1",
        "strategy_version": "v1",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-02-01T00:00:00Z",
        "initial_capital": 100000,
    }
    started = perf_counter()
    responses = [
        client.post("/v1/backtest-portfolio/backtests", headers=headers, json=payload)
        for _ in range(50)
    ]
    elapsed = perf_counter() - started
    assert all(response.status_code == 202 for response in responses)
    assert elapsed < 5.0
