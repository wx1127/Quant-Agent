from typing import Any

from core.app import create_app  # initialize app import order
from fastapi.testclient import TestClient
from routes.research import TushareResearchProvider

from quant_agent.data.providers.base import ProviderError


class FakeTushare:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], tuple[str, ...] | None]] = []

    def fetch_raw_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((endpoint, params, fields))
        if endpoint == "trade_cal":
            return {"data": {"items": [["SSE", "20260909", "1", "20260908"]]}}
        if endpoint == "index_daily":
            return {
                "data": {
                    "items": [
                        ["000001.SH", "20260909", "3000", "3010", "2990", "3005", "100", "1000"]
                    ]
                }
            }
        if endpoint == "daily":
            return {
                "data": {
                    "items": [
                        ["000001.SZ", "20260909", "10", "100", "1000", "2.5"],
                        ["000002.SZ", "20260909", "10", "100", "1000", "8.0"],
                    ]
                }
            }
        raise AssertionError(endpoint)

    @staticmethod
    def decode_raw_rows(payload: dict[str, Any], *, endpoint: str) -> tuple[dict[str, Any], ...]:
        fields = {
            "trade_cal": ("exchange", "cal_date", "is_open", "pretrade_date"),
            "index_daily": (
                "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"
            ),
            "daily": ("ts_code", "trade_date", "close", "vol", "amount", "pct_chg"),
        }[endpoint]
        return tuple(dict(zip(fields, row, strict=True)) for row in payload["data"]["items"])


def test_tushare_research_provider_uses_trade_date_and_sorts_leaders() -> None:
    fake = FakeTushare()
    provider = TushareResearchProvider(fake)  # type: ignore[arg-type]
    result = provider.get("leaders")

    assert result is not None
    assert result.data_version == "tushare-live"
    assert result.freshness == "live"
    assert [item["ts_code"] for item in result.items] == ["000002.SZ", "000001.SZ"]
    assert fake.calls[0][0] == "trade_cal"
    assert fake.calls[1][0] == "daily"
    assert "pct_chg" in (fake.calls[1][2] or ())

    cached = provider.get("leaders")
    assert cached is not None
    assert cached.freshness == "cached"
    assert len(fake.calls) == 2


def test_tushare_research_provider_reads_market_index() -> None:
    result = TushareResearchProvider(FakeTushare()).get("market")  # type: ignore[arg-type]

    assert result is not None
    assert result.items[0]["ts_code"] == "000001.SH"


def test_tushare_research_provider_can_disable_cache() -> None:
    fake = FakeTushare()
    provider = TushareResearchProvider(fake, cache_ttl_seconds=0)

    provider.get("market")
    provider.get("market")

    assert [call[0] for call in fake.calls] == [
        "trade_cal",
        "index_daily",
        "trade_cal",
        "index_daily",
    ]


def test_tushare_provider_reads_bounded_cache_ttl(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_DATA_TOKEN", "test-token")
    monkeypatch.setenv("RESEARCH_CACHE_TTL_SECONDS", "999")
    bounded = TushareResearchProvider.from_env()
    assert bounded is not None
    assert bounded._cache_ttl_seconds == 300.0
    bounded._provider.close()

    monkeypatch.setenv("RESEARCH_CACHE_TTL_SECONDS", "invalid")
    fallback = TushareResearchProvider.from_env()
    assert fallback is not None
    assert fallback._cache_ttl_seconds == 30.0
    fallback._provider.close()


class FailingResearchProvider:
    def get(self, kind: str, symbol: str | None = None) -> None:
        del kind, symbol
        raise ProviderError("tushare", "daily", "token=should-not-escape", retryable=False)


def test_provider_error_is_redacted_at_api_boundary() -> None:
    client = TestClient(create_app(research_provider=FailingResearchProvider()))
    response = client.get(
        "/v1/research/market",
        headers={"Authorization": "Bearer analyst:research"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["message"] == "research data source unavailable: tushare"
    assert "should-not-escape" not in response.text
