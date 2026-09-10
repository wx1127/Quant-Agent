from typing import Any

from core.app import create_app  # noqa: F401  # initialize app import order
from routes.research import TushareResearchProvider


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
    assert [item["ts_code"] for item in result.items] == ["000002.SZ", "000001.SZ"]
    assert fake.calls[0][0] == "trade_cal"
    assert fake.calls[1][0] == "daily"
    assert "pct_chg" in (fake.calls[1][2] or ())


def test_tushare_research_provider_reads_market_index() -> None:
    result = TushareResearchProvider(FakeTushare()).get("market")  # type: ignore[arg-type]

    assert result is not None
    assert result.items[0]["ts_code"] == "000001.SH"
