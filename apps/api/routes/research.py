"""Read-only research endpoints with explicit freshness metadata."""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from core.app import Principal, _principal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from quant_agent.data.providers.base import ProviderError
from quant_agent.data.providers.tushare import TushareHttpProvider


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str
    as_of: datetime
    data_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    items: tuple[dict[str, Any], ...] = ()


class ResearchProvider(Protocol):
    def get(self, kind: str, symbol: str | None = None) -> ResearchResult | None: ...


class EmptyResearchProvider:
    def get(self, kind: str, symbol: str | None = None) -> ResearchResult | None:
        del kind, symbol
        return None


class TushareResearchProvider:
    """Read-only live adapter over the existing Tushare HTTP boundary."""

    def __init__(self, provider: TushareHttpProvider, *, cache_ttl_seconds: float = 30.0) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        self._provider = provider
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple[str, str | None], tuple[float, ResearchResult | None]] = {}

    @classmethod
    def from_env(cls) -> TushareResearchProvider | None:
        token = os.getenv("MARKET_DATA_TOKEN") or os.getenv("TUSHARE_TOKEN")
        return cls(TushareHttpProvider(token)) if token else None

    def get(self, kind: str, symbol: str | None = None) -> ResearchResult | None:
        cache_key = (kind, symbol)
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]
        trade_date = self._latest_trade_date()
        if kind == "market":
            rows = self._rows("index_daily", {"ts_code": "000001.SH", "trade_date": trade_date})
        elif kind in {"leaders", "candidates"}:
            rows = self._rows(
                "daily",
                {"trade_date": trade_date},
                fields=("ts_code", "trade_date", "close", "vol", "amount", "pct_chg"),
            )
            rows = sorted(rows, key=lambda row: float(row.get("pct_chg") or 0), reverse=True)[:20]
        elif kind == "stock" and symbol:
            rows = self._rows("daily", {"ts_code": symbol, "trade_date": trade_date})
        else:
            result = None
        if kind in {"market", "leaders", "candidates"} or (kind == "stock" and symbol):
            result = self._result(kind, rows)
        self._cache[cache_key] = (now + self._cache_ttl_seconds, result)
        return result

    def _latest_trade_date(self) -> str:
        today = date.today()
        payload = self._provider.fetch_raw_json(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": (today - timedelta(days=14)).strftime("%Y%m%d"),
                "end_date": today.strftime("%Y%m%d"),
            },
        )
        rows = self._provider.decode_raw_rows(payload, endpoint="trade_cal")
        open_days = [str(row["cal_date"]) for row in rows if str(row.get("is_open")) == "1"]
        return max(open_days) if open_days else today.strftime("%Y%m%d")

    def _rows(
        self,
        endpoint: str,
        params: dict[str, str],
        *,
        fields: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        payload = self._provider.fetch_raw_json(endpoint, params, fields=fields)
        return self._provider.decode_raw_rows(payload, endpoint=endpoint)

    @staticmethod
    def _result(kind: str, rows: tuple[dict[str, Any], ...]) -> ResearchResult | None:
        if not rows:
            return None
        return ResearchResult(
            kind=kind,
            as_of=datetime.now(UTC),
            data_version="tushare-live",
            model_version="raw-provider-v1",
            items=rows,
        )


class InMemoryResearchProvider:
    """Deterministic provider for local runs and contract tests.

    Production adapters can implement the same protocol against frozen snapshots
    without changing the HTTP layer.
    """

    def __init__(self, results: tuple[ResearchResult, ...] = ()) -> None:
        self._results = {
            (result.kind, item.get("symbol")): result
            for result in results
            for item in result.items
        }
        self._by_kind = {result.kind: result for result in results}

    def get(self, kind: str, symbol: str | None = None) -> ResearchResult | None:
        if symbol is None:
            return self._by_kind.get(kind)
        result = self._results.get((kind, symbol)) or self._by_kind.get("stock")
        if result is None:
            return None
        items = tuple(item for item in result.items if item.get("symbol") == symbol)
        return result.model_copy(update={"items": items})


def build_research_router(provider: ResearchProvider | None = None) -> APIRouter:
    source = provider or EmptyResearchProvider()
    router = APIRouter(prefix="/research", tags=["research"])

    def read(kind: str, symbol: str | None, principal: Principal) -> ResearchResult:
        del principal
        try:
            result = source.get(kind, symbol)
        except ProviderError as error:
            # Keep provider details in server logs only; never expose response text
            # or credentials through the public API error body.
            raise HTTPException(
                status_code=503,
                detail=f"research data source unavailable: {error.provider}",
            ) from error
        if result is None:
            raise HTTPException(status_code=404, detail=f"research data unavailable: {kind}")
        return result

    @router.get("/market", response_model=ResearchResult)
    async def market(principal: Principal = Depends(_principal)) -> ResearchResult:  # noqa: B008
        return read("market", None, principal)

    @router.get("/mainlines", response_model=ResearchResult)
    async def mainlines(principal: Principal = Depends(_principal)) -> ResearchResult:  # noqa: B008
        return read("mainlines", None, principal)

    @router.get("/leaders", response_model=ResearchResult)
    async def leaders(
        limit: int = Query(default=50, ge=1, le=500),
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> ResearchResult:
        result = read("leaders", None, principal)
        return result.model_copy(update={"items": result.items[:limit]})

    @router.get("/candidates", response_model=ResearchResult)
    async def candidates(
        limit: int = Query(default=50, ge=1, le=500),
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> ResearchResult:
        result = read("candidates", None, principal)
        return result.model_copy(update={"items": result.items[:limit]})

    @router.get("/stocks/{symbol}", response_model=ResearchResult)
    async def stock(symbol: str, principal: Principal = Depends(_principal)) -> ResearchResult:  # noqa: B008
        return read("stock", symbol.upper(), principal)

    return router
