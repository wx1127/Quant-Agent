"""Read-only research endpoints with explicit freshness metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from core.app import Principal, _principal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


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
        result = source.get(kind, symbol)
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
