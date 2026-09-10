"""Data-source readiness endpoints; credentials are never returned."""

from __future__ import annotations

import os
from typing import Any

from core.app import Principal, _principal
from fastapi import APIRouter, Depends


def build_data_router(research_provider: Any = None) -> APIRouter:
    router = APIRouter(prefix="/data", tags=["data"])

    @router.get("/tushare-status")
    async def tushare_status(principal: Principal = Depends(_principal)) -> dict[str, object]:  # noqa: B008
        del principal
        token = os.getenv("MARKET_DATA_TOKEN") or os.getenv("TUSHARE_TOKEN")
        return {
            "provider": "tushare",
            "configured": bool(token),
            "token_present": bool(token),
            "sync_command": "quant-agent data sync --help",
            # This is adapter readiness, not a claim that Tushare is reachable.
            "research_provider_connected": research_provider is not None,
            "message": (
                "token configured; run data sync before research queries"
                if token
                else "set MARKET_DATA_TOKEN or TUSHARE_TOKEN before data sync"
            ),
        }

    return router
