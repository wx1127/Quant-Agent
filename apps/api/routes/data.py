"""Data-source readiness endpoints; credentials are never returned."""

from __future__ import annotations

import os

from core.app import Principal, _principal
from fastapi import APIRouter, Depends


def build_data_router() -> APIRouter:
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
            "research_provider_connected": False,
            "message": (
                "token configured; run data sync before research queries"
                if token
                else "set MARKET_DATA_TOKEN or TUSHARE_TOKEN before data sync"
            ),
        }

    return router
