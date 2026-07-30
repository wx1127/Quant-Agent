"""Read-only market, mainline, leader, candidate and evidence APIs."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from apps.api.core.auth import Principal, Role, require_roles
from apps.api.core.services import ResearchRecord

router = APIRouter(tags=["research"])
_viewer = require_roles(*tuple(Role))


def _record(item: ResearchRecord) -> dict[str, object]:
    return {
        "id": item.record_id,
        "as_of": item.as_of,
        "data_version": item.data_version,
        **item.payload,
    }


def _page(
    request: Request,
    kind: str,
    offset: int,
    limit: int,
    filters: dict[str, str],
) -> dict[str, object]:
    rows, total = request.app.state.services.research.list(
        kind,
        offset=offset,
        limit=limit,
        filters=filters,
    )
    return {
        "request_id": request.state.request_id,
        "data": {
            "items": [_record(item) for item in rows],
            "offset": offset,
            "limit": limit,
            "total": total,
        },
    }


@router.get("/market/regime")
def market_regime(
    request: Request,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    return {
        "request_id": request.state.request_id,
        "data": _record(request.app.state.services.research.regime()),
    }


@router.get("/themes")
def themes(
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    state: str | None = None,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    return _page(request, "themes", offset, limit, {"state": state} if state else {})


@router.get("/themes/{theme_id}/leaders")
def leaders(
    request: Request,
    theme_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    return _page(request, "leaders", offset, limit, {"theme_id": theme_id})


@router.get("/candidates")
def candidates(
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    tier: str | None = None,
    theme_id: str | None = None,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    filters = {
        key: value for key, value in (("tier", tier), ("theme_id", theme_id)) if value is not None
    }
    return _page(request, "candidates", offset, limit, filters)


@router.get("/instruments/{instrument_id}/evidence")
def instrument_evidence(
    request: Request,
    instrument_id: str,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    item = request.app.state.services.research.get("evidence", instrument_id)
    return {"request_id": request.state.request_id, "data": _record(item)}
