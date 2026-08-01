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


def _stock_record(item: ResearchRecord) -> dict[str, object]:
    payload = _record(item)
    payload["id"] = item.payload.get("instrument_id", item.record_id)
    return payload


def _ranked_stock_page(
    request: Request,
    kind: str,
    offset: int,
    limit: int,
    filters: dict[str, str],
) -> dict[str, object]:
    rows, total = request.app.state.services.research.list(
        kind,
        offset=0,
        limit=10_000,
        filters=filters,
    )
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item.payload.get("rank", 10_000)),
            -float(item.payload.get("score", 0)),
            str(item.payload.get("instrument_id", item.record_id)),
        ),
    )
    return {
        "request_id": request.state.request_id,
        "data": {
            "items": [_stock_record(item) for item in ordered[offset : offset + limit]],
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


@router.get("/themes/{theme_id}")
def theme_detail(
    request: Request,
    theme_id: str,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    item = request.app.state.services.research.get("themes", theme_id)
    return {"request_id": request.state.request_id, "data": _record(item)}


@router.get("/themes/{theme_id}/leaders")
def leaders(
    request: Request,
    theme_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    return _ranked_stock_page(
        request,
        "leaders",
        offset,
        limit,
        {"theme_id": theme_id},
    )


@router.get("/themes/{theme_id}/stocks")
def theme_stocks(
    request: Request,
    theme_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    request.app.state.services.research.get("themes", theme_id)
    return _ranked_stock_page(
        request,
        "theme_members",
        offset,
        limit,
        {"theme_id": theme_id},
    )


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


@router.get("/instruments/{instrument_id}")
def instrument_detail(
    request: Request,
    instrument_id: str,
    _principal: Principal = Depends(_viewer),
) -> dict[str, object]:
    item = request.app.state.services.research.get("stock_details", instrument_id)
    return {"request_id": request.state.request_id, "data": _record(item)}
