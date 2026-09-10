"""FastAPI application factory with request correlation and auth gates."""

from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .alerts import evaluate_alerts
from .metrics import metrics, timer

request_id_context: ContextVar[str] = ContextVar("request_id", default="")


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    request_id: str


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject: str
    roles: frozenset[str] = frozenset()


def _principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not token or ":" not in token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    subject, raw_roles = token.split(":", 1)
    return Principal(subject=subject, roles=frozenset(filter(None, raw_roles.split(","))))


def require_role(role: str):
    def dependency(principal: Principal = Depends(_principal)) -> Principal:  # noqa: B008
        if role not in principal.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient role")
        return principal

    return dependency


def create_app(
    research_provider=None,
    backtest_service=None,
    orders_service=None,
    report_provider=None,
) -> FastAPI:  # type: ignore[no-untyped-def]
    app = FastAPI(title="Quant-Agent API", version="1.0.0")
    router = APIRouter(prefix="/v1")

    @app.middleware("http")
    async def correlation(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = timer()
        supplied = request.headers.get("X-Request-ID")
        try:
            request_id = str(UUID(supplied)) if supplied else str(uuid4())
        except ValueError:
            request_id = str(uuid4())
        token = request_id_context.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            metrics.observe_request(timer() - started, response.status_code >= 500)
            return response
        finally:
            request_id_context.reset(token)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):  # type: ignore[no-untyped-def]
        del request
        payload = ApiError(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            request_id=request_id_context.get(),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @router.get("/health", dependencies=[Depends(_principal)])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version, "request_id": request_id_context.get()}

    @router.get("/metrics", dependencies=[Depends(require_role("admin"))])
    async def metrics_endpoint() -> dict[str, float | int]:
        return metrics.snapshot()

    @router.get("/alerts", dependencies=[Depends(require_role("admin"))])
    async def alerts_endpoint() -> list[dict[str, str]]:
        snapshot = metrics.snapshot()
        alerts = evaluate_alerts(
            requests=int(snapshot["api_requests_total"]),
            errors=int(snapshot["api_errors_total"]),
            average_seconds=float(snapshot["api_request_seconds_average"]),
        )
        return [
            {"code": alert.code, "severity": alert.severity.value, "message": alert.message}
            for alert in alerts
        ]

    @router.get("/admin/ping", dependencies=[Depends(require_role("admin"))])
    async def admin_ping() -> dict[str, str]:
        return {"status": "ok", "role": "admin", "request_id": request_id_context.get()}

    app.include_router(router)
    try:
        from routes.agent import build_agent_router
        from routes.backtest_portfolio import build_backtest_portfolio_router
        from routes.data import build_data_router
        from routes.orders import build_orders_router
        from routes.reports import build_reports_router
        from routes.research import build_research_router

        app.include_router(build_research_router(research_provider), prefix="/v1")
        app.include_router(build_backtest_portfolio_router(backtest_service), prefix="/v1")
        app.include_router(build_data_router(), prefix="/v1")
        app.include_router(build_agent_router(), prefix="/v1")
        app.include_router(build_orders_router(orders_service), prefix="/v1")
        app.include_router(build_reports_router(report_provider), prefix="/v1")
    except ModuleNotFoundError:
        pass
    return app


app = create_app()
