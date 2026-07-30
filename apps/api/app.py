"""Quant Agent FastAPI application factory."""

import hashlib
import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from apps.api.core.auth import AuthService
from apps.api.core.errors import install_error_handling
from apps.api.core.services import (
    ApiServices,
    BacktestTaskStore,
    DecisionStore,
    OrderApprovalService,
    PortfolioService,
    ResearchCatalog,
)
from apps.api.routes import agent, backtest_portfolio, orders, research
from quant_agent import __version__
from quant_agent.agent.responses import AgentAnswer
from quant_agent.agent.security import AgentSecurityGuard
from quant_agent.config.models import AppEnvironment, RuntimeMode, RuntimeSettings
from quant_agent.core.time import shanghai_now
from quant_agent.observability.audit import AuditEvent, AuditSink, JsonLinesAuditSink
from quant_agent.observability.monitoring import MonitoringRegistry
from quant_agent.risk.kill_switch import KillSwitch, KillSwitchScope


def create_app(
    *,
    services: ApiServices | None = None,
    auth: AuthService | None = None,
    agent_security: AgentSecurityGuard | None = None,
    runtime_settings: RuntimeSettings | None = None,
    kill_switch: KillSwitch | None = None,
    audit_sink: AuditSink | None = None,
    monitoring: MonitoringRegistry | None = None,
) -> FastAPI:
    runtime = runtime_settings or _runtime_from_environment()
    deployment_kill_switch = kill_switch or KillSwitch()
    deployment_audit = audit_sink or _audit_from_environment(runtime)
    monitoring_registry = monitoring or MonitoringRegistry()
    owned_services = services is None
    api_services = services or _default_services(
        deployment_kill_switch,
        deployment_audit,
        monitoring_registry,
    )
    if runtime.app_env is AppEnvironment.PRODUCTION:
        if api_services.orders.kill_switch is not deployment_kill_switch:
            raise ValueError("production services must share the deployment Kill Switch")
        _secure_production_boot(deployment_kill_switch, deployment_audit)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if owned_services:
            api_services.backtests.shutdown()

    app = FastAPI(
        title="Quant Agent API",
        version=__version__,
        description="Versioned, authenticated research and controlled paper-trading API.",
        lifespan=lifespan,
    )
    app.state.services = api_services
    app.state.auth = auth or AuthService()
    app.state.runtime = runtime
    app.state.kill_switch = deployment_kill_switch
    app.state.monitoring = monitoring_registry
    app.state.agent_security = agent_security or AgentSecurityGuard(
        configured_mode=runtime.mode,
        allowed_instruments=set(),
    )
    install_error_handling(app)

    @app.middleware("http")
    async def monitor_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = perf_counter()
        success = False
        try:
            response = await call_next(request)
            success = response.status_code < 500
            return response
        finally:
            route = request.scope.get("route")
            operation = getattr(route, "path", "unmatched")
            monitoring_registry.record_service(
                "api",
                operation,
                success=success,
                latency_seconds=perf_counter() - started,
            )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> dict[str, str | bool]:
        kill_switch_active = not deployment_kill_switch.order_allowed("__readiness_probe__")
        monitoring_registry.set_kill_switch(kill_switch_active)
        return {
            "status": "ready",
            "version": __version__,
            "environment": runtime.app_env,
            "mode": runtime.mode,
            "kill_switch_active": kill_switch_active,
        }

    @app.get("/internal/metrics", include_in_schema=False)
    def internal_metrics() -> PlainTextResponse:
        monitoring_registry.set_kill_switch(
            not deployment_kill_switch.order_allowed("__metrics_probe__")
        )
        return PlainTextResponse(
            monitoring_registry.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    app.include_router(research.router, prefix="/v1")
    app.include_router(backtest_portfolio.router, prefix="/v1")
    app.include_router(orders.router, prefix="/v1")
    app.include_router(agent.router, prefix="/v1")

    web_root = Path(__file__).resolve().parents[1] / "web"
    static_root = web_root / "static"
    if static_root.exists():
        app.mount("/web/static", StaticFiles(directory=static_root), name="web-static")

    def page(filename: str) -> FileResponse:
        return FileResponse(web_root / filename, media_type="text/html")

    @app.get("/web", include_in_schema=False)
    def web_home() -> FileResponse:
        return page("research.html")

    @app.get("/web/research", include_in_schema=False)
    def web_research() -> FileResponse:
        return page("research.html")

    @app.get("/web/trading", include_in_schema=False)
    def web_trading() -> FileResponse:
        return page("trading.html")

    @app.get("/web/chat", include_in_schema=False)
    def web_chat() -> FileResponse:
        return page("chat.html")

    @app.get("/web/evidence", include_in_schema=False)
    def web_evidence() -> FileResponse:
        return page("evidence.html")

    return app


def _default_services(
    kill_switch: KillSwitch | None = None,
    audit_sink: AuditSink | None = None,
    monitoring: MonitoringRegistry | None = None,
) -> ApiServices:
    backtests = BacktestTaskStore(
        {("etf_rotation", "etf_rotation_v1"), ("mainline_leader", "mainline_leader_v1")},
        lambda strategy, parameters, data: {
            "strategy_id": strategy,
            "parameter_version": parameters,
            "data_version": data,
            "message": "No production runner is configured.",
        },
    )
    portfolios = PortfolioService()
    orders_service = OrderApprovalService(
        secrets.token_bytes(32),
        kill_switch or KillSwitch(),
        lambda draft: {
            "status": "PAPER_SUBMITTED",
            "draft_id": draft.draft_id,
            "batch_hash": draft.batch_hash,
        },
        audit_sink,
        monitoring=monitoring,
    )

    def no_evidence(_message: str, _user_id: str) -> AgentAnswer:
        return AgentAnswer.insufficient(decision_id=None, as_of=None)

    return ApiServices(
        research=ResearchCatalog(),
        backtests=backtests,
        portfolios=portfolios,
        orders=orders_service,
        decisions=DecisionStore(),
        agent_handler=no_evidence,
        reconcile_handler=lambda account_id: {
            "account_id": account_id,
            "status": "UNAVAILABLE",
            "available": False,
            "differences": [],
            "message": "No reconciliation service is configured.",
        },
        risk_handler=lambda account_id, decision_id, target: {
            "passed": False,
            "available": False,
            "account_id": account_id,
            "decision_id": decision_id,
            "violations": ["No risk service is configured."],
        },
    )


def _runtime_from_environment() -> RuntimeSettings:
    app_env = AppEnvironment(os.getenv("QUANT_AGENT_APP_ENV", AppEnvironment.LOCAL))
    mode = RuntimeMode(os.getenv("QUANT_AGENT_RUNTIME_MODE", RuntimeMode.RESEARCH))
    return RuntimeSettings(app_env=app_env, mode=mode, allow_live_auto=False)


def _audit_from_environment(runtime: RuntimeSettings) -> AuditSink | None:
    path = os.getenv("QUANT_AGENT_AUDIT_LOG_PATH")
    if path:
        return JsonLinesAuditSink(path)
    if runtime.app_env is AppEnvironment.PRODUCTION:
        raise ValueError("production requires QUANT_AGENT_AUDIT_LOG_PATH")
    return None


def _secure_production_boot(kill_switch: KillSwitch, audit_sink: AuditSink | None) -> None:
    occurred_at = shanghai_now()
    snapshot_hash = hashlib.sha256(
        f"production-boot|{__version__}|{occurred_at.isoformat()}".encode()
    ).hexdigest()
    if kill_switch.order_allowed("__production_boot__"):
        kill_switch.trigger(
            scope=KillSwitchScope.GLOBAL,
            account_id=None,
            reason="production starts with Kill Switch active",
            actor_id="deployment",
            actor_role="SYSTEM",
            occurred_at=occurred_at,
            incident_snapshot_hash=snapshot_hash,
        )
    if audit_sink is not None:
        audit_sink.append(
            AuditEvent(
                event_type="deployment",
                actor_id="deployment",
                action="PRODUCTION_BOOT_KILL_SWITCH",
                result="active",
                request_id=f"boot-{snapshot_hash[:16]}",
                occurred_at=occurred_at,
                metadata={"version": __version__, "snapshot_hash": snapshot_hash},
            )
        )


app = create_app()
