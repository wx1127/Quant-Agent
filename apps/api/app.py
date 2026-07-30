"""Quant Agent FastAPI application factory."""

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
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
from quant_agent.config.models import RuntimeMode
from quant_agent.risk.kill_switch import KillSwitch


def create_app(
    *,
    services: ApiServices | None = None,
    auth: AuthService | None = None,
    agent_security: AgentSecurityGuard | None = None,
) -> FastAPI:
    owned_services = services is None
    api_services = services or _default_services()

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
    app.state.agent_security = agent_security or AgentSecurityGuard(
        configured_mode=RuntimeMode.RESEARCH,
        allowed_instruments=set(),
    )
    install_error_handling(app)

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

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


def _default_services() -> ApiServices:
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
        KillSwitch(),
        lambda draft: {
            "status": "PAPER_SUBMITTED",
            "draft_id": draft.draft_id,
            "batch_hash": draft.batch_hash,
        },
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
            "differences": [],
            "message": "No reconciliation service is configured.",
        },
        risk_handler=lambda account_id, decision_id, target: {
            "passed": False,
            "account_id": account_id,
            "decision_id": decision_id,
            "violations": ["No risk service is configured."],
        },
    )


app = create_app()
