"""Asynchronous backtest and non-executable portfolio proposal APIs."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Request, status

from apps.api.core.auth import Principal, Role, require_account, require_roles
from apps.api.routes.schemas import (
    BacktestCreateRequest,
    PortfolioProposalRequest,
    RiskCheckRequest,
)

router = APIRouter(tags=["backtest", "portfolio"])


@router.post("/backtests", status_code=status.HTTP_202_ACCEPTED)
def create_backtest(
    payload: BacktestCreateRequest,
    request: Request,
    _principal: Principal = Depends(require_roles(Role.RESEARCHER)),
) -> dict[str, object]:
    task = request.app.state.services.backtests.submit(
        payload.strategy_id,
        payload.parameter_version,
        payload.data_version,
    )
    return {"request_id": request.state.request_id, "data": asdict(task)}


@router.get("/backtests/{run_id}")
def get_backtest(
    run_id: str,
    request: Request,
    _principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.TRADER, Role.APPROVER)),
) -> dict[str, object]:
    task = request.app.state.services.backtests.get(run_id)
    return {"request_id": request.state.request_id, "data": asdict(task)}


@router.get("/portfolios/{account_id}")
def get_portfolio(
    account_id: str,
    request: Request,
    principal: Principal = Depends(
        require_roles(Role.VIEWER, Role.RESEARCHER, Role.TRADER, Role.APPROVER)
    ),
) -> dict[str, object]:
    require_account(principal, account_id)
    data = request.app.state.services.portfolios.get_snapshot(account_id)
    return {"request_id": request.state.request_id, "data": data}


@router.post("/portfolio-proposals", status_code=status.HTTP_201_CREATED)
def create_portfolio_proposal(
    payload: PortfolioProposalRequest,
    request: Request,
    principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.TRADER)),
) -> dict[str, object]:
    require_account(principal, payload.account_id)
    risk_result = request.app.state.services.risk_handler(
        payload.account_id,
        payload.decision_id,
        payload.target,
    )
    proposal = request.app.state.services.portfolios.create_proposal(
        account_id=payload.account_id,
        decision_id=payload.decision_id,
        as_of=payload.as_of,
        data_version=payload.data_version,
        target=payload.target,
        risk_result=risk_result,
    )
    return {"request_id": request.state.request_id, "data": asdict(proposal)}


@router.post("/risk/checks")
def check_risk(
    payload: RiskCheckRequest,
    request: Request,
    principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.TRADER)),
) -> dict[str, object]:
    require_account(principal, payload.account_id)
    result = request.app.state.services.risk_handler(
        payload.account_id,
        payload.decision_id,
        payload.target,
    )
    return {"request_id": request.state.request_id, "data": result}


@router.get("/portfolio-proposals/{proposal_id}")
def get_portfolio_proposal(
    proposal_id: str,
    request: Request,
    principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.TRADER, Role.APPROVER)),
) -> dict[str, object]:
    proposal = request.app.state.services.portfolios.get_proposal(proposal_id)
    require_account(principal, proposal.account_id)
    return {"request_id": request.state.request_id, "data": asdict(proposal)}
