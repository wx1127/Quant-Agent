"""Human-only order approval, paper submission and reconciliation entry points."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, Request

from apps.api.core.auth import Principal, Role, require_account, require_roles
from apps.api.routes.schemas import ApprovalRequest, SubmitRequest
from quant_agent.core.errors import ErrorCode, QuantAgentError
from quant_agent.core.time import shanghai_now

router = APIRouter(tags=["orders"])


@router.get("/order-drafts/{draft_id}")
def get_order_draft(
    draft_id: str,
    request: Request,
    principal: Principal = Depends(require_roles(Role.TRADER, Role.APPROVER)),
) -> dict[str, object]:
    draft = request.app.state.services.orders.get_draft(draft_id)
    require_account(principal, draft.account_id)
    return {"request_id": request.state.request_id, "data": asdict(draft)}


@router.post("/order-drafts/{draft_id}/approve")
def approve_order_draft(
    draft_id: str,
    _payload: ApprovalRequest,
    request: Request,
    principal: Principal = Depends(require_roles(Role.APPROVER)),
) -> dict[str, object]:
    draft = request.app.state.services.orders.get_draft(draft_id)
    require_account(principal, draft.account_id)
    approval = request.app.state.services.orders.approve(
        draft_id,
        approver_id=principal.user_id,
        approved_at=shanghai_now(),
        request_id=request.state.request_id,
    )
    return {
        "request_id": request.state.request_id,
        "data": {
            "approval_token": approval.token,
            "draft_id": draft_id,
            "batch_hash": draft.batch_hash,
            "expires_at": approval.expires_at,
        },
    }


@router.post("/order-drafts/{draft_id}/submit")
def submit_order_draft(
    draft_id: str,
    payload: SubmitRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_roles(Role.TRADER, Role.APPROVER)),
) -> dict[str, object]:
    draft = request.app.state.services.orders.get_draft(draft_id)
    require_account(principal, draft.account_id)
    if not idempotency_key:
        raise QuantAgentError(ErrorCode.INVALID_ARGUMENT, "Idempotency-Key header is required")
    result = request.app.state.services.orders.submit(
        draft_id,
        approval_token=payload.approval_token,
        idempotency_key=idempotency_key,
        submitted_at=shanghai_now(),
        submitted_by=principal.user_id,
        request_id=request.state.request_id,
    )
    return {"request_id": request.state.request_id, "data": result}


@router.post("/accounts/{account_id}/reconcile")
def reconcile_account(
    account_id: str,
    request: Request,
    principal: Principal = Depends(require_roles(Role.TRADER, Role.APPROVER, Role.RISK_ADMIN)),
) -> dict[str, object]:
    require_account(principal, account_id)
    result = request.app.state.services.reconcile_handler(account_id)
    return {
        "request_id": request.state.request_id,
        "data": result,
    }
