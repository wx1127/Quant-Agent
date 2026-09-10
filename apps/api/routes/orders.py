"""Order approval facade; execution remains behind injected services."""

from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from core.app import Principal, _principal, require_role
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class OrderState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    SUBMITTED = "submitted"


class OrderDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    draft_id: str
    decision_id: str = Field(min_length=1)
    state: OrderState = OrderState.DRAFT
    lines: tuple[dict[str, object], ...] = ()
    approval_token: str | None = None


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    idempotency_key: str = Field(min_length=8)


class PaperSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_token: str = Field(min_length=1)


class OrderService:
    def __init__(self, drafts: tuple[OrderDraft, ...] = ()) -> None:
        self._drafts = {draft.draft_id: draft for draft in drafts}
        self._idempotent: dict[tuple[str, str], OrderDraft] = {}

    def get(self, draft_id: str) -> OrderDraft:
        draft = self._drafts.get(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="order draft not found")
        return draft

    def approve(self, draft_id: str, key: str) -> OrderDraft:
        draft = self.get(draft_id)
        cached = self._idempotent.get((draft_id, key))
        if cached is not None:
            return cached
        if draft.state is not OrderState.DRAFT:
            raise HTTPException(status_code=409, detail="order draft is not pending approval")
        approved = draft.model_copy(
            update={"state": OrderState.APPROVED, "approval_token": str(uuid4())}
        )
        self._drafts[draft_id] = approved
        self._idempotent[(draft_id, key)] = approved
        return approved

    def submit_paper(self, draft_id: str, approval_token: str) -> OrderDraft:
        draft = self.get(draft_id)
        if draft.state is not OrderState.APPROVED or draft.approval_token != approval_token:
            raise HTTPException(status_code=409, detail="valid approval token required")
        submitted = draft.model_copy(update={"state": OrderState.SUBMITTED})
        self._drafts[draft_id] = submitted
        return submitted


def build_orders_router(service: OrderService | None = None) -> APIRouter:
    backend = service or OrderService()
    router = APIRouter(prefix="/orders", tags=["orders"])

    @router.get("/drafts/{draft_id}", response_model=OrderDraft)
    async def get_draft(draft_id: str, principal: Principal = Depends(_principal)) -> OrderDraft:  # noqa: B008
        del principal
        return backend.get(draft_id)

    @router.post("/drafts/{draft_id}/approve", response_model=OrderDraft)
    async def approve(
        draft_id: str,
        request: ApprovalRequest,
        principal: Principal = Depends(require_role("approver")),  # noqa: B008
    ) -> OrderDraft:
        del principal
        return backend.approve(draft_id, request.idempotency_key)

    @router.post("/drafts/{draft_id}/submit-paper", response_model=OrderDraft)
    async def submit_paper(
        draft_id: str,
        request: PaperSubmitRequest,
        principal: Principal = Depends(require_role("approver")),  # noqa: B008
    ) -> OrderDraft:
        del principal
        return backend.submit_paper(draft_id, request.approval_token)

    return router
