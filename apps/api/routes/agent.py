"""Controlled conversational entry point for the Agent Harness."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from core.app import Principal, _principal
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: str
    decision_id: str
    as_of: datetime
    data_version: str
    tool_references: tuple[str, ...] = ()
    requires_human_approval: bool = True


class AgentChatService(Protocol):
    def answer(self, message: str, principal: Principal) -> ChatResponse: ...


class DefaultAgentChatService:
    def answer(self, message: str, principal: Principal) -> ChatResponse:
        del principal
        return ChatResponse(
            answer=(
                f"已收到研究请求: {message}。当前为受控研究入口, "
                "结果需要通过数据快照和人工审批流程后才能形成交易动作。"
            ),
            decision_id=str(uuid4()),
            as_of=datetime.now(UTC),
            data_version="unbound",
        )


def build_agent_router(service: AgentChatService | None = None) -> APIRouter:
    backend = service or DefaultAgentChatService()
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> ChatResponse:
        return backend.answer(request.message, principal)

    return router
