"""Natural-language research entry point without approval privileges."""

from fastapi import APIRouter, Depends, Request

from apps.api.core.auth import Principal, Role, require_roles
from apps.api.routes.schemas import AgentMessageRequest
from quant_agent.agent.responses import AgentAnswer, AnswerType
from quant_agent.agent.security import AgentSecurityGuard

router = APIRouter(tags=["agent"])


@router.post("/agent/messages")
def agent_message(
    payload: AgentMessageRequest,
    request: Request,
    principal: Principal = Depends(require_roles(Role.VIEWER, Role.RESEARCHER)),
) -> dict[str, object]:
    guard: AgentSecurityGuard = request.app.state.agent_security
    assessment = guard.assess_user_instruction(payload.message)
    if not assessment.allowed:
        answer = AgentAnswer(
            decision_id=None,
            as_of=None,
            answer_type=AnswerType.REJECTED,
            summary="运行模式由服务端配置决定, 对话请求不能切换到模拟盘或实盘。",
            risks=("未授权的模式提升请求已被拒绝。",),
        )
    else:
        answer = request.app.state.services.agent_handler(payload.message, principal.user_id)
    data = answer.model_dump(mode="json")
    references = sorted(
        {reference.tool_call_id for fact in answer.facts for reference in fact.evidence}
        | {
            reference.tool_call_id
            for inference in answer.inferences
            for reference in inference.basis
        }
    )
    data["agent_state"] = (
        "REJECTED"
        if answer.answer_type in {AnswerType.REJECTED, AnswerType.INSUFFICIENT_EVIDENCE}
        else "REPORTED"
    )
    data["tool_references"] = references
    data["decision_url"] = f"/v1/decisions/{answer.decision_id}" if answer.decision_id else None
    decision_id = answer.decision_id
    if decision_id:
        request.app.state.services.decisions.records[decision_id] = data
    return {"request_id": request.state.request_id, "data": data}


@router.get("/decisions/{decision_id}")
def get_decision(
    decision_id: str,
    request: Request,
    _principal: Principal = Depends(require_roles(*tuple(Role))),
) -> dict[str, object]:
    return {
        "request_id": request.state.request_id,
        "data": request.app.state.services.decisions.get(decision_id),
    }
