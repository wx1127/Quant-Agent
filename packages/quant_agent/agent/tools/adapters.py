"""Registration helpers for research and controlled portfolio/execution tools."""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_agent.agent.runtime import AgentState
from quant_agent.agent.tools.registry import (
    AgentTool,
    ToolCallContext,
    ToolCategory,
    ToolEffect,
    ToolRegistry,
)
from quant_agent.config.models import RuntimeMode
from quant_agent.core.errors import ErrorCode, QuantAgentError

READ_MODES = frozenset(
    {
        RuntimeMode.RESEARCH,
        RuntimeMode.BACKTEST,
        RuntimeMode.PAPER,
        RuntimeMode.LIVE_ASSISTED,
    }
)
TRADING_MODES = frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED})


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstrumentArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument_id: str = Field(pattern=r"^[A-Z0-9_.-]{2,32}$")


class PayloadArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict[str, Any]


class RiskApprovedPayloadArguments(BaseModel):
    """A draft can only be requested with a deterministic passed risk decision."""

    model_config = ConfigDict(extra="forbid")
    risk_passed: Literal[True]
    risk_request_id: str = Field(min_length=1)
    payload: dict[str, Any]


def _handler(function: Callable[..., Any]) -> Callable[[BaseModel, ToolCallContext], Any]:
    def call(arguments: BaseModel, context: ToolCallContext) -> Any:
        return function(
            arguments.model_dump(),
            decision_id=context.snapshot.decision_id,
            as_of=context.snapshot.as_of,
            data_version=context.snapshot.data_version,
        )

    return call


def _risk_handler(function: Callable[..., Any]) -> Callable[[BaseModel, ToolCallContext], Any]:
    base = _handler(function)

    def call(arguments: BaseModel, context: ToolCallContext) -> Any:
        result = base(arguments, context)
        passed = (
            result.get("passed") if isinstance(result, dict) else getattr(result, "passed", False)
        )
        request_id = (
            result.get("request_id")
            if isinstance(result, dict)
            else getattr(result, "request_id", None)
        )
        if passed is True and isinstance(request_id, str) and request_id:
            context.risk_approved_request_ids.add(request_id)
        return result

    return call


def _draft_handler(function: Callable[..., Any]) -> Callable[[BaseModel, ToolCallContext], Any]:
    base = _handler(function)

    def call(arguments: BaseModel, context: ToolCallContext) -> Any:
        request_id = getattr(arguments, "risk_request_id", None)
        if request_id not in context.risk_approved_request_ids:
            raise QuantAgentError(
                ErrorCode.RISK_REJECTED,
                "order draft requires a passed risk result from this decision runtime",
            )
        return base(arguments, context)

    return call


def register_research_tools(
    registry: ToolRegistry,
    handlers: dict[str, Callable[..., Any]],
    *,
    version: str = "research_tools_v1",
) -> None:
    """Expose only deterministic, read-only analysis capabilities."""

    definitions: dict[str, tuple[ToolCategory, type[BaseModel]]] = {
        "get_market_snapshot": (ToolCategory.DATA, EmptyArguments),
        "validate_market_data": (ToolCategory.VALIDATION, EmptyArguments),
        "detect_market_regime": (ToolCategory.ANALYSIS, EmptyArguments),
        "rank_market_themes": (ToolCategory.ANALYSIS, EmptyArguments),
        "rank_theme_leaders": (ToolCategory.ANALYSIS, PayloadArguments),
        "rank_stock_candidates": (ToolCategory.ANALYSIS, PayloadArguments),
        "explain_candidate": (ToolCategory.ANALYSIS, InstrumentArguments),
    }
    for name, (category, argument_model) in definitions.items():
        if name not in handlers:
            continue
        registry.register(
            AgentTool(
                name=name,
                category=category,
                effect=ToolEffect.READ,
                argument_model=argument_model,
                handler=_handler(handlers[name]),
                allowed_modes=READ_MODES,
                allowed_states=frozenset({AgentState.SNAPSHOT_READY, AgentState.ANALYZED}),
                service=name.replace("_", "-"),
                version=version,
            )
        )


def register_portfolio_execution_tools(
    registry: ToolRegistry,
    handlers: dict[str, Callable[..., Any]],
    *,
    version: str = "portfolio_execution_tools_v1",
) -> None:
    """Expose staged portfolio, risk, draft, paper and reconciliation operations."""

    definitions = {
        "get_portfolio_snapshot": (
            ToolCategory.PORTFOLIO,
            ToolEffect.READ,
            frozenset({AgentState.ANALYZED}),
        ),
        "build_target_portfolio": (
            ToolCategory.PORTFOLIO,
            ToolEffect.READ,
            frozenset({AgentState.ANALYZED}),
        ),
        "check_portfolio_risk": (
            ToolCategory.RISK,
            ToolEffect.READ,
            frozenset({AgentState.ANALYZED}),
        ),
        "create_order_draft": (
            ToolCategory.ORDER_DRAFT,
            ToolEffect.WRITE,
            frozenset({AgentState.RISK_CHECKED}),
        ),
        "submit_paper_orders": (
            ToolCategory.PAPER_EXECUTION,
            ToolEffect.WRITE,
            frozenset({AgentState.EXECUTING}),
        ),
        "reconcile_account": (
            ToolCategory.RECONCILIATION,
            ToolEffect.READ,
            frozenset({AgentState.RECONCILING}),
        ),
    }
    for name, (category, effect, states) in definitions.items():
        if name not in handlers:
            continue
        modes = frozenset({RuntimeMode.PAPER}) if name == "submit_paper_orders" else TRADING_MODES
        registry.register(
            AgentTool(
                name=name,
                category=category,
                effect=effect,
                argument_model=(
                    RiskApprovedPayloadArguments
                    if name == "create_order_draft"
                    else PayloadArguments
                ),
                handler=(
                    _risk_handler(handlers[name])
                    if name == "check_portfolio_risk"
                    else _draft_handler(handlers[name])
                    if name == "create_order_draft"
                    else _handler(handlers[name])
                ),
                allowed_modes=modes,
                allowed_states=states,
                service=name.replace("_", "-"),
                version=version,
            )
        )
