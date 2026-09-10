"""Fail-closed workflow and stage-tool rules for the explicit Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from quant_agent.agent.runtime.contracts import (
    TERMINAL_AGENT_RUN_STATES,
    AgentRunEventKind,
    AgentRunGoal,
    AgentRunState,
)
from quant_agent.config import RuntimeMode


class AgentRuntimeRuleError(ValueError):
    """Base error for a rejected runtime rule decision."""


class AgentGoalModeMismatch(AgentRuntimeRuleError):
    """Raised when a server-selected goal is incompatible with the fixed mode."""


class InvalidAgentStateTransition(AgentRuntimeRuleError):
    """Raised when an event attempts an unsupported workflow transition."""


TERMINAL_STATES: Final[frozenset[AgentRunState]] = TERMINAL_AGENT_RUN_STATES
EXECUTION_GOALS: Final[frozenset[AgentRunGoal]] = frozenset(
    {
        AgentRunGoal.PAPER_EXECUTION,
        AgentRunGoal.LIVE_ASSISTED_EXECUTION,
    }
)

GOAL_MODE_COMPATIBILITY: Final[Mapping[AgentRunGoal, frozenset[RuntimeMode]]] = MappingProxyType(
    {
        AgentRunGoal.MARKET_RESEARCH: frozenset(
            {
                RuntimeMode.RESEARCH,
                RuntimeMode.BACKTEST,
                RuntimeMode.PAPER,
                RuntimeMode.LIVE_ASSISTED,
            }
        ),
        AgentRunGoal.BACKTEST_REPORT: frozenset({RuntimeMode.BACKTEST}),
        AgentRunGoal.PORTFOLIO_REPORT: frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}),
        AgentRunGoal.ORDER_DRAFT: frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}),
        AgentRunGoal.PAPER_EXECUTION: frozenset({RuntimeMode.PAPER}),
        AgentRunGoal.LIVE_ASSISTED_EXECUTION: frozenset({RuntimeMode.LIVE_ASSISTED}),
    }
)

_DATA_TOOLS: Final[frozenset[str]] = frozenset({"get_market_snapshot", "validate_market_data"})
_RESEARCH_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "detect_market_regime",
        "rank_market_themes",
        "rank_theme_leaders",
        "rank_stock_candidates",
        "explain_candidate",
    }
)
_PORTFOLIO_TOOLS: Final[frozenset[str]] = frozenset(
    {"get_portfolio_snapshot", "build_target_portfolio"}
)
_REPORT_TOOL: Final[frozenset[str]] = frozenset({"generate_decision_report"})
_DRAFT_READ_TOOL: Final[frozenset[str]] = frozenset({"get_order_draft"})

_REPORT_READY_BY_GOAL: Final[Mapping[AgentRunGoal, AgentRunState]] = MappingProxyType(
    {
        AgentRunGoal.MARKET_RESEARCH: AgentRunState.ANALYZED,
        AgentRunGoal.BACKTEST_REPORT: AgentRunState.ANALYZED,
        AgentRunGoal.PORTFOLIO_REPORT: AgentRunState.RISK_CHECKED,
        AgentRunGoal.ORDER_DRAFT: AgentRunState.DRAFT_READY,
    }
)
REPORT_READY_STATES: Final[frozenset[AgentRunState]] = frozenset(_REPORT_READY_BY_GOAL.values())


def _goal_path(goal: AgentRunGoal) -> tuple[AgentRunState, ...]:
    prefix = (
        AgentRunState.RECEIVED,
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.DATA_VALIDATED,
        AgentRunState.ANALYZED,
    )
    if goal in {AgentRunGoal.MARKET_RESEARCH, AgentRunGoal.BACKTEST_REPORT}:
        return (*prefix, AgentRunState.REPORTED)
    portfolio = (*prefix, AgentRunState.PORTFOLIO_READY, AgentRunState.RISK_CHECKED)
    if goal is AgentRunGoal.PORTFOLIO_REPORT:
        return (*portfolio, AgentRunState.REPORTED)
    draft = (*portfolio, AgentRunState.DRAFT_READY)
    if goal is AgentRunGoal.ORDER_DRAFT:
        return (*draft, AgentRunState.REPORTED)
    if goal is AgentRunGoal.PAPER_EXECUTION:
        return (
            *draft,
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
            AgentRunState.COMPLETED,
        )
    if goal is AgentRunGoal.LIVE_ASSISTED_EXECUTION:
        return (
            *draft,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
            AgentRunState.COMPLETED,
        )
    raise AgentRuntimeRuleError("unknown Agent run goal")


_PRE_EXECUTION_STATES: Final[frozenset[AgentRunState]] = frozenset(
    {
        AgentRunState.RECEIVED,
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.DATA_VALIDATED,
        AgentRunState.ANALYZED,
        AgentRunState.PORTFOLIO_READY,
        AgentRunState.RISK_CHECKED,
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
    }
)
_REJECTABLE_STATES: Final[frozenset[AgentRunState]] = frozenset(
    {
        AgentRunState.RECEIVED,
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.DATA_VALIDATED,
        AgentRunState.ANALYZED,
        AgentRunState.PORTFOLIO_READY,
        AgentRunState.RISK_CHECKED,
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
    }
)
_DATA_FAILURE_STATES: Final[frozenset[AgentRunState]] = frozenset(
    {
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.DATA_VALIDATED,
        AgentRunState.ANALYZED,
        AgentRunState.PORTFOLIO_READY,
        AgentRunState.RISK_CHECKED,
        AgentRunState.DRAFT_READY,
    }
)
_EXPIRABLE_STATES: Final[frozenset[AgentRunState]] = frozenset(
    {
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
    }
)

_SAME_STATE_OBSERVATIONS: Final[frozenset[AgentRunEventKind]] = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_RESERVED,
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
        AgentRunEventKind.CANCELLATION_REQUESTED,
    }
)


def validate_goal_mode(goal: AgentRunGoal, mode: RuntimeMode) -> None:
    """Reject non-enum inputs and every goal/mode pair outside the fixed matrix."""

    if type(goal) is not AgentRunGoal:
        raise AgentGoalModeMismatch("run goal must be an exact AgentRunGoal")
    if type(mode) is not RuntimeMode or mode is RuntimeMode.LIVE_AUTO:
        raise AgentGoalModeMismatch("runtime mode is unsupported for Agent execution")
    if mode not in GOAL_MODE_COMPATIBILITY[goal]:
        raise AgentGoalModeMismatch(
            f"goal {goal.value} is not permitted in runtime mode {mode.value}"
        )


def report_ready(goal: AgentRunGoal, state: AgentRunState) -> bool:
    """Return whether this non-execution goal reached its reportable milestone."""

    if type(goal) is not AgentRunGoal or type(state) is not AgentRunState:
        return False
    expected = _REPORT_READY_BY_GOAL.get(goal)
    return expected is state


def legal_transition_targets(
    state: AgentRunState,
    *,
    goal: AgentRunGoal,
    mode: RuntimeMode,
) -> frozenset[AgentRunState]:
    """Return exact next states for the fixed goal, excluding observations."""

    validate_goal_mode(goal, mode)
    if type(state) is not AgentRunState:
        raise InvalidAgentStateTransition("state must be an exact AgentRunState")
    if state in TERMINAL_STATES:
        return frozenset()

    path = _goal_path(goal)
    targets: set[AgentRunState] = set()
    try:
        path_index = path.index(state)
    except ValueError:
        path_index = -1
    if path_index >= 0 and path_index + 1 < len(path):
        targets.add(path[path_index + 1])

    if state in _PRE_EXECUTION_STATES:
        targets.update(
            {
                AgentRunState.CANCELLED,
                AgentRunState.TIMED_OUT,
                AgentRunState.LIMIT_EXCEEDED,
                AgentRunState.FAILED,
                AgentRunState.INCIDENT,
            }
        )
    if state in _REJECTABLE_STATES:
        targets.add(AgentRunState.REJECTED)
    if state in _DATA_FAILURE_STATES:
        targets.add(AgentRunState.DATA_INVALID)
    if state in _EXPIRABLE_STATES:
        targets.add(AgentRunState.EXPIRED)

    # EXECUTING is the submission-ready stage.  A trusted service may still
    # cancel or expire it before a write reservation exists; replay performs
    # that pending-call distinction.  RECONCILING means a receipt exists and
    # can only converge to a confirmed result or an incident.
    if state is AgentRunState.EXECUTING:
        return frozenset(
            {
                AgentRunState.RECONCILING,
                AgentRunState.CANCELLED,
                AgentRunState.TIMED_OUT,
                AgentRunState.EXPIRED,
                AgentRunState.INCIDENT,
            }
        )
    if state is AgentRunState.RECONCILING:
        return frozenset({AgentRunState.COMPLETED, AgentRunState.INCIDENT})
    return frozenset(targets)


LEGAL_STATE_EDGES: Final[Mapping[AgentRunState, frozenset[AgentRunState]]] = MappingProxyType(
    {
        state: frozenset(
            target
            for goal, modes in GOAL_MODE_COMPATIBILITY.items()
            for mode in modes
            for target in legal_transition_targets(state, goal=goal, mode=mode)
        )
        for state in AgentRunState
    }
)


def _validate_changed_event_kind(
    before: AgentRunState,
    after: AgentRunState,
    event_kind: AgentRunEventKind,
) -> None:
    if event_kind in {
        AgentRunEventKind.RUN_CREATED,
        AgentRunEventKind.TOOL_CALL_RESERVED,
    }:
        raise InvalidAgentStateTransition(f"{event_kind.value} cannot change the Agent run state")
    if after is AgentRunState.APPROVED and (
        before is not AgentRunState.PENDING_APPROVAL
        or event_kind is not AgentRunEventKind.APPROVAL_RECORDED
    ):
        raise InvalidAgentStateTransition("only trusted approval evidence may enter APPROVED")
    if event_kind is AgentRunEventKind.APPROVAL_RECORDED and (
        before is not AgentRunState.PENDING_APPROVAL or after is not AgentRunState.APPROVED
    ):
        raise InvalidAgentStateTransition(
            "approval evidence must move PENDING_APPROVAL to APPROVED"
        )
    if event_kind is AgentRunEventKind.APPROVAL_REJECTED and (
        before is not AgentRunState.PENDING_APPROVAL or after is not AgentRunState.REJECTED
    ):
        raise InvalidAgentStateTransition(
            "approval rejection must move PENDING_APPROVAL to REJECTED"
        )
    if event_kind is AgentRunEventKind.CANCELLATION_REQUESTED and after not in {
        AgentRunState.CANCELLED,
        AgentRunState.INCIDENT,
    }:
        raise InvalidAgentStateTransition(
            "a changing cancellation event must enter CANCELLED or INCIDENT"
        )
    if event_kind is AgentRunEventKind.TIMEOUT_RECORDED and after not in {
        AgentRunState.TIMED_OUT,
        AgentRunState.EXPIRED,
        AgentRunState.INCIDENT,
    }:
        raise InvalidAgentStateTransition(
            "a timeout event must enter TIMED_OUT, EXPIRED, or INCIDENT"
        )
    if event_kind is AgentRunEventKind.LIMIT_REACHED and after not in {
        AgentRunState.LIMIT_EXCEEDED,
        AgentRunState.INCIDENT,
    }:
        raise InvalidAgentStateTransition("a limit event must enter LIMIT_EXCEEDED or INCIDENT")
    if event_kind is AgentRunEventKind.INCIDENT_RECORDED and (after is not AgentRunState.INCIDENT):
        raise InvalidAgentStateTransition("an incident event must enter INCIDENT")

    target_event_kinds = {
        AgentRunState.CANCELLED: frozenset({AgentRunEventKind.CANCELLATION_REQUESTED}),
        AgentRunState.TIMED_OUT: frozenset({AgentRunEventKind.TIMEOUT_RECORDED}),
        AgentRunState.EXPIRED: frozenset({AgentRunEventKind.TIMEOUT_RECORDED}),
        AgentRunState.LIMIT_EXCEEDED: frozenset({AgentRunEventKind.LIMIT_REACHED}),
    }.get(after)
    if target_event_kinds is not None and event_kind not in target_event_kinds:
        raise InvalidAgentStateTransition(f"{event_kind.value} cannot enter {after.value}")


def validate_transition(
    before: AgentRunState,
    after: AgentRunState,
    *,
    goal: AgentRunGoal,
    mode: RuntimeMode,
    event_kind: AgentRunEventKind,
) -> None:
    """Validate one event transition, including same-state observation rules."""

    validate_goal_mode(goal, mode)
    if type(before) is not AgentRunState or type(after) is not AgentRunState:
        raise InvalidAgentStateTransition("states must be exact AgentRunState values")
    if type(event_kind) is not AgentRunEventKind:
        raise InvalidAgentStateTransition("event_kind must be an exact AgentRunEventKind")
    if before in TERMINAL_STATES:
        raise InvalidAgentStateTransition("terminal Agent run states are absorbing")
    if before is after:
        if event_kind is AgentRunEventKind.RUN_CREATED and before is AgentRunState.RECEIVED:
            return
        if event_kind is AgentRunEventKind.CANCELLATION_REQUESTED:
            if before in {AgentRunState.EXECUTING, AgentRunState.RECONCILING}:
                return
            raise InvalidAgentStateTransition(
                "cancellation may remain observational only after the execution boundary"
            )
        if event_kind not in _SAME_STATE_OBSERVATIONS:
            raise InvalidAgentStateTransition(
                f"{event_kind.value} is not a permitted same-state observation"
            )
        return

    targets = legal_transition_targets(before, goal=goal, mode=mode)
    if after not in targets:
        raise InvalidAgentStateTransition(
            f"illegal {goal.value}/{mode.value} transition: {before.value} -> {after.value}"
        )
    _validate_changed_event_kind(before, after, event_kind)


def stage_tool_names(
    state: AgentRunState,
    *,
    goal: AgentRunGoal,
    mode: RuntimeMode,
) -> frozenset[str]:
    """Return the maximum Agent tool set for one workflow stage.

    The runtime must still intersect this set with ``AgentToolRegistry.catalog``
    and enforce artifact prerequisites before invocation.
    """

    validate_goal_mode(goal, mode)
    if type(state) is not AgentRunState:
        raise InvalidAgentStateTransition("state must be an exact AgentRunState")
    if state in TERMINAL_STATES or state is AgentRunState.RECEIVED:
        return frozenset()
    if state is AgentRunState.SNAPSHOT_READY:
        return _DATA_TOOLS
    if state is AgentRunState.DATA_VALIDATED:
        if goal is AgentRunGoal.BACKTEST_REPORT:
            return frozenset({"run_backtest"})
        return _RESEARCH_TOOLS
    if state is AgentRunState.ANALYZED:
        explanation = frozenset({"explain_candidate"})
        if goal in {AgentRunGoal.MARKET_RESEARCH, AgentRunGoal.BACKTEST_REPORT}:
            return _REPORT_TOOL | explanation
        return _PORTFOLIO_TOOLS | explanation
    if state is AgentRunState.PORTFOLIO_READY:
        return frozenset({"check_portfolio_risk"})
    if state is AgentRunState.RISK_CHECKED:
        if goal is AgentRunGoal.PORTFOLIO_REPORT:
            return _REPORT_TOOL
        return frozenset({"create_order_draft"})
    if state is AgentRunState.DRAFT_READY:
        tools = _DRAFT_READ_TOOL
        if goal is AgentRunGoal.ORDER_DRAFT:
            tools = tools | _REPORT_TOOL
        return tools
    if state in {AgentRunState.PENDING_APPROVAL, AgentRunState.APPROVED}:
        return _DRAFT_READ_TOOL
    if state is AgentRunState.EXECUTING:
        if goal is AgentRunGoal.PAPER_EXECUTION and mode is RuntimeMode.PAPER:
            return frozenset({"submit_paper_orders"})
        if goal is AgentRunGoal.LIVE_ASSISTED_EXECUTION and mode is RuntimeMode.LIVE_ASSISTED:
            return frozenset({"submit_approved_orders"})
        return frozenset()
    if state is AgentRunState.RECONCILING:
        return frozenset({"reconcile_account"})
    return frozenset()


__all__ = [
    "EXECUTION_GOALS",
    "GOAL_MODE_COMPATIBILITY",
    "LEGAL_STATE_EDGES",
    "REPORT_READY_STATES",
    "TERMINAL_STATES",
    "AgentGoalModeMismatch",
    "AgentRuntimeRuleError",
    "InvalidAgentStateTransition",
    "legal_transition_targets",
    "report_ready",
    "stage_tool_names",
    "validate_goal_mode",
    "validate_transition",
]
