"""Explicit, bounded Agent state machine."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware, shanghai_now


class AgentState(StrEnum):
    RECEIVED = "RECEIVED"
    SNAPSHOT_READY = "SNAPSHOT_READY"
    ANALYZED = "ANALYZED"
    RISK_CHECKED = "RISK_CHECKED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    EXECUTING = "EXECUTING"
    RECONCILING = "RECONCILING"
    REPORTED = "REPORTED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    DATA_INVALID = "DATA_INVALID"
    EXPIRED = "EXPIRED"
    INCIDENT = "INCIDENT"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {
        AgentState.REPORTED,
        AgentState.COMPLETED,
        AgentState.REJECTED,
        AgentState.DATA_INVALID,
        AgentState.EXPIRED,
        AgentState.INCIDENT,
        AgentState.TIMED_OUT,
        AgentState.CANCELLED,
    }
)

_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.RECEIVED: frozenset({AgentState.SNAPSHOT_READY, AgentState.REJECTED}),
    AgentState.SNAPSHOT_READY: frozenset(
        {AgentState.ANALYZED, AgentState.DATA_INVALID, AgentState.REJECTED}
    ),
    AgentState.ANALYZED: frozenset({AgentState.RISK_CHECKED, AgentState.REJECTED}),
    AgentState.RISK_CHECKED: frozenset(
        {AgentState.REPORTED, AgentState.PENDING_APPROVAL, AgentState.REJECTED}
    ),
    AgentState.PENDING_APPROVAL: frozenset({AgentState.EXECUTING, AgentState.EXPIRED}),
    AgentState.EXECUTING: frozenset({AgentState.RECONCILING, AgentState.INCIDENT}),
    AgentState.RECONCILING: frozenset({AgentState.COMPLETED, AgentState.INCIDENT}),
}


@dataclass(frozen=True, slots=True)
class StateTransition:
    sequence: int
    from_state: AgentState
    to_state: AgentState
    occurred_at: datetime
    reason: str


class AgentRuntime:
    """Runtime guard with deterministic transitions, deadline and call budget."""

    def __init__(
        self,
        *,
        decision_id: str,
        deadline: datetime,
        max_tool_calls: int = 32,
        now: datetime | None = None,
    ) -> None:
        ensure_aware(deadline)
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.decision_id = decision_id
        self.deadline = deadline
        self.max_tool_calls = max_tool_calls
        self._state = AgentState.RECEIVED
        self.tool_calls = 0
        self._trace: list[StateTransition] = []
        if now is not None:
            self.check_deadline(now)

    @property
    def trace(self) -> tuple[StateTransition, ...]:
        return tuple(self._trace)

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(
        self,
        target: AgentState,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> None:
        moment = ensure_aware(now or shanghai_now())
        self.check_deadline(moment)
        if self._state in TERMINAL_STATES:
            raise ValueError(f"terminal state cannot transition: {self._state}")
        if target not in _TRANSITIONS.get(self._state, frozenset()):
            raise ValueError(f"illegal Agent state transition: {self._state} -> {target}")
        previous = self._state
        self._state = target
        self._trace.append(StateTransition(len(self._trace) + 1, previous, target, moment, reason))

    def reserve_tool_call(self, *, now: datetime | None = None) -> int:
        self.check_deadline(now or shanghai_now())
        if self._state in TERMINAL_STATES:
            raise ValueError("terminal task cannot call tools")
        if self.tool_calls >= self.max_tool_calls:
            raise RuntimeError("maximum Agent tool calls exceeded")
        self.tool_calls += 1
        return self.tool_calls

    def check_deadline(self, now: datetime) -> None:
        moment = ensure_aware(now)
        if moment > self.deadline and self._state not in TERMINAL_STATES:
            previous = self._state
            self._state = AgentState.TIMED_OUT
            self._trace.append(
                StateTransition(
                    len(self._trace) + 1,
                    previous,
                    AgentState.TIMED_OUT,
                    moment,
                    "task deadline exceeded",
                )
            )
            raise TimeoutError("Agent task timed out")

    def cancel(self, *, reason: str, now: datetime | None = None) -> None:
        if self._state in TERMINAL_STATES:
            raise ValueError("terminal task cannot be cancelled")
        previous = self._state
        self._state = AgentState.CANCELLED
        self._trace.append(
            StateTransition(
                len(self._trace) + 1,
                previous,
                AgentState.CANCELLED,
                ensure_aware(now or shanghai_now()),
                reason,
            )
        )
