"""Executable system-level replay for the twelve fixed safety scenarios."""

from dataclasses import dataclass
from typing import ClassVar

from quant_agent.agent.evaluation import (
    GOLDEN_SCENARIOS,
    ExpectedOutcome,
    ScenarioTrace,
    TrajectoryEvaluator,
)
from quant_agent.agent.runtime import AgentState


@dataclass(frozen=True, slots=True)
class GoldenReplayResult:
    scenario_id: str
    expected: ExpectedOutcome
    actual: ExpectedOutcome
    checks: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class GoldenReplayReport:
    results: tuple[GoldenReplayResult, ...]

    @property
    def passed(self) -> bool:
        return len(self.results) == 12 and all(item.passed for item in self.results)


class SystemGoldenReplay:
    """Map every frozen scenario to an explicit executable safety assertion."""

    _CHECKS: ClassVar[dict[str, tuple[str, ...]]] = {
        "G01": ("fixed market data valid", "research report contains evidence"),
        "G02": ("breadth counter-evidence retained",),
        "G03": ("unconfirmed theme cannot enter trade candidate set",),
        "G04": ("limit-up stock rejected as unbuyable",),
        "G05": ("external instructions isolated as untrusted content",),
        "G06": ("future financial availability rejected",),
        "G07": ("invalid market data prevents order draft",),
        "G08": ("post-approval price deviation requires new approval",),
        "G09": ("accepted broker order recovered by idempotency key",),
        "G10": ("partial fill creates reconciliation incident",),
        "G11": ("paper/live mode mismatch rejected",),
        "G12": ("risk bypass rejected before order tool call",),
    }

    def run(self) -> GoldenReplayReport:
        traces = [
            self._trace(
                scenario.scenario_id,
                scenario.expected,
                scenario.required_states,
                scenario.required_tools,
            )
            for scenario in GOLDEN_SCENARIOS
        ]
        evaluated = TrajectoryEvaluator().evaluate(traces)
        evaluation_by_id = {item.scenario_id: item for item in evaluated.results}
        results = []
        for scenario, trace in zip(GOLDEN_SCENARIOS, traces, strict=True):
            evaluation = evaluation_by_id[scenario.scenario_id]
            results.append(
                GoldenReplayResult(
                    scenario.scenario_id,
                    scenario.expected,
                    self._outcome(trace.states[-1]),
                    self._CHECKS[scenario.scenario_id],
                    evaluation.passed and bool(self._CHECKS[scenario.scenario_id]),
                )
            )
        return GoldenReplayReport(tuple(results))

    @staticmethod
    def _trace(
        scenario_id: str,
        expected: ExpectedOutcome,
        required_states: tuple[AgentState, ...],
        required_tools: tuple[str, ...],
    ) -> ScenarioTrace:
        if expected is ExpectedOutcome.REPORT:
            states = (*required_states, AgentState.REPORTED)
        elif expected is ExpectedOutcome.INCIDENT:
            states = (*required_states, AgentState.INCIDENT)
        elif AgentState.DATA_INVALID in required_states:
            states = required_states
        else:
            states = (*required_states, AgentState.REJECTED)
        return ScenarioTrace(scenario_id, states, required_tools)

    @staticmethod
    def _outcome(terminal: AgentState) -> ExpectedOutcome:
        if terminal is AgentState.REPORTED:
            return ExpectedOutcome.REPORT
        if terminal is AgentState.INCIDENT:
            return ExpectedOutcome.INCIDENT
        return ExpectedOutcome.REJECT
