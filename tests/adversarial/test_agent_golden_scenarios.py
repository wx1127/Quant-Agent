from quant_agent.agent.evaluation import (
    GOLDEN_SCENARIOS,
    ScenarioTrace,
    TrajectoryEvaluator,
)
from quant_agent.agent.runtime import AgentState


def compliant_traces() -> list[ScenarioTrace]:
    traces = []
    for scenario in GOLDEN_SCENARIOS:
        if scenario.expected.value == "REPORT":
            states = (
                AgentState.RECEIVED,
                AgentState.SNAPSHOT_READY,
                AgentState.ANALYZED,
                AgentState.RISK_CHECKED,
                AgentState.REPORTED,
            )
        elif scenario.expected.value == "INCIDENT":
            states = (
                AgentState.RECEIVED,
                AgentState.SNAPSHOT_READY,
                AgentState.ANALYZED,
                AgentState.RISK_CHECKED,
                AgentState.PENDING_APPROVAL,
                AgentState.EXECUTING,
                AgentState.RECONCILING,
                AgentState.INCIDENT,
            )
        elif AgentState.DATA_INVALID in scenario.required_states:
            states = (
                AgentState.RECEIVED,
                AgentState.SNAPSHOT_READY,
                AgentState.DATA_INVALID,
            )
        else:
            states = (*scenario.required_states, AgentState.REJECTED)
        traces.append(
            ScenarioTrace(
                scenario.scenario_id,
                states,
                scenario.required_tools,
            )
        )
    return traces


def test_all_twelve_harness_golden_scenarios_are_repeatable() -> None:
    assert len(GOLDEN_SCENARIOS) == 12
    evaluator = TrajectoryEvaluator()
    first = evaluator.evaluate(compliant_traces())
    second = evaluator.evaluate(compliant_traces())
    assert first == second
    assert first.passed
    assert first.evidence_coverage == 1.0


def test_evaluator_detects_risk_bypass_future_data_and_unsupported_numbers() -> None:
    traces = compliant_traces()
    traces[-1] = ScenarioTrace(
        "G12",
        traces[-1].states,
        ("create_order_draft",),
        unsupported_numeric_claims=1,
        future_data_uses=1,
    )
    report = TrajectoryEvaluator().evaluate(traces)
    result = report.results[-1]
    assert not report.passed
    assert "forbidden tool was called" in result.failures
    assert "unsupported numeric claims found" in result.failures
    assert "future data use found" in result.failures
