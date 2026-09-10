from quant_agent.evaluation import (
    AgentTrajectory,
    ToolCall,
    TrajectoryIssue,
    evaluate_agent_trajectory,
)


def test_agent_trajectory_evaluation_passes_bound_calls() -> None:
    result = evaluate_agent_trajectory(
        AgentTrajectory(
            "t1",
            (ToolCall("market", ("e1",)), ToolCall("research", ("e2",))),
            frozenset({"market", "research"}),
            ("market", "research"),
        )
    )
    assert result.passed is True


def test_agent_trajectory_evaluation_catches_bypass_and_missing_evidence() -> None:
    result = evaluate_agent_trajectory(
        AgentTrajectory(
            "t2",
            (ToolCall("execute", requires_approval=True),),
            frozenset({"execute"}),
        )
    )
    assert result.passed is False
    assert TrajectoryIssue.APPROVAL_BYPASS in result.issues
    assert TrajectoryIssue.MISSING_EVIDENCE in result.issues
