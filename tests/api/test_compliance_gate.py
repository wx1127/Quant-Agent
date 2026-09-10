import pytest
from apps.execution_gateway.compliance import (
    BrokerExecutionGate,
    ComplianceChecklist,
    ExecutionEnvironment,
    SmallCapitalAcceptance,
)


def test_shadow_and_paper_are_safe_defaults() -> None:
    BrokerExecutionGate(ExecutionEnvironment.SHADOW, ComplianceChecklist()).authorize()
    BrokerExecutionGate(ExecutionEnvironment.PAPER, ComplianceChecklist()).authorize()


def test_live_requires_complete_compliance_checklist() -> None:
    with pytest.raises(PermissionError):
        BrokerExecutionGate(ExecutionEnvironment.LIVE, ComplianceChecklist()).authorize()
    ready = ComplianceChecklist(True, True, True, True, "approval-v1")
    BrokerExecutionGate(ExecutionEnvironment.LIVE, ready).authorize()


def test_small_capital_acceptance_enforces_limits_and_controls() -> None:
    acceptance = SmallCapitalAcceptance(10000, kill_switch_ready=True, rollback_ready=True)
    acceptance.validate(proposed_position=1000, daily_loss=100)
    with pytest.raises(PermissionError):
        acceptance.validate(proposed_position=1001, daily_loss=100)
    with pytest.raises(PermissionError):
        acceptance.validate(proposed_position=1000, daily_loss=201)
