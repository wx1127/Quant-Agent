import pytest
from apps.execution_gateway.compliance import (
    BrokerExecutionGate,
    ComplianceChecklist,
    ExecutionEnvironment,
)


def test_shadow_and_paper_are_safe_defaults() -> None:
    BrokerExecutionGate(ExecutionEnvironment.SHADOW, ComplianceChecklist()).authorize()
    BrokerExecutionGate(ExecutionEnvironment.PAPER, ComplianceChecklist()).authorize()


def test_live_requires_complete_compliance_checklist() -> None:
    with pytest.raises(PermissionError):
        BrokerExecutionGate(ExecutionEnvironment.LIVE, ComplianceChecklist()).authorize()
    ready = ComplianceChecklist(True, True, True, True, "approval-v1")
    BrokerExecutionGate(ExecutionEnvironment.LIVE, ready).authorize()
