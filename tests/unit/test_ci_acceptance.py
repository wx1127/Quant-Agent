from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_ci_workflow_has_branch_pull_request_manual_fault_and_reports() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "\n  push:\n" in workflow
    assert "\n  pull_request:\n" in workflow
    assert "\n  workflow_dispatch:\n" in workflow
    assert "fault_injection:" in workflow
    assert "python -m pip install -r requirements-dev.lock" in workflow
    assert "python -m mypy packages apps" in workflow
    assert "python scripts/verify_ci_gates.py" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "QUANT_AGENT_RUNTIME_MODE: RESEARCH" in workflow
