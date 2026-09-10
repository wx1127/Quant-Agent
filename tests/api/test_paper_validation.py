from apps.worker.paper_validation import PaperState, PaperValidation


def test_paper_cycle_requires_approval_and_reconciles() -> None:
    validation = PaperValidation()
    cycle = validation.create("draft-1")
    try:
        validation.simulate_fill(cycle.cycle_id, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("paper fill must require approval")
    validation.approve(cycle.cycle_id)
    validation.simulate_fill(cycle.cycle_id, 2)
    result = validation.reconcile(cycle.cycle_id, 2)
    assert result.state is PaperState.RECONCILED
    assert result.reconciliation_ok is True
