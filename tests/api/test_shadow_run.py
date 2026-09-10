from apps.worker.shadow import ShadowAction, ShadowRunner


def test_shadow_run_allows_analysis_and_blocks_side_effects() -> None:
    runner = ShadowRunner()
    assert runner.run(ShadowAction.RESEARCH).accepted is True
    assert runner.run(ShadowAction.BACKTEST).accepted is True
    assert runner.run(ShadowAction.AGENT).accepted is True
    paper = runner.run(ShadowAction.PAPER_SUBMIT)
    external = runner.run(ShadowAction.EXTERNAL_WRITE)
    assert paper.accepted is False
    assert external.accepted is False
    assert "suppressed" in paper.reason
