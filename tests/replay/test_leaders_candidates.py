from datetime import date

import pytest

from quant_agent.evaluation.leaders_candidates import (
    CandidateOutcome,
    CandidatePrediction,
    CandidateReplayEvaluator,
)
from quant_agent.regime.models import MarketRegime


def test_candidate_replay_is_segmented_and_cost_aware() -> None:
    predictions = [
        CandidatePrediction(date(2025, 1, 2), "A", 1, "A", MarketRegime.UPTREND, True),
        CandidatePrediction(date(2026, 1, 2), "B", 1, "B", MarketRegime.DIVERGENT, True),
    ]
    outcomes = [
        CandidateOutcome(day, instrument, horizon, value, -0.03, 0.002, executable)
        for day, instrument, value, executable in (
            (date(2025, 1, 2), "A", 0.05, True),
            (date(2026, 1, 2), "B", -0.01, False),
        )
        for horizon in (5, 10, 20)
    ]
    evaluator = CandidateReplayEvaluator(top_k=1)
    first = evaluator.evaluate(snapshot_version="snap", predictions=predictions, outcomes=outcomes)
    second = evaluator.evaluate(
        snapshot_version="snap",
        predictions=list(reversed(predictions)),
        outcomes=list(reversed(outcomes)),
    )
    assert first.content_hash == second.content_hash
    assert len(first.segments) == 6
    assert {item.year for item in first.segments} == {2025, 2026}
    assert any(item.unexecutable_ratio == 1 for item in first.segments)
    assert all(
        item.mean_return_after_cost < item.mean_return_before_cost for item in first.segments
    )


def test_candidate_replay_validates_inputs() -> None:
    with pytest.raises(ValueError):
        CandidateReplayEvaluator(top_k=0)
    evaluator = CandidateReplayEvaluator()
    with pytest.raises(ValueError, match="snapshot"):
        evaluator.evaluate(snapshot_version="", predictions=[], outcomes=[])
    with pytest.raises(ValueError):
        CandidateOutcome(date.today(), "A", 7, 0, 0, 0, True)
    prediction = CandidatePrediction(date.today(), "A", 1, "A", MarketRegime.UPTREND, True)
    with pytest.raises(ValueError, match="unique"):
        evaluator.evaluate(
            snapshot_version="s",
            predictions=[prediction, prediction],
            outcomes=[],
        )
