from datetime import date, timedelta

import pytest

from quant_agent.evaluation import (
    ForwardIndustryReturn,
    RegimeObservation,
    RegimeThemeReplayEvaluator,
    ThemePrediction,
)
from quant_agent.regime.models import MarketRegime


def test_chronological_replay_metrics_and_hash_are_reproducible() -> None:
    start = date(2026, 1, 1)
    regimes = [
        RegimeObservation(start, MarketRegime.DIVERGENT),
        RegimeObservation(start + timedelta(days=1), MarketRegime.DIVERGENT),
        RegimeObservation(start + timedelta(days=2), MarketRegime.UPTREND),
    ]
    predictions = [
        ThemePrediction(start, "TECH", 1),
        ThemePrediction(start, "BANK", 2),
        ThemePrediction(start, "OTHER", 4),
    ]
    returns = [
        ForwardIndustryReturn(start, industry, horizon, value)
        for horizon in (5, 10, 20)
        for industry, value in (("TECH", 0.03), ("BANK", -0.01), ("OTHER", 0.5))
    ]
    evaluator = RegimeThemeReplayEvaluator(top_k=2)
    first = evaluator.evaluate(
        snapshot_version="snapshot-1",
        regimes=regimes,
        predictions=predictions,
        forward_returns=returns,
    )
    second = evaluator.evaluate(
        snapshot_version="snapshot-1",
        regimes=list(reversed(regimes)),
        predictions=list(reversed(predictions)),
        forward_returns=list(reversed(returns)),
    )
    assert first.content_hash == second.content_hash
    assert first.regime.transition_count == 1
    assert first.regime.longest_run == 2
    assert first.themes.precision_at_k_by_horizon[5] == 0.5
    assert first.themes.mean_relative_return_by_horizon[20] == pytest.approx(0.01)
    assert len(first.themes.effective_examples) == 5
    assert len(first.themes.failure_examples) == 5


def test_replay_rejects_invalid_or_ambiguous_inputs() -> None:
    evaluator = RegimeThemeReplayEvaluator()
    with pytest.raises(ValueError):
        RegimeThemeReplayEvaluator(top_k=0)
    with pytest.raises(ValueError, match="snapshot"):
        evaluator.evaluate(
            snapshot_version=" ",
            regimes=[],
            predictions=[],
            forward_returns=[],
        )
    with pytest.raises(ValueError):
        ForwardIndustryReturn(date.today(), "X", 7, 0.1)
    with pytest.raises(ValueError):
        ThemePrediction(date.today(), "X", 0)
    duplicate = RegimeObservation(date.today(), MarketRegime.UPTREND)
    with pytest.raises(ValueError, match="unique dates"):
        evaluator.evaluate(
            snapshot_version="s",
            regimes=[duplicate, duplicate],
            predictions=[],
            forward_returns=[],
        )
