from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_agent.features.market_breadth import MarketBreadthResult
from quant_agent.features.market_trend import MarketTrendResult
from quant_agent.regime.classifier import MarketRegimeClassifier, RegimeConfig
from quant_agent.regime.models import MarketRegime
from quant_agent.regime.transitions import RegimeTransitionEngine, TransitionConfig

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def trend(score: float, when: datetime = NOW) -> MarketTrendResult:
    return MarketTrendResult(when, "snap", "trend-v1", score, (), ())


def breadth(score: float, when: datetime = NOW, low: float = 0.1) -> MarketBreadthResult:
    return MarketBreadthResult(
        when,
        "snap",
        "breadth-v1",
        10,
        6,
        3,
        1,
        0.6,
        0.6,
        0.6,
        0.1,
        low,
        0.1,
        low,
        1000,
        score / 100,
        0.01,
        0.1,
        score,
        score,
        score,
        score,
        score,
        (),
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (80, MarketRegime.UPTREND),
        (60, MarketRegime.RANGE_STRONG),
        (45, MarketRegime.DIVERGENT),
        (30, MarketRegime.DOWNTREND),
    ],
)
def test_regime_thresholds(score: float, expected: MarketRegime) -> None:
    result = MarketRegimeClassifier().classify(trend(score), breadth(score))
    assert result.regime is expected
    assert result.score == pytest.approx(score)
    assert 0 <= result.confidence <= 1
    assert result.invalidations


def test_bottom_recovery_and_contract_validation() -> None:
    classifier = MarketRegimeClassifier()
    old = breadth(30, NOW - timedelta(days=2), 0.3)
    mid = breadth(40, NOW - timedelta(days=1), 0.2)
    current = replace(breadth(50, NOW, 0.1), median_daily_return=0.01)
    result = classifier.classify(trend(50), current, breadth_history=[old, mid])
    assert result.regime is MarketRegime.BOTTOM_RECOVERY
    assert result.max_risk_budget == 0.3
    with pytest.raises(ValueError, match="as_of"):
        classifier.classify(trend(50), replace(current, as_of=NOW + timedelta(days=1)))
    with pytest.raises(ValueError, match="data version"):
        classifier.classify(trend(50), replace(current, data_version="other"))
    with pytest.raises(ValueError):
        RegimeConfig(trend_weight=0.5)


def test_transition_confirmation_urgent_path_and_ordering() -> None:
    classifier = MarketRegimeClassifier()
    engine = RegimeTransitionEngine(TransitionConfig(confirmation_days=2))
    first = classifier.classify(trend(60), breadth(60))
    record = engine.apply(first)
    assert record.changed and record.final_regime is MarketRegime.RANGE_STRONG
    second = classifier.classify(
        trend(80, NOW + timedelta(days=1)),
        breadth(80, NOW + timedelta(days=1)),
    )
    pending = engine.apply(second)
    assert not pending.changed and pending.candidate_count == 1
    third = classifier.classify(
        trend(80, NOW + timedelta(days=2)),
        breadth(80, NOW + timedelta(days=2)),
    )
    confirmed = engine.apply(third)
    assert confirmed.changed and confirmed.final_regime is MarketRegime.UPTREND
    urgent = classifier.classify(
        trend(10, NOW + timedelta(days=3)),
        breadth(10, NOW + timedelta(days=3)),
    )
    assert engine.apply(urgent).reason == "urgent downside threshold"
    with pytest.raises(ValueError, match="increasing"):
        engine.apply(urgent)
