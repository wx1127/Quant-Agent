from datetime import date

import pytest

from quant_agent.backtest.validation import (
    FrozenParameters,
    ParameterRegistry,
    TimeSeriesSplit,
    WalkForwardSplitter,
    parameter_stability,
)
from quant_agent.backtest.vectorized import (
    ResearchObservation,
    VectorizedResearchEngine,
)


def split() -> TimeSeriesSplit:
    return TimeSeriesSplit(
        date(2020, 1, 1),
        date(2022, 12, 31),
        date(2023, 1, 1),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 12, 31),
    )


def test_vectorized_research_lags_signals_and_matches_manual_quantiles() -> None:
    observations = [
        ResearchObservation(
            date(2025, 1, 1),
            date(2025, 1, 2),
            str(index),
            float(index),
            index / 100,
            0.01,
        )
        for index in range(1, 11)
    ]
    report = VectorizedResearchEngine().analyze(observations, quantile_count=5)
    assert [item.count for item in report.quantiles] == [2, 2, 2, 2, 2]
    assert report.quantiles[0].mean_return == pytest.approx(0.015)
    assert report.top_minus_bottom == pytest.approx(0.08)
    assert report.information_coefficient == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lag"):
        ResearchObservation(date.today(), date.today(), "X", 1, 0, 0)


def test_parameters_are_chronological_frozen_and_stability_is_reported() -> None:
    parameters = FrozenParameters("strategy", "p1", {"window": 20}, date(2024, 1, 1), split())
    registry = ParameterRegistry()
    registry.freeze(parameters)
    registry.freeze(parameters)
    assert registry.get("strategy", "p1").content_hash == parameters.content_hash
    with pytest.raises(TypeError):
        parameters.values["window"] = 60  # type: ignore[index]
    changed = FrozenParameters("strategy", "p1", {"window": 60}, date(2024, 1, 1), split())
    with pytest.raises(ValueError, match="cannot be changed"):
        registry.freeze(changed)
    assert parameter_stability(1.0, [0.9, 1.1]) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        parameter_stability(1.0, [])
    with pytest.raises(ValueError, match="ordered"):
        TimeSeriesSplit(
            date(2020, 1, 1),
            date(2022, 1, 1),
            date(2021, 1, 1),
            date(2023, 1, 1),
            date(2024, 1, 1),
            date(2025, 1, 1),
        )
    dates = [date(2020, 1, day) for day in range(1, 13)]
    windows = WalkForwardSplitter().generate(
        dates,
        train_size=4,
        validation_size=2,
        out_of_sample_size=2,
    )
    assert len(windows) == 3
    assert windows[1].train_start > windows[0].train_start
