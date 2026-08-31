"""Tests for point-in-time multi-index market trend features."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.features.core import FeatureObservation
from quant_agent.features.market_trend import (
    InsufficientMarketData,
    MarketTrendAnalyzer,
    MarketTrendConfig,
    MissingIndexPolicy,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
START = datetime(2026, 1, 2, 15, 0, tzinfo=SHANGHAI)


def _series(
    index_id: str,
    count: int,
    price: Callable[[int], Decimal],
) -> list[FeatureObservation]:
    del index_id
    return [
        FeatureObservation(
            observation_key=f"day-{day:03d}",
            observed_at=START + timedelta(days=day),
            available_at=START + timedelta(days=day, minutes=30),
            value=price(day),
            revision="v1",
        )
        for day in range(count)
    ]


def _small_config(
    *,
    missing_index_policy: MissingIndexPolicy = MissingIndexPolicy.FAIL,
) -> MarketTrendConfig:
    return MarketTrendConfig(
        version="test-v1",
        windows=(3, 5, 8),
        window_weights=(Decimal("0.2"), Decimal("0.3"), Decimal("0.5")),
        slope_lookback=2,
        missing_index_policy=missing_index_policy,
    )


def test_rising_flat_and_falling_indices_aggregate_transparently() -> None:
    observations = {
        "RISING": _series("RISING", 12, lambda day: Decimal(100 + day)),
        "FLAT": _series("FLAT", 12, lambda _: Decimal(100)),
        "FALLING": _series("FALLING", 12, lambda day: Decimal(120 - day)),
    }
    snapshot = MarketTrendAnalyzer(_small_config()).analyze(
        session_date=(START + timedelta(days=11)).date(),
        required_indices=("RISING", "FLAT", "FALLING"),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )

    scores = {item.index_id: item.score for item in snapshot.indices}
    assert scores["RISING"] > 0
    assert scores["FLAT"] == 0
    assert scores["FALLING"] < 0
    assert snapshot.aggregate_score == sum(scores.values(), Decimal(0)) / 3
    assert [item.window for item in snapshot.indices[0].windows] == [3, 5, 8]


def test_metrics_match_manual_window_calculation() -> None:
    series = _series("IDX", 10, lambda day: Decimal(day + 1))
    snapshot = MarketTrendAnalyzer(_small_config()).analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations={"IDX": series},
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )

    metrics = snapshot.indices[0].windows[0]
    assert metrics.latest_close == Decimal(10)
    assert metrics.moving_average == Decimal(9)
    assert metrics.period_return == Decimal(10) / Decimal(8) - 1
    assert metrics.position_vs_average == Decimal(10) / Decimal(9) - 1
    assert metrics.average_slope == Decimal(9) / Decimal(7) - 1


def test_future_price_does_not_change_historical_snapshot() -> None:
    history = _series("IDX", 10, lambda day: Decimal(100 + day))
    as_of = START + timedelta(days=10)
    analyzer = MarketTrendAnalyzer(_small_config())
    original = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations={"IDX": history},
        as_of=as_of,
        data_version="prices-v1",
    )
    future = FeatureObservation(
        observation_key="future",
        observed_at=START + timedelta(days=11),
        available_at=START + timedelta(days=11, minutes=30),
        value=Decimal("999999"),
        revision="v1",
    )
    repeated = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations={"IDX": [*history, future]},
        as_of=as_of,
        data_version="prices-v1",
    )

    assert repeated.aggregate_score == original.aggregate_score
    assert repeated.cache_key == original.cache_key


def test_missing_required_index_fails_by_default() -> None:
    with pytest.raises(InsufficientMarketData, match="MISSING"):
        MarketTrendAnalyzer(_small_config()).analyze(
            session_date=(START + timedelta(days=9)).date(),
            required_indices=("IDX", "MISSING"),
            observations={
                "IDX": _series("IDX", 10, lambda day: Decimal(100 + day)),
            },
            as_of=START + timedelta(days=20),
            data_version="prices-v1",
        )


def test_degrade_policy_preserves_missing_index_evidence() -> None:
    snapshot = MarketTrendAnalyzer(
        _small_config(missing_index_policy=MissingIndexPolicy.DEGRADE)
    ).analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX", "SHORT", "ABSENT"),
        observations={
            "IDX": _series("IDX", 10, lambda day: Decimal(100 + day)),
            "SHORT": _series("SHORT", 3, lambda day: Decimal(100 + day)),
        },
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )

    assert [item.index_id for item in snapshot.indices] == ["IDX"]
    assert snapshot.missing_indices == ("ABSENT", "SHORT")


def test_no_sufficient_index_still_fails_in_degrade_mode() -> None:
    with pytest.raises(InsufficientMarketData, match="no required index"):
        MarketTrendAnalyzer(_small_config(missing_index_policy=MissingIndexPolicy.DEGRADE)).analyze(
            session_date=(START + timedelta(days=1)).date(),
            required_indices=("SHORT",),
            observations={"SHORT": _series("SHORT", 2, lambda _: Decimal(100))},
            as_of=START + timedelta(days=20),
            data_version="prices-v1",
        )


def test_snapshot_is_reproducible_and_bound_to_data_version() -> None:
    analyzer = MarketTrendAnalyzer(_small_config())
    observations = {"IDX": _series("IDX", 10, lambda day: Decimal(100 + day))}
    first = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )
    repeated = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )
    changed = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("IDX",),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v2",
    )

    assert first == repeated
    assert first.cache_key != changed.cache_key
    assert first.identity_payload()["feature_version"] == "test-v1"


def test_stale_or_cross_index_session_is_not_ranked() -> None:
    analyzer = MarketTrendAnalyzer(_small_config())
    history = _series("IDX", 10, lambda day: Decimal(100 + day))

    with pytest.raises(InsufficientMarketData, match="IDX"):
        analyzer.analyze(
            session_date=(START + timedelta(days=10)).date(),
            required_indices=("IDX",),
            observations={"IDX": history},
            as_of=START + timedelta(days=20),
            data_version="prices-v1",
        )
    with pytest.raises(InsufficientMarketData, match="STALE"):
        analyzer.analyze(
            session_date=(START + timedelta(days=9)).date(),
            required_indices=("CURRENT", "STALE"),
            observations={
                "CURRENT": history,
                "STALE": history[:-1],
            },
            as_of=START + timedelta(days=20),
            data_version="prices-v1",
        )


def test_required_index_set_order_does_not_change_identity() -> None:
    observations = {
        "A": _series("A", 10, lambda day: Decimal(100 + day)),
        "B": _series("B", 10, lambda day: Decimal(200 + day)),
    }
    analyzer = MarketTrendAnalyzer(_small_config())
    first = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("A", "B"),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )
    reordered = analyzer.analyze(
        session_date=(START + timedelta(days=9)).date(),
        required_indices=("B", "A"),
        observations=observations,
        as_of=START + timedelta(days=20),
        data_version="prices-v1",
    )

    assert reordered == first
    assert replace(_small_config(), score_scale=Decimal("500.0")).config_hash == (
        _small_config().config_hash
    )


def test_cross_index_internal_session_gap_fails_alignment() -> None:
    complete = _series("A", 11, lambda day: Decimal(100 + day))
    gapped = [
        item
        for item in _series("B", 11, lambda day: Decimal(200 + day))
        if item.observed_at.date() != (START + timedelta(days=5)).date()
    ]

    with pytest.raises(InsufficientMarketData, match="session-aligned"):
        MarketTrendAnalyzer(_small_config()).analyze(
            session_date=(START + timedelta(days=10)).date(),
            required_indices=("A", "B"),
            observations={"A": complete, "B": gapped},
            as_of=START + timedelta(days=20),
            data_version="prices-v1",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MarketTrendConfig(windows=()),
        lambda: MarketTrendConfig(windows=(20, 20, 120)),
        lambda: MarketTrendConfig(window_weights=(Decimal("1"),)),
        lambda: MarketTrendConfig(slope_lookback=0),
        lambda: MarketTrendConfig(return_weight=Decimal("0.5")),
        lambda: MarketTrendConfig(score_scale=Decimal(0)),
        lambda: MarketTrendConfig(score_scale=Decimal("Infinity")),
        lambda: MarketTrendConfig(return_weight=Decimal("NaN")),
    ],
)
def test_invalid_configuration_is_rejected(factory: Callable[[], MarketTrendConfig]) -> None:
    with pytest.raises(ValueError):
        factory()
