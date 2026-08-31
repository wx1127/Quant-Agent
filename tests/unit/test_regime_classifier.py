"""Tests for deterministic and point-in-time-bound market regime classification."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.features.market_breadth import MarketBreadthSnapshot
from quant_agent.features.market_trend import (
    IndexTrendResult,
    MarketTrendSnapshot,
    WindowTrendMetrics,
)
from quant_agent.regime import (
    ConfidenceMeaning,
    EvidenceSide,
    InsufficientRegimeData,
    MarketRegime,
    MarketRegimeClassifier,
    RegimeClassifierConfig,
    RegimeInputMismatch,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(2026, 8, 28, 16, 0, tzinfo=SHANGHAI)
SESSION_DATE = date(2026, 8, 28)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _window(window: int, score: Decimal) -> WindowTrendMetrics:
    return WindowTrendMetrics(
        window=window,
        latest_close=Decimal(100),
        moving_average=Decimal(100),
        period_return=Decimal(0),
        position_vs_average=Decimal(0),
        average_slope=Decimal(0),
        score=score,
        input_hash=HASH_A,
    )


def _trend(
    aggregate: str,
    *,
    short: str | None = None,
    long: str | None = None,
    data_version: str = "snapshot-v1",
    as_of: datetime = AS_OF,
) -> MarketTrendSnapshot:
    aggregate_score = Decimal(aggregate)
    short_score = aggregate_score if short is None else Decimal(short)
    long_score = aggregate_score if long is None else Decimal(long)
    windows = (_window(20, short_score), _window(120, long_score))
    return MarketTrendSnapshot(
        session_date=SESSION_DATE,
        as_of=as_of,
        data_version=data_version,
        feature_version="market-trend-test-v1",
        config_hash=HASH_A,
        indices=(
            IndexTrendResult(
                index_id="000001.SH",
                windows=windows,
                observation_dates=(SESSION_DATE,),
                score=aggregate_score,
            ),
        ),
        missing_indices=(),
        aggregate_score=aggregate_score,
        cache_key=HASH_B,
    )


def _breadth(
    *,
    advance_decline: str,
    above_average: str,
    new_high: str,
    new_low: str,
    turnover: str,
    downside_volatility: str,
    data_version: str = "snapshot-v1",
    as_of: datetime = AS_OF,
) -> MarketBreadthSnapshot:
    advance_count = int(Decimal(advance_decline) * 100)
    above_count = int(Decimal(above_average) * 100)
    high_count = int(Decimal(new_high) * 100)
    low_count = int(Decimal(new_low) * 100)
    return MarketBreadthSnapshot(
        session_date=SESSION_DATE,
        as_of=as_of,
        data_version=data_version,
        feature_version="market-breadth-test-v1",
        config_hash=HASH_C,
        expected_active_count=100,
        received_active_count=100,
        current_coverage=Decimal(1),
        historical_session_count=60,
        minimum_historical_coverage_observed=Decimal(1),
        missing_instruments=(),
        suspended_count=0,
        advancing_count=advance_count,
        declining_count=100 - advance_count,
        unchanged_count=0,
        return_denominator=100,
        above_average_count=above_count,
        moving_average_denominator=100,
        above_average_ratio=Decimal(above_average),
        new_high_count=high_count,
        new_low_count=low_count,
        high_low_denominator=100,
        new_high_ratio=Decimal(new_high),
        new_low_ratio=Decimal(new_low),
        total_turnover=Decimal("1000000000"),
        turnover_percentile=Decimal(turnover),
        turnover_history_count=60,
        downside_volatility=Decimal(downside_volatility),
        downside_observation_count=20,
        cache_key=HASH_D,
    )


SCENARIOS = (
    (
        MarketRegime.UPTREND,
        _trend("70"),
        _breadth(
            advance_decline="0.70",
            above_average="0.75",
            new_high="0.20",
            new_low="0.01",
            turnover="0.80",
            downside_volatility="0.005",
        ),
    ),
    (
        MarketRegime.RANGE_STRONG,
        _trend("25"),
        _breadth(
            advance_decline="0.58",
            above_average="0.62",
            new_high="0.08",
            new_low="0.03",
            turnover="0.65",
            downside_volatility="0.015",
        ),
    ),
    (
        MarketRegime.DIVERGENT,
        _trend("55"),
        _breadth(
            advance_decline="0.42",
            above_average="0.38",
            new_high="0.03",
            new_low="0.08",
            turnover="0.70",
            downside_volatility="0.025",
        ),
    ),
    (
        MarketRegime.DOWNTREND,
        _trend("-70"),
        _breadth(
            advance_decline="0.25",
            above_average="0.20",
            new_high="0.01",
            new_low="0.20",
            turnover="0.25",
            downside_volatility="0.04",
        ),
    ),
    (
        MarketRegime.BOTTOM_RECOVERY,
        _trend("-20", short="15", long="-30"),
        _breadth(
            advance_decline="0.60",
            above_average="0.55",
            new_high="0.06",
            new_low="0.03",
            turnover="0.55",
            downside_volatility="0.02",
        ),
    ),
)


@pytest.mark.parametrize(("expected", "trend", "breadth"), SCENARIOS)
def test_fixed_scenarios_cover_all_five_regimes(
    expected: MarketRegime,
    trend: MarketTrendSnapshot,
    breadth: MarketBreadthSnapshot,
) -> None:
    config = RegimeClassifierConfig()
    result = MarketRegimeClassifier(config).classify(
        trend_snapshot=trend,
        breadth_snapshot=breadth,
    )

    assert result.regime is expected
    assert Decimal(0) <= result.score <= Decimal(100)
    assert result.risk_budget_max == config.risk_budget_for(expected)
    assert result.evidence
    assert result.invalidations


def _trend_only_config() -> RegimeClassifierConfig:
    return RegimeClassifierConfig(
        version="boundary-v1",
        trend_weight=Decimal(1),
        breadth_weight=Decimal(0),
        turnover_weight=Decimal(0),
        new_high_low_weight=Decimal(0),
        downside_risk_weight=Decimal(0),
        divergence_trend_min=Decimal(100),
    )


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    (
        ("40", MarketRegime.UPTREND),
        ("10", MarketRegime.RANGE_STRONG),
        ("-20", MarketRegime.DIVERGENT),
        ("-20.0002", MarketRegime.DOWNTREND),
    ),
)
def test_score_boundaries_are_inclusive_only_on_the_upper_state(
    aggregate: str,
    expected: MarketRegime,
) -> None:
    breadth = _breadth(
        advance_decline="0.50",
        above_average="0.50",
        new_high="0.05",
        new_low="0.05",
        turnover="0.50",
        downside_volatility="0.02",
    )
    result = MarketRegimeClassifier(_trend_only_config()).classify(
        trend_snapshot=_trend(aggregate),
        breadth_snapshot=breadth,
    )

    assert result.regime is expected


def test_environment_score_exposes_each_weighted_term() -> None:
    result = MarketRegimeClassifier().classify(
        trend_snapshot=_trend("70"),
        breadth_snapshot=SCENARIOS[0][2],
    )

    assert result.components.trend == Decimal("85")
    assert result.components.breadth == Decimal("72.50")
    assert result.components.turnover == Decimal("80.00")
    assert result.components.new_high_low == Decimal("59.50")
    assert result.components.downside_risk == Decimal("87.500")
    assert result.score == Decimal("76.27500")


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("above_average_ratio", None),
        ("new_high_ratio", None),
        ("new_low_ratio", None),
        ("turnover_percentile", None),
        ("downside_volatility", None),
        ("return_denominator", 0),
        ("moving_average_denominator", 0),
        ("high_low_denominator", 0),
        ("turnover_history_count", 0),
        ("downside_observation_count", 0),
    ),
)
def test_missing_critical_breadth_metric_fails_closed(
    field_name: str,
    value: Decimal | int | None,
) -> None:
    breadth = replace(SCENARIOS[0][2], **{field_name: value})

    with pytest.raises(InsufficientRegimeData, match="critical breadth"):
        MarketRegimeClassifier().classify(
            trend_snapshot=SCENARIOS[0][1],
            breadth_snapshot=breadth,
        )


def test_regime_coverage_floor_is_independent_of_upstream_breadth_config() -> None:
    breadth = replace(
        SCENARIOS[0][2],
        current_coverage=Decimal("0.90"),
        received_active_count=90,
        missing_instruments=tuple(f"S{i}" for i in range(10)),
    )

    with pytest.raises(InsufficientRegimeData, match="coverage"):
        MarketRegimeClassifier().classify(
            trend_snapshot=SCENARIOS[0][1],
            breadth_snapshot=breadth,
        )


def test_mixed_as_of_or_data_version_fails_before_classification() -> None:
    with pytest.raises(RegimeInputMismatch, match="as_of"):
        MarketRegimeClassifier().classify(
            trend_snapshot=SCENARIOS[0][1],
            breadth_snapshot=replace(
                SCENARIOS[0][2],
                as_of=AS_OF + timedelta(minutes=1),
            ),
        )

    with pytest.raises(RegimeInputMismatch, match="data_version"):
        MarketRegimeClassifier().classify(
            trend_snapshot=SCENARIOS[0][1],
            breadth_snapshot=replace(SCENARIOS[0][2], data_version="snapshot-v2"),
        )

    with pytest.raises(RegimeInputMismatch, match="session_date"):
        MarketRegimeClassifier().classify(
            trend_snapshot=replace(
                SCENARIOS[0][1],
                session_date=SESSION_DATE - timedelta(days=1),
            ),
            breadth_snapshot=SCENARIOS[0][2],
        )


def test_result_is_reproducible_and_bound_to_pit_input_versions() -> None:
    classifier = MarketRegimeClassifier()
    first = classifier.classify(
        trend_snapshot=SCENARIOS[0][1],
        breadth_snapshot=SCENARIOS[0][2],
    )
    repeated = classifier.classify(
        trend_snapshot=SCENARIOS[0][1],
        breadth_snapshot=SCENARIOS[0][2],
    )
    changed = classifier.classify(
        trend_snapshot=replace(SCENARIOS[0][1], data_version="snapshot-v2"),
        breadth_snapshot=replace(SCENARIOS[0][2], data_version="snapshot-v2"),
    )

    assert repeated == first
    assert repeated.result_hash == first.result_hash
    assert changed.score == first.score
    assert changed.input_hash != first.input_hash
    assert changed.result_hash != first.result_hash
    assert first.input_identity.trend_cache_key == SCENARIOS[0][1].cache_key
    assert first.input_identity.breadth_cache_key == SCENARIOS[0][2].cache_key


def test_all_thresholds_and_model_version_participate_in_config_hash() -> None:
    baseline = RegimeClassifierConfig()
    threshold_changed = replace(baseline, bottom_new_low_max=Decimal("0.07"))
    version_changed = replace(baseline, version="market-regime-v2")
    evidence_weight_changed = replace(
        baseline,
        divergent_evidence_weights=(
            Decimal("0.21"),
            Decimal("0.24"),
            Decimal("0.20"),
            Decimal("0.20"),
            Decimal("0.15"),
        ),
    )

    assert len(baseline.config_hash) == 64
    assert threshold_changed.config_hash != baseline.config_hash
    assert version_changed.config_hash != baseline.config_hash
    assert evidence_weight_changed.config_hash != baseline.config_hash
    assert (
        replace(
            baseline,
            uptrend_risk_budget_max=Decimal("0.800"),
        ).config_hash
        == baseline.config_hash
    )


def test_confidence_is_rule_consistency_not_upward_probability() -> None:
    result = MarketRegimeClassifier().classify(
        trend_snapshot=SCENARIOS[2][1],
        breadth_snapshot=SCENARIOS[2][2],
    )
    evaluated = (*result.evidence, *result.counter_evidence)

    assert result.confidence_meaning is ConfidenceMeaning.RULE_EVIDENCE_CONSISTENCY
    assert result.confidence_is_probability is False
    assert "不是市场上涨概率" in result.confidence_definition
    assert result.confidence == sum(
        (item.rule_weight for item in result.evidence),
        Decimal(0),
    )
    assert sum((item.rule_weight for item in evaluated), Decimal(0)) == Decimal(1)
    assert all(item.side is EvidenceSide.SUPPORTING for item in result.supporting_evidence)
    assert all(item.side is EvidenceSide.OPPOSING for item in result.opposing_evidence)


def test_harness_aliases_and_stable_labels_are_explicit() -> None:
    result = MarketRegimeClassifier().classify(
        trend_snapshot=SCENARIOS[1][1],
        breadth_snapshot=SCENARIOS[1][2],
    )

    assert result.raw_score == result.score
    assert result.max_risk_budget == result.risk_budget_max
    assert result.identity_payload()["result_hash"] == result.result_hash
    assert MarketRegime.HIGH_LEVEL_DIVERGENCE is MarketRegime.DIVERGENT
    assert {regime.display_name for regime in MarketRegime} == {
        "上升",
        "震荡偏强",
        "高位分化",
        "下跌",
        "底部修复",
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"version": " "}, "version"),
        ({"trend_weight": Decimal("-0.1")}, "non-negative"),
        ({"trend_weight": Decimal("0.31")}, "sum to one"),
        ({"range_strong_min_score": Decimal("75")}, "ordered"),
        ({"neutral_component_score": Decimal("101")}, "neutral"),
        ({"minimum_breadth_coverage": Decimal("1.01")}, "ratio"),
        ({"divergence_high_low_spread_max": Decimal("1.01")}, "spread"),
        ({"downside_volatility_ceiling": Decimal(0)}, "volatility"),
        ({"minimum_return_denominator": 0}, "minimum counts"),
        ({"divergence_min_weak_signals": 0}, "weak_signals"),
        ({"bottom_short_window": 1}, "bottom windows"),
        ({"bottom_min_confirming_signals": 0}, "confirming_signals"),
        ({"divergence_trend_min": Decimal("101")}, "trend thresholds"),
        ({"downtrend_risk_budget_max": Decimal("1.01")}, "risk budget"),
        ({"neutral_component_score": Decimal("NaN")}, "finite"),
        (
            {"uptrend_evidence_weights": (Decimal("0.25"),) * 4},
            "exactly five",
        ),
        (
            {"uptrend_evidence_weights": (Decimal("0.10"),) * 5},
            "sum to one",
        ),
    ),
)
def test_invalid_config_fails_early(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RegimeClassifierConfig(**changes)  # type: ignore[arg-type]


def test_absent_bottom_windows_do_not_invent_a_recovery_signal() -> None:
    base = _trend("-20")
    without_bottom_windows = replace(
        base,
        indices=(
            IndexTrendResult(
                index_id="000001.SH",
                windows=(_window(60, Decimal("-20")),),
                observation_dates=(SESSION_DATE,),
                score=Decimal("-20"),
            ),
        ),
    )
    result = MarketRegimeClassifier().classify(
        trend_snapshot=without_bottom_windows,
        breadth_snapshot=_breadth(
            advance_decline="0.60",
            above_average="0.55",
            new_high="0.06",
            new_low="0.03",
            turnover="0.55",
            downside_volatility="0.02",
        ),
    )

    assert result.regime is not MarketRegime.BOTTOM_RECOVERY
