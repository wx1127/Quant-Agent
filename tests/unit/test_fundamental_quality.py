"""Fixed point-in-time samples for fundamental quality and risk features."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import FundamentalPoint
from quant_agent.features.fundamental_quality import (
    FundamentalComponent,
    FundamentalEvidenceSide,
    FundamentalFlag,
    FundamentalFlagCode,
    FundamentalFlagKind,
    FundamentalMetricMapping,
    FundamentalQualityAnalyzer,
    FundamentalQualityConfig,
    FundamentalQualityInputError,
    FundamentalQualityRequest,
    FundamentalQualityStatus,
)

TZ = ZoneInfo("Asia/Shanghai")
INSTRUMENT = "CN.SZ.000001"
AS_OF = datetime(2026, 7, 22, 15, tzinfo=TZ)
PERIOD_1 = date(2025, 12, 31)
PERIOD_2 = date(2026, 3, 31)
PERIOD_3 = date(2026, 6, 30)


def _point(
    metric_name: str,
    value: str,
    *,
    report_period: date = PERIOD_3,
    announced_on: date = date(2026, 7, 20),
    revision: str = "r1",
    instrument_id: str = INSTRUMENT,
) -> FundamentalPoint:
    announced_at = datetime(
        announced_on.year,
        announced_on.month,
        announced_on.day,
        18,
        tzinfo=TZ,
    )
    available_day = announced_on + timedelta(days=1)
    return FundamentalPoint(
        instrument_id=instrument_id,
        report_period=report_period,
        metric_name=metric_name,
        metric_value=Decimal(value),
        announced_at=announced_at,
        available_at=datetime(
            available_day.year,
            available_day.month,
            available_day.day,
            9,
            tzinfo=TZ,
        ),
        provider_revision=revision,
        source="fixed-test-provider",
    )


def _complete_points(
    *,
    profitability: str = "12",
    cash_flow: str = "0.8",
    leverage: str = "46",
    growth: tuple[str, str, str] = ("20", "25", "30"),
) -> tuple[FundamentalPoint, ...]:
    return (
        _point("roe", profitability),
        _point("ocfps", cash_flow),
        _point("debt_to_assets", leverage),
        _point(
            "netprofit_yoy",
            growth[0],
            report_period=PERIOD_1,
            announced_on=date(2026, 3, 31),
            revision="growth-r1",
        ),
        _point(
            "netprofit_yoy",
            growth[1],
            report_period=PERIOD_2,
            announced_on=date(2026, 4, 30),
            revision="growth-r2",
        ),
        _point(
            "netprofit_yoy",
            growth[2],
            revision="growth-r3",
        ),
    )


def test_fixed_sample_matches_manual_quality_score_and_source_lineage() -> None:
    points = _complete_points()
    analyzer = FundamentalQualityAnalyzer()

    first = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=points,
    )
    second = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=reversed(points),
    )

    # 80*0.30 + 80*0.25 + 80*0.20 + 100*0.25 = 85.
    assert first == second
    assert first.status is FundamentalQualityStatus.READY
    assert first.quality_score == Decimal(85)
    assert first.risk_penalty == Decimal(0)
    assert first.adjusted_quality_score == Decimal(85)
    assert tuple(item.normalized_score for item in first.components) == (
        Decimal(80),
        Decimal(80),
        Decimal(80),
        Decimal(100),
    )
    assert tuple(item.contribution for item in first.components) == (
        Decimal(24),
        Decimal(20),
        Decimal(16),
        Decimal(25),
    )
    assert len(first.supporting_evidence) == 4
    assert not first.counter_evidence
    assert all(
        item.side is FundamentalEvidenceSide.SUPPORTING for item in first.supporting_evidence
    )
    assert not first.flags
    assert len(first.selected_sources) == 6
    assert {item.report_period for item in first.selected_sources} == {
        PERIOD_1,
        PERIOD_2,
        PERIOD_3,
    }
    assert all(item.provider_revision for item in first.selected_sources)
    assert all(component.sources for component in first.components)
    assert first.data_version == "fundamentals-v1"
    assert first.feature_version == "fundamental-quality-v1"
    assert first.metric_mapping_version == "fundamental-metric-map-tushare-v1"
    assert first.identity_payload()["result_hash"] == first.result_hash
    assert first.score == first.quality_score
    assert all(
        len(value) == 64
        for value in (
            first.metric_mapping_hash,
            first.config_hash,
            first.input_hash,
            first.cache_key,
            first.result_hash,
        )
    )


def test_latest_available_revision_is_pit_and_future_append_cannot_rewrite_history() -> None:
    points = _complete_points()
    analyzer = FundamentalQualityAnalyzer()
    baseline = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=points,
    )
    future_revision = _point(
        "roe",
        "3",
        announced_on=date(2026, 8, 1),
        revision="r2",
    )

    appended = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=(*points, future_revision),
    )
    later = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=datetime(2026, 8, 3, 15, tzinfo=TZ),
        data_version="fundamentals-v1",
        points=(*points, future_revision),
    )

    assert appended == baseline
    assert appended.input_hash == baseline.input_hash
    assert later.input_hash != baseline.input_hash
    assert later.result_hash != baseline.result_hash
    profitability = later.components[0]
    assert profitability.raw_value == Decimal(3)
    assert profitability.sources[0].provider_revision == "r2"
    assert profitability.normalized_score == Decimal(20)
    assert later.quality_score == Decimal(67)


def test_missing_components_are_flagged_and_penalized_not_scored_as_neutral() -> None:
    analyzer = FundamentalQualityAnalyzer()
    roe_only = (_point("roe", "12"),)
    partial = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=roe_only,
    )
    unavailable = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=(),
    )

    assert partial.status is FundamentalQualityStatus.PARTIAL
    assert partial.quality_score == Decimal(24)
    assert partial.risk_penalty == Decimal(30)
    assert partial.adjusted_quality_score == Decimal(0)
    assert len(partial.supporting_evidence) == 1
    assert len(partial.counter_evidence) == 3
    assert {item.code for item in partial.flags} == {
        FundamentalFlagCode.MISSING_OPERATING_CASH_FLOW,
        FundamentalFlagCode.MISSING_LEVERAGE,
        FundamentalFlagCode.MISSING_GROWTH_HISTORY,
    }
    assert all(item.penalty == Decimal(10) for item in partial.flags)
    assert all(item.normalized_score is None for item in partial.components[1:])
    assert all(item.contribution == 0 for item in partial.components[1:])

    assert unavailable.status is FundamentalQualityStatus.INSUFFICIENT_DATA
    assert unavailable.quality_score == 0
    assert unavailable.risk_penalty == 40
    assert unavailable.adjusted_quality_score == 0
    assert len(unavailable.flags) == 4
    assert not unavailable.selected_sources


def test_core_metrics_never_backfill_old_period_but_same_period_roic_can_replace_roe() -> None:
    analyzer = FundamentalQualityAnalyzer()
    without_current_cash = tuple(
        point for point in _complete_points() if point.metric_name != "ocfps"
    )
    old_cash = _point(
        "ocfps",
        "9",
        report_period=PERIOD_1,
        announced_on=date(2026, 3, 31),
        revision="old-cash",
    )
    no_backfill = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=(*without_current_cash, old_cash),
    )

    cash_component = next(
        item
        for item in no_backfill.components
        if item.component is FundamentalComponent.OPERATING_CASH_FLOW
    )
    assert cash_component.missing
    assert cash_component.normalized_score is None
    assert not cash_component.sources
    assert FundamentalFlagCode.MISSING_OPERATING_CASH_FLOW in {
        item.code for item in no_backfill.flags
    }
    assert all(source.provider_revision != "old-cash" for source in no_backfill.selected_sources)

    roic_points = tuple(
        point.model_copy(update={"metric_name": "roic"}) if point.metric_name == "roe" else point
        for point in _complete_points()
    )
    roic = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=roic_points,
    )
    assert roic.status is FundamentalQualityStatus.READY
    assert roic.components[0].raw_value == Decimal(12)
    assert roic.components[0].sources[0].metric_name == "roic"


def test_financial_anomalies_create_traceable_risk_penalties() -> None:
    analyzer = FundamentalQualityAnalyzer()
    adverse = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=_complete_points(
            profitability="-5",
            cash_flow="-1",
            leverage="75",
            growth=("-100", "50", "150"),
        ),
    )
    mismatch = analyzer.analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=_complete_points(cash_flow="-1"),
    )

    assert adverse.quality_score == 0
    assert adverse.risk_penalty == 75
    assert adverse.adjusted_quality_score == 0
    assert {item.code for item in adverse.flags} == {
        FundamentalFlagCode.NEGATIVE_PROFITABILITY,
        FundamentalFlagCode.NEGATIVE_OPERATING_CASH_FLOW,
        FundamentalFlagCode.HIGH_LEVERAGE,
        FundamentalFlagCode.UNSTABLE_GROWTH,
        FundamentalFlagCode.EXTREME_GROWTH,
    }
    assert all(item.sources for item in adverse.flags)
    assert all(item.sources[0].provider_revision for item in adverse.flags)
    assert len(adverse.counter_evidence) == 4

    assert {item.code for item in mismatch.flags} == {
        FundamentalFlagCode.NEGATIVE_OPERATING_CASH_FLOW,
        FundamentalFlagCode.CASH_EARNINGS_MISMATCH,
    }
    assert mismatch.risk_penalty == Decimal(25)
    mismatch_flag = next(
        item for item in mismatch.flags if item.code is FundamentalFlagCode.CASH_EARNINGS_MISMATCH
    )
    assert len(mismatch_flag.sources) == 2


def test_versioned_metric_mapping_controls_provider_names_and_hash_identity() -> None:
    mapping = FundamentalMetricMapping(
        version="custom-map-v2",
        profitability=("custom_roe",),
        operating_cash_flow=("custom_ocf",),
        leverage=("custom_leverage",),
        growth=("custom_growth",),
    )
    config = FundamentalQualityConfig(metric_mapping=mapping)
    points = tuple(
        point.model_copy(
            update={"metric_name": replacement},
        )
        for point, replacement in zip(
            _complete_points(),
            (
                "custom_roe",
                "custom_ocf",
                "custom_leverage",
                "custom_growth",
                "custom_growth",
                "custom_growth",
            ),
            strict=True,
        )
    )

    result = FundamentalQualityAnalyzer(config).evaluate(
        request=FundamentalQualityRequest(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="custom-data-v2",
        ),
        points=points,
    )

    assert result.status is FundamentalQualityStatus.READY
    assert result.quality_score == Decimal(85)
    assert result.metric_mapping_version == "custom-map-v2"
    assert result.metric_mapping_hash == mapping.mapping_hash
    assert result.config_hash == config.config_hash
    assert result.config_hash != FundamentalQualityConfig().config_hash
    assert {item.metric_name for item in result.selected_sources} == {
        "custom_roe",
        "custom_ocf",
        "custom_leverage",
        "custom_growth",
    }


def test_rejects_mixed_instruments_nonfinite_values_and_time_order_errors() -> None:
    analyzer = FundamentalQualityAnalyzer()
    valid = _point("roe", "12")

    with pytest.raises(FundamentalQualityInputError, match="mix instruments"):
        analyzer.analyze(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="fundamentals-v1",
            points=(valid, valid.model_copy(update={"instrument_id": "CN.SH.600000"})),
        )

    with pytest.raises(FundamentalQualityInputError, match="finite"):
        analyzer.analyze(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="fundamentals-v1",
            points=(valid.model_copy(update={"metric_value": Decimal("NaN")}),),
        )

    with pytest.raises(FundamentalQualityInputError, match="report_period"):
        analyzer.analyze(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="fundamentals-v1",
            points=(valid.model_copy(update={"report_period": date(2026, 12, 31)}),),
        )

    with pytest.raises(FundamentalQualityInputError, match="available_at"):
        analyzer.analyze(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="fundamentals-v1",
            points=(
                valid.model_copy(
                    update={"available_at": valid.announced_at - timedelta(minutes=1)}
                ),
            ),
        )


def test_rejects_ambiguous_same_timestamp_revisions_and_invalid_config() -> None:
    point = _point("roe", "12")
    ambiguous = point.model_copy(update={"metric_value": Decimal(13), "provider_revision": "other"})
    with pytest.raises(FundamentalQualityInputError, match="ambiguous"):
        FundamentalQualityAnalyzer().analyze(
            instrument_id=INSTRUMENT,
            as_of=AS_OF,
            data_version="fundamentals-v1",
            points=(point, ambiguous),
        )

    with pytest.raises(ValueError, match="sum to one"):
        FundamentalQualityConfig(profitability_weight=Decimal("0.31"))
    with pytest.raises(ValueError, match="multiple fundamental components"):
        FundamentalMetricMapping(
            profitability=("same",),
            operating_cash_flow=("same",),
        )
    with pytest.raises(ValueError, match="growth window"):
        FundamentalQualityConfig(growth_window_periods=2, minimum_growth_periods=3)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"version": ""}, "non-empty"),
        ({"growth_window_periods": 1}, "at least two"),
        ({"profitability_zero_score": Decimal(15)}, "profitability"),
        ({"cash_flow_zero_score": Decimal(1)}, "cash-flow"),
        ({"leverage_full_score_max": Decimal(70)}, "leverage"),
        ({"growth_range_full_score_max": Decimal(40)}, "growth-range"),
        ({"evidence_support_score_min": Decimal(101)}, "evidence support"),
        ({"extreme_growth_absolute_threshold": Decimal(0)}, "extreme growth"),
        ({"missing_component_penalty": Decimal(-1)}, "cannot be negative"),
        ({"maximum_risk_penalty": Decimal(0)}, "maximum risk penalty"),
    ),
)
def test_config_contract_rejects_ambiguous_scoring_semantics(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FundamentalQualityConfig(**changes)  # type: ignore[arg-type]


def test_output_contracts_reject_broken_lineage_scores_and_flags() -> None:
    snapshot = FundamentalQualityAnalyzer().analyze(
        instrument_id=INSTRUMENT,
        as_of=AS_OF,
        data_version="fundamentals-v1",
        points=_complete_points(),
    )
    source = snapshot.selected_sources[0]
    component = snapshot.components[0]

    with pytest.raises(ValueError, match="report_period"):
        replace(source, report_period=date(2027, 1, 1))
    with pytest.raises(ValueError, match="available_at"):
        replace(source, available_at=source.announced_at - timedelta(minutes=1))
    with pytest.raises(ValueError, match="component weight"):
        replace(component, configured_weight=Decimal(2))
    with pytest.raises(ValueError, match="within"):
        replace(component, normalized_score=Decimal(101))
    with pytest.raises(ValueError, match="requires value"):
        replace(component, raw_value=None)
    with pytest.raises(ValueError, match="equal score times weight"):
        replace(component, contribution=Decimal(1))
    with pytest.raises(ValueError, match="within"):
        replace(snapshot.supporting_evidence[0], normalized_score=Decimal(101))
    with pytest.raises(ValueError, match="within"):
        replace(
            FundamentalFlag(
                code=FundamentalFlagCode.MISSING_PROFITABILITY,
                kind=FundamentalFlagKind.MISSING,
                component=FundamentalComponent.PROFITABILITY,
                penalty=Decimal(0),
                rationale="missing",
                sources=(),
            ),
            penalty=Decimal(101),
        )
    with pytest.raises(ValueError, match="require financial source"):
        FundamentalFlag(
            code=FundamentalFlagCode.NEGATIVE_PROFITABILITY,
            kind=FundamentalFlagKind.ANOMALY,
            component=FundamentalComponent.PROFITABILITY,
            penalty=Decimal(1),
            rationale="observed",
            sources=(),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(snapshot, result_hash="bad")
    with pytest.raises(ValueError, match="deduct the risk penalty"):
        replace(snapshot, adjusted_quality_score=Decimal(1))
    with pytest.raises(ValueError, match="canonical order"):
        replace(snapshot, components=tuple(reversed(snapshot.components)))
    with pytest.raises(ValueError, match="must be unique"):
        replace(snapshot, selected_sources=(source, source))
