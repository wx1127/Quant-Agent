from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.features.core import (
    FeatureMetadata,
    InsufficientObservationsError,
    MissingValuePolicy,
    RollingCalculator,
    TimeSeriesPoint,
    clamp_score,
)

AS_OF = datetime(2026, 1, 10, 16, tzinfo=UTC)


def metadata(window: int = 3, minimum: int = 2) -> FeatureMetadata:
    return FeatureMetadata("close", "v1", "snapshot-1", window, minimum, {"x": 1})


def test_prepare_is_point_in_time_safe_and_deduplicates() -> None:
    points = [
        TimeSeriesPoint(date(2026, 1, 8), AS_OF - timedelta(days=2), 1.0),
        TimeSeriesPoint(date(2026, 1, 8), AS_OF - timedelta(days=1), 2.0),
        TimeSeriesPoint(date(2026, 1, 9), AS_OF - timedelta(hours=1), 4.0),
        TimeSeriesPoint(date(2026, 1, 10), AS_OF + timedelta(hours=1), 100.0),
        TimeSeriesPoint(date(2026, 1, 11), AS_OF - timedelta(hours=1), 100.0),
    ]
    result = RollingCalculator.mean(points, as_of=AS_OF, metadata=metadata())
    assert result.value == 3.0
    assert result.observations_used == 2
    assert result.cache_key == metadata().cache_key(AS_OF)
    assert result.cache_key != metadata().cache_key(AS_OF + timedelta(days=1))


def test_missing_policy_and_statistics() -> None:
    points = [
        TimeSeriesPoint(date(2026, 1, 7), AS_OF - timedelta(days=3), 1.0),
        TimeSeriesPoint(date(2026, 1, 8), AS_OF - timedelta(days=2), None),
        TimeSeriesPoint(date(2026, 1, 9), AS_OF - timedelta(days=1), 3.0),
    ]
    with pytest.raises(ValueError, match="missing"):
        RollingCalculator.prepare(points, as_of=AS_OF, metadata=metadata())
    prepared = RollingCalculator.prepare(
        points,
        as_of=AS_OF,
        metadata=metadata(),
        missing_policy=MissingValuePolicy.DROP,
    )
    assert len(prepared) == 2
    assert (
        RollingCalculator.total_return(
            points,
            as_of=AS_OF,
            metadata=metadata(),
            missing_policy=MissingValuePolicy.DROP,
        ).value
        == 2.0
    )
    assert RollingCalculator.slope(
        [point for point in points if point.value is not None],
        as_of=AS_OF,
        metadata=metadata(),
    ).value == pytest.approx(2.0)
    assert RollingCalculator.standard_deviation(
        [point for point in points if point.value is not None],
        as_of=AS_OF,
        metadata=metadata(),
    ).value == pytest.approx(2**0.5)
    assert RollingCalculator.percentile_rank([1, 2, 3], 2) == pytest.approx(2 / 3)


def test_feature_validation_and_failures() -> None:
    with pytest.raises(ValueError):
        FeatureMetadata("x", "v", "d", 0, 1)
    with pytest.raises(ValueError):
        FeatureMetadata("x", "v", "", 2, 1)
    with pytest.raises(ValueError):
        TimeSeriesPoint(date.today(), datetime.now(), 1.0)
    with pytest.raises(InsufficientObservationsError):
        RollingCalculator.mean([], as_of=AS_OF, metadata=metadata())
    with pytest.raises(InsufficientObservationsError):
        RollingCalculator.percentile_rank([], 1)
    with pytest.raises(ValueError):
        RollingCalculator.total_return(
            [
                TimeSeriesPoint(date(2026, 1, 8), AS_OF, 0),
                TimeSeriesPoint(date(2026, 1, 9), AS_OF, 1),
            ],
            as_of=AS_OF,
            metadata=metadata(window=2, minimum=2),
        )
    with pytest.raises(ValueError):
        clamp_score(1, 2, 2)
    assert clamp_score(-1, 0, 2) == 0
    assert clamp_score(3, 0, 2) == 100
