"""Tests for deterministic point-in-time rolling features."""

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.features.core import (
    FeatureDefinition,
    FeatureObservation,
    FeatureRequest,
    FeatureStatus,
    MissingValuePolicy,
    RollingFeatureEngine,
)
from quant_agent.features.core.engine import describe_cache_identity

SHANGHAI = ZoneInfo("Asia/Shanghai")
BASE = datetime(2026, 8, 3, 15, 0, tzinfo=SHANGHAI)


def _observation(
    day: int,
    value: str | None,
    *,
    available_delay: timedelta = timedelta(minutes=30),
    revision: str = "v1",
) -> FeatureObservation:
    observed_at = BASE + timedelta(days=day)
    return FeatureObservation(
        observation_key=f"day-{day}",
        observed_at=observed_at,
        available_at=observed_at + available_delay,
        value=Decimal(value) if value is not None else None,
        revision=revision,
    )


def _definition(**overrides: object) -> FeatureDefinition:
    values: dict[str, object] = {
        "name": "close_mean",
        "version": "1.0.0",
        "calculator_id": "arithmetic-mean-v1",
        "window": 3,
        "min_observations": 2,
    }
    values.update(overrides)
    return FeatureDefinition.create(**values)  # type: ignore[arg-type]


def _request(*, as_of: datetime | None = None, data_version: str = "snapshot-v1") -> FeatureRequest:
    return FeatureRequest(
        entity_id="CN.SH.510300",
        as_of=as_of or BASE + timedelta(days=10),
        data_version=data_version,
    )


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def test_engine_uses_latest_chronological_window() -> None:
    result = RollingFeatureEngine().evaluate(
        definition=_definition(),
        request=_request(),
        observations=[
            _observation(3, "4"),
            _observation(0, "1"),
            _observation(2, "3"),
            _observation(1, "2"),
        ],
        calculator=_mean,
    )

    assert result.status is FeatureStatus.READY
    assert result.value == Decimal("3")
    assert result.observation_count == 3
    assert result.window_start == BASE + timedelta(days=1)
    assert result.window_end == BASE + timedelta(days=3)


def test_future_data_and_future_revision_cannot_change_earlier_result() -> None:
    original = _observation(0, "10")
    revised = FeatureObservation(
        observation_key=original.observation_key,
        observed_at=original.observed_at,
        available_at=BASE + timedelta(days=5),
        value=Decimal("999"),
        revision="v2",
    )
    future = _observation(4, "777")
    request = _request(as_of=BASE + timedelta(days=2))
    engine = RollingFeatureEngine()

    without_future = engine.evaluate(
        definition=_definition(window=2, min_observations=1),
        request=request,
        observations=[original],
        calculator=_mean,
    )
    with_future = engine.evaluate(
        definition=_definition(window=2, min_observations=1),
        request=request,
        observations=[original, revised, future],
        calculator=_mean,
    )

    assert with_future.value == Decimal("10")
    assert with_future.input_hash == without_future.input_hash
    assert with_future.cache_key == without_future.cache_key


def test_latest_revision_known_at_as_of_wins() -> None:
    original = _observation(0, "10")
    revised = FeatureObservation(
        observation_key=original.observation_key,
        observed_at=original.observed_at,
        available_at=BASE + timedelta(days=1),
        value=Decimal("11"),
        revision="v2",
    )

    result = RollingFeatureEngine().evaluate(
        definition=_definition(window=1, min_observations=1),
        request=_request(as_of=BASE + timedelta(days=2)),
        observations=[original, revised],
        calculator=_mean,
    )

    assert result.value == Decimal("11")


def test_insufficient_window_is_explicit_and_does_not_call_calculator() -> None:
    called = False

    def calculator(values: tuple[Decimal, ...]) -> Decimal:
        nonlocal called
        called = True
        return _mean(values)

    result = RollingFeatureEngine().evaluate(
        definition=_definition(window=5, min_observations=3),
        request=_request(),
        observations=[_observation(0, "1"), _observation(1, None)],
        calculator=calculator,
    )

    assert result.status is FeatureStatus.INSUFFICIENT_DATA
    assert result.value is None
    assert result.observation_count == 1
    assert result.reason == "requires at least 3 observations; received 1"
    assert not called


def test_missing_policy_can_require_a_complete_selected_window() -> None:
    result = RollingFeatureEngine().evaluate(
        definition=_definition(
            missing_value_policy=MissingValuePolicy.REQUIRE_COMPLETE,
        ),
        request=_request(),
        observations=[
            _observation(0, "1"),
            _observation(1, None),
            _observation(2, "3"),
        ],
        calculator=_mean,
    )

    assert result.status is FeatureStatus.INSUFFICIENT_DATA
    assert result.reason == "selected window contains missing values"


def test_cache_identity_binds_snapshot_parameters_and_inputs() -> None:
    engine = RollingFeatureEngine()
    inputs = [_observation(0, "1"), _observation(1, "2")]
    base = engine.evaluate(
        definition=_definition(parameters={"lag": 1}),
        request=_request(),
        observations=inputs,
        calculator=_mean,
    )
    reordered_parameters = engine.evaluate(
        definition=_definition(parameters={"lag": 1}),
        request=_request(),
        observations=list(reversed(inputs)),
        calculator=_mean,
    )
    changed_snapshot = engine.evaluate(
        definition=_definition(parameters={"lag": 1}),
        request=_request(data_version="snapshot-v2"),
        observations=inputs,
        calculator=_mean,
    )
    changed_parameters = engine.evaluate(
        definition=_definition(parameters={"lag": 2}),
        request=_request(),
        observations=inputs,
        calculator=_mean,
    )

    assert base.cache_key == reordered_parameters.cache_key
    assert base.cache_key != changed_snapshot.cache_key
    assert base.cache_key != changed_parameters.cache_key
    assert f'"cache_key":"{base.cache_key}"' in describe_cache_identity(base)


def test_ambiguous_same_time_revision_is_rejected() -> None:
    original = _observation(0, "1")
    conflicting = FeatureObservation(
        observation_key=original.observation_key,
        observed_at=original.observed_at,
        available_at=original.available_at,
        value=Decimal("2"),
        revision="v2",
    )

    with pytest.raises(ValueError, match="ambiguous revisions"):
        RollingFeatureEngine().evaluate(
            definition=_definition(window=1, min_observations=1),
            request=_request(),
            observations=[original, conflicting],
            calculator=_mean,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: FeatureDefinition.create(
                name="x",
                version="v1",
                calculator_id="identity-v1",
                window=0,
                min_observations=1,
            ),
            "window must be positive",
        ),
        (
            lambda: FeatureDefinition.create(
                name="x",
                version="v1",
                calculator_id="identity-v1",
                window=2,
                min_observations=3,
            ),
            "min_observations",
        ),
        (
            lambda: FeatureRequest(
                entity_id="x",
                as_of=datetime(2026, 1, 1),
                data_version="v1",
            ),
            "timezone",
        ),
        (
            lambda: FeatureObservation(
                observation_key="x",
                observed_at=BASE,
                available_at=BASE - timedelta(seconds=1),
                value=Decimal("1"),
                revision="v1",
            ),
            "cannot precede",
        ),
    ],
)
def test_contracts_reject_invalid_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_non_finite_calculator_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        RollingFeatureEngine().evaluate(
            definition=_definition(window=1, min_observations=1),
            request=_request(),
            observations=[_observation(0, "1")],
            calculator=lambda _: float("nan"),
        )


def test_public_window_selection_has_the_same_point_in_time_semantics() -> None:
    future = _observation(3, "99")
    selected = RollingFeatureEngine().select_window(
        observations=[_observation(0, "1"), _observation(1, "2"), future],
        as_of=BASE + timedelta(days=2),
        window=1,
    )

    assert [item.value for item in selected] == [Decimal("2")]
    with pytest.raises(ValueError, match="window must be positive"):
        RollingFeatureEngine().select_window(
            observations=(),
            as_of=BASE,
            window=0,
        )


def test_calculator_identity_is_part_of_the_cache_key() -> None:
    engine = RollingFeatureEngine()
    request = _request()
    observations = [_observation(0, "2")]
    identity = engine.evaluate(
        definition=_definition(
            window=1,
            min_observations=1,
            calculator_id="identity-v1",
        ),
        request=request,
        observations=observations,
        calculator=lambda values: values[0],
    )
    doubled = engine.evaluate(
        definition=_definition(
            window=1,
            min_observations=1,
            calculator_id="double-v1",
        ),
        request=request,
        observations=observations,
        calculator=lambda values: values[0] * 2,
    )

    assert identity.value == 2
    assert doubled.value == 4
    assert identity.cache_key != doubled.cache_key
