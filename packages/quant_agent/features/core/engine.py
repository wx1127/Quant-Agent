"""Point-in-time-safe rolling feature evaluation."""

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from decimal import Decimal

from quant_agent.features.core.contracts import (
    FeatureDefinition,
    FeatureObservation,
    FeatureRequest,
    FeatureResult,
    FeatureStatus,
    MissingValuePolicy,
    finite_decimal,
    stable_feature_hash,
)

FeatureCalculator = Callable[[tuple[Decimal, ...]], Decimal | int | float | str]


def _select_point_in_time_revisions(
    observations: Iterable[FeatureObservation],
    *,
    as_of: datetime,
) -> list[FeatureObservation]:
    selected: dict[str, FeatureObservation] = {}
    for observation in observations:
        if observation.observed_at > as_of or observation.available_at > as_of:
            continue
        current = selected.get(observation.observation_key)
        if current is None or observation.available_at > current.available_at:
            selected[observation.observation_key] = observation
            continue
        if observation.available_at == current.available_at and observation != current:
            raise ValueError(
                "ambiguous revisions share an observation key and available_at timestamp"
            )
    return sorted(
        selected.values(),
        key=lambda item: (item.observed_at, item.observation_key),
    )


def _input_hash(observations: tuple[FeatureObservation, ...]) -> str:
    return stable_feature_hash(
        {"observations": [item.fingerprint_payload() for item in observations]}
    )


def _cache_key(
    *,
    definition: FeatureDefinition,
    request: FeatureRequest,
    input_hash: str,
) -> str:
    return stable_feature_hash(
        {
            "as_of": request.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": request.data_version,
            "definition_hash": definition.definition_hash,
            "entity_id": request.entity_id,
            "input_hash": input_hash,
        }
    )


class RollingFeatureEngine:
    """Evaluate a pure calculator over the latest eligible chronological window."""

    def select_window(
        self,
        *,
        observations: Iterable[FeatureObservation],
        as_of: datetime,
        window: int,
    ) -> tuple[FeatureObservation, ...]:
        """Select latest known revisions behind an aware point-in-time boundary."""

        if window <= 0:
            raise ValueError("window must be positive")
        revisions = _select_point_in_time_revisions(observations, as_of=as_of)
        return tuple(revisions[-window:])

    def evaluate(
        self,
        *,
        definition: FeatureDefinition,
        request: FeatureRequest,
        observations: Iterable[FeatureObservation],
        calculator: FeatureCalculator,
    ) -> FeatureResult:
        """Compute a feature without exposing records unavailable at ``request.as_of``."""

        raw_window = self.select_window(
            observations=observations,
            as_of=request.as_of,
            window=definition.window,
        )
        input_hash = _input_hash(raw_window)
        cache_key = _cache_key(
            definition=definition,
            request=request,
            input_hash=input_hash,
        )
        present = tuple(item for item in raw_window if item.value is not None)
        has_missing = len(present) != len(raw_window)
        insufficient = len(present) < definition.min_observations
        if definition.missing_value_policy is MissingValuePolicy.REQUIRE_COMPLETE and has_missing:
            insufficient = True
            reason = "selected window contains missing values"
        elif insufficient:
            reason = (
                f"requires at least {definition.min_observations} observations; "
                f"received {len(present)}"
            )
        else:
            reason = None
        if insufficient:
            return FeatureResult(
                definition=definition,
                request=request,
                status=FeatureStatus.INSUFFICIENT_DATA,
                value=None,
                observation_count=len(present),
                window_start=raw_window[0].observed_at if raw_window else None,
                window_end=raw_window[-1].observed_at if raw_window else None,
                input_hash=input_hash,
                cache_key=cache_key,
                reason=reason,
            )

        values = tuple(item.value for item in present)
        if any(value is None for value in values):  # pragma: no cover - filtered above
            raise AssertionError("missing observation passed to calculator")
        calculated = finite_decimal(
            calculator(tuple(value for value in values if value is not None))
        )
        return FeatureResult(
            definition=definition,
            request=request,
            status=FeatureStatus.READY,
            value=calculated,
            observation_count=len(present),
            window_start=raw_window[0].observed_at,
            window_end=raw_window[-1].observed_at,
            input_hash=input_hash,
            cache_key=cache_key,
        )


def describe_cache_identity(result: FeatureResult) -> str:
    """Return a stable JSON description useful in manifests and diagnostics."""

    return json.dumps(
        {
            "cache_key": result.cache_key,
            "data_version": result.request.data_version,
            "definition_hash": result.definition.definition_hash,
            "entity_id": result.request.entity_id,
            "input_hash": result.input_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
