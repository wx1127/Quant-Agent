"""Versioned rolling-feature primitives with point-in-time safety."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from quant_agent.core.time import ensure_aware


class MissingValuePolicy(StrEnum):
    """Consistent behavior for missing observations."""

    FAIL = "FAIL"
    DROP = "DROP"


class InsufficientObservationsError(ValueError):
    """A rolling window cannot satisfy its declared minimum."""


@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    """One value with event date and legal availability time."""

    observed_on: date
    available_at: datetime
    value: float | None

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("feature observation must be finite")


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Version and reproducibility metadata for one feature."""

    name: str
    version: str
    dataset_version: str
    window: int
    min_observations: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be positive")
        if not 1 <= self.min_observations <= self.window:
            raise ValueError("min_observations must be inside the window")
        if not self.dataset_version:
            raise ValueError("dataset_version is required")

    def cache_key(self, as_of: datetime) -> str:
        """Return a stable key based on frozen data and feature parameters."""

        ensure_aware(as_of)
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "dataset_version": self.dataset_version,
                "window": self.window,
                "min_observations": self.min_observations,
                "parameters": self.parameters,
                "as_of": as_of.isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """One deterministic feature result."""

    metadata: FeatureMetadata
    as_of: datetime
    value: float
    observations_used: int
    cache_key: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not math.isfinite(self.value):
            raise ValueError("feature result must be finite")


class RollingCalculator:
    """Point-in-time-safe rolling statistics."""

    @staticmethod
    def prepare(
        points: list[TimeSeriesPoint],
        *,
        as_of: datetime,
        metadata: FeatureMetadata,
        missing_policy: MissingValuePolicy = MissingValuePolicy.FAIL,
    ) -> tuple[TimeSeriesPoint, ...]:
        """Filter unavailable/future data, deduplicate dates and apply a window."""

        ensure_aware(as_of)
        latest_by_date: dict[date, TimeSeriesPoint] = {}
        for point in points:
            if point.observed_on > as_of.date() or point.available_at > as_of:
                continue
            current = latest_by_date.get(point.observed_on)
            if current is None or point.available_at > current.available_at:
                latest_by_date[point.observed_on] = point
        ordered = [latest_by_date[key] for key in sorted(latest_by_date)]
        windowed = ordered[-metadata.window :]
        if missing_policy is MissingValuePolicy.FAIL and any(
            point.value is None for point in windowed
        ):
            raise ValueError(f"{metadata.name} contains missing values")
        prepared = tuple(point for point in windowed if point.value is not None)
        if len(prepared) < metadata.min_observations:
            raise InsufficientObservationsError(
                f"{metadata.name} requires {metadata.min_observations} observations; "
                f"got {len(prepared)}"
            )
        return prepared

    @staticmethod
    def _values(points: tuple[TimeSeriesPoint, ...]) -> list[float]:
        return [float(point.value) for point in points if point.value is not None]

    @classmethod
    def mean(
        cls,
        points: list[TimeSeriesPoint],
        *,
        as_of: datetime,
        metadata: FeatureMetadata,
        missing_policy: MissingValuePolicy = MissingValuePolicy.FAIL,
    ) -> FeatureResult:
        prepared = cls.prepare(
            points,
            as_of=as_of,
            metadata=metadata,
            missing_policy=missing_policy,
        )
        return FeatureResult(
            metadata=metadata,
            as_of=as_of,
            value=statistics.fmean(cls._values(prepared)),
            observations_used=len(prepared),
            cache_key=metadata.cache_key(as_of),
        )

    @classmethod
    def total_return(
        cls,
        points: list[TimeSeriesPoint],
        *,
        as_of: datetime,
        metadata: FeatureMetadata,
        missing_policy: MissingValuePolicy = MissingValuePolicy.FAIL,
    ) -> FeatureResult:
        prepared = cls.prepare(
            points,
            as_of=as_of,
            metadata=metadata,
            missing_policy=missing_policy,
        )
        values = cls._values(prepared)
        if values[0] == 0:
            raise ValueError("return start value cannot be zero")
        return FeatureResult(
            metadata=metadata,
            as_of=as_of,
            value=values[-1] / values[0] - 1.0,
            observations_used=len(prepared),
            cache_key=metadata.cache_key(as_of),
        )

    @classmethod
    def slope(
        cls,
        points: list[TimeSeriesPoint],
        *,
        as_of: datetime,
        metadata: FeatureMetadata,
        normalize: bool = True,
    ) -> FeatureResult:
        prepared = cls.prepare(points, as_of=as_of, metadata=metadata)
        values = cls._values(prepared)
        if normalize:
            if values[0] == 0:
                raise ValueError("slope normalization value cannot be zero")
            values = [value / values[0] for value in values]
        x_mean = (len(values) - 1) / 2
        y_mean = statistics.fmean(values)
        numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
        denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
        if denominator == 0:
            raise InsufficientObservationsError("slope requires at least two observations")
        return FeatureResult(
            metadata=metadata,
            as_of=as_of,
            value=numerator / denominator,
            observations_used=len(prepared),
            cache_key=metadata.cache_key(as_of),
        )

    @classmethod
    def standard_deviation(
        cls,
        points: list[TimeSeriesPoint],
        *,
        as_of: datetime,
        metadata: FeatureMetadata,
    ) -> FeatureResult:
        prepared = cls.prepare(points, as_of=as_of, metadata=metadata)
        values = cls._values(prepared)
        value = statistics.stdev(values) if len(values) > 1 else 0.0
        return FeatureResult(
            metadata=metadata,
            as_of=as_of,
            value=value,
            observations_used=len(prepared),
            cache_key=metadata.cache_key(as_of),
        )

    @staticmethod
    def percentile_rank(values: list[float], current: float) -> float:
        """Return an inclusive empirical percentile in [0, 1]."""

        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            raise InsufficientObservationsError("percentile rank requires observations")
        return sum(value <= current for value in finite) / len(finite)


def clamp_score(value: float, lower: float, upper: float) -> float:
    """Linearly map a value to [0, 100] with clipping."""

    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return max(0.0, min(100.0, (value - lower) / (upper - lower) * 100.0))
