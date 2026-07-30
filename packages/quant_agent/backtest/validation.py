"""Chronological splits, immutable parameters and walk-forward records."""

import hashlib
import json
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class TimeSeriesSplit:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    out_of_sample_start: date
    out_of_sample_end: date

    def __post_init__(self) -> None:
        if not (
            self.train_start
            <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.out_of_sample_start
            <= self.out_of_sample_end
        ):
            raise ValueError("time-series split periods must be ordered and disjoint")


@dataclass(frozen=True, slots=True)
class FrozenParameters:
    strategy_id: str
    version: str
    values: Mapping[str, Any]
    frozen_on: date
    source_split: TimeSeriesSplit

    def __post_init__(self) -> None:
        if not (
            self.source_split.validation_end
            <= self.frozen_on
            <= self.source_split.out_of_sample_start
        ):
            raise ValueError("parameters must freeze after validation and before out-of-sample")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "strategy_id": self.strategy_id,
                    "version": self.version,
                    "values": dict(self.values),
                    "frozen_on": self.frozen_on.isoformat(),
                    "source_split": {
                        key: value.isoformat() for key, value in asdict(self.source_split).items()
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


class ParameterRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], FrozenParameters] = {}

    def freeze(self, parameters: FrozenParameters) -> None:
        key = (parameters.strategy_id, parameters.version)
        existing = self._items.get(key)
        if existing is not None and existing.content_hash != parameters.content_hash:
            raise ValueError("frozen parameter version cannot be changed")
        self._items[key] = parameters

    def get(self, strategy_id: str, version: str) -> FrozenParameters:
        return self._items[(strategy_id, version)]


class WalkForwardSplitter:
    """Generate ordered validation and out-of-sample windows."""

    def generate(
        self,
        dates: list[date],
        *,
        train_size: int,
        validation_size: int,
        out_of_sample_size: int,
        step_size: int | None = None,
    ) -> tuple[TimeSeriesSplit, ...]:
        sizes = (train_size, validation_size, out_of_sample_size)
        if any(size < 1 for size in sizes):
            raise ValueError("walk-forward window sizes must be positive")
        ordered = sorted(set(dates))
        required = sum(sizes)
        step = step_size or out_of_sample_size
        if step < 1:
            raise ValueError("walk-forward step must be positive")
        splits = []
        start = 0
        while start + required <= len(ordered):
            train_end = start + train_size - 1
            validation_start = train_end + 1
            validation_end = validation_start + validation_size - 1
            oos_start = validation_end + 1
            oos_end = oos_start + out_of_sample_size - 1
            splits.append(
                TimeSeriesSplit(
                    train_start=ordered[start],
                    train_end=ordered[train_end],
                    validation_start=ordered[validation_start],
                    validation_end=ordered[validation_end],
                    out_of_sample_start=ordered[oos_start],
                    out_of_sample_end=ordered[oos_end],
                )
            )
            start += step
        return tuple(splits)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    parameter_version: str
    split: TimeSeriesSplit
    out_of_sample_metric: float
    perturbation_metrics: tuple[float, ...]
    stability: float


def parameter_stability(base_metric: float, perturbations: list[float]) -> float:
    """Return [0, 1], penalizing dispersion and collapse around the base."""

    if not perturbations:
        raise ValueError("parameter stability requires perturbation results")
    scale = max(abs(base_metric), 1e-9)
    mean_deviation = statistics.fmean(abs(value - base_metric) / scale for value in perturbations)
    return max(0.0, min(1.0, 1.0 - mean_deviation))
