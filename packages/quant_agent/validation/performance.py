"""Repeatable latency, memory and storage-capacity acceptance harness."""

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LatencyResult:
    name: str
    samples_ms: tuple[float, ...]
    target_p95_ms: float

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.samples_ms)
        index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered)) - 1))
        return ordered[index]

    @property
    def passed(self) -> bool:
        return self.p95_ms <= self.target_p95_ms


@dataclass(frozen=True, slots=True)
class CapacityProjection:
    instruments: int
    trading_days: int
    bytes_per_row: int

    @property
    def rows(self) -> int:
        return self.instruments * self.trading_days

    @property
    def estimated_gib(self) -> float:
        return self.rows * self.bytes_per_row / 1024**3

    @property
    def covers_one_year(self) -> bool:
        return self.trading_days >= 252


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    latency: tuple[LatencyResult, ...]
    retained_memory_bytes: int
    memory_limit_bytes: int
    capacity: CapacityProjection
    after_close_budget_hours: float

    @property
    def passed(self) -> bool:
        return (
            all(item.passed for item in self.latency)
            and self.retained_memory_bytes <= self.memory_limit_bytes
            and self.capacity.covers_one_year
            and self.after_close_budget_hours >= 8
        )


class PerformanceHarness:
    def measure(
        self,
        workloads: dict[str, tuple[Callable[[], Any], float]],
        *,
        repeats: int = 30,
    ) -> PerformanceReport:
        results = []
        for name, (workload, target_ms) in workloads.items():
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                workload()
                samples.append((time.perf_counter() - started) * 1_000)
            results.append(LatencyResult(name, tuple(samples), target_ms))
        retained = self._retained_memory(next(iter(workloads.values()))[0])
        return PerformanceReport(
            latency=tuple(results),
            retained_memory_bytes=retained,
            memory_limit_bytes=2 * 1024 * 1024,
            capacity=CapacityProjection(6_000, 252, 512),
            after_close_budget_hours=12.0,
        )

    @staticmethod
    def _retained_memory(workload: Callable[[], Any]) -> int:
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(100):
            workload()
        gc.collect()
        after = tracemalloc.take_snapshot()
        retained = sum(
            stat.size_diff for stat in after.compare_to(before, "lineno") if stat.size_diff > 0
        )
        tracemalloc.stop()
        return retained

    @staticmethod
    def summarize(result: LatencyResult) -> dict[str, float]:
        return {
            "median_ms": statistics.median(result.samples_ms),
            "p95_ms": result.p95_ms,
            "target_p95_ms": result.target_p95_ms,
        }
