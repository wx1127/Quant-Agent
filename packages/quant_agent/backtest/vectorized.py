"""Fast lagged-signal factor and quantile research."""

import statistics
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ResearchObservation:
    signal_date: date
    execution_date: date
    instrument_id: str
    score: float
    forward_return: float
    benchmark_return: float

    def __post_init__(self) -> None:
        if self.execution_date <= self.signal_date:
            raise ValueError("execution must lag signal to prevent alignment leakage")


@dataclass(frozen=True, slots=True)
class QuantileResult:
    quantile: int
    count: int
    mean_return: float
    mean_excess_return: float


@dataclass(frozen=True, slots=True)
class VectorizedResearchReport:
    quantiles: tuple[QuantileResult, ...]
    top_minus_bottom: float
    information_coefficient: float


class VectorizedResearchEngine:
    def analyze(
        self,
        observations: list[ResearchObservation],
        *,
        quantile_count: int = 5,
    ) -> VectorizedResearchReport:
        if quantile_count < 2:
            raise ValueError("quantile_count must be at least two")
        if len(observations) < quantile_count:
            raise ValueError("insufficient observations for quantiles")
        ordered = sorted(observations, key=lambda item: (item.score, item.instrument_id))
        buckets: list[list[ResearchObservation]] = [[] for _ in range(quantile_count)]
        for index, item in enumerate(ordered):
            bucket = min(quantile_count - 1, index * quantile_count // len(ordered))
            buckets[bucket].append(item)
        results = tuple(
            QuantileResult(
                quantile=index + 1,
                count=len(bucket),
                mean_return=statistics.fmean(item.forward_return for item in bucket),
                mean_excess_return=statistics.fmean(
                    item.forward_return - item.benchmark_return for item in bucket
                ),
            )
            for index, bucket in enumerate(buckets)
        )
        scores = [item.score for item in observations]
        returns = [item.forward_return for item in observations]
        score_mean = statistics.fmean(scores)
        return_mean = statistics.fmean(returns)
        numerator = sum(
            (score - score_mean) * (value - return_mean)
            for score, value in zip(scores, returns, strict=True)
        )
        denominator = (
            sum((score - score_mean) ** 2 for score in scores)
            * sum((value - return_mean) ** 2 for value in returns)
        ) ** 0.5
        correlation = numerator / denominator if denominator else 0.0
        return VectorizedResearchReport(
            quantiles=results,
            top_minus_bottom=results[-1].mean_excess_return - results[0].mean_excess_return,
            information_coefficient=correlation,
        )
