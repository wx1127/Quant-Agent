"""Strict-time replay metrics for leader and candidate rankings."""

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import date

from quant_agent.regime.models import MarketRegime


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    signal_date: date
    instrument_id: str
    rank: int
    tier: str
    regime: MarketRegime
    tradable: bool

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("candidate rank must be positive")


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    signal_date: date
    instrument_id: str
    horizon: int
    excess_return: float
    maximum_adverse_excursion: float
    transaction_cost: float
    executable: bool

    def __post_init__(self) -> None:
        if self.horizon not in {5, 10, 20}:
            raise ValueError("candidate horizon must be 5, 10 or 20")
        if self.transaction_cost < 0:
            raise ValueError("transaction cost cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateSegmentMetrics:
    year: int
    regime: str
    horizon: int
    tier: str
    sample_count: int
    precision_at_k: float
    mean_return_before_cost: float
    mean_return_after_cost: float
    mean_maximum_adverse_excursion: float
    unexecutable_ratio: float


@dataclass(frozen=True, slots=True)
class CandidateReplayReport:
    snapshot_version: str
    top_k: int
    segments: tuple[CandidateSegmentMetrics, ...]
    content_hash: str


class CandidateReplayEvaluator:
    def __init__(self, *, top_k: int = 10) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._top_k = top_k

    def evaluate(
        self,
        *,
        snapshot_version: str,
        predictions: list[CandidatePrediction],
        outcomes: list[CandidateOutcome],
    ) -> CandidateReplayReport:
        if not snapshot_version.strip():
            raise ValueError("snapshot_version is required")
        prediction_keys = [
            (item.signal_date, item.instrument_id, item.rank) for item in predictions
        ]
        outcome_keys = [(item.signal_date, item.instrument_id, item.horizon) for item in outcomes]
        if len(set(prediction_keys)) != len(prediction_keys):
            raise ValueError("candidate predictions must be unique")
        if len(set(outcome_keys)) != len(outcome_keys):
            raise ValueError("candidate outcomes must be unique")
        lookup = {(item.signal_date, item.instrument_id, item.horizon): item for item in outcomes}
        groups: dict[
            tuple[int, str, int, str], list[tuple[CandidatePrediction, CandidateOutcome]]
        ] = {}
        for prediction in predictions:
            if prediction.rank > self._top_k:
                continue
            for horizon in (5, 10, 20):
                outcome = lookup.get((prediction.signal_date, prediction.instrument_id, horizon))
                if outcome is not None:
                    key = (
                        prediction.signal_date.year,
                        prediction.regime.value,
                        horizon,
                        prediction.tier,
                    )
                    groups.setdefault(key, []).append((prediction, outcome))
        segments = []
        for (year, regime, horizon, tier), rows in sorted(groups.items()):
            values = [outcome.excess_return for _prediction, outcome in rows]
            segments.append(
                CandidateSegmentMetrics(
                    year=year,
                    regime=regime,
                    horizon=horizon,
                    tier=tier,
                    sample_count=len(rows),
                    precision_at_k=sum(value > 0 for value in values) / len(values),
                    mean_return_before_cost=statistics.fmean(values),
                    mean_return_after_cost=statistics.fmean(
                        outcome.excess_return - outcome.transaction_cost
                        for _prediction, outcome in rows
                    ),
                    mean_maximum_adverse_excursion=statistics.fmean(
                        outcome.maximum_adverse_excursion for _prediction, outcome in rows
                    ),
                    unexecutable_ratio=sum(not outcome.executable for _prediction, outcome in rows)
                    / len(rows),
                )
            )
        payload = {
            "snapshot_version": snapshot_version,
            "top_k": self._top_k,
            "segments": [asdict(item) for item in segments],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CandidateReplayReport(
            snapshot_version=snapshot_version,
            top_k=self._top_k,
            segments=tuple(segments),
            content_hash=digest,
        )
