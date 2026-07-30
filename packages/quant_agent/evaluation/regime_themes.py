"""Frozen-snapshot chronological replay metrics for P2 models."""

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date

from quant_agent.regime.models import MarketRegime


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    """One point-in-time regime output."""

    trade_date: date
    regime: MarketRegime


@dataclass(frozen=True, slots=True)
class ThemePrediction:
    """One point-in-time industry rank."""

    trade_date: date
    industry_id: str
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("theme rank must be positive")


@dataclass(frozen=True, slots=True)
class ForwardIndustryReturn:
    """Realized relative return labelled by its original signal date."""

    signal_date: date
    industry_id: str
    horizon: int
    relative_return: float

    def __post_init__(self) -> None:
        if self.horizon not in {5, 10, 20}:
            raise ValueError("forward horizon must be 5, 10 or 20")


@dataclass(frozen=True, slots=True)
class RegimeReplayReport:
    """Duration and transition stability metrics."""

    observation_count: int
    transition_count: int
    duration_by_regime: dict[str, int]
    longest_run: int


@dataclass(frozen=True, slots=True)
class ThemeReplayReport:
    """Top-K forward relative-return metrics."""

    evaluated_count_by_horizon: dict[int, int]
    precision_at_k_by_horizon: dict[int, float]
    mean_relative_return_by_horizon: dict[int, float]
    effective_examples: tuple[str, ...]
    failure_examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MarketThemeReplayReport:
    """Reproducible combined P2 replay report."""

    snapshot_version: str
    top_k: int
    regime: RegimeReplayReport
    themes: ThemeReplayReport
    content_hash: str


class RegimeThemeReplayEvaluator:
    """Evaluate chronological outputs without random splits or future joins."""

    def __init__(self, *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._top_k = top_k

    @staticmethod
    def _regime_report(
        observations: list[RegimeObservation],
    ) -> RegimeReplayReport:
        ordered = sorted(observations, key=lambda item: item.trade_date)
        if len({item.trade_date for item in ordered}) != len(ordered):
            raise ValueError("regime observations must have unique dates")
        durations: Counter[str] = Counter(item.regime.value for item in ordered)
        transitions = 0
        longest = 0
        current_run = 0
        previous: MarketRegime | None = None
        for item in ordered:
            if previous is not None and item.regime is not previous:
                transitions += 1
                current_run = 0
            current_run += 1
            longest = max(longest, current_run)
            previous = item.regime
        return RegimeReplayReport(
            observation_count=len(ordered),
            transition_count=transitions,
            duration_by_regime=dict(sorted(durations.items())),
            longest_run=longest,
        )

    def _theme_report(
        self,
        predictions: list[ThemePrediction],
        returns: list[ForwardIndustryReturn],
    ) -> ThemeReplayReport:
        keys = [(item.trade_date, item.industry_id, item.rank) for item in predictions]
        if len(set(keys)) != len(keys):
            raise ValueError("theme predictions must be unique")
        return_keys = [(item.signal_date, item.industry_id, item.horizon) for item in returns]
        if len(set(return_keys)) != len(return_keys):
            raise ValueError("forward returns must be unique")
        lookup = {
            (item.signal_date, item.industry_id, item.horizon): item.relative_return
            for item in returns
        }
        counts: dict[int, int] = {}
        precision: dict[int, float] = {}
        means: dict[int, float] = {}
        examples: list[tuple[float, str]] = []
        selected = sorted(
            (item for item in predictions if item.rank <= self._top_k),
            key=lambda item: (item.trade_date, item.rank, item.industry_id),
        )
        for horizon in (5, 10, 20):
            joined = [
                (
                    item,
                    lookup[(item.trade_date, item.industry_id, horizon)],
                )
                for item in selected
                if (item.trade_date, item.industry_id, horizon) in lookup
            ]
            values = [value for _item, value in joined]
            counts[horizon] = len(values)
            precision[horizon] = sum(value > 0 for value in values) / len(values) if values else 0.0
            means[horizon] = statistics.fmean(values) if values else 0.0
            examples.extend(
                (
                    value,
                    f"{item.trade_date.isoformat()} {item.industry_id} {horizon}d {value:.4f}",
                )
                for item, value in joined
            )
        effective = tuple(text for _value, text in sorted(examples, reverse=True)[:5])
        failures = tuple(text for _value, text in sorted(examples)[:5])
        return ThemeReplayReport(
            evaluated_count_by_horizon=counts,
            precision_at_k_by_horizon=precision,
            mean_relative_return_by_horizon=means,
            effective_examples=effective,
            failure_examples=failures,
        )

    def evaluate(
        self,
        *,
        snapshot_version: str,
        regimes: list[RegimeObservation],
        predictions: list[ThemePrediction],
        forward_returns: list[ForwardIndustryReturn],
    ) -> MarketThemeReplayReport:
        """Build a stable report tied to one immutable dataset snapshot."""

        if not snapshot_version.strip():
            raise ValueError("snapshot_version is required")
        regime_report = self._regime_report(regimes)
        theme_report = self._theme_report(predictions, forward_returns)
        payload = {
            "snapshot_version": snapshot_version,
            "top_k": self._top_k,
            "regime": asdict(regime_report),
            "themes": asdict(theme_report),
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return MarketThemeReplayReport(
            snapshot_version=snapshot_version,
            top_k=self._top_k,
            regime=regime_report,
            themes=theme_report,
            content_hash=digest,
        )
