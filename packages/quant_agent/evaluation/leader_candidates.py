"""Historical evaluation for frozen leader/candidate scores and execution labels.

Forward labels are joined only after point-in-time candidate snapshots and Top-K
selections are frozen.  This keeps future returns, costs, adverse excursions, and
fillability outside the feature-generation boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, localcontext
from enum import StrEnum
from itertools import pairwise
from typing import cast

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.leaders.candidates import CandidateSnapshot, CandidateTier
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash

LEADER_CANDIDATE_EVALUATION_CODE_VERSION = "leader-candidate-evaluation-code-v1"


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: tuple[Decimal, ...]) -> Decimal:
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


class LeaderCandidateEvaluationError(ValueError):
    """Raised when replay inputs are incomplete, ambiguous, or leakage-prone."""


class CandidateSliceKind(StrEnum):
    """Deterministic reporting dimensions required by the project plan."""

    OVERALL = "OVERALL"
    YEAR = "YEAR"
    REGIME = "REGIME"
    TIER = "TIER"
    YEAR_REGIME = "YEAR_REGIME"


@dataclass(frozen=True, slots=True)
class LeaderCandidateEvaluationConfig:
    """Versioned Top-K, horizon, success, and period-coverage semantics."""

    version: str = "leader-candidate-evaluation-config-v1"
    top_k: int = 10
    horizons: tuple[int, ...] = (5, 10, 20)
    positive_excess_return_threshold: Decimal = Decimal(0)
    minimum_years_for_coverage: int = 2
    minimum_regimes_for_coverage: int = 2

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("candidate evaluation config version must be non-empty")
        integer_values = (
            self.top_k,
            self.minimum_years_for_coverage,
            self.minimum_regimes_for_coverage,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integer_values
        ):
            raise ValueError("candidate evaluation counts must be positive integers")
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in self.horizons
            )
        ):
            raise ValueError("candidate evaluation horizons must be positive and increasing")
        canonical_decimal(
            self.positive_excess_return_threshold,
            field_name="positive excess-return threshold",
        )

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ForwardCandidateLabel:
    """One evaluation-only future path summary for a frozen candidate signal."""

    signal_date: date
    instrument_id: str
    horizon: int
    future_sessions: tuple[date, ...]
    gross_return: Decimal
    benchmark_return: Decimal
    estimated_round_trip_cost: Decimal
    maximum_adverse_excursion: Decimal
    fillable: bool
    unfillable_reason: str | None
    available_at: datetime
    data_version: str
    label_version: str
    source_hash: str

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("candidate label instrument_id must be non-empty")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("candidate label horizon must be a positive integer")
        if len(self.future_sessions) != self.horizon:
            raise ValueError("candidate label future_sessions must equal horizon")
        previous = self.signal_date
        for session in self.future_sessions:
            if session <= previous:
                raise ValueError("candidate label sessions must be strictly after the signal")
            previous = session
        for value in (
            self.gross_return,
            self.benchmark_return,
            self.estimated_round_trip_cost,
            self.maximum_adverse_excursion,
        ):
            canonical_decimal(value, field_name="candidate forward label value")
        if self.estimated_round_trip_cost < 0:
            raise ValueError("candidate label transaction cost cannot be negative")
        if self.maximum_adverse_excursion > 0:
            raise ValueError("maximum_adverse_excursion must be non-positive")
        if not isinstance(self.fillable, bool):
            raise ValueError("candidate label fillable must be boolean")
        if self.fillable != (self.unfillable_reason is None):
            raise ValueError("unfillable_reason must be present exactly when fillable is false")
        if self.unfillable_reason is not None and not self.unfillable_reason.strip():
            raise ValueError("unfillable_reason must be non-empty")
        ensure_aware(self.available_at)
        if self.available_at.astimezone(SHANGHAI_TZ).date() < self.future_sessions[-1]:
            raise ValueError("candidate label cannot be available before its future window ends")
        if not self.data_version.strip() or not self.label_version.strip():
            raise ValueError("candidate label versions must be non-empty")
        _validate_hash(self.source_hash, "candidate label source_hash")

    @property
    def gross_excess_return(self) -> Decimal:
        return self.gross_return - self.benchmark_return

    @property
    def net_excess_return(self) -> Decimal | None:
        if not self.fillable:
            return None
        return self.gross_excess_return - self.estimated_round_trip_cost

    @property
    def label_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class EvaluationDistribution:
    """Sorted empirical values and deterministic descriptive statistics."""

    sample_count: int
    values: tuple[Decimal, ...]
    minimum: Decimal | None
    median: Decimal | None
    mean: Decimal | None
    maximum: Decimal | None

    def __post_init__(self) -> None:
        if self.sample_count != len(self.values):
            raise ValueError("distribution sample_count must match values")
        if tuple(sorted(self.values)) != self.values:
            raise ValueError("distribution values must be sorted")
        for value in self.values:
            canonical_decimal(value, field_name="evaluation distribution value")
        statistics = (self.minimum, self.median, self.mean, self.maximum)
        if not self.values:
            if any(value is not None for value in statistics):
                raise ValueError("empty distribution cannot contain statistics")
            return
        if any(value is None for value in statistics):
            raise ValueError("non-empty distribution requires statistics")
        if (
            self.minimum != self.values[0]
            or self.maximum != self.values[-1]
            or self.median != _median(self.values)
            or self.mean != _mean(self.values)
        ):
            raise ValueError("distribution statistics must match values")


@dataclass(frozen=True, slots=True)
class CandidatePrecisionAtK:
    """Positive excess-return precision for selected Top-K observations."""

    top_k: int
    evaluated_count: int
    successful_count: int
    precision: Decimal | None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("precision top_k must be positive")
        if not 0 <= self.successful_count <= self.evaluated_count:
            raise ValueError("precision counts must be non-negative and ordered")
        expected = (
            _ratio(self.successful_count, self.evaluated_count) if self.evaluated_count else None
        )
        if self.precision != expected:
            raise ValueError("precision must match evaluation counts")


@dataclass(frozen=True, slots=True)
class CandidateMetricSlice:
    """One horizon and reporting-dimension slice with cost/fillability metrics."""

    kind: CandidateSliceKind
    horizon: int
    year: int | None
    regime: MarketRegime | None
    tier: CandidateTier | None
    selected_count: int
    fillable_count: int
    unfillable_count: int
    unfillable_ratio: Decimal
    gross_precision_at_k: CandidatePrecisionAtK
    net_precision_at_k: CandidatePrecisionAtK
    gross_excess_returns: EvaluationDistribution
    net_excess_returns: EvaluationDistribution
    maximum_adverse_excursions: EvaluationDistribution

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("candidate metric horizon must be positive")
        if self.selected_count != self.fillable_count + self.unfillable_count:
            raise ValueError("candidate fillable counts must equal selected_count")
        if self.unfillable_ratio != _ratio(self.unfillable_count, self.selected_count):
            raise ValueError("candidate unfillable_ratio must match counts")
        if self.gross_precision_at_k.evaluated_count != self.selected_count:
            raise ValueError("gross precision count must match selected_count")
        if self.net_precision_at_k.evaluated_count != self.fillable_count:
            raise ValueError("net precision count must match fillable_count")
        if self.gross_excess_returns.sample_count != self.selected_count:
            raise ValueError("gross distribution count must match selected_count")
        if self.net_excess_returns.sample_count != self.fillable_count:
            raise ValueError("net distribution count must match fillable_count")
        if self.maximum_adverse_excursions.sample_count != self.selected_count:
            raise ValueError("adverse-excursion count must match selected_count")
        dimensions = (
            self.year is not None,
            self.regime is not None,
            self.tier is not None,
        )
        expected_dimensions = {
            CandidateSliceKind.OVERALL: (False, False, False),
            CandidateSliceKind.YEAR: (True, False, False),
            CandidateSliceKind.REGIME: (False, True, False),
            CandidateSliceKind.TIER: (False, False, True),
            CandidateSliceKind.YEAR_REGIME: (True, True, False),
        }[self.kind]
        if dimensions != expected_dimensions:
            raise ValueError("candidate metric dimensions do not match slice kind")


@dataclass(frozen=True, slots=True)
class CandidatePeriodCoverage:
    """Observed years/regimes and whether multi-period interpretation is supported."""

    years: tuple[int, ...]
    regimes: tuple[MarketRegime, ...]
    minimum_years: int
    minimum_regimes: int
    sufficient: bool
    interpretation: str

    def __post_init__(self) -> None:
        if self.minimum_years < 1 or self.minimum_regimes < 1:
            raise ValueError("candidate coverage minimums must be positive")
        if tuple(sorted(set(self.years))) != self.years:
            raise ValueError("candidate coverage years must be unique and sorted")
        expected_regimes = tuple(value for value in MarketRegime if value in set(self.regimes))
        if self.regimes != expected_regimes:
            raise ValueError("candidate coverage regimes must use canonical enum order")
        expected = (
            len(self.years) >= self.minimum_years and len(self.regimes) >= self.minimum_regimes
        )
        if self.sufficient is not expected:
            raise ValueError("candidate period coverage flag does not match observed periods")
        if not self.interpretation.strip():
            raise ValueError("candidate coverage interpretation must be non-empty")


@dataclass(frozen=True, slots=True)
class LeaderCandidateEvaluationVersions:
    """Frozen signal, label, evaluator, and configuration identities."""

    signal_data_version: str
    candidate_feature_version: str
    candidate_config_hash: str
    label_data_version: str
    label_version: str
    evaluation_code_version: str
    evaluation_config_version: str
    evaluation_config_hash: str

    def __post_init__(self) -> None:
        versions = (
            self.signal_data_version,
            self.candidate_feature_version,
            self.label_data_version,
            self.label_version,
            self.evaluation_code_version,
            self.evaluation_config_version,
        )
        if any(not value.strip() for value in versions):
            raise ValueError("candidate evaluation versions must be non-empty")
        _validate_hash(self.candidate_config_hash, "candidate config hash")
        _validate_hash(self.evaluation_config_hash, "evaluation config hash")


@dataclass(frozen=True, slots=True)
class LeaderCandidateEvaluationReport:
    """Reproducible, descriptive P3-T08 evaluation artifact."""

    versions: LeaderCandidateEvaluationVersions
    period_coverage: CandidatePeriodCoverage
    metrics: tuple[CandidateMetricSlice, ...]
    signal_input_hash: str
    label_input_hash: str
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.signal_input_hash, "signal input hash"),
            (self.label_input_hash, "label input hash"),
            (self.input_hash, "evaluation input hash"),
            (self.result_hash, "evaluation result hash"),
        ):
            _validate_hash(value, field_name)
        if not self.metrics:
            raise ValueError("candidate evaluation report requires metrics")
        expected_input_hash = stable_hash(
            {
                "config_hash": self.versions.evaluation_config_hash,
                "label_input_hash": self.label_input_hash,
                "signal_input_hash": self.signal_input_hash,
                "versions": asdict(self.versions),
            }
        )
        if self.input_hash != expected_input_hash:
            raise ValueError("candidate evaluation input hash does not match inputs")
        expected_result_hash = stable_hash(
            {
                "input_hash": self.input_hash,
                "metrics": [asdict(item) for item in self.metrics],
                "period_coverage": asdict(self.period_coverage),
                "versions": asdict(self.versions),
            }
        )
        if self.result_hash != expected_result_hash:
            raise ValueError("candidate evaluation result hash does not match report")

    def identity_payload(self) -> dict[str, str]:
        return {
            "config_hash": self.versions.evaluation_config_hash,
            "config_version": self.versions.evaluation_config_version,
            "input_hash": self.input_hash,
            "label_input_hash": self.label_input_hash,
            "result_hash": self.result_hash,
            "signal_input_hash": self.signal_input_hash,
        }


@dataclass(frozen=True, slots=True)
class _Prediction:
    signal_date: date
    instrument_id: str
    rank: int
    tier: CandidateTier
    regime: MarketRegime


@dataclass(frozen=True, slots=True)
class _PreparedSignals:
    snapshots: tuple[CandidateSnapshot, ...]
    predictions: tuple[_Prediction, ...]
    signal_input_hash: str
    data_version: str
    feature_version: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class _Observation:
    prediction: _Prediction
    label: ForwardCandidateLabel


def _prepare_signals(
    snapshots: Iterable[CandidateSnapshot],
    config: LeaderCandidateEvaluationConfig,
) -> _PreparedSignals:
    frozen = tuple(snapshots)
    if not frozen:
        raise LeaderCandidateEvaluationError("candidate evaluation requires signal snapshots")
    first = frozen[0]
    identity = (first.data_version, first.feature_version, first.config_hash)
    previous_date: date | None = None
    predictions: list[_Prediction] = []
    for snapshot in frozen:
        if not isinstance(snapshot, CandidateSnapshot):
            raise LeaderCandidateEvaluationError(
                "candidate signals must be immutable CandidateSnapshot values"
            )
        if previous_date is not None and snapshot.session_date <= previous_date:
            raise LeaderCandidateEvaluationError(
                "candidate signal dates must be strictly increasing"
            )
        if (snapshot.data_version, snapshot.feature_version, snapshot.config_hash) != identity:
            raise LeaderCandidateEvaluationError(
                "candidate signal history mixes data, feature, or config versions"
            )
        selected = tuple(
            item
            for item in snapshot.candidates
            if item.eligible_rank is not None and item.eligible_rank <= config.top_k
        )
        ranks = tuple(cast(int, item.eligible_rank) for item in selected)
        if tuple(sorted(ranks)) != tuple(range(1, len(ranks) + 1)):
            raise LeaderCandidateEvaluationError("candidate Top-K ranks contain a gap")
        for item in sorted(selected, key=lambda value: cast(int, value.eligible_rank)):
            if item.eligible_rank is None:  # pragma: no cover - filtered above
                continue
            predictions.append(
                _Prediction(
                    signal_date=snapshot.session_date,
                    instrument_id=item.instrument_id,
                    rank=item.eligible_rank,
                    tier=item.tier,
                    regime=snapshot.regime,
                )
            )
        previous_date = snapshot.session_date
    if not predictions:
        raise LeaderCandidateEvaluationError("candidate evaluation requires Top-K predictions")
    signal_input_hash = stable_hash(
        {
            "snapshots": [
                {
                    "input_hash": item.input_hash,
                    "result_hash": item.result_hash,
                    "session_date": item.session_date,
                }
                for item in frozen
            ]
        }
    )
    return _PreparedSignals(
        snapshots=frozen,
        predictions=tuple(predictions),
        signal_input_hash=signal_input_hash,
        data_version=identity[0],
        feature_version=identity[1],
        config_hash=identity[2],
    )


def _prepare_labels(
    labels: Iterable[ForwardCandidateLabel],
    prepared: _PreparedSignals,
    config: LeaderCandidateEvaluationConfig,
) -> tuple[dict[tuple[date, str, int], ForwardCandidateLabel], str, str, str]:
    frozen = tuple(labels)
    if not frozen:
        raise LeaderCandidateEvaluationError("candidate evaluation requires forward labels")
    expected = {
        (prediction.signal_date, prediction.instrument_id, horizon)
        for prediction in prepared.predictions
        for horizon in config.horizons
    }
    signal_as_of = {item.session_date: item.as_of for item in prepared.snapshots}
    by_key: dict[tuple[date, str, int], ForwardCandidateLabel] = {}
    calendars: dict[tuple[date, int], tuple[date, ...]] = {}
    by_prediction: dict[tuple[date, str], list[ForwardCandidateLabel]] = {}
    previous_key: tuple[date, str, int] | None = None
    label_identity: tuple[str, str] | None = None
    for label in frozen:
        if not isinstance(label, ForwardCandidateLabel):
            raise LeaderCandidateEvaluationError(
                "labels must be immutable ForwardCandidateLabel values"
            )
        key = (label.signal_date, label.instrument_id, label.horizon)
        if key in by_key:
            raise LeaderCandidateEvaluationError("candidate labels contain a duplicate key")
        if previous_key is not None and key <= previous_key:
            raise LeaderCandidateEvaluationError(
                "candidate labels must use canonical chronological key order"
            )
        previous_key = key
        if key not in expected:
            raise LeaderCandidateEvaluationError(
                "candidate label does not match a configured Top-K prediction"
            )
        if label.available_at <= signal_as_of[label.signal_date]:
            raise LeaderCandidateEvaluationError(
                "candidate future label must become available after the signal snapshot"
            )
        identity = (label.data_version, label.label_version)
        if label_identity is None:
            label_identity = identity
        elif identity != label_identity:
            raise LeaderCandidateEvaluationError("candidate labels mix data or label-code versions")
        calendar_key = (label.signal_date, label.horizon)
        existing_calendar = calendars.get(calendar_key)
        if existing_calendar is not None and existing_calendar != label.future_sessions:
            raise LeaderCandidateEvaluationError(
                "candidate labels on one signal/horizon use different future sessions"
            )
        calendars[calendar_key] = label.future_sessions
        by_prediction.setdefault((label.signal_date, label.instrument_id), []).append(label)
        by_key[key] = label
    missing = sorted(expected - set(by_key))
    if missing:
        raise LeaderCandidateEvaluationError(f"missing candidate forward label: {missing[0]}")
    for grouped in by_prediction.values():
        ordered = sorted(grouped, key=lambda item: item.horizon)
        for shorter, longer in pairwise(ordered):
            if shorter.future_sessions != longer.future_sessions[: shorter.horizon]:
                raise LeaderCandidateEvaluationError(
                    "candidate label horizons must share nested future-session prefixes"
                )
    if label_identity is None:  # pragma: no cover - non-empty input
        raise AssertionError("candidate label identity was not initialized")
    label_input_hash = stable_hash(
        {"labels": [{"key": key, "label_hash": by_key[key].label_hash} for key in sorted(by_key)]}
    )
    return by_key, label_identity[0], label_identity[1], label_input_hash


def _distribution(values: Iterable[Decimal]) -> EvaluationDistribution:
    ordered = tuple(sorted(values))
    if not ordered:
        return EvaluationDistribution(0, (), None, None, None, None)
    return EvaluationDistribution(
        sample_count=len(ordered),
        values=ordered,
        minimum=ordered[0],
        median=_median(ordered),
        mean=_mean(ordered),
        maximum=ordered[-1],
    )


def _precision(
    values: tuple[Decimal, ...],
    config: LeaderCandidateEvaluationConfig,
) -> CandidatePrecisionAtK:
    successes = sum(value > config.positive_excess_return_threshold for value in values)
    return CandidatePrecisionAtK(
        top_k=config.top_k,
        evaluated_count=len(values),
        successful_count=successes,
        precision=_ratio(successes, len(values)) if values else None,
    )


def _metric_slice(
    *,
    kind: CandidateSliceKind,
    horizon: int,
    observations: tuple[_Observation, ...],
    config: LeaderCandidateEvaluationConfig,
    year: int | None = None,
    regime: MarketRegime | None = None,
    tier: CandidateTier | None = None,
) -> CandidateMetricSlice:
    gross = tuple(item.label.gross_excess_return for item in observations)
    net = tuple(
        value for item in observations if (value := item.label.net_excess_return) is not None
    )
    adverse = tuple(item.label.maximum_adverse_excursion for item in observations)
    fillable = len(net)
    selected = len(observations)
    return CandidateMetricSlice(
        kind=kind,
        horizon=horizon,
        year=year,
        regime=regime,
        tier=tier,
        selected_count=selected,
        fillable_count=fillable,
        unfillable_count=selected - fillable,
        unfillable_ratio=_ratio(selected - fillable, selected),
        gross_precision_at_k=_precision(gross, config),
        net_precision_at_k=_precision(net, config),
        gross_excess_returns=_distribution(gross),
        net_excess_returns=_distribution(net),
        maximum_adverse_excursions=_distribution(adverse),
    )


def _metrics(
    prepared: _PreparedSignals,
    labels: dict[tuple[date, str, int], ForwardCandidateLabel],
    config: LeaderCandidateEvaluationConfig,
) -> tuple[CandidateMetricSlice, ...]:
    result: list[CandidateMetricSlice] = []
    for horizon in config.horizons:
        observations = tuple(
            _Observation(
                prediction=prediction,
                label=labels[(prediction.signal_date, prediction.instrument_id, horizon)],
            )
            for prediction in prepared.predictions
        )
        result.append(
            _metric_slice(
                kind=CandidateSliceKind.OVERALL,
                horizon=horizon,
                observations=observations,
                config=config,
            )
        )
        years = sorted({item.prediction.signal_date.year for item in observations})
        regimes = tuple(
            value
            for value in MarketRegime
            if any(item.prediction.regime is value for item in observations)
        )
        tiers = tuple(
            value
            for value in (CandidateTier.A, CandidateTier.B, CandidateTier.WATCH)
            if any(item.prediction.tier is value for item in observations)
        )
        for year in years:
            selected = tuple(
                item for item in observations if item.prediction.signal_date.year == year
            )
            result.append(
                _metric_slice(
                    kind=CandidateSliceKind.YEAR,
                    horizon=horizon,
                    year=year,
                    observations=selected,
                    config=config,
                )
            )
        for regime in regimes:
            selected = tuple(item for item in observations if item.prediction.regime is regime)
            result.append(
                _metric_slice(
                    kind=CandidateSliceKind.REGIME,
                    horizon=horizon,
                    regime=regime,
                    observations=selected,
                    config=config,
                )
            )
        for tier in tiers:
            selected = tuple(item for item in observations if item.prediction.tier is tier)
            result.append(
                _metric_slice(
                    kind=CandidateSliceKind.TIER,
                    horizon=horizon,
                    tier=tier,
                    observations=selected,
                    config=config,
                )
            )
        for year in years:
            for regime in regimes:
                selected = tuple(
                    item
                    for item in observations
                    if item.prediction.signal_date.year == year and item.prediction.regime is regime
                )
                if selected:
                    result.append(
                        _metric_slice(
                            kind=CandidateSliceKind.YEAR_REGIME,
                            horizon=horizon,
                            year=year,
                            regime=regime,
                            observations=selected,
                            config=config,
                        )
                    )
    return tuple(result)


def evaluate_leader_candidates(
    snapshots: Iterable[CandidateSnapshot],
    labels: Iterable[ForwardCandidateLabel],
    config: LeaderCandidateEvaluationConfig | None = None,
) -> LeaderCandidateEvaluationReport:
    """Evaluate frozen candidate ranks against future gross/net excess returns."""

    active_config = config or LeaderCandidateEvaluationConfig()
    prepared = _prepare_signals(snapshots, active_config)
    labels_by_key, label_data_version, label_version, label_input_hash = _prepare_labels(
        labels,
        prepared,
        active_config,
    )
    metrics = _metrics(prepared, labels_by_key, active_config)
    years = tuple(sorted({item.signal_date.year for item in prepared.predictions}))
    regimes = tuple(
        value
        for value in MarketRegime
        if any(item.regime is value for item in prepared.predictions)
    )
    sufficient = (
        len(years) >= active_config.minimum_years_for_coverage
        and len(regimes) >= active_config.minimum_regimes_for_coverage
    )
    coverage = CandidatePeriodCoverage(
        years=years,
        regimes=regimes,
        minimum_years=active_config.minimum_years_for_coverage,
        minimum_regimes=active_config.minimum_regimes_for_coverage,
        sufficient=sufficient,
        interpretation=(
            "descriptive multi-period coverage reached; no causal or guaranteed-return claim"
            if sufficient
            else (
                "insufficient period coverage; results are descriptive and cannot "
                "establish validity"
            )
        ),
    )
    versions = LeaderCandidateEvaluationVersions(
        signal_data_version=prepared.data_version,
        candidate_feature_version=prepared.feature_version,
        candidate_config_hash=prepared.config_hash,
        label_data_version=label_data_version,
        label_version=label_version,
        evaluation_code_version=LEADER_CANDIDATE_EVALUATION_CODE_VERSION,
        evaluation_config_version=active_config.version,
        evaluation_config_hash=active_config.config_hash,
    )
    input_hash = stable_hash(
        {
            "config_hash": active_config.config_hash,
            "label_input_hash": label_input_hash,
            "signal_input_hash": prepared.signal_input_hash,
            "versions": asdict(versions),
        }
    )
    result_hash = stable_hash(
        {
            "input_hash": input_hash,
            "metrics": [asdict(item) for item in metrics],
            "period_coverage": asdict(coverage),
            "versions": asdict(versions),
        }
    )
    return LeaderCandidateEvaluationReport(
        versions=versions,
        period_coverage=coverage,
        metrics=metrics,
        signal_input_hash=prepared.signal_input_hash,
        label_input_hash=label_input_hash,
        input_hash=input_hash,
        result_hash=result_hash,
    )


evaluate_leader_candidate_history = evaluate_leader_candidates


__all__ = [
    "LEADER_CANDIDATE_EVALUATION_CODE_VERSION",
    "CandidateMetricSlice",
    "CandidatePeriodCoverage",
    "CandidatePrecisionAtK",
    "CandidateSliceKind",
    "EvaluationDistribution",
    "ForwardCandidateLabel",
    "LeaderCandidateEvaluationConfig",
    "LeaderCandidateEvaluationError",
    "LeaderCandidateEvaluationReport",
    "LeaderCandidateEvaluationVersions",
    "evaluate_leader_candidate_history",
    "evaluate_leader_candidates",
]
