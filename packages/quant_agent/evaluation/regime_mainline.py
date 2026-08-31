"""Deterministic historical evaluation for frozen regime and mainline outputs.

The evaluator deliberately accepts already-computed point-in-time signals.  Forward
returns are a separate label contract and are joined only after signal validation and
selection, so a label can never become an input to regime or mainline generation.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum
from itertools import pairwise

from quant_agent.features.core.identity import canonical_decimal
from quant_agent.features.mainline import MainlineSnapshot, MainlineState
from quant_agent.regime import MarketRegime, RegimeTransitionResult
from quant_agent.regime.contracts import stable_hash

REGIME_MAINLINE_EVALUATION_CODE_VERSION = "regime-mainline-evaluation-code-v1"
_SHA256_LENGTH = 64


class RegimeMainlineEvaluationError(ValueError):
    """Raised when replay inputs cannot be evaluated without ambiguity or leakage."""


class EvaluationPeriod(StrEnum):
    """Whether the point-in-time mainline lifecycle signal was active or inactive."""

    EFFECTIVE = "EFFECTIVE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class RegimeMainlineEvaluationConfig:
    """Versioned definition of prediction, horizon, and success semantics."""

    version: str = "regime-mainline-evaluation-config-v1"
    top_k: int = 5
    horizons: tuple[int, ...] = (5, 10, 20)
    positive_return_threshold: Decimal = Decimal(0)
    effective_states: tuple[MainlineState, ...] = (
        MainlineState.CONFIRMED,
        MainlineState.CROWDED,
    )

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("evaluation config version must be non-empty")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("evaluation top_k must be a positive integer")
        if (
            not self.horizons
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in self.horizons
            )
            or tuple(sorted(set(self.horizons))) != self.horizons
        ):
            raise ValueError("evaluation horizons must be unique, positive, and increasing")
        canonical_decimal(
            self.positive_return_threshold,
            field_name="positive return threshold",
        )
        if (
            not self.effective_states
            or len(set(self.effective_states)) != len(self.effective_states)
            or any(not isinstance(value, MainlineState) for value in self.effective_states)
        ):
            raise ValueError("effective mainline states must be non-empty and unique")

    @property
    def config_hash(self) -> str:
        """Return a stable digest over all evaluation choices."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ForwardRelativeReturnLabel:
    """One frozen, evaluation-only future industry relative-return label.

    ``future_sessions`` makes the label window auditable.  It must contain exactly
    ``horizon`` strictly increasing sessions, all later than ``signal_date``.
    """

    signal_date: date
    industry_id: str
    horizon: int
    future_sessions: tuple[date, ...]
    relative_return: Decimal
    data_version: str
    label_version: str
    source_hash: str

    def __post_init__(self) -> None:
        if type(self.signal_date) is not date:
            raise ValueError("label signal_date must be a date")
        if not self.industry_id.strip():
            raise ValueError("label industry_id must be non-empty")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("label horizon must be a positive integer")
        if len(self.future_sessions) != self.horizon:
            raise ValueError("label future_sessions must contain exactly horizon sessions")
        previous = self.signal_date
        for session in self.future_sessions:
            if type(session) is not date:
                raise ValueError("label future sessions must be dates")
            if session <= previous:
                raise ValueError(
                    "future label window must be strictly increasing and cannot overlap "
                    "the signal date"
                )
            previous = session
        canonical_decimal(self.relative_return, field_name="future relative return")
        if not self.data_version.strip() or not self.label_version.strip():
            raise ValueError("label data and code versions must be non-empty")
        if len(self.source_hash) != _SHA256_LENGTH or any(
            value not in "0123456789abcdef" for value in self.source_hash
        ):
            raise ValueError("label source_hash must be a SHA-256 digest")

    @property
    def window_start_date(self) -> date:
        """Return the first future session included in the label."""

        return self.future_sessions[0]

    @property
    def window_end_date(self) -> date:
        """Return the last future session included in the label."""

        return self.future_sessions[-1]

    @property
    def label_hash(self) -> str:
        """Return a stable digest over the complete immutable label."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class RegimeDurationSegment:
    """One contiguous observed-session run of the same stabilized market state."""

    regime: MarketRegime
    start_date: date
    end_date: date
    duration_sessions: int


@dataclass(frozen=True, slots=True)
class RegimeDurationSummary:
    """Empirical duration statistics for one market state."""

    regime: MarketRegime
    segment_count: int
    total_sessions: int
    minimum_sessions: int | None
    median_sessions: Decimal | None
    mean_sessions: Decimal | None
    maximum_sessions: int | None


@dataclass(frozen=True, slots=True)
class RegimeStabilityMetrics:
    """State duration and adjacent-session switching statistics."""

    observation_count: int
    transition_opportunity_count: int
    switch_count: int
    switch_rate: Decimal
    segments: tuple[RegimeDurationSegment, ...]
    duration_by_regime: tuple[RegimeDurationSummary, ...]


@dataclass(frozen=True, slots=True)
class PrecisionAtK:
    """Positive-forward-return precision among actual point-in-time Top-K selections."""

    top_k: int
    horizon: int
    period: EvaluationPeriod | None
    prediction_count: int
    successful_count: int
    precision: Decimal | None


@dataclass(frozen=True, slots=True)
class RelativeReturnDistribution:
    """Complete sorted empirical values plus deterministic summary statistics."""

    sample_count: int
    values: tuple[Decimal, ...]
    positive_count: int
    positive_rate: Decimal | None
    minimum: Decimal | None
    median: Decimal | None
    mean: Decimal | None
    maximum: Decimal | None


@dataclass(frozen=True, slots=True)
class HorizonMainlineMetrics:
    """Precision and return distributions for one future-session horizon."""

    horizon: int
    overall_precision_at_k: PrecisionAtK
    effective_period_precision_at_k: PrecisionAtK
    invalid_period_precision_at_k: PrecisionAtK
    overall_returns: RelativeReturnDistribution
    effective_period_returns: RelativeReturnDistribution
    invalid_period_returns: RelativeReturnDistribution


@dataclass(frozen=True, slots=True)
class EvaluationVersions:
    """Every data, code, model, feature, and configuration identity in the replay."""

    signal_data_version: str
    label_data_version: str
    label_version: str
    evaluation_code_version: str
    evaluation_config_version: str
    evaluation_config_hash: str
    regime_model_version: str
    regime_classifier_config_hash: str
    regime_transition_version: str
    regime_transition_config_hash: str
    mainline_model_version: str
    mainline_config_hash: str
    industry_feature_version: str
    industry_feature_config_hash: str
    classification_version: str
    industry_level: int


@dataclass(frozen=True, slots=True)
class RegimeMainlineEvaluationReport:
    """Reproducible P2-T08 market-state and mainline evaluation result."""

    versions: EvaluationVersions
    regime_stability: RegimeStabilityMetrics
    horizon_metrics: tuple[HorizonMainlineMetrics, ...]
    signal_input_hash: str
    label_input_hash: str
    input_hash: str
    result_hash: str

    @property
    def data_version(self) -> str:
        """Return the frozen point-in-time signal data version."""

        return self.versions.signal_data_version

    @property
    def code_version(self) -> str:
        """Return the evaluator implementation contract version."""

        return self.versions.evaluation_code_version

    @property
    def config_version(self) -> str:
        """Return the evaluation configuration version."""

        return self.versions.evaluation_config_version

    @property
    def config_hash(self) -> str:
        """Return the evaluation configuration hash."""

        return self.versions.evaluation_config_hash

    def identity_payload(self) -> dict[str, str]:
        """Return stable identities suitable for an artifact manifest."""

        return {
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "config_version": self.config_version,
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "label_input_hash": self.label_input_hash,
            "result_hash": self.result_hash,
            "signal_input_hash": self.signal_input_hash,
        }


@dataclass(frozen=True, slots=True)
class _SignalIdentity:
    data_version: str
    regime_model_version: str
    regime_classifier_config_hash: str
    regime_transition_version: str
    regime_transition_config_hash: str
    mainline_model_version: str
    mainline_config_hash: str
    industry_feature_version: str
    industry_feature_config_hash: str
    classification_version: str
    industry_level: int
    trend_feature_version: str
    trend_config_hash: str
    breadth_feature_version: str
    breadth_config_hash: str

    @classmethod
    def from_pair(
        cls,
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
    ) -> "_SignalIdentity":
        raw = regime.raw_result
        identity = mainline.input_identity
        return cls(
            data_version=mainline.data_version,
            regime_model_version=raw.model_version,
            regime_classifier_config_hash=raw.config_hash,
            regime_transition_version=regime.transition_config_version,
            regime_transition_config_hash=regime.transition_config_hash,
            mainline_model_version=mainline.model_version,
            mainline_config_hash=mainline.config_hash,
            industry_feature_version=identity.industry_feature_version,
            industry_feature_config_hash=identity.industry_config_hash,
            classification_version=mainline.classification_version,
            industry_level=mainline.industry_level,
            trend_feature_version=raw.input_identity.trend_feature_version,
            trend_config_hash=raw.input_identity.trend_config_hash,
            breadth_feature_version=raw.input_identity.breadth_feature_version,
            breadth_config_hash=raw.input_identity.breadth_config_hash,
        )


@dataclass(frozen=True, slots=True)
class _Prediction:
    signal_date: date
    industry_id: str
    rank: int
    state: MainlineState
    period: EvaluationPeriod


@dataclass(frozen=True, slots=True)
class _PreparedSignals:
    regime_results: tuple[RegimeTransitionResult, ...]
    mainline_snapshots: tuple[MainlineSnapshot, ...]
    identity: _SignalIdentity
    predictions: tuple[_Prediction, ...]
    signal_input_hash: str


def _ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def _decimal_mean(values: tuple[Decimal, ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal(0)) / Decimal(len(values))


def _decimal_median(values: tuple[Decimal, ...]) -> Decimal:
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / Decimal(2)


def _validate_pair_bindings(
    regime: RegimeTransitionResult,
    mainline: MainlineSnapshot,
) -> None:
    raw = regime.raw_result
    event = regime.event
    identity = mainline.input_identity
    if (
        mainline.session_date != raw.session_date
        or raw.session_date != event.session_date
        or mainline.as_of != raw.as_of
        or raw.as_of != event.as_of
    ):
        raise RegimeMainlineEvaluationError("regime and mainline dates/as_of values are misaligned")
    if mainline.data_version != raw.data_version:
        raise RegimeMainlineEvaluationError("regime and mainline data versions are misaligned")
    if mainline.market_regime is not regime.final_regime:
        raise RegimeMainlineEvaluationError(
            "mainline market state does not match the stabilized regime"
        )
    if event.source_result_hash != raw.result_hash:
        raise RegimeMainlineEvaluationError("regime event does not bind its raw frozen result")
    if (
        event.transition_config_version != regime.transition_config_version
        or event.transition_config_hash != regime.transition_config_hash
    ):
        raise RegimeMainlineEvaluationError(
            "regime event and result transition versions are misaligned"
        )
    expected_bindings = (
        (identity.classification_version, mainline.classification_version),
        (identity.industry_level, mainline.industry_level),
        (identity.regime_model_version, raw.model_version),
        (identity.regime_classifier_config_hash, raw.config_hash),
        (identity.regime_input_hash, raw.input_hash),
        (
            identity.regime_trend_feature_version,
            raw.input_identity.trend_feature_version,
        ),
        (identity.regime_trend_config_hash, raw.input_identity.trend_config_hash),
        (
            identity.regime_breadth_feature_version,
            raw.input_identity.breadth_feature_version,
        ),
        (identity.regime_breadth_config_hash, raw.input_identity.breadth_config_hash),
        (identity.regime_transition_version, regime.transition_config_version),
        (identity.regime_transition_config_hash, regime.transition_config_hash),
        (identity.regime_transition_result_hash, regime.result_hash),
    )
    if any(actual != expected for actual, expected in expected_bindings):
        raise RegimeMainlineEvaluationError(
            "mainline snapshot does not bind the supplied frozen regime identity"
        )


def _prepare_signals(
    regime_results: Iterable[RegimeTransitionResult],
    mainline_snapshots: Iterable[MainlineSnapshot],
    config: RegimeMainlineEvaluationConfig,
) -> _PreparedSignals:
    regimes = tuple(regime_results)
    mainlines = tuple(mainline_snapshots)
    if not regimes or not mainlines:
        raise RegimeMainlineEvaluationError("evaluation requires non-empty signal histories")
    if len(regimes) != len(mainlines):
        raise RegimeMainlineEvaluationError(
            "regime and mainline histories must contain the same sessions"
        )

    sequence_identity: _SignalIdentity | None = None
    previous_date: date | None = None
    previous_regime_event_hash: str | None = None
    previous_mainline_result_hash: str | None = None
    predictions: list[_Prediction] = []

    for index, (regime, mainline) in enumerate(zip(regimes, mainlines, strict=True)):
        if not isinstance(regime, RegimeTransitionResult) or not isinstance(
            mainline, MainlineSnapshot
        ):
            raise RegimeMainlineEvaluationError(
                "evaluation accepts frozen RegimeTransitionResult and MainlineSnapshot values"
            )
        _validate_pair_bindings(regime, mainline)
        if previous_date is not None and mainline.session_date <= previous_date:
            raise RegimeMainlineEvaluationError(
                "signal session dates must be strictly increasing without duplicates"
            )
        if index > 0:
            if regime.event.previous_event_hash != previous_regime_event_hash:
                raise RegimeMainlineEvaluationError("regime frozen-result hash chain is broken")
            if mainline.previous_result_hash != previous_mainline_result_hash:
                raise RegimeMainlineEvaluationError("mainline frozen-result hash chain is broken")

        identity = _SignalIdentity.from_pair(regime, mainline)
        if sequence_identity is None:
            sequence_identity = identity
        elif identity != sequence_identity:
            raise RegimeMainlineEvaluationError(
                "signal history mixes data, code, feature, model, or config versions"
            )

        selected = tuple(
            item
            for item in mainline.industries
            if item.current_rank is not None and item.current_rank <= config.top_k
        )
        selected_ranks = tuple(item.current_rank for item in selected)
        if len(set(selected_ranks)) != len(selected_ranks):
            raise RegimeMainlineEvaluationError("mainline snapshot contains duplicate Top-K ranks")
        for item in sorted(
            selected,
            key=lambda value: (
                value.current_rank if value.current_rank is not None else config.top_k + 1,
                value.industry_id,
            ),
        ):
            if item.current_rank is None:  # pragma: no cover - filtered above
                continue
            period = (
                EvaluationPeriod.EFFECTIVE
                if item.state in config.effective_states
                else EvaluationPeriod.INVALID
            )
            predictions.append(
                _Prediction(
                    signal_date=mainline.session_date,
                    industry_id=item.industry_id,
                    rank=item.current_rank,
                    state=item.state,
                    period=period,
                )
            )

        previous_date = mainline.session_date
        previous_regime_event_hash = regime.event.event_hash
        previous_mainline_result_hash = mainline.result_hash

    if sequence_identity is None:  # pragma: no cover - guarded by non-empty histories
        raise AssertionError("signal sequence identity was not initialized")
    if not predictions:
        raise RegimeMainlineEvaluationError("evaluation requires at least one Top-K prediction")

    signal_input_hash = stable_hash(
        {
            "mainline": [
                {
                    "as_of": item.as_of,
                    "input_hash": item.input_hash,
                    "result_hash": item.result_hash,
                    "session_date": item.session_date,
                }
                for item in mainlines
            ],
            "regime": [
                {
                    "event_hash": item.event.event_hash,
                    "raw_result_hash": item.raw_result.result_hash,
                    "result_hash": item.result_hash,
                    "session_date": item.raw_result.session_date,
                }
                for item in regimes
            ],
        }
    )
    return _PreparedSignals(
        regime_results=regimes,
        mainline_snapshots=mainlines,
        identity=sequence_identity,
        predictions=tuple(predictions),
        signal_input_hash=signal_input_hash,
    )


def _validate_labels(
    labels: Iterable[ForwardRelativeReturnLabel],
    prepared: _PreparedSignals,
    config: RegimeMainlineEvaluationConfig,
) -> tuple[
    tuple[ForwardRelativeReturnLabel, ...],
    dict[tuple[date, str, int], ForwardRelativeReturnLabel],
    str,
    str,
]:
    frozen = tuple(labels)
    if not frozen:
        raise RegimeMainlineEvaluationError("evaluation requires future-return labels")

    signal_dates = tuple(item.session_date for item in prepared.mainline_snapshots)
    signal_date_set = set(signal_dates)
    date_positions = {value: index for index, value in enumerate(signal_dates)}
    by_key: dict[tuple[date, str, int], ForwardRelativeReturnLabel] = {}
    calendars: dict[tuple[date, int], tuple[date, ...]] = {}
    by_signal_industry: dict[tuple[date, str], list[ForwardRelativeReturnLabel]] = {}
    previous_key: tuple[date, str, int] | None = None
    label_identity: tuple[str, str] | None = None

    for label in frozen:
        if not isinstance(label, ForwardRelativeReturnLabel):
            raise RegimeMainlineEvaluationError(
                "labels must be frozen ForwardRelativeReturnLabel values"
            )
        key = (label.signal_date, label.industry_id, label.horizon)
        if key in by_key:
            raise RegimeMainlineEvaluationError("future-return labels contain a duplicate key")
        if previous_key is not None and key <= previous_key:
            raise RegimeMainlineEvaluationError(
                "future-return labels must use canonical chronological key order"
            )
        previous_key = key
        if label.signal_date not in signal_date_set:
            raise RegimeMainlineEvaluationError(
                "future-return label date is not aligned to a signal session"
            )
        if label.horizon not in config.horizons:
            raise RegimeMainlineEvaluationError(
                "future-return label horizon is not configured for this evaluation"
            )
        current_identity = (label.data_version, label.label_version)
        if label_identity is None:
            label_identity = current_identity
        elif current_identity != label_identity:
            raise RegimeMainlineEvaluationError(
                "future-return labels mix data or label-code versions"
            )

        calendar_key = (label.signal_date, label.horizon)
        prior_calendar = calendars.get(calendar_key)
        if prior_calendar is not None and label.future_sessions != prior_calendar:
            raise RegimeMainlineEvaluationError(
                "industries on one signal/horizon use inconsistent future sessions"
            )
        calendars[calendar_key] = label.future_sessions

        position = date_positions[label.signal_date]
        known_future = signal_dates[position + 1 : position + 1 + label.horizon]
        if label.future_sessions[: len(known_future)] != known_future:
            raise RegimeMainlineEvaluationError(
                "future label window is misaligned or overlaps the PIT signal history"
            )

        by_key[key] = label
        by_signal_industry.setdefault((label.signal_date, label.industry_id), []).append(label)

    missing = [
        (prediction.signal_date, prediction.industry_id, horizon)
        for prediction in prepared.predictions
        for horizon in config.horizons
        if (prediction.signal_date, prediction.industry_id, horizon) not in by_key
    ]
    if missing:
        first = missing[0]
        raise RegimeMainlineEvaluationError(
            f"missing future-return label for Top-K prediction {first[0]}/{first[1]}/{first[2]}"
        )

    for grouped in by_signal_industry.values():
        ordered = sorted(grouped, key=lambda item: item.horizon)
        for shorter, longer in pairwise(ordered):
            if shorter.future_sessions != longer.future_sessions[: shorter.horizon]:
                raise RegimeMainlineEvaluationError(
                    "future label horizons must share exact nested session prefixes"
                )

    for prior_date, current_date in pairwise(signal_dates):
        for horizon in config.horizons:
            prior_calendar = calendars.get((prior_date, horizon))
            current_calendar = calendars.get((current_date, horizon))
            if prior_calendar is None or current_calendar is None:
                continue
            if prior_calendar[1:] != current_calendar[: horizon - 1]:
                raise RegimeMainlineEvaluationError(
                    "adjacent future label windows have an inconsistent overlap"
                )

    if label_identity is None:  # pragma: no cover - guarded by non-empty labels
        raise AssertionError("label identity was not initialized")

    label_input_hash = stable_hash(
        {
            "labels": [
                {
                    "key": (item.signal_date, item.industry_id, item.horizon),
                    "label_hash": item.label_hash,
                }
                for item in frozen
            ]
        }
    )
    return frozen, by_key, label_identity[0], label_input_hash


def _regime_stability(
    regimes: tuple[RegimeTransitionResult, ...],
) -> RegimeStabilityMetrics:
    segments: list[RegimeDurationSegment] = []
    start = regimes[0].raw_result.session_date
    current = regimes[0].final_regime
    duration = 1
    previous_date = start
    for result in regimes[1:]:
        session_date = result.raw_result.session_date
        if result.final_regime is current:
            duration += 1
        else:
            segments.append(
                RegimeDurationSegment(
                    regime=current,
                    start_date=start,
                    end_date=previous_date,
                    duration_sessions=duration,
                )
            )
            start = session_date
            current = result.final_regime
            duration = 1
        previous_date = session_date
    segments.append(
        RegimeDurationSegment(
            regime=current,
            start_date=start,
            end_date=previous_date,
            duration_sessions=duration,
        )
    )

    summaries: list[RegimeDurationSummary] = []
    for regime in MarketRegime:
        durations = tuple(item.duration_sessions for item in segments if item.regime is regime)
        if not durations:
            summaries.append(
                RegimeDurationSummary(
                    regime=regime,
                    segment_count=0,
                    total_sessions=0,
                    minimum_sessions=None,
                    median_sessions=None,
                    mean_sessions=None,
                    maximum_sessions=None,
                )
            )
            continue
        decimal_durations = tuple(Decimal(value) for value in sorted(durations))
        summaries.append(
            RegimeDurationSummary(
                regime=regime,
                segment_count=len(durations),
                total_sessions=sum(durations),
                minimum_sessions=min(durations),
                median_sessions=_decimal_median(decimal_durations),
                mean_sessions=_decimal_mean(decimal_durations),
                maximum_sessions=max(durations),
            )
        )

    opportunities = len(regimes) - 1
    switches = len(segments) - 1
    return RegimeStabilityMetrics(
        observation_count=len(regimes),
        transition_opportunity_count=opportunities,
        switch_count=switches,
        switch_rate=_ratio(switches, opportunities),
        segments=tuple(segments),
        duration_by_regime=tuple(summaries),
    )


def _distribution(
    values: Iterable[Decimal],
    threshold: Decimal,
) -> RelativeReturnDistribution:
    ordered = tuple(sorted(values))
    if not ordered:
        return RelativeReturnDistribution(
            sample_count=0,
            values=(),
            positive_count=0,
            positive_rate=None,
            minimum=None,
            median=None,
            mean=None,
            maximum=None,
        )
    positive_count = sum(value > threshold for value in ordered)
    return RelativeReturnDistribution(
        sample_count=len(ordered),
        values=ordered,
        positive_count=positive_count,
        positive_rate=_ratio(positive_count, len(ordered)),
        minimum=ordered[0],
        median=_decimal_median(ordered),
        mean=_decimal_mean(ordered),
        maximum=ordered[-1],
    )


def _precision(
    *,
    values: tuple[Decimal, ...],
    config: RegimeMainlineEvaluationConfig,
    horizon: int,
    period: EvaluationPeriod | None,
) -> PrecisionAtK:
    successful = sum(value > config.positive_return_threshold for value in values)
    return PrecisionAtK(
        top_k=config.top_k,
        horizon=horizon,
        period=period,
        prediction_count=len(values),
        successful_count=successful,
        precision=_ratio(successful, len(values)) if values else None,
    )


def _horizon_metrics(
    prepared: _PreparedSignals,
    labels: dict[tuple[date, str, int], ForwardRelativeReturnLabel],
    config: RegimeMainlineEvaluationConfig,
) -> tuple[HorizonMainlineMetrics, ...]:
    metrics: list[HorizonMainlineMetrics] = []
    for horizon in config.horizons:
        all_values: list[Decimal] = []
        effective_values: list[Decimal] = []
        invalid_values: list[Decimal] = []
        for prediction in prepared.predictions:
            value = labels[
                (prediction.signal_date, prediction.industry_id, horizon)
            ].relative_return
            all_values.append(value)
            target = (
                effective_values
                if prediction.period is EvaluationPeriod.EFFECTIVE
                else invalid_values
            )
            target.append(value)

        frozen_all = tuple(all_values)
        frozen_effective = tuple(effective_values)
        frozen_invalid = tuple(invalid_values)
        metrics.append(
            HorizonMainlineMetrics(
                horizon=horizon,
                overall_precision_at_k=_precision(
                    values=frozen_all,
                    config=config,
                    horizon=horizon,
                    period=None,
                ),
                effective_period_precision_at_k=_precision(
                    values=frozen_effective,
                    config=config,
                    horizon=horizon,
                    period=EvaluationPeriod.EFFECTIVE,
                ),
                invalid_period_precision_at_k=_precision(
                    values=frozen_invalid,
                    config=config,
                    horizon=horizon,
                    period=EvaluationPeriod.INVALID,
                ),
                overall_returns=_distribution(
                    frozen_all,
                    config.positive_return_threshold,
                ),
                effective_period_returns=_distribution(
                    frozen_effective,
                    config.positive_return_threshold,
                ),
                invalid_period_returns=_distribution(
                    frozen_invalid,
                    config.positive_return_threshold,
                ),
            )
        )
    return tuple(metrics)


def evaluate_regime_mainline(
    regime_results: Iterable[RegimeTransitionResult],
    mainline_snapshots: Iterable[MainlineSnapshot],
    labels: Iterable[ForwardRelativeReturnLabel],
    config: RegimeMainlineEvaluationConfig | None = None,
) -> RegimeMainlineEvaluationReport:
    """Evaluate one chronological, frozen, point-in-time replay.

    Input order is never sorted.  Signal histories must be strictly increasing and
    labels must use ``(signal_date, industry_id, horizon)`` order.  Labels are first
    touched only after immutable signals have been validated and Top-K predictions
    have been selected.
    """

    active_config = config or RegimeMainlineEvaluationConfig()
    prepared = _prepare_signals(regime_results, mainline_snapshots, active_config)
    frozen_labels, labels_by_key, label_data_version, label_input_hash = _validate_labels(
        labels,
        prepared,
        active_config,
    )
    label_version = frozen_labels[0].label_version
    identity = prepared.identity
    versions = EvaluationVersions(
        signal_data_version=identity.data_version,
        label_data_version=label_data_version,
        label_version=label_version,
        evaluation_code_version=REGIME_MAINLINE_EVALUATION_CODE_VERSION,
        evaluation_config_version=active_config.version,
        evaluation_config_hash=active_config.config_hash,
        regime_model_version=identity.regime_model_version,
        regime_classifier_config_hash=identity.regime_classifier_config_hash,
        regime_transition_version=identity.regime_transition_version,
        regime_transition_config_hash=identity.regime_transition_config_hash,
        mainline_model_version=identity.mainline_model_version,
        mainline_config_hash=identity.mainline_config_hash,
        industry_feature_version=identity.industry_feature_version,
        industry_feature_config_hash=identity.industry_feature_config_hash,
        classification_version=identity.classification_version,
        industry_level=identity.industry_level,
    )
    regime_stability = _regime_stability(prepared.regime_results)
    horizon_metrics = _horizon_metrics(prepared, labels_by_key, active_config)
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
            "horizon_metrics": [asdict(item) for item in horizon_metrics],
            "input_hash": input_hash,
            "regime_stability": asdict(regime_stability),
            "versions": asdict(versions),
        }
    )
    return RegimeMainlineEvaluationReport(
        versions=versions,
        regime_stability=regime_stability,
        horizon_metrics=horizon_metrics,
        signal_input_hash=prepared.signal_input_hash,
        label_input_hash=label_input_hash,
        input_hash=input_hash,
        result_hash=result_hash,
    )


# Domain-readable aliases for callers using the wording in the project plan.
FutureIndustryRelativeReturnLabel = ForwardRelativeReturnLabel
evaluate_regime_mainline_history = evaluate_regime_mainline


__all__ = [
    "REGIME_MAINLINE_EVALUATION_CODE_VERSION",
    "EvaluationPeriod",
    "EvaluationVersions",
    "ForwardRelativeReturnLabel",
    "FutureIndustryRelativeReturnLabel",
    "HorizonMainlineMetrics",
    "PrecisionAtK",
    "RegimeDurationSegment",
    "RegimeDurationSummary",
    "RegimeMainlineEvaluationConfig",
    "RegimeMainlineEvaluationError",
    "RegimeMainlineEvaluationReport",
    "RegimeStabilityMetrics",
    "RelativeReturnDistribution",
    "evaluate_regime_mainline",
    "evaluate_regime_mainline_history",
]
