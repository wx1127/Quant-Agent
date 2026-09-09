"""Deterministic orchestration for decision-bound Agent research tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from pydantic import TypeAdapter, ValidationError

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.data.quality import QualityEngine, QualityReport
from quant_agent.data.snapshots import SnapshotManifest
from quant_agent.features.industry_strength import IndustryStrengthSnapshot
from quant_agent.features.mainline import (
    MainlineConfig,
    MainlineInputMismatch,
    MainlineSnapshot,
    apply_mainline_states,
)
from quant_agent.features.market_breadth import MarketBreadthSnapshot
from quant_agent.features.market_trend import MarketTrendSnapshot
from quant_agent.leaders import (
    LeaderConfig,
    LeaderEngine,
    LeaderInputError,
    LeaderSnapshot,
)
from quant_agent.leaders.candidates import (
    CandidateConfig,
    CandidateEngine,
    CandidateInputError,
    CandidateSnapshot,
    CandidateSupplementalSignal,
)
from quant_agent.regime import (
    InsufficientRegimeData,
    MarketRegimeClassifier,
    RegimeClassifierConfig,
    RegimeInputMismatch,
    RegimeTransitionConfig,
    RegimeTransitionInputMismatch,
    RegimeTransitionResult,
    apply_regime_transitions,
)

from .inputs import (
    AccountBoundLeaderInputs,
    DataBoundQualityContext,
    RegimeFeaturePair,
    ResearchInputInvalid,
    ResearchInputSource,
)

_MAX_HISTORY_SESSIONS = 10_000
_MAX_SECURITY_INPUTS = 20_000
_MAX_SUPPLEMENTAL_SIGNALS = 40_000


def _detach_exact[T](value: object, expected_type: type[T], label: str) -> T:
    """Round-trip one trusted object and reject forged or weakened instances."""

    if type(value) is not expected_type:
        raise ResearchInputInvalid(f"{label} has an unsupported type")
    adapter = TypeAdapter(expected_type)
    failed = False
    detached: T | None = None
    try:
        encoded = adapter.dump_json(value, warnings="error")
        detached = adapter.validate_json(encoded, strict=True)
    except (TypeError, ValueError, ValidationError):
        failed = True
    if failed or type(detached) is not expected_type:
        raise ResearchInputInvalid(f"{label} failed strict revalidation") from None
    return detached


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _require_exact_decision(value: object) -> DecisionSnapshot:
    if type(value) is not DecisionSnapshot:
        raise ResearchInputInvalid("research pipeline requires an exact DecisionSnapshot")
    failed = False
    detached: DecisionSnapshot | None = None
    try:
        detached = DecisionSnapshot.from_json(value.to_json())
    except (TypeError, ValueError):
        failed = True
    if failed or detached is None:
        raise ResearchInputInvalid("decision snapshot failed strict revalidation") from None
    return detached


def _require_exact_tuple(value: object, label: str, *, maximum: int) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ResearchInputInvalid(f"{label} must be an exact tuple")
    items = cast(tuple[object, ...], value)
    if not items:
        raise ResearchInputInvalid(f"{label} cannot be empty")
    if len(items) > maximum:
        raise ResearchInputInvalid(f"{label} exceeds its resource limit")
    return items


def _require_callable_source(source: object) -> ResearchInputSource:
    methods = (
        "load_verified_manifest",
        "load_quality_context",
        "load_regime_history",
        "load_industry_history",
        "load_leader_inputs",
        "load_candidate_signals",
    )
    if any(not callable(getattr(source, method, None)) for method in methods):
        raise TypeError("research input source does not implement the complete read-only protocol")
    return cast(ResearchInputSource, source)


class ResearchPipeline:
    """Run existing deterministic engines behind one verified input boundary."""

    __slots__ = (
        "_candidate_engine",
        "_candidate_version",
        "_leader_engine",
        "_leader_version",
        "_mainline_config",
        "_quality_engine",
        "_regime_classifier",
        "_source",
        "_transition_config",
    )

    def __init__(
        self,
        source: ResearchInputSource,
        *,
        regime_config: RegimeClassifierConfig | None = None,
        transition_config: RegimeTransitionConfig | None = None,
        mainline_config: MainlineConfig | None = None,
        leader_config: LeaderConfig | None = None,
        candidate_config: CandidateConfig | None = None,
    ) -> None:
        self._source = _require_callable_source(source)
        regime = _detach_exact(
            regime_config or RegimeClassifierConfig(),
            RegimeClassifierConfig,
            "regime config",
        )
        transition = _detach_exact(
            transition_config or RegimeTransitionConfig(),
            RegimeTransitionConfig,
            "regime transition config",
        )
        mainline = _detach_exact(
            mainline_config or MainlineConfig(),
            MainlineConfig,
            "mainline config",
        )
        leader = _detach_exact(
            leader_config or LeaderConfig(),
            LeaderConfig,
            "leader config",
        )
        candidate = _detach_exact(
            candidate_config or CandidateConfig(),
            CandidateConfig,
            "candidate config",
        )
        self._regime_classifier = MarketRegimeClassifier(regime)
        self._transition_config = transition
        self._mainline_config = mainline
        self._leader_engine = LeaderEngine(leader)
        self._candidate_engine = CandidateEngine(candidate)
        self._quality_engine = QualityEngine()
        self._leader_version = leader.version
        self._candidate_version = candidate.version

    @property
    def transition_version(self) -> str:
        return self._transition_config.version

    @property
    def mainline_version(self) -> str:
        return self._mainline_config.version

    @property
    def leader_version(self) -> str:
        return self._leader_version

    @property
    def candidate_version(self) -> str:
        return self._candidate_version

    def load_manifest(self, decision: DecisionSnapshot) -> SnapshotManifest:
        """Load and revalidate the manifest named by the frozen decision."""

        frozen = _require_exact_decision(decision)
        manifest = _detach_exact(
            self._source.load_verified_manifest(frozen),
            SnapshotManifest,
            "snapshot manifest",
        )
        if (
            manifest.data_version != frozen.data_version
            or manifest.content_hash != frozen.data_content_hash
        ):
            raise ResearchInputInvalid("snapshot manifest does not match the decision")
        return manifest

    def validate_market_data(self, decision: DecisionSnapshot) -> QualityReport:
        """Run quality rules without a database session at the exact decision time."""

        frozen = _require_exact_decision(decision)
        self.load_manifest(frozen)
        bound = _detach_exact(
            self._source.load_quality_context(frozen),
            DataBoundQualityContext,
            "quality context",
        )
        if (
            bound.data_version != frozen.data_version
            or bound.data_content_hash != frozen.data_content_hash
        ):
            raise ResearchInputInvalid("quality context does not match the decision dataset")
        report = self._quality_engine.run(
            frozen.data_version,
            bound.context,
            session=None,
            observed_at=frozen.as_of,
        )
        if report.data_version != frozen.data_version or _utc(report.observed_at) != _utc(
            frozen.as_of
        ):
            raise ResearchInputInvalid("quality report drifted from the decision context")
        return report

    def detect_market_regime(self, decision: DecisionSnapshot) -> RegimeTransitionResult:
        """Classify and transition the complete chronological market history."""

        frozen = _require_exact_decision(decision)
        self.load_manifest(frozen)
        return self._regime_history(frozen)[-1]

    def rank_market_themes(self, decision: DecisionSnapshot) -> MainlineSnapshot:
        """Apply the mainline state machine to complete aligned histories."""

        frozen = _require_exact_decision(decision)
        self.load_manifest(frozen)
        transitions = self._regime_history(frozen)
        return self._mainline_history(frozen, transitions)[-1]

    def rank_theme_leaders(self, decision: DecisionSnapshot) -> LeaderSnapshot:
        """Rank leaders from the final mainline and same-account feature inputs."""

        frozen = _require_exact_decision(decision)
        self.load_manifest(frozen)
        transitions = self._regime_history(frozen)
        mainline = self._mainline_history(frozen, transitions)[-1]
        inputs = self._leader_inputs(frozen, mainline)
        try:
            result = self._leader_engine.rank(
                mainline=mainline,
                stock_strength=inputs.stock_strength,
                tradeability=inputs.tradeability,
                fundamentals=inputs.fundamentals,
            )
        except LeaderInputError as error:
            raise ResearchInputInvalid("leader inputs failed domain alignment") from error
        self._validate_final_identity(
            as_of=result.as_of,
            data_version=result.data_version,
            decision=frozen,
            label="leader result",
        )
        if result.mainline_result_hash != mainline.result_hash:
            raise ResearchInputInvalid("leader result does not bind the final mainline")
        return result

    def rank_stock_candidates(self, decision: DecisionSnapshot) -> CandidateSnapshot:
        """Rank candidates from freshly recomputed, hash-linked upstream results."""

        frozen = _require_exact_decision(decision)
        self.load_manifest(frozen)
        transitions = self._regime_history(frozen)
        regime = transitions[-1]
        mainline = self._mainline_history(frozen, transitions)[-1]
        inputs = self._leader_inputs(frozen, mainline)
        try:
            leaders = self._leader_engine.rank(
                mainline=mainline,
                stock_strength=inputs.stock_strength,
                tradeability=inputs.tradeability,
                fundamentals=inputs.fundamentals,
            )
        except LeaderInputError as error:
            raise ResearchInputInvalid("leader inputs failed domain alignment") from error

        ranked_ids = frozenset(item.instrument_id for item in leaders.leaders)
        stocks = tuple(item for item in inputs.stock_strength if item.instrument_id in ranked_ids)
        trades = tuple(
            item for item in inputs.tradeability if item.request.instrument_id in ranked_ids
        )
        fundamentals = tuple(
            item for item in inputs.fundamentals if item.instrument_id in ranked_ids
        )
        signals = self._candidate_signals(frozen)
        try:
            result = self._candidate_engine.rank(
                regime=regime,
                mainline=mainline,
                leaders=leaders,
                stock_strength=stocks,
                tradeability=trades,
                fundamentals=fundamentals,
                supplemental_signals=signals,
            )
        except CandidateInputError as error:
            raise ResearchInputInvalid("candidate inputs failed domain alignment") from error
        self._validate_final_identity(
            as_of=result.as_of,
            data_version=result.data_version,
            decision=frozen,
            label="candidate result",
        )
        if result.calibrated or result.calibration_version is not None:
            raise ResearchInputInvalid("v1 candidate results must remain explicitly uncalibrated")
        return result

    def _regime_history(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[RegimeTransitionResult, ...]:
        raw_history = _require_exact_tuple(
            self._source.load_regime_history(decision),
            "regime history",
            maximum=_MAX_HISTORY_SESSIONS,
        )
        pairs = tuple(
            _detach_exact(value, RegimeFeaturePair, "regime feature pair") for value in raw_history
        )
        self._validate_regime_pairs(pairs, decision)
        try:
            classified = tuple(
                self._regime_classifier.classify(
                    trend_snapshot=pair.trend,
                    breadth_snapshot=pair.breadth,
                )
                for pair in pairs
            )
            transitions = apply_regime_transitions(classified, self._transition_config)
        except (
            InsufficientRegimeData,
            RegimeInputMismatch,
            RegimeTransitionInputMismatch,
        ) as error:
            raise ResearchInputInvalid("regime history failed domain alignment") from error
        if len(transitions) != len(pairs):
            raise ResearchInputInvalid("regime transition history is incomplete")
        self._validate_final_identity(
            as_of=transitions[-1].raw_result.as_of,
            data_version=transitions[-1].raw_result.data_version,
            decision=decision,
            label="regime result",
        )
        return transitions

    def _mainline_history(
        self,
        decision: DecisionSnapshot,
        transitions: tuple[RegimeTransitionResult, ...],
    ) -> tuple[MainlineSnapshot, ...]:
        raw_history = _require_exact_tuple(
            self._source.load_industry_history(decision),
            "industry history",
            maximum=_MAX_HISTORY_SESSIONS,
        )
        industries = tuple(
            _detach_exact(value, IndustryStrengthSnapshot, "industry strength snapshot")
            for value in raw_history
        )
        if len(industries) != len(transitions):
            raise ResearchInputInvalid("industry and regime histories must cover the same sessions")
        previous: datetime | None = None
        for industry, transition in zip(industries, transitions, strict=True):
            self._validate_final_identity(
                as_of=industry.as_of,
                data_version=industry.data_version,
                decision=decision,
                label="industry history row",
                allow_before=True,
            )
            if industry.session_date != transition.raw_result.session_date or _utc(
                industry.as_of
            ) != _utc(transition.raw_result.as_of):
                raise ResearchInputInvalid("industry and regime histories are not aligned")
            if previous is not None and _utc(industry.as_of) <= previous:
                raise ResearchInputInvalid("industry history must be strictly chronological")
            previous = _utc(industry.as_of)
        if previous != _utc(decision.as_of):
            raise ResearchInputInvalid("industry history must end at the decision time")
        try:
            snapshots = apply_mainline_states(
                zip(industries, transitions, strict=True),
                self._mainline_config,
            )
        except MainlineInputMismatch as error:
            raise ResearchInputInvalid("mainline history failed domain alignment") from error
        if len(snapshots) != len(industries):
            raise ResearchInputInvalid("mainline history is incomplete")
        self._validate_final_identity(
            as_of=snapshots[-1].as_of,
            data_version=snapshots[-1].data_version,
            decision=decision,
            label="mainline result",
        )
        return snapshots

    def _leader_inputs(
        self,
        decision: DecisionSnapshot,
        mainline: MainlineSnapshot,
    ) -> AccountBoundLeaderInputs:
        inputs = _detach_exact(
            self._source.load_leader_inputs(decision),
            AccountBoundLeaderInputs,
            "account-bound leader inputs",
        )
        if (
            inputs.account_id != decision.account_id
            or inputs.account_snapshot_id != decision.account_snapshot_id
            or inputs.account_snapshot_hash != decision.account_snapshot_hash
        ):
            raise ResearchInputInvalid("leader inputs do not match the decision account")
        if not inputs.stock_strength or len(inputs.stock_strength) > _MAX_SECURITY_INPUTS:
            raise ResearchInputInvalid("stock strength inputs have an invalid size")
        if not inputs.tradeability or len(inputs.tradeability) > _MAX_SECURITY_INPUTS:
            raise ResearchInputInvalid("tradeability inputs have an invalid size")
        if not inputs.fundamentals or len(inputs.fundamentals) > _MAX_SECURITY_INPUTS:
            raise ResearchInputInvalid("fundamental inputs have an invalid size")
        self._validate_security_inputs(inputs, decision, mainline)
        return inputs

    @staticmethod
    def _validate_security_inputs(
        inputs: AccountBoundLeaderInputs,
        decision: DecisionSnapshot,
        mainline: MainlineSnapshot,
    ) -> None:
        stock_ids: list[str] = []
        for stock in inputs.stock_strength:
            if (
                stock.data_version != decision.data_version
                or _utc(stock.as_of) != _utc(decision.as_of)
                or stock.session_date != mainline.session_date
            ):
                raise ResearchInputInvalid("stock strength input is not decision-bound")
            stock_ids.append(stock.instrument_id)
        trade_ids: list[str] = []
        for trade in inputs.tradeability:
            request = trade.request
            if (
                request.data_version != decision.data_version
                or _utc(request.as_of) != _utc(decision.as_of)
                or request.session_date != mainline.session_date
            ):
                raise ResearchInputInvalid("tradeability input is not decision-bound")
            trade_ids.append(request.instrument_id)
        fundamental_ids: list[str] = []
        for fundamental in inputs.fundamentals:
            if fundamental.data_version != decision.data_version or _utc(fundamental.as_of) != _utc(
                decision.as_of
            ):
                raise ResearchInputInvalid("fundamental input is not decision-bound")
            fundamental_ids.append(fundamental.instrument_id)
        if (
            len(stock_ids) != len(set(stock_ids))
            or len(trade_ids) != len(set(trade_ids))
            or len(fundamental_ids) != len(set(fundamental_ids))
            or set(stock_ids) != set(trade_ids)
            or set(stock_ids) != set(fundamental_ids)
        ):
            raise ResearchInputInvalid(
                "leader feature collections must cover identical unique instruments"
            )

    def _candidate_signals(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[CandidateSupplementalSignal, ...]:
        value = self._source.load_candidate_signals(decision)
        if type(value) is not tuple:
            raise ResearchInputInvalid("candidate signals must be an exact tuple")
        if len(value) > _MAX_SUPPLEMENTAL_SIGNALS:
            raise ResearchInputInvalid("candidate signals exceed their resource limit")
        signals = tuple(
            _detach_exact(item, CandidateSupplementalSignal, "candidate signal") for item in value
        )
        for signal in signals:
            if signal.data_version != decision.data_version or _utc(signal.as_of) > _utc(
                decision.as_of
            ):
                raise ResearchInputInvalid("candidate signal is not point-in-time aligned")
        return signals

    @staticmethod
    def _validate_regime_pairs(
        pairs: tuple[RegimeFeaturePair, ...],
        decision: DecisionSnapshot,
    ) -> None:
        previous: datetime | None = None
        for pair in pairs:
            trend: MarketTrendSnapshot = pair.trend
            breadth: MarketBreadthSnapshot = pair.breadth
            if (
                trend.data_version != decision.data_version
                or breadth.data_version != decision.data_version
            ):
                raise ResearchInputInvalid("regime history data version does not match decision")
            if trend.session_date != breadth.session_date or _utc(trend.as_of) != _utc(
                breadth.as_of
            ):
                raise ResearchInputInvalid("trend and breadth histories are not aligned")
            current = _utc(trend.as_of)
            if current > _utc(decision.as_of):
                raise ResearchInputInvalid("regime history contains a future observation")
            if previous is not None and current <= previous:
                raise ResearchInputInvalid("regime history must be strictly chronological")
            previous = current
        if previous != _utc(decision.as_of):
            raise ResearchInputInvalid("regime history must end at the decision time")

    @staticmethod
    def _validate_final_identity(
        *,
        as_of: datetime,
        data_version: str,
        decision: DecisionSnapshot,
        label: str,
        allow_before: bool = False,
    ) -> None:
        observed = _utc(as_of)
        boundary = _utc(decision.as_of)
        if data_version != decision.data_version or (
            observed > boundary if allow_before else observed != boundary
        ):
            raise ResearchInputInvalid(f"{label} does not match the decision boundary")


__all__ = ["ResearchPipeline"]
