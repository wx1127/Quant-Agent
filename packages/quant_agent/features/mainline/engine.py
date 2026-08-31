"""Sequential Top-K persistence and lifecycle states for mainline industries."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal

from quant_agent.features.industry_strength import (
    IndustryStrengthResult,
    IndustryStrengthSnapshot,
    IndustryStrengthStatus,
)
from quant_agent.features.mainline.contracts import (
    MainlineConfig,
    MainlineEvidence,
    MainlineEvidenceSide,
    MainlineIndustryResult,
    MainlineInputIdentity,
    MainlineInputMismatch,
    MainlineSnapshot,
    MainlineState,
    TopKPersistence,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.regime.transitions import RegimeTransitionResult


@dataclass(frozen=True, slots=True)
class _SequenceIdentity:
    industry_feature_version: str
    industry_config_hash: str
    classification_version: str
    industry_level: int
    regime_model_version: str
    regime_classifier_config_hash: str
    trend_feature_version: str
    trend_config_hash: str
    breadth_feature_version: str
    breadth_config_hash: str
    transition_version: str
    transition_config_hash: str

    @classmethod
    def from_inputs(
        cls,
        industry_snapshot: IndustryStrengthSnapshot,
        regime_result: RegimeTransitionResult,
    ) -> "_SequenceIdentity":
        raw = regime_result.raw_result
        return cls(
            industry_feature_version=industry_snapshot.feature_version,
            industry_config_hash=industry_snapshot.config_hash,
            classification_version=industry_snapshot.classification_version,
            industry_level=industry_snapshot.industry_level,
            regime_model_version=raw.model_version,
            regime_classifier_config_hash=raw.config_hash,
            trend_feature_version=raw.input_identity.trend_feature_version,
            trend_config_hash=raw.input_identity.trend_config_hash,
            breadth_feature_version=raw.input_identity.breadth_feature_version,
            breadth_config_hash=raw.input_identity.breadth_config_hash,
            transition_version=regime_result.transition_config_version,
            transition_config_hash=regime_result.transition_config_hash,
        )


@dataclass(frozen=True, slots=True)
class _TrackedState:
    state: MainlineState
    fading_sessions: int


def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


class MainlineEngine:
    """No-lookahead state machine over chronological PIT industry snapshots.

    The engine counts observed trading sessions, not calendar days.  An industry
    cannot reach ``CONFIRMED`` or ``CROWDED`` until the configured multi-session
    Top-K requirement is met, even when its first-session strength is extreme.
    """

    def __init__(self, config: MainlineConfig | None = None) -> None:
        self._config = config or MainlineConfig()
        self._sequence_identity: _SequenceIdentity | None = None
        self._last_session_date: date | None = None
        self._last_as_of: datetime | None = None
        self._rank_histories: dict[str, tuple[int | None, ...]] = {}
        self._tracked: dict[str, _TrackedState] = {}
        self._snapshots: list[MainlineSnapshot] = []
        self._processed_sessions = 0

    @property
    def config(self) -> MainlineConfig:
        """Return the immutable scoring and transition policy."""

        return self._config

    @property
    def current_snapshot(self) -> MainlineSnapshot | None:
        """Return the latest immutable result, if any."""

        return self._snapshots[-1] if self._snapshots else None

    @property
    def snapshots(self) -> tuple[MainlineSnapshot, ...]:
        """Return the append-only decision history."""

        return tuple(self._snapshots)

    def process(
        self,
        industry_snapshot: IndustryStrengthSnapshot,
        regime_result: RegimeTransitionResult,
    ) -> MainlineSnapshot:
        """Process one strictly later, time/version-aligned pair of inputs."""

        self._validate_next(industry_snapshot, regime_result)
        source_rows = self._source_rows(industry_snapshot)
        next_histories = self._next_histories(source_rows)
        candidate_ids = {
            industry_id
            for industry_id, row in source_rows.items()
            if row.status is IndustryStrengthStatus.READY
            and row.rank is not None
            and row.rank <= self._config.top_k
        } | set(self._tracked)

        next_tracked: dict[str, _TrackedState] = {}
        decisions: list[MainlineIndustryResult] = []
        for industry_id in sorted(candidate_ids):
            decision = self._decide_industry(
                industry_id=industry_id,
                row=source_rows.get(industry_id),
                ranks=next_histories[industry_id],
                previous=self._tracked.get(industry_id),
                regime_result=regime_result,
            )
            if decision is None:
                continue
            result, tracked = decision
            decisions.append(result)
            next_tracked[industry_id] = tracked

        decisions.sort(
            key=lambda item: (
                item.current_rank
                if item.current_rank is not None and item.current_rank <= self._config.top_k
                else self._config.top_k + 1,
                item.industry_id,
            )
        )
        input_identity = self._input_identity(industry_snapshot, regime_result)
        input_hash = stable_hash(
            {
                "as_of": industry_snapshot.as_of,
                "data_version": industry_snapshot.data_version,
                "input_identity": asdict(input_identity),
                "market_regime": regime_result.final_regime,
                "session_date": industry_snapshot.session_date,
            }
        )
        previous_result_hash = self._snapshots[-1].result_hash if self._snapshots else None
        result_hash = stable_hash(
            {
                "config_hash": self._config.config_hash,
                "decisions": [asdict(item) for item in decisions],
                "input_hash": input_hash,
                "market_regime": regime_result.final_regime,
                "previous_result_hash": previous_result_hash,
            }
        )
        output = MainlineSnapshot(
            session_date=industry_snapshot.session_date,
            as_of=industry_snapshot.as_of,
            data_version=industry_snapshot.data_version,
            classification_version=industry_snapshot.classification_version,
            industry_level=industry_snapshot.industry_level,
            market_regime=regime_result.final_regime,
            model_version=self._config.version,
            config_hash=self._config.config_hash,
            input_identity=input_identity,
            input_hash=input_hash,
            industries=tuple(decisions),
            previous_result_hash=previous_result_hash,
            result_hash=result_hash,
        )

        # Commit mutable engine state only after all contracts and hashes succeeded.
        self._sequence_identity = self._sequence_identity or _SequenceIdentity.from_inputs(
            industry_snapshot,
            regime_result,
        )
        self._last_session_date = industry_snapshot.session_date
        self._last_as_of = industry_snapshot.as_of
        self._rank_histories = next_histories
        self._tracked = next_tracked
        self._snapshots.append(output)
        self._processed_sessions += 1
        return output

    def process_all(
        self,
        inputs: Iterable[tuple[IndustryStrengthSnapshot, RegimeTransitionResult]],
    ) -> tuple[MainlineSnapshot, ...]:
        """Process an already chronological iterable without sorting or lookahead."""

        return tuple(self.process(industry, regime) for industry, regime in inputs)

    def _validate_next(
        self,
        industry_snapshot: IndustryStrengthSnapshot,
        regime_result: RegimeTransitionResult,
    ) -> None:
        raw = regime_result.raw_result
        event = regime_result.event
        if (
            industry_snapshot.session_date != raw.session_date
            or raw.session_date != event.session_date
        ):
            raise MainlineInputMismatch(
                "industry and stabilized regime session_date values must match"
            )
        if industry_snapshot.as_of != raw.as_of or raw.as_of != event.as_of:
            raise MainlineInputMismatch("industry and stabilized regime as_of values must match")
        if industry_snapshot.data_version != raw.data_version:
            raise MainlineInputMismatch(
                "industry and stabilized regime data_version values must match"
            )
        if event.source_result_hash != raw.result_hash:
            raise MainlineInputMismatch("stabilized regime event must bind the raw result hash")
        if (
            event.transition_config_version != regime_result.transition_config_version
            or event.transition_config_hash != regime_result.transition_config_hash
        ):
            raise MainlineInputMismatch(
                "stabilized regime event and result transition versions must match"
            )
        if industry_snapshot.session_date > industry_snapshot.as_of.date():
            raise MainlineInputMismatch("mainline inputs cannot be dated after their as_of")
        if (
            self._last_session_date is not None
            and industry_snapshot.session_date <= self._last_session_date
        ):
            raise MainlineInputMismatch("mainline session_date must be strictly increasing")
        if self._last_as_of is not None and industry_snapshot.as_of <= self._last_as_of:
            raise MainlineInputMismatch("mainline as_of must be strictly increasing")
        sequence_identity = _SequenceIdentity.from_inputs(industry_snapshot, regime_result)
        if self._sequence_identity is not None and sequence_identity != self._sequence_identity:
            raise MainlineInputMismatch(
                "mainline sequence must keep feature, classification, model, "
                "and config versions aligned"
            )

    @staticmethod
    def _source_rows(
        snapshot: IndustryStrengthSnapshot,
    ) -> dict[str, IndustryStrengthResult]:
        rows: dict[str, IndustryStrengthResult] = {}
        ready_ranks: set[int] = set()
        for row in snapshot.industries:
            if row.industry_id in rows:
                raise MainlineInputMismatch(
                    "industry snapshot contains duplicate industry_id values"
                )
            rows[row.industry_id] = row
            if row.status is IndustryStrengthStatus.READY:
                if row.rank is None:  # pragma: no cover - upstream contract rejects this
                    raise MainlineInputMismatch("ready industry requires a rank")
                if row.rank in ready_ranks:
                    raise MainlineInputMismatch("industry snapshot contains duplicate ready ranks")
                ready_ranks.add(row.rank)
        if ready_ranks and ready_ranks != set(range(1, len(ready_ranks) + 1)):
            raise MainlineInputMismatch("ready industry ranks must be contiguous from one")
        return rows

    def _next_histories(
        self,
        rows: dict[str, IndustryStrengthResult],
    ) -> dict[str, tuple[int | None, ...]]:
        all_ids = set(self._rank_histories) | set(rows)
        histories: dict[str, tuple[int | None, ...]] = {}
        prior_slots = min(self._processed_sessions, self._config.persistence_window_sessions - 1)
        for industry_id in all_ids:
            row = rows.get(industry_id)
            current_rank = (
                row.rank if row is not None and row.status is IndustryStrengthStatus.READY else None
            )
            prior = self._rank_histories.get(industry_id, (None,) * prior_slots)
            histories[industry_id] = (*prior, current_rank)[
                -self._config.persistence_window_sessions :
            ]
        return histories

    def _decide_industry(
        self,
        *,
        industry_id: str,
        row: IndustryStrengthResult | None,
        ranks: tuple[int | None, ...],
        previous: _TrackedState | None,
        regime_result: RegimeTransitionResult,
    ) -> tuple[MainlineIndustryResult, _TrackedState] | None:
        persistence = self._persistence(ranks)
        current_rank = (
            row.rank if row is not None and row.status is IndustryStrengthStatus.READY else None
        )
        current_strength = (
            row.score if row is not None and row.status is IndustryStrengthStatus.READY else None
        )
        current_top_k = current_rank is not None and current_rank <= self._config.top_k
        market_eligible = regime_result.final_regime in self._config.confirmation_regimes
        strength_eligible = (
            current_strength is not None
            and current_strength >= self._config.minimum_confirmed_strength_score
        )
        persistence_eligible = persistence.top_k_hits >= self._config.minimum_top_k_hits
        confirmation_eligible = market_eligible and strength_eligible and persistence_eligible
        crowding_signals = self._crowding_signals(row)
        crowded = (
            confirmation_eligible
            and current_top_k
            and len(crowding_signals) >= self._config.minimum_crowding_signals
        )

        prior_state = previous.state if previous is not None else None
        if confirmation_eligible:
            state = MainlineState.CROWDED if crowded else MainlineState.CONFIRMED
            fading_sessions = 0
            reason = (
                "confirmed persistent Top-K leadership and current crowding signals"
                if crowded
                else "confirmed persistent Top-K leadership under the stabilized market state"
            )
        elif current_top_k and not (
            previous is not None
            and previous.state in (MainlineState.CONFIRMED, MainlineState.CROWDED)
            and (not market_eligible or not strength_eligible)
        ):
            state = MainlineState.EMERGING
            fading_sessions = 0
            reason = "current Top-K leadership is still below the multi-session confirmation gate"
        elif previous is not None:
            state = MainlineState.FADING
            fading_sessions = previous.fading_sessions + 1 if previous.state is state else 1
            reason = self._fading_reason(
                current_top_k=current_top_k,
                market_eligible=market_eligible,
                strength_eligible=strength_eligible,
                persistence_eligible=persistence_eligible,
            )
            if not current_top_k and fading_sessions > self._config.fading_retention_sessions:
                return None
        else:  # pragma: no cover - candidate construction makes this unreachable
            return None

        mainline_score = self._mainline_score(
            current_rank=current_rank,
            current_strength=current_strength,
            persistence=persistence,
        )
        crowding_score = Decimal(len(crowding_signals)) / Decimal(3) * Decimal(100)
        supporting, counter = self._evidence(
            state=state,
            current_rank=current_rank,
            current_strength=current_strength,
            persistence=persistence,
            market_eligible=market_eligible,
            market_regime=regime_result.final_regime.value,
            crowding_signals=crowding_signals,
            prior_state=prior_state,
        )
        result = MainlineIndustryResult(
            industry_id=industry_id,
            state=state,
            previous_state=prior_state,
            current_rank=current_rank,
            current_strength_score=current_strength,
            mainline_score=mainline_score,
            crowding_score=crowding_score,
            persistence=persistence,
            crowding_signals=crowding_signals,
            supporting_evidence=supporting,
            counter_evidence=counter,
            invalidations=self._invalidations(),
            transition_reason=reason,
        )
        return result, _TrackedState(state=state, fading_sessions=fading_sessions)

    def _persistence(self, ranks: tuple[int | None, ...]) -> TopKPersistence:
        hits = sum(rank is not None and rank <= self._config.top_k for rank in ranks)
        consecutive = 0
        for rank in reversed(ranks):
            if rank is None or rank > self._config.top_k:
                break
            consecutive += 1
        return TopKPersistence(
            top_k=self._config.top_k,
            window_sessions=self._config.persistence_window_sessions,
            observed_sessions=len(ranks),
            top_k_hits=hits,
            consecutive_top_k_sessions=consecutive,
            recent_ranks=ranks,
        )

    def _mainline_score(
        self,
        *,
        current_rank: int | None,
        current_strength: Decimal | None,
        persistence: TopKPersistence,
    ) -> Decimal:
        normalized_strength = (
            _clamp((current_strength + Decimal(100)) / Decimal(200), Decimal(0), Decimal(1))
            if current_strength is not None
            else Decimal(0)
        )
        rank_component = (
            Decimal(self._config.top_k - current_rank + 1) / self._config.top_k
            if current_rank is not None and current_rank <= self._config.top_k
            else Decimal(0)
        )
        return Decimal(100) * (
            normalized_strength * self._config.strength_weight
            + rank_component * self._config.rank_weight
            + persistence.hit_ratio * self._config.persistence_weight
        )

    def _crowding_signals(
        self,
        row: IndustryStrengthResult | None,
    ) -> tuple[str, ...]:
        if row is None or row.status is not IndustryStrengthStatus.READY:
            return ()
        signals: list[str] = []
        if (
            row.turnover_growth is not None
            and row.turnover_growth >= self._config.crowding_turnover_growth_min
        ):
            signals.append("TURNOVER_GROWTH")
        if (
            row.new_high_ratio is not None
            and row.new_high_ratio >= self._config.crowding_new_high_ratio_min
        ):
            signals.append("NEW_HIGH_HEAT")
        top_member_share = max(
            (item.turnover_share for item in row.contributions),
            default=Decimal(0),
        )
        if top_member_share >= self._config.crowding_top_member_turnover_share_min:
            signals.append("TURNOVER_CONCENTRATION")
        return tuple(signals)

    def _evidence(
        self,
        *,
        state: MainlineState,
        current_rank: int | None,
        current_strength: Decimal | None,
        persistence: TopKPersistence,
        market_eligible: bool,
        market_regime: str,
        crowding_signals: tuple[str, ...],
        prior_state: MainlineState | None,
    ) -> tuple[tuple[MainlineEvidence, ...], tuple[MainlineEvidence, ...]]:
        supporting: list[MainlineEvidence] = []
        opposing: list[MainlineEvidence] = []

        def evidence(
            feature: str,
            value: Decimal,
            criterion: str,
            side: MainlineEvidenceSide,
            rationale: str,
        ) -> MainlineEvidence:
            return MainlineEvidence(
                feature=feature,
                value=value,
                criterion=criterion,
                side=side,
                rationale=rationale,
            )

        current_top_k = current_rank is not None and current_rank <= self._config.top_k
        persistence_met = persistence.top_k_hits >= self._config.minimum_top_k_hits
        strength_met = (
            current_strength is not None
            and current_strength >= self._config.minimum_confirmed_strength_score
        )
        if state is MainlineState.FADING:
            if not current_top_k:
                supporting.append(
                    evidence(
                        "current_rank",
                        Decimal(current_rank or 0),
                        f"rank is absent or above {self._config.top_k}",
                        MainlineEvidenceSide.SUPPORTING,
                        "the industry is no longer in the current Top-K",
                    )
                )
            if not persistence_met:
                supporting.append(
                    evidence(
                        "top_k_hits",
                        Decimal(persistence.top_k_hits),
                        f"< {self._config.minimum_top_k_hits}",
                        MainlineEvidenceSide.SUPPORTING,
                        "rolling Top-K persistence no longer meets confirmation",
                    )
                )
            if not market_eligible:
                supporting.append(
                    evidence(
                        "market_regime_eligible",
                        Decimal(0),
                        "= 0",
                        MainlineEvidenceSide.SUPPORTING,
                        f"stabilized market state {market_regime} blocks confirmation",
                    )
                )
            if not strength_met:
                supporting.append(
                    evidence(
                        "strength_score",
                        (current_strength if current_strength is not None else Decimal(-100)),
                        f"< {self._config.minimum_confirmed_strength_score}",
                        MainlineEvidenceSide.SUPPORTING,
                        "current industry strength is absent or below the confirmation floor",
                    )
                )
            if current_top_k:
                opposing.append(
                    evidence(
                        "current_rank",
                        Decimal(current_rank if current_rank is not None else 0),
                        f"<= {self._config.top_k}",
                        MainlineEvidenceSide.OPPOSING,
                        "current Top-K rank is counter-evidence to continued fading",
                    )
                )
            if persistence_met:
                opposing.append(
                    evidence(
                        "top_k_hits",
                        Decimal(persistence.top_k_hits),
                        f">= {self._config.minimum_top_k_hits}",
                        MainlineEvidenceSide.OPPOSING,
                        "rolling leadership remains persistent despite the fading state",
                    )
                )
            if not supporting:
                supporting.append(
                    evidence(
                        "prior_tracked_state",
                        Decimal(1),
                        "previous state existed",
                        MainlineEvidenceSide.SUPPORTING,
                        f"the previously tracked {prior_state} state has lost a confirmation gate",
                    )
                )
        else:
            if current_top_k:
                supporting.append(
                    evidence(
                        "current_rank",
                        Decimal(current_rank if current_rank is not None else 0),
                        f"<= {self._config.top_k}",
                        MainlineEvidenceSide.SUPPORTING,
                        "the industry is in the current point-in-time Top-K",
                    )
                )
            if persistence_met:
                supporting.append(
                    evidence(
                        "top_k_hits",
                        Decimal(persistence.top_k_hits),
                        f">= {self._config.minimum_top_k_hits}",
                        MainlineEvidenceSide.SUPPORTING,
                        "recent Top-K frequency meets the multi-session confirmation rule",
                    )
                )
            else:
                opposing.append(
                    evidence(
                        "top_k_hits",
                        Decimal(persistence.top_k_hits),
                        f"< {self._config.minimum_top_k_hits}",
                        MainlineEvidenceSide.OPPOSING,
                        "recent Top-K frequency is not yet sufficient for confirmation",
                    )
                )
            if market_eligible:
                supporting.append(
                    evidence(
                        "market_regime_eligible",
                        Decimal(1),
                        "= 1",
                        MainlineEvidenceSide.SUPPORTING,
                        f"stabilized market state {market_regime} permits confirmation",
                    )
                )
            else:
                opposing.append(
                    evidence(
                        "market_regime_eligible",
                        Decimal(0),
                        "= 0",
                        MainlineEvidenceSide.OPPOSING,
                        f"stabilized market state {market_regime} blocks confirmation",
                    )
                )
            if strength_met:
                supporting.append(
                    evidence(
                        "strength_score",
                        current_strength if current_strength is not None else Decimal(-100),
                        f">= {self._config.minimum_confirmed_strength_score}",
                        MainlineEvidenceSide.SUPPORTING,
                        "current industry strength meets the confirmation floor",
                    )
                )
            else:
                opposing.append(
                    evidence(
                        "strength_score",
                        (current_strength if current_strength is not None else Decimal(-100)),
                        f"< {self._config.minimum_confirmed_strength_score}",
                        MainlineEvidenceSide.OPPOSING,
                        "current industry strength is absent or below the confirmation floor",
                    )
                )
            if state is MainlineState.CROWDED:
                supporting.append(
                    evidence(
                        "crowding_signal_count",
                        Decimal(len(crowding_signals)),
                        f">= {self._config.minimum_crowding_signals}",
                        MainlineEvidenceSide.SUPPORTING,
                        "multiple current heat and concentration signals support CROWDED",
                    )
                )

        for signal in crowding_signals:
            opposing.append(
                evidence(
                    f"crowding:{signal.lower()}",
                    Decimal(1),
                    "triggered",
                    MainlineEvidenceSide.OPPOSING,
                    "crowding is counter-evidence to durable forward leadership",
                )
            )
        return tuple(supporting), tuple(opposing)

    def _invalidations(self) -> tuple[str, ...]:
        regimes = ",".join(value.value for value in self._config.confirmation_regimes)
        return (
            f"rolling Top-{self._config.top_k} hits fall below "
            f"{self._config.minimum_top_k_hits} in "
            f"{self._config.persistence_window_sessions} observed sessions",
            f"stabilized market regime leaves [{regimes}]",
            "current industry strength score falls below "
            f"{self._config.minimum_confirmed_strength_score}",
        )

    @staticmethod
    def _fading_reason(
        *,
        current_top_k: bool,
        market_eligible: bool,
        strength_eligible: bool,
        persistence_eligible: bool,
    ) -> str:
        failed: list[str] = []
        if not current_top_k:
            failed.append("current Top-K rank")
        if not persistence_eligible:
            failed.append("rolling persistence")
        if not market_eligible:
            failed.append("stabilized market regime")
        if not strength_eligible:
            failed.append("industry strength floor")
        return "fading because confirmation lost: " + ", ".join(failed)

    @staticmethod
    def _input_identity(
        industry_snapshot: IndustryStrengthSnapshot,
        regime_result: RegimeTransitionResult,
    ) -> MainlineInputIdentity:
        raw = regime_result.raw_result
        return MainlineInputIdentity(
            industry_feature_version=industry_snapshot.feature_version,
            industry_config_hash=industry_snapshot.config_hash,
            industry_cache_key=industry_snapshot.cache_key,
            classification_version=industry_snapshot.classification_version,
            industry_level=industry_snapshot.industry_level,
            regime_model_version=raw.model_version,
            regime_classifier_config_hash=raw.config_hash,
            regime_input_hash=raw.input_hash,
            regime_trend_feature_version=raw.input_identity.trend_feature_version,
            regime_trend_config_hash=raw.input_identity.trend_config_hash,
            regime_breadth_feature_version=raw.input_identity.breadth_feature_version,
            regime_breadth_config_hash=raw.input_identity.breadth_config_hash,
            regime_transition_version=regime_result.transition_config_version,
            regime_transition_config_hash=regime_result.transition_config_hash,
            regime_transition_result_hash=regime_result.result_hash,
        )


# Explicit aliases for consumers using state-machine or scoring terminology.
MainlineStateMachine = MainlineEngine
MainlineScoringEngine = MainlineEngine


def apply_mainline_states(
    inputs: Iterable[tuple[IndustryStrengthSnapshot, RegimeTransitionResult]],
    config: MainlineConfig | None = None,
) -> tuple[MainlineSnapshot, ...]:
    """Apply a fresh engine to a chronological sequence of PIT input pairs."""

    return MainlineEngine(config).process_all(inputs)


__all__ = [
    "MainlineEngine",
    "MainlineScoringEngine",
    "MainlineStateMachine",
    "apply_mainline_states",
]
