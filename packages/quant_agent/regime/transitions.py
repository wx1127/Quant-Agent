"""Point-in-time-only hysteresis for raw market-regime classifications.

The classifier intentionally emits the state supported by *today's* evidence.  This
module adds a small, deterministic state machine for consumers that need a less
noisy effective state.  It never revises an earlier output and never looks ahead.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime.contracts import (
    ConfidenceMeaning,
    MarketRegime,
    MarketRegimeResult,
    RegimeEvidence,
    stable_hash,
)


class RegimeTransitionInputMismatch(ValueError):
    """Raised when a sequence is out of order or mixes incompatible identities."""


class TransitionAction(StrEnum):
    """The decision made by the transition engine for one input session."""

    INITIALIZED = "INITIALIZED"
    STABLE = "STABLE"
    CANDIDATE_STARTED = "CANDIDATE_STARTED"
    CANDIDATE_ADVANCED = "CANDIDATE_ADVANCED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    CANDIDATE_CANCELLED = "CANDIDATE_CANCELLED"
    TRANSITION_CONFIRMED = "TRANSITION_CONFIRMED"


@dataclass(frozen=True, slots=True)
class RegimeTransitionConfig:
    """Versioned confirmation and confidence gates for regime changes.

    ``confirmation_sessions`` counts consecutive *observed trading sessions*, not
    calendar days.  Bottom recovery may deliberately require a longer confirmation
    period.  A sufficiently severe and well-supported downtrend may use the shorter
    emergency period, but the decision is still based only on the current record and
    records already processed.
    """

    version: str = "market-regime-transition-v1"
    confirmation_sessions: int = 3
    bottom_recovery_confirmation_sessions: int = 3
    minimum_confidence: Decimal = Decimal("0.60")
    severe_downtrend_enabled: bool = True
    severe_downtrend_score_max: Decimal = Decimal("25")
    severe_downtrend_minimum_confidence: Decimal = Decimal("0.80")
    severe_downtrend_confirmation_sessions: int = 1

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("transition config version must be non-empty")
        counts = (
            self.confirmation_sessions,
            self.bottom_recovery_confirmation_sessions,
            self.severe_downtrend_confirmation_sessions,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in counts
        ):
            raise ValueError("transition confirmation sessions must be positive integers")
        if not isinstance(self.severe_downtrend_enabled, bool):
            raise ValueError("severe_downtrend_enabled must be boolean")
        if self.severe_downtrend_confirmation_sessions > self.confirmation_sessions:
            raise ValueError("severe downtrend confirmation cannot be slower than normal")
        for value in (
            self.minimum_confidence,
            self.severe_downtrend_minimum_confidence,
            self.severe_downtrend_score_max,
        ):
            canonical_decimal(value, field_name="transition numeric parameter")
        if not Decimal(0) <= self.minimum_confidence <= Decimal(1):
            raise ValueError("minimum transition confidence must be within 0..1")
        if not Decimal(0) <= self.severe_downtrend_minimum_confidence <= Decimal(1):
            raise ValueError("severe downtrend confidence must be within 0..1")
        if self.severe_downtrend_minimum_confidence < self.minimum_confidence:
            raise ValueError("severe downtrend confidence cannot be below the normal minimum")
        if not Decimal(0) <= self.severe_downtrend_score_max <= Decimal(100):
            raise ValueError("severe downtrend score threshold must be within 0..100")

    @property
    def config_hash(self) -> str:
        """Return a stable hash over the complete transition policy."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """Immutable audit record for one sequential transition decision."""

    session_date: date
    as_of: datetime
    raw_regime: MarketRegime
    previous_regime: MarketRegime | None
    final_regime: MarketRegime
    action: TransitionAction
    candidate_regime: MarketRegime | None
    candidate_count: int
    required_confirmation_sessions: int | None
    raw_score: Decimal
    raw_confidence: Decimal
    reason: str
    supporting_evidence: tuple[RegimeEvidence, ...]
    source_result_hash: str
    transition_config_version: str
    transition_config_hash: str
    previous_event_hash: str | None
    event_hash: str
    confidence_meaning: ConfidenceMeaning = field(
        default=ConfidenceMeaning.RULE_EVIDENCE_CONSISTENCY,
        init=False,
    )

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.candidate_count < 0:
            raise ValueError("transition candidate count cannot be negative")
        if self.required_confirmation_sessions is not None:
            if self.required_confirmation_sessions < 1:
                raise ValueError("required confirmation sessions must be positive")
            if self.candidate_regime is None:
                raise ValueError("a required count needs a candidate regime")
        if not self.reason.strip() or not self.transition_config_version.strip():
            raise ValueError("transition event reason and config version must be non-empty")
        if not self.supporting_evidence:
            raise ValueError("transition events require the raw classifier's evidence")
        hashes = (
            self.source_result_hash,
            self.transition_config_hash,
            self.event_hash,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("transition event hashes must be SHA-256 digests")
        if self.previous_event_hash is not None and len(self.previous_event_hash) != 64:
            raise ValueError("previous transition event hash must be a SHA-256 digest")
        if not Decimal(0) <= self.raw_score <= Decimal(100):
            raise ValueError("raw transition score must be within 0..100")
        if not Decimal(0) <= self.raw_confidence <= Decimal(1):
            raise ValueError("raw transition confidence must be within 0..1")

    @property
    def evidence(self) -> tuple[RegimeEvidence, ...]:
        """Compatibility alias for the supporting evidence behind the raw state."""

        return self.supporting_evidence

    @property
    def confidence_is_probability(self) -> bool:
        """The copied classifier confidence remains rule consistency, not probability."""

        return False


@dataclass(frozen=True, slots=True)
class RegimeTransitionResult:
    """Raw and effective states after processing one session."""

    raw_result: MarketRegimeResult
    raw_regime: MarketRegime
    final_regime: MarketRegime
    pending_candidate: MarketRegime | None
    pending_count: int
    required_confirmation_sessions: int | None
    event: TransitionEvent
    transition_config_version: str
    transition_config_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if self.raw_regime is not self.raw_result.regime:
            raise ValueError("raw transition state must match the classifier result")
        if self.event.raw_regime is not self.raw_regime:
            raise ValueError("transition event must describe the same raw state")
        if self.event.final_regime is not self.final_regime:
            raise ValueError("transition event must describe the same final state")
        if self.pending_count < 0:
            raise ValueError("pending transition count cannot be negative")
        if self.pending_candidate is None:
            if self.pending_count != 0 or self.required_confirmation_sessions is not None:
                raise ValueError("an empty pending transition cannot retain confirmation state")
        elif (
            self.pending_count < 1
            or self.required_confirmation_sessions is None
            or self.pending_count >= self.required_confirmation_sessions
        ):
            raise ValueError("pending transition count must be below its required count")
        if not self.transition_config_version.strip():
            raise ValueError("transition config version must be non-empty")
        if any(len(value) != 64 for value in (self.transition_config_hash, self.result_hash)):
            raise ValueError("transition result hashes must be SHA-256 digests")

    @property
    def regime(self) -> MarketRegime:
        """Return the debounced state consumed by downstream systems."""

        return self.final_regime

    @property
    def raw_score(self) -> Decimal:
        """Return the classifier score without reinterpreting it."""

        return self.raw_result.score

    @property
    def confidence(self) -> Decimal:
        """Return raw rule-evidence consistency, never a price probability."""

        return self.raw_result.confidence

    @property
    def transitioned(self) -> bool:
        """Whether this observation confirmed a change of effective state."""

        return self.event.action is TransitionAction.TRANSITION_CONFIRMED

    @property
    def confidence_is_probability(self) -> bool:
        """The transition layer does not change classifier confidence semantics."""

        return False


@dataclass(frozen=True, slots=True)
class _SequenceIdentity:
    data_version: str
    model_version: str
    classifier_config_hash: str
    trend_feature_version: str
    trend_config_hash: str
    breadth_feature_version: str
    breadth_config_hash: str

    @classmethod
    def from_result(cls, result: MarketRegimeResult) -> "_SequenceIdentity":
        identity = result.input_identity
        return cls(
            data_version=result.data_version,
            model_version=result.model_version,
            classifier_config_hash=result.config_hash,
            trend_feature_version=identity.trend_feature_version,
            trend_config_hash=identity.trend_config_hash,
            breadth_feature_version=identity.breadth_feature_version,
            breadth_config_hash=identity.breadth_config_hash,
        )


class RegimeTransitionEngine:
    """Sequential, deterministic, no-lookahead market-regime transition engine."""

    def __init__(self, config: RegimeTransitionConfig | None = None) -> None:
        self._config = config or RegimeTransitionConfig()
        self._sequence_identity: _SequenceIdentity | None = None
        self._last_session_date: date | None = None
        self._current_regime: MarketRegime | None = None
        self._pending_candidate: MarketRegime | None = None
        self._pending_count = 0
        self._pending_required_sessions: int | None = None
        self._events: list[TransitionEvent] = []

    @property
    def config(self) -> RegimeTransitionConfig:
        """Return the immutable transition policy."""

        return self._config

    @property
    def current_regime(self) -> MarketRegime | None:
        """Return the effective state after the most recently processed session."""

        return self._current_regime

    @property
    def pending_candidate(self) -> MarketRegime | None:
        """Return the candidate currently accumulating consecutive confirmations."""

        return self._pending_candidate

    @property
    def pending_count(self) -> int:
        """Return consecutive qualifying sessions for the pending candidate."""

        return self._pending_count

    @property
    def events(self) -> tuple[TransitionEvent, ...]:
        """Return a frozen view of the append-only audit history."""

        return tuple(self._events)

    def process(self, result: MarketRegimeResult) -> RegimeTransitionResult:
        """Process exactly one later session and return its immutable decision."""

        self._validate_next(result)
        previous_regime = self._current_regime
        previous_event_hash = self._events[-1].event_hash if self._events else None

        if previous_regime is None:
            final_regime = result.regime
            pending_candidate = None
            pending_count = 0
            pending_required = None
            action = TransitionAction.INITIALIZED
            candidate_for_event = None
            count_for_event = 0
            required_for_event = None
            reason = "initialized effective state from the first chronological observation"
        elif result.regime is previous_regime:
            final_regime = previous_regime
            pending_candidate = None
            pending_count = 0
            pending_required = None
            if self._pending_candidate is None:
                action = TransitionAction.STABLE
                candidate_for_event = None
                count_for_event = 0
                required_for_event = None
                reason = "raw state agrees with the effective state"
            else:
                action = TransitionAction.CANDIDATE_CANCELLED
                candidate_for_event = self._pending_candidate
                count_for_event = self._pending_count
                required_for_event = self._pending_required_sessions
                reason = "raw state returned to the effective state before confirmation"
        else:
            required = self._required_sessions(result)
            if result.confidence < self._config.minimum_confidence:
                final_regime = previous_regime
                pending_candidate = None
                pending_count = 0
                pending_required = None
                action = TransitionAction.CANDIDATE_REJECTED
                candidate_for_event = result.regime
                count_for_event = 0
                required_for_event = required
                reason = (
                    "candidate rule consistency is below the configured minimum "
                    f"{self._config.minimum_confidence}"
                )
            else:
                same_candidate = self._pending_candidate is result.regime
                candidate_count = self._pending_count + 1 if same_candidate else 1
                if candidate_count >= required:
                    final_regime = result.regime
                    pending_candidate = None
                    pending_count = 0
                    pending_required = None
                    action = TransitionAction.TRANSITION_CONFIRMED
                    candidate_for_event = result.regime
                    count_for_event = candidate_count
                    required_for_event = required
                    if self._is_severe_downtrend(result):
                        reason = "severe downtrend met the configured accelerated confirmation gate"
                    else:
                        reason = "candidate reached its consecutive-session confirmation count"
                else:
                    final_regime = previous_regime
                    pending_candidate = result.regime
                    pending_count = candidate_count
                    pending_required = required
                    action = (
                        TransitionAction.CANDIDATE_ADVANCED
                        if same_candidate
                        else TransitionAction.CANDIDATE_STARTED
                    )
                    candidate_for_event = result.regime
                    count_for_event = candidate_count
                    required_for_event = required
                    reason = (
                        "candidate is accumulating consecutive qualifying sessions "
                        f"({candidate_count}/{required})"
                    )

        event = self._event(
            result=result,
            previous_regime=previous_regime,
            final_regime=final_regime,
            action=action,
            candidate_regime=candidate_for_event,
            candidate_count=count_for_event,
            required_confirmation_sessions=required_for_event,
            reason=reason,
            previous_event_hash=previous_event_hash,
        )
        result_hash = stable_hash(
            {
                "event_hash": event.event_hash,
                "final_regime": final_regime,
                "pending_candidate": pending_candidate,
                "pending_count": pending_count,
                "raw_result_hash": result.result_hash,
                "required_confirmation_sessions": pending_required,
                "transition_config_hash": self._config.config_hash,
            }
        )
        output = RegimeTransitionResult(
            raw_result=result,
            raw_regime=result.regime,
            final_regime=final_regime,
            pending_candidate=pending_candidate,
            pending_count=pending_count,
            required_confirmation_sessions=pending_required,
            event=event,
            transition_config_version=self._config.version,
            transition_config_hash=self._config.config_hash,
            result_hash=result_hash,
        )

        # Mutate only after all validation and immutable output construction succeeds.
        self._sequence_identity = self._sequence_identity or _SequenceIdentity.from_result(result)
        self._last_session_date = result.session_date
        self._current_regime = final_regime
        self._pending_candidate = pending_candidate
        self._pending_count = pending_count
        self._pending_required_sessions = pending_required
        self._events.append(event)
        return output

    def process_all(
        self,
        results: Iterable[MarketRegimeResult],
    ) -> tuple[RegimeTransitionResult, ...]:
        """Process an already chronological iterable without sorting or lookahead."""

        return tuple(self.process(result) for result in results)

    def _validate_next(self, result: MarketRegimeResult) -> None:
        if self._last_session_date is not None and result.session_date <= self._last_session_date:
            raise RegimeTransitionInputMismatch(
                "regime session_date must be strictly increasing; "
                f"received {result.session_date} after {self._last_session_date}"
            )
        identity = _SequenceIdentity.from_result(result)
        if self._sequence_identity is not None and identity != self._sequence_identity:
            raise RegimeTransitionInputMismatch(
                "regime sequence must keep data, model, classifier, and feature versions aligned"
            )

    def _required_sessions(self, result: MarketRegimeResult) -> int:
        if self._is_severe_downtrend(result):
            return self._config.severe_downtrend_confirmation_sessions
        if result.regime is MarketRegime.BOTTOM_RECOVERY:
            return self._config.bottom_recovery_confirmation_sessions
        return self._config.confirmation_sessions

    def _is_severe_downtrend(self, result: MarketRegimeResult) -> bool:
        return (
            self._config.severe_downtrend_enabled
            and result.regime is MarketRegime.DOWNTREND
            and result.score <= self._config.severe_downtrend_score_max
            and result.confidence >= self._config.severe_downtrend_minimum_confidence
        )

    def _event(
        self,
        *,
        result: MarketRegimeResult,
        previous_regime: MarketRegime | None,
        final_regime: MarketRegime,
        action: TransitionAction,
        candidate_regime: MarketRegime | None,
        candidate_count: int,
        required_confirmation_sessions: int | None,
        reason: str,
        previous_event_hash: str | None,
    ) -> TransitionEvent:
        evidence_payload = tuple(
            {
                "contribution": item.contribution,
                "criterion": item.criterion,
                "feature": item.feature,
                "rationale": item.rationale,
                "rule_weight": item.rule_weight,
                "side": item.side,
                "value": item.value,
            }
            for item in result.evidence
        )
        payload = {
            "action": action,
            "as_of": result.as_of,
            "candidate_count": candidate_count,
            "candidate_regime": candidate_regime,
            "final_regime": final_regime,
            "previous_event_hash": previous_event_hash,
            "previous_regime": previous_regime,
            "raw_confidence": result.confidence,
            "raw_regime": result.regime,
            "raw_score": result.score,
            "reason": reason,
            "required_confirmation_sessions": required_confirmation_sessions,
            "session_date": result.session_date,
            "source_result_hash": result.result_hash,
            "supporting_evidence": evidence_payload,
            "transition_config_hash": self._config.config_hash,
            "transition_config_version": self._config.version,
        }
        return TransitionEvent(
            session_date=result.session_date,
            as_of=result.as_of,
            raw_regime=result.regime,
            previous_regime=previous_regime,
            final_regime=final_regime,
            action=action,
            candidate_regime=candidate_regime,
            candidate_count=candidate_count,
            required_confirmation_sessions=required_confirmation_sessions,
            raw_score=result.score,
            raw_confidence=result.confidence,
            reason=reason,
            supporting_evidence=result.evidence,
            source_result_hash=result.result_hash,
            transition_config_version=self._config.version,
            transition_config_hash=self._config.config_hash,
            previous_event_hash=previous_event_hash,
            event_hash=stable_hash(payload),
        )


# Explicit long name for callers that prefer the domain-qualified type.
MarketRegimeTransitionEngine = RegimeTransitionEngine


def apply_regime_transitions(
    results: Iterable[MarketRegimeResult],
    config: RegimeTransitionConfig | None = None,
) -> tuple[RegimeTransitionResult, ...]:
    """Apply a fresh transition engine to a chronological sequence."""

    return RegimeTransitionEngine(config).process_all(results)


__all__ = [
    "MarketRegimeTransitionEngine",
    "RegimeTransitionConfig",
    "RegimeTransitionEngine",
    "RegimeTransitionInputMismatch",
    "RegimeTransitionResult",
    "TransitionAction",
    "TransitionEvent",
    "apply_regime_transitions",
]
