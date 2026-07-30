"""Past-only regime confirmation and transition records."""

from dataclasses import dataclass
from datetime import datetime

from quant_agent.regime.models import MarketRegime, MarketRegimeResult


@dataclass(frozen=True, slots=True)
class TransitionConfig:
    """Versioned confirmation and urgent-risk rules."""

    version: str = "regime_transition_v1"
    confirmation_days: int = 2
    urgent_downtrend_score: float = 25.0

    def __post_init__(self) -> None:
        if self.confirmation_days < 1:
            raise ValueError("confirmation_days must be positive")
        if not 0 <= self.urgent_downtrend_score <= 100:
            raise ValueError("urgent_downtrend_score must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class RegimeTransitionRecord:
    """Raw state, final debounced state and transition explanation."""

    as_of: datetime
    raw_regime: MarketRegime
    final_regime: MarketRegime
    changed: bool
    candidate_count: int
    reason: str
    config_version: str


class RegimeTransitionEngine:
    """Sequential state machine that never reads future classifications."""

    def __init__(self, config: TransitionConfig | None = None) -> None:
        self.config = config or TransitionConfig()
        self._current: MarketRegime | None = None
        self._candidate: MarketRegime | None = None
        self._candidate_count = 0
        self._last_as_of: datetime | None = None

    def apply(self, raw: MarketRegimeResult) -> RegimeTransitionRecord:
        """Apply one strictly chronological raw classification."""

        if self._last_as_of is not None and raw.as_of <= self._last_as_of:
            raise ValueError("regime transitions require strictly increasing as_of")
        self._last_as_of = raw.as_of
        if self._current is None:
            self._current = raw.regime
            return RegimeTransitionRecord(
                as_of=raw.as_of,
                raw_regime=raw.regime,
                final_regime=self._current,
                changed=True,
                candidate_count=0,
                reason="initial state",
                config_version=self.config.version,
            )
        if raw.regime is MarketRegime.DOWNTREND and raw.score <= self.config.urgent_downtrend_score:
            changed = self._current is not MarketRegime.DOWNTREND
            self._current = MarketRegime.DOWNTREND
            self._candidate = None
            self._candidate_count = 0
            return RegimeTransitionRecord(
                as_of=raw.as_of,
                raw_regime=raw.regime,
                final_regime=self._current,
                changed=changed,
                candidate_count=0,
                reason="urgent downside threshold",
                config_version=self.config.version,
            )
        if raw.regime is self._current:
            self._candidate = None
            self._candidate_count = 0
            return RegimeTransitionRecord(
                as_of=raw.as_of,
                raw_regime=raw.regime,
                final_regime=self._current,
                changed=False,
                candidate_count=0,
                reason="raw state confirms current state",
                config_version=self.config.version,
            )
        if raw.regime is self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw.regime
            self._candidate_count = 1
        if self._candidate_count >= self.config.confirmation_days:
            self._current = raw.regime
            self._candidate = None
            self._candidate_count = 0
            return RegimeTransitionRecord(
                as_of=raw.as_of,
                raw_regime=raw.regime,
                final_regime=self._current,
                changed=True,
                candidate_count=0,
                reason="candidate state reached confirmation requirement",
                config_version=self.config.version,
            )
        return RegimeTransitionRecord(
            as_of=raw.as_of,
            raw_regime=raw.regime,
            final_regime=self._current,
            changed=False,
            candidate_count=self._candidate_count,
            reason="candidate state awaiting confirmation",
            config_version=self.config.version,
        )
