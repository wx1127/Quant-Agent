"""Contracts for deterministic, point-in-time mainline-industry states."""

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime.contracts import MarketRegime, stable_hash


class MainlineInputMismatch(ValueError):
    """Raised when industry and stabilized-regime inputs cannot be combined safely."""


class MainlineState(StrEnum):
    """Lifecycle states exposed for an active or recently active industry."""

    EMERGING = "EMERGING"
    CONFIRMED = "CONFIRMED"
    CROWDED = "CROWDED"
    FADING = "FADING"


class MainlineEvidenceSide(StrEnum):
    """Whether one fact supports or weakens the assigned lifecycle state."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"


@dataclass(frozen=True, slots=True)
class MainlineConfig:
    """Versioned persistence, scoring, crowding, and state-transition policy."""

    version: str = "mainline-v1"
    top_k: int = 5
    persistence_window_sessions: int = 5
    minimum_top_k_hits: int = 3
    minimum_confirmed_strength_score: Decimal = Decimal("0")
    confirmation_regimes: tuple[MarketRegime, ...] = (
        MarketRegime.UPTREND,
        MarketRegime.RANGE_STRONG,
        MarketRegime.DIVERGENT,
        MarketRegime.BOTTOM_RECOVERY,
    )
    fading_retention_sessions: int = 2
    crowding_turnover_growth_min: Decimal = Decimal("0.50")
    crowding_new_high_ratio_min: Decimal = Decimal("0.50")
    crowding_top_member_turnover_share_min: Decimal = Decimal("0.35")
    minimum_crowding_signals: int = 2
    strength_weight: Decimal = Decimal("0.45")
    rank_weight: Decimal = Decimal("0.20")
    persistence_weight: Decimal = Decimal("0.35")

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("mainline config version must be non-empty")
        integer_values = (
            self.top_k,
            self.persistence_window_sessions,
            self.minimum_top_k_hits,
            self.fading_retention_sessions,
            self.minimum_crowding_signals,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integer_values
        ):
            raise ValueError("mainline session counts and top_k must be positive integers")
        if self.persistence_window_sessions < 2:
            raise ValueError("mainline persistence window must contain at least two sessions")
        if not 2 <= self.minimum_top_k_hits <= self.persistence_window_sessions:
            raise ValueError(
                "mainline confirmation requires two or more hits within the persistence window"
            )
        if not self.confirmation_regimes or len(set(self.confirmation_regimes)) != len(
            self.confirmation_regimes
        ):
            raise ValueError("mainline confirmation regimes must be non-empty and unique")
        if any(not isinstance(value, MarketRegime) for value in self.confirmation_regimes):
            raise ValueError("mainline confirmation regimes must contain MarketRegime values")
        if not 1 <= self.minimum_crowding_signals <= 3:
            raise ValueError("minimum_crowding_signals must be within one to three")

        decimal_values = (
            self.minimum_confirmed_strength_score,
            self.crowding_turnover_growth_min,
            self.crowding_new_high_ratio_min,
            self.crowding_top_member_turnover_share_min,
            self.strength_weight,
            self.rank_weight,
            self.persistence_weight,
        )
        for value in decimal_values:
            canonical_decimal(value, field_name="mainline numeric parameter")
        if not Decimal(-100) <= self.minimum_confirmed_strength_score <= Decimal(100):
            raise ValueError("minimum confirmed strength score must be within -100..100")
        if self.crowding_turnover_growth_min < 0:
            raise ValueError("crowding turnover growth threshold cannot be negative")
        ratios = (
            self.crowding_new_high_ratio_min,
            self.crowding_top_member_turnover_share_min,
        )
        if any(value < 0 or value > 1 for value in ratios):
            raise ValueError("mainline crowding ratios must be within 0..1")
        weights = (self.strength_weight, self.rank_weight, self.persistence_weight)
        if any(value < 0 for value in weights):
            raise ValueError("mainline score weights cannot be negative")
        if sum(weights, Decimal(0)) != Decimal(1):
            raise ValueError("mainline score weights must sum to one")

    @property
    def config_hash(self) -> str:
        """Return a stable digest over every state and scoring parameter."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class TopKPersistence:
    """Recent observed-session ranks and derived persistence statistics."""

    top_k: int
    window_sessions: int
    observed_sessions: int
    top_k_hits: int
    consecutive_top_k_sessions: int
    recent_ranks: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.window_sessions < 1:
            raise ValueError("top-K persistence dimensions must be positive")
        if self.observed_sessions != len(self.recent_ranks):
            raise ValueError("observed_sessions must equal the number of recent ranks")
        if not 0 <= self.observed_sessions <= self.window_sessions:
            raise ValueError("observed persistence sessions cannot exceed the window")
        if any(rank is not None and rank < 1 for rank in self.recent_ranks):
            raise ValueError("observed industry ranks must be positive")
        expected_hits = sum(rank is not None and rank <= self.top_k for rank in self.recent_ranks)
        if self.top_k_hits != expected_hits:
            raise ValueError("top_k_hits must match recent ranks")
        expected_consecutive = 0
        for rank in reversed(self.recent_ranks):
            if rank is None or rank > self.top_k:
                break
            expected_consecutive += 1
        if self.consecutive_top_k_sessions != expected_consecutive:
            raise ValueError("consecutive top-K sessions must match recent ranks")

    @property
    def hit_ratio(self) -> Decimal:
        """Return hits divided by the configured window, including unobserved warm-up slots."""

        return Decimal(self.top_k_hits) / self.window_sessions


@dataclass(frozen=True, slots=True)
class MainlineEvidence:
    """One auditable fact used by the mainline state machine."""

    feature: str
    value: Decimal
    criterion: str
    side: MainlineEvidenceSide
    rationale: str

    def __post_init__(self) -> None:
        if not self.feature.strip() or not self.criterion.strip() or not self.rationale.strip():
            raise ValueError("mainline evidence text fields must be non-empty")
        if not self.value.is_finite():
            raise ValueError("mainline evidence values must be finite")


@dataclass(frozen=True, slots=True)
class MainlineInputIdentity:
    """Exact upstream industry-feature and stabilized-regime identities."""

    industry_feature_version: str
    industry_config_hash: str
    industry_cache_key: str
    classification_version: str
    industry_level: int
    regime_model_version: str
    regime_classifier_config_hash: str
    regime_input_hash: str
    regime_trend_feature_version: str
    regime_trend_config_hash: str
    regime_breadth_feature_version: str
    regime_breadth_config_hash: str
    regime_transition_version: str
    regime_transition_config_hash: str
    regime_transition_result_hash: str

    def __post_init__(self) -> None:
        versions = (
            self.industry_feature_version,
            self.classification_version,
            self.regime_model_version,
            self.regime_trend_feature_version,
            self.regime_breadth_feature_version,
            self.regime_transition_version,
        )
        if any(not value.strip() for value in versions):
            raise ValueError("mainline upstream versions must be non-empty")
        hashes = (
            self.industry_config_hash,
            self.industry_cache_key,
            self.regime_classifier_config_hash,
            self.regime_input_hash,
            self.regime_trend_config_hash,
            self.regime_breadth_config_hash,
            self.regime_transition_config_hash,
            self.regime_transition_result_hash,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("mainline upstream identities must contain SHA-256 hashes")
        if self.industry_level not in (1, 2, 3):
            raise ValueError("mainline industry level must be 1, 2, or 3")


@dataclass(frozen=True, slots=True)
class MainlineIndustryResult:
    """One current or retained industry lifecycle decision."""

    industry_id: str
    state: MainlineState
    previous_state: MainlineState | None
    current_rank: int | None
    current_strength_score: Decimal | None
    mainline_score: Decimal
    crowding_score: Decimal
    persistence: TopKPersistence
    crowding_signals: tuple[str, ...]
    supporting_evidence: tuple[MainlineEvidence, ...]
    counter_evidence: tuple[MainlineEvidence, ...]
    invalidations: tuple[str, ...]
    transition_reason: str

    def __post_init__(self) -> None:
        if not self.industry_id.strip():
            raise ValueError("mainline industry_id must be non-empty")
        if self.current_rank is not None and self.current_rank < 1:
            raise ValueError("current mainline rank must be positive")
        if self.current_strength_score is not None and not self.current_strength_score.is_finite():
            raise ValueError("current industry strength score must be finite")
        for score in (self.mainline_score, self.crowding_score):
            if score < 0 or score > 100 or not score.is_finite():
                raise ValueError("mainline and crowding scores must be finite within 0..100")
        if len(set(self.crowding_signals)) != len(self.crowding_signals):
            raise ValueError("mainline crowding signals must be unique")
        if not self.supporting_evidence:
            raise ValueError("mainline results require supporting evidence")
        if any(
            item.side is not MainlineEvidenceSide.SUPPORTING for item in self.supporting_evidence
        ):
            raise ValueError("supporting_evidence can contain only supporting facts")
        if any(item.side is not MainlineEvidenceSide.OPPOSING for item in self.counter_evidence):
            raise ValueError("counter_evidence can contain only opposing facts")
        if not self.invalidations or any(not value.strip() for value in self.invalidations):
            raise ValueError("mainline results require non-empty invalidation conditions")
        if not self.transition_reason.strip():
            raise ValueError("mainline transition reason must be non-empty")

    @property
    def status(self) -> MainlineState:
        """Compatibility alias for consumers that call lifecycle states statuses."""

        return self.state

    @property
    def rank(self) -> int | None:
        """Return the current industry-strength rank."""

        return self.current_rank

    @property
    def top_k_hits(self) -> int:
        """Return the recent-window Top-K hit count."""

        return self.persistence.top_k_hits


@dataclass(frozen=True, slots=True)
class MainlineSnapshot:
    """Append-only mainline decisions for one point-in-time market session."""

    session_date: date
    as_of: datetime
    data_version: str
    classification_version: str
    industry_level: int
    market_regime: MarketRegime
    model_version: str
    config_hash: str
    input_identity: MainlineInputIdentity
    input_hash: str
    industries: tuple[MainlineIndustryResult, ...]
    previous_result_hash: str | None
    result_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.session_date > self.as_of.date():
            raise ValueError("mainline session_date cannot be after as_of")
        versions = (self.data_version, self.classification_version, self.model_version)
        if any(not value.strip() for value in versions):
            raise ValueError("mainline snapshot versions must be non-empty")
        if self.industry_level not in (1, 2, 3):
            raise ValueError("mainline industry_level must be 1, 2, or 3")
        hashes = (self.config_hash, self.input_hash, self.result_hash)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("mainline snapshot hashes must be SHA-256 digests")
        if self.previous_result_hash is not None and len(self.previous_result_hash) != 64:
            raise ValueError("previous mainline result hash must be a SHA-256 digest")
        industry_ids = tuple(item.industry_id for item in self.industries)
        if len(set(industry_ids)) != len(industry_ids):
            raise ValueError("mainline snapshot industry identifiers must be unique")

    @property
    def results(self) -> tuple[MainlineIndustryResult, ...]:
        """Compatibility alias for the per-industry decisions."""

        return self.industries

    @property
    def top_k_industries(self) -> tuple[MainlineIndustryResult, ...]:
        """Return currently ranked Top-K rows, including emerging candidates."""

        return tuple(
            item
            for item in self.industries
            if item.current_rank is not None and item.current_rank <= item.persistence.top_k
        )

    def identity_payload(self) -> dict[str, str]:
        """Return stable identities needed to reproduce this session's result."""

        return {
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "input_hash": self.input_hash,
            "model_version": self.model_version,
            "result_hash": self.result_hash,
        }


# Domain-qualified aliases for callers that prefer longer names.
MainlineScoringConfig = MainlineConfig
MainlineStatus = MainlineState


__all__ = [
    "MainlineConfig",
    "MainlineEvidence",
    "MainlineEvidenceSide",
    "MainlineIndustryResult",
    "MainlineInputIdentity",
    "MainlineInputMismatch",
    "MainlineScoringConfig",
    "MainlineSnapshot",
    "MainlineState",
    "MainlineStatus",
    "TopKPersistence",
]
