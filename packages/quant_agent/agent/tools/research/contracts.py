"""Strict Agent-visible contracts for deterministic research tools.

These models intentionally project domain records into schema-closed Pydantic
objects.  Model-controlled arguments contain only bounded presentation choices;
decision, dataset, account, and model identities always come from the trusted
``ToolExecutionContext`` and research-input boundary.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from quant_agent.agent.tools.contracts import ToolArguments, ToolOutput
from quant_agent.core.time import ensure_aware
from quant_agent.data.quality import QualitySeverity
from quant_agent.features.mainline.contracts import (
    MainlineEvidenceSide,
    MainlineState,
)
from quant_agent.leaders.candidates.contracts import (
    CandidateComponent,
    CandidateEvidenceSide,
    CandidateExclusionCode,
    CandidateRiskCode,
    CandidateTier,
)
from quant_agent.leaders.contracts import (
    LeaderComponent,
    LeaderEvidenceSide,
    LeaderExclusionCode,
    LeaderRiskCode,
    LeaderType,
)
from quant_agent.regime.contracts import (
    ConfidenceMeaning,
    EvidenceSide,
    MarketRegime,
)
from quant_agent.regime.transitions import TransitionAction


def _validate_exact_text(value: str) -> str:
    if value != value.strip() or not value.isprintable():
        raise ValueError("text must be printable and have no surrounding whitespace")
    return value


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=256),
    AfterValidator(_validate_exact_text),
]
ShortText = Annotated[
    str,
    Field(min_length=1, max_length=512),
    AfterValidator(_validate_exact_text),
]
LongText = Annotated[
    str,
    Field(min_length=1, max_length=4_000),
    AfterValidator(_validate_exact_text),
]
Version = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]
Score = Annotated[Decimal, Field(ge=0, le=100, allow_inf_nan=False)]
Ratio = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveRank = Annotated[int, Field(ge=1, le=1_000_000)]
NonNegativeCount = Annotated[int, Field(ge=0, le=1_000_000_000_000)]
IndustryLevel = Annotated[int, Field(ge=1, le=3)]


def _validate_aware(value: datetime) -> datetime:
    """Reject naive timestamps without changing the represented offset."""

    ensure_aware(value)
    return value


class _DecisionBoundOutput(ToolOutput):
    """Fields repeated in data payloads to make their binding self-contained."""

    decision_id: Annotated[str, Field(min_length=1, max_length=128)]
    as_of: datetime
    data_version: Version

    _aware_as_of = field_validator("as_of")(_validate_aware)


class _AccountDecisionBoundOutput(_DecisionBoundOutput):
    """Decision binding for account-derived research calculations."""

    account_id: Identifier
    account_snapshot_id: Identifier
    account_snapshot_hash: Sha256


class GetMarketSnapshotArguments(ToolArguments):
    """Bound the number of manifest files returned to the model."""

    max_files: Annotated[int, Field(ge=1, le=100)] = 50


class ValidateMarketDataArguments(ToolArguments):
    """Bound the number of quality issues returned to the model."""

    max_issues: Annotated[int, Field(ge=1, le=500)] = 100


class DetectMarketRegimeArguments(ToolArguments):
    """The decision context supplies every regime input and identity."""


class RankedListArguments(ToolArguments):
    """Presentation-only limit shared by theme and leader rankings."""

    limit: Annotated[int, Field(ge=1, le=50)] = 10


class RankCandidatesArguments(ToolArguments):
    """Presentation-only controls for the deterministic candidate snapshot."""

    limit: Annotated[int, Field(ge=1, le=50)] = 20
    include_excluded: bool = False


class ExplainCandidateArguments(ToolArguments):
    """Select one already-computed candidate without accepting hidden context."""

    instrument_id: Identifier


class SnapshotFileOutput(ToolOutput):
    """Closed projection of one immutable snapshot file."""

    name: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    row_count: NonNegativeCount
    sha256: Sha256


class MarketSnapshotOutput(_DecisionBoundOutput):
    """Verified manifest summary for the authorized market-data snapshot."""

    market: Literal["CN_A"]
    data_content_hash: Sha256
    created_at: Annotated[str, Field(min_length=1, max_length=128)]
    total_file_count: Annotated[int, Field(ge=1, le=10_000)]
    returned_file_count: Annotated[int, Field(ge=0, le=100)]
    total_row_count: NonNegativeCount
    files_truncated: bool
    files: Annotated[tuple[SnapshotFileOutput, ...], Field(max_length=100)]

    @model_validator(mode="after")
    def validate_file_counts(self) -> Self:
        if self.returned_file_count != len(self.files):
            raise ValueError("returned_file_count must equal files length")
        if self.returned_file_count > self.total_file_count:
            raise ValueError("returned_file_count cannot exceed total_file_count")
        if self.files_truncated != (self.returned_file_count < self.total_file_count):
            raise ValueError("files_truncated must match the returned file count")
        return self


class QualityIssueOutput(ToolOutput):
    """One bounded data-quality finding."""

    rule_id: Annotated[str, Field(min_length=1, max_length=128)]
    severity: QualitySeverity
    entity_key: Annotated[str, Field(min_length=1, max_length=512)] | None
    message: LongText


class MarketDataQualityOutput(_DecisionBoundOutput):
    """Complete quality outcome with a bounded issue projection."""

    data_content_hash: Sha256
    observed_at: datetime
    qualified: bool
    total_issue_count: NonNegativeCount
    blocking_issue_count: NonNegativeCount
    returned_issue_count: Annotated[int, Field(ge=0, le=500)]
    issues_truncated: bool
    issues: Annotated[tuple[QualityIssueOutput, ...], Field(max_length=500)]

    _aware_observed_at = field_validator("observed_at")(_validate_aware)

    @model_validator(mode="after")
    def validate_issue_counts(self) -> Self:
        if self.observed_at != self.as_of:
            raise ValueError("quality observed_at must equal the decision as_of")
        if self.returned_issue_count != len(self.issues):
            raise ValueError("returned_issue_count must equal issues length")
        if self.returned_issue_count > self.total_issue_count:
            raise ValueError("returned_issue_count cannot exceed total_issue_count")
        if self.blocking_issue_count > self.total_issue_count:
            raise ValueError("blocking_issue_count cannot exceed total_issue_count")
        if self.qualified != (self.blocking_issue_count == 0):
            raise ValueError("qualified must reflect the blocking issue count")
        if self.issues_truncated != (self.returned_issue_count < self.total_issue_count):
            raise ValueError("issues_truncated must match the returned issue count")
        return self


class RegimeComponentScoresOutput(ToolOutput):
    """Classifier components and their exact weighted terms."""

    trend: Score
    breadth: Score
    turnover: Score
    new_high_low: Score
    downside_risk: Score
    weighted_trend: Score
    weighted_breadth: Score
    weighted_turnover: Score
    weighted_new_high_low: Score
    weighted_downside_risk: Score
    total: Score

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        expected = sum(
            (
                self.weighted_trend,
                self.weighted_breadth,
                self.weighted_turnover,
                self.weighted_new_high_low,
                self.weighted_downside_risk,
            ),
            Decimal(0),
        )
        if self.total != expected:
            raise ValueError("regime component total must equal weighted terms")
        return self


class RegimeEvidenceOutput(ToolOutput):
    """One exact classifier fact and its confidence contribution."""

    feature: ShortText
    value: FiniteDecimal
    criterion: LongText
    rule_weight: Annotated[Decimal, Field(gt=0, le=1, allow_inf_nan=False)]
    contribution: Score
    side: EvidenceSide
    rationale: LongText


class RegimeInputIdentityOutput(ToolOutput):
    """Exact trend and breadth identities used by the classifier."""

    trend_feature_version: Version
    trend_config_hash: Sha256
    trend_cache_key: Sha256
    breadth_feature_version: Version
    breadth_config_hash: Sha256
    breadth_cache_key: Sha256


class RegimeClassificationOutput(ToolOutput):
    """Full raw classification before transition hysteresis."""

    as_of: datetime
    session_date: date
    data_version: Version
    regime: MarketRegime
    raw_score: Score
    confidence: Ratio
    confidence_meaning: ConfidenceMeaning
    confidence_definition: ShortText
    confidence_is_probability: Literal[False] = False
    risk_budget_max: Ratio
    components: RegimeComponentScoresOutput
    supporting_evidence: Annotated[
        tuple[RegimeEvidenceOutput, ...], Field(min_length=1, max_length=64)
    ]
    counter_evidence: Annotated[tuple[RegimeEvidenceOutput, ...], Field(max_length=64)]
    invalidations: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    model_version: Version
    config_hash: Sha256
    input_identity: RegimeInputIdentityOutput
    input_hash: Sha256
    result_hash: Sha256

    _aware_as_of = field_validator("as_of")(_validate_aware)

    @model_validator(mode="after")
    def validate_score(self) -> Self:
        if self.raw_score != self.components.total:
            raise ValueError("raw_score must equal the component total")
        return self


class RegimeTransitionEventOutput(ToolOutput):
    """Complete append-only event emitted by regime transition hysteresis."""

    session_date: date
    as_of: datetime
    raw_regime: MarketRegime
    previous_regime: MarketRegime | None
    final_regime: MarketRegime
    action: TransitionAction
    candidate_regime: MarketRegime | None
    candidate_count: NonNegativeCount
    required_confirmation_sessions: Annotated[int, Field(ge=1, le=1_000)] | None
    raw_score: Score
    raw_confidence: Ratio
    confidence_meaning: ConfidenceMeaning
    confidence_is_probability: Literal[False] = False
    reason: LongText
    supporting_evidence: Annotated[
        tuple[RegimeEvidenceOutput, ...], Field(min_length=1, max_length=64)
    ]
    source_result_hash: Sha256
    transition_config_version: Version
    transition_config_hash: Sha256
    previous_event_hash: Sha256 | None
    event_hash: Sha256

    _aware_as_of = field_validator("as_of")(_validate_aware)


class MarketRegimeOutput(_DecisionBoundOutput):
    """Raw and stabilized regime result for the authorized decision."""

    session_date: date
    raw: RegimeClassificationOutput
    effective_regime: MarketRegime
    pending_candidate: MarketRegime | None
    pending_count: NonNegativeCount
    required_confirmation_sessions: Annotated[int, Field(ge=1, le=1_000)] | None
    transitioned: bool
    confidence_is_probability: Literal[False] = False
    event: RegimeTransitionEventOutput
    transition_config_version: Version
    transition_config_hash: Sha256
    result_hash: Sha256

    @model_validator(mode="after")
    def validate_transition_projection(self) -> Self:
        if self.raw.as_of != self.as_of or self.raw.data_version != self.data_version:
            raise ValueError("raw regime identity must match the decision-bound output")
        if (
            self.raw.session_date != self.session_date
            or self.event.session_date != self.session_date
        ):
            raise ValueError("regime session dates must match")
        if self.event.as_of != self.as_of:
            raise ValueError("transition event as_of must match the decision-bound output")
        if self.event.raw_regime is not self.raw.regime:
            raise ValueError("transition event raw regime must match the classifier")
        if self.event.final_regime is not self.effective_regime:
            raise ValueError("transition event final regime must match the effective regime")
        if self.event.source_result_hash != self.raw.result_hash:
            raise ValueError("transition event must bind the raw classifier result")
        if self.event.transition_config_version != self.transition_config_version:
            raise ValueError("transition config versions must match")
        if self.event.transition_config_hash != self.transition_config_hash:
            raise ValueError("transition config hashes must match")
        if self.transitioned != (self.event.action is TransitionAction.TRANSITION_CONFIRMED):
            raise ValueError("transitioned must reflect the transition event action")
        if self.pending_candidate is None:
            if self.pending_count != 0 or self.required_confirmation_sessions is not None:
                raise ValueError("empty pending state cannot retain counters")
        elif (
            self.pending_count < 1
            or self.required_confirmation_sessions is None
            or self.pending_count >= self.required_confirmation_sessions
        ):
            raise ValueError("pending state requires an incomplete confirmation count")
        return self


class MainlinePersistenceOutput(ToolOutput):
    """Top-K history and persistence statistics for one industry."""

    top_k: Annotated[int, Field(ge=1, le=10_000)]
    window_sessions: Annotated[int, Field(ge=1, le=512)]
    observed_sessions: Annotated[int, Field(ge=0, le=512)]
    top_k_hits: Annotated[int, Field(ge=0, le=512)]
    consecutive_top_k_sessions: Annotated[int, Field(ge=0, le=512)]
    recent_ranks: Annotated[tuple[PositiveRank | None, ...], Field(max_length=512)]
    hit_ratio: Ratio

    @model_validator(mode="after")
    def validate_persistence(self) -> Self:
        if self.observed_sessions != len(self.recent_ranks):
            raise ValueError("observed_sessions must equal recent_ranks length")
        if self.observed_sessions > self.window_sessions:
            raise ValueError("observed_sessions cannot exceed window_sessions")
        expected_hits = sum(rank is not None and rank <= self.top_k for rank in self.recent_ranks)
        if self.top_k_hits != expected_hits:
            raise ValueError("top_k_hits must match recent_ranks")
        expected_consecutive = 0
        for rank in reversed(self.recent_ranks):
            if rank is None or rank > self.top_k:
                break
            expected_consecutive += 1
        if self.consecutive_top_k_sessions != expected_consecutive:
            raise ValueError("consecutive_top_k_sessions must match recent_ranks")
        if self.hit_ratio != Decimal(self.top_k_hits) / self.window_sessions:
            raise ValueError("hit_ratio must equal top_k_hits divided by window_sessions")
        return self


class MainlineEvidenceOutput(ToolOutput):
    """One exact supporting or opposing mainline fact."""

    feature: ShortText
    value: FiniteDecimal
    criterion: LongText
    side: MainlineEvidenceSide
    rationale: LongText


class MainlineInputIdentityOutput(ToolOutput):
    """Exact feature, classification, and regime identities for mainlines."""

    industry_feature_version: Version
    industry_config_hash: Sha256
    industry_cache_key: Sha256
    classification_version: Version
    industry_level: IndustryLevel
    regime_model_version: Version
    regime_classifier_config_hash: Sha256
    regime_input_hash: Sha256
    regime_trend_feature_version: Version
    regime_trend_config_hash: Sha256
    regime_breadth_feature_version: Version
    regime_breadth_config_hash: Sha256
    regime_transition_version: Version
    regime_transition_config_hash: Sha256
    regime_transition_result_hash: Sha256


class ThemeRowOutput(ToolOutput):
    """One current or retained industry lifecycle result."""

    industry_id: Identifier
    state: MainlineState
    previous_state: MainlineState | None
    current_rank: PositiveRank | None
    current_strength_score: FiniteDecimal | None
    mainline_score: Score
    crowding_score: Score
    persistence: MainlinePersistenceOutput
    crowding_signals: Annotated[tuple[ShortText, ...], Field(max_length=64)]
    supporting_evidence: Annotated[
        tuple[MainlineEvidenceOutput, ...], Field(min_length=1, max_length=64)
    ]
    counter_evidence: Annotated[tuple[MainlineEvidenceOutput, ...], Field(max_length=64)]
    invalidations: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    transition_reason: LongText


class ThemeRankingOutput(_DecisionBoundOutput):
    """Bounded projection of the deterministic mainline-industry snapshot."""

    session_date: date
    classification_version: Version
    industry_level: IndustryLevel
    market_regime: MarketRegime
    model_version: Version
    config_hash: Sha256
    input_identity: MainlineInputIdentityOutput
    input_hash: Sha256
    previous_result_hash: Sha256 | None
    result_hash: Sha256
    total_theme_count: NonNegativeCount
    returned_theme_count: Annotated[int, Field(ge=0, le=50)]
    themes_truncated: bool
    themes: Annotated[tuple[ThemeRowOutput, ...], Field(max_length=50)]

    @model_validator(mode="after")
    def validate_theme_counts(self) -> Self:
        if self.returned_theme_count != len(self.themes):
            raise ValueError("returned_theme_count must equal themes length")
        if self.returned_theme_count > self.total_theme_count:
            raise ValueError("returned_theme_count cannot exceed total_theme_count")
        if self.themes_truncated != (self.returned_theme_count < self.total_theme_count):
            raise ValueError("themes_truncated must match the returned theme count")
        return self


class LeaderUpstreamIdentityOutput(ToolOutput):
    """Exact mainline and stock-feature identities for one leader."""

    mainline_model_version: Version
    mainline_result_hash: Sha256
    stock_feature_version: Version
    stock_config_hash: Sha256
    stock_input_hash: Sha256
    stock_result_hash: Sha256
    tradeability_feature_version: Version
    tradeability_config_hash: Sha256
    tradeability_rule_version: Version
    tradeability_rule_hash: Sha256
    tradeability_result_hash: Sha256
    fundamental_feature_version: Version
    fundamental_config_hash: Sha256
    fundamental_data_version: Version
    fundamental_input_hash: Sha256
    fundamental_result_hash: Sha256


class LeaderComponentScoreOutput(ToolOutput):
    """One leader score component and its exact weighted contribution."""

    component: LeaderComponent
    score: Score
    weight: Ratio
    contribution: Score

    @model_validator(mode="after")
    def validate_contribution(self) -> Self:
        if self.contribution != self.score * self.weight:
            raise ValueError("leader contribution must equal score times weight")
        return self


class LeaderEvidenceOutput(ToolOutput):
    """One exact supporting or opposing leader fact."""

    feature: ShortText
    value: FiniteDecimal
    criterion: LongText
    side: LeaderEvidenceSide
    rationale: LongText


class LeaderRiskOutput(ToolOutput):
    """One applied leader risk penalty."""

    code: LeaderRiskCode
    penalty: Annotated[Decimal, Field(gt=0, le=100, allow_inf_nan=False)]
    rationale: LongText


class LeaderRowOutput(ToolOutput):
    """One fully auditable ranked theme leader."""

    instrument_id: Identifier
    industry_id: Identifier
    mainline_state: MainlineState
    overall_rank: PositiveRank
    theme_rank: PositiveRank
    leader_type: LeaderType
    gross_score: Score
    risk_penalty: Score
    score: Score
    components: Annotated[
        tuple[LeaderComponentScoreOutput, ...], Field(min_length=1, max_length=16)
    ]
    supporting_evidence: Annotated[tuple[LeaderEvidenceOutput, ...], Field(max_length=128)]
    counter_evidence: Annotated[tuple[LeaderEvidenceOutput, ...], Field(max_length=128)]
    risks: Annotated[tuple[LeaderRiskOutput, ...], Field(max_length=32)]
    candidate_eligible: bool
    ineligibility_reasons: Annotated[tuple[LongText, ...], Field(max_length=32)]
    observation_conditions: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    invalidations: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    upstream_identity: LeaderUpstreamIdentityOutput

    @model_validator(mode="after")
    def validate_scores_and_gate(self) -> Self:
        if tuple(item.component for item in self.components) != tuple(LeaderComponent):
            raise ValueError("leader components must use canonical order")
        if sum((item.weight for item in self.components), Decimal(0)) != Decimal(1):
            raise ValueError("leader component weights must sum to one")
        if sum((item.contribution for item in self.components), Decimal(0)) != self.gross_score:
            raise ValueError("leader gross_score must equal component contributions")
        if sum((item.penalty for item in self.risks), Decimal(0)) != self.risk_penalty:
            raise ValueError("leader risk_penalty must equal risk records")
        if self.score != max(Decimal(0), self.gross_score - self.risk_penalty):
            raise ValueError("leader score must deduct risk_penalty")
        if self.candidate_eligible == bool(self.ineligibility_reasons):
            raise ValueError("candidate eligibility must match ineligibility reasons")
        return self


class LeaderExclusionOutput(ToolOutput):
    """One stock excluded before leader ranking."""

    instrument_id: Identifier
    industry_id: Identifier | None
    code: LeaderExclusionCode
    reason: LongText
    stock_result_hash: Sha256


class LeaderRankingOutput(_AccountDecisionBoundOutput):
    """Bounded projection of an exact account-bound leader snapshot."""

    session_date: date
    classification_version: Version
    industry_level: IndustryLevel
    feature_version: Version
    config_hash: Sha256
    mainline_result_hash: Sha256
    input_hash: Sha256
    result_hash: Sha256
    total_leader_count: NonNegativeCount
    returned_leader_count: Annotated[int, Field(ge=0, le=50)]
    leaders_truncated: bool
    total_exclusion_count: NonNegativeCount
    returned_exclusion_count: Annotated[int, Field(ge=0, le=200)]
    exclusions_truncated: bool
    leaders: Annotated[tuple[LeaderRowOutput, ...], Field(max_length=50)]
    exclusions: Annotated[tuple[LeaderExclusionOutput, ...], Field(max_length=200)]

    @model_validator(mode="after")
    def validate_result_counts(self) -> Self:
        if self.returned_leader_count != len(self.leaders):
            raise ValueError("returned_leader_count must equal leaders length")
        if self.returned_exclusion_count != len(self.exclusions):
            raise ValueError("returned_exclusion_count must equal exclusions length")
        if self.returned_leader_count > self.total_leader_count:
            raise ValueError("returned_leader_count cannot exceed total_leader_count")
        if self.returned_exclusion_count > self.total_exclusion_count:
            raise ValueError("returned_exclusion_count cannot exceed total_exclusion_count")
        if self.leaders_truncated != (self.returned_leader_count < self.total_leader_count):
            raise ValueError("leaders_truncated must match the returned leader count")
        if self.exclusions_truncated != (
            self.returned_exclusion_count < self.total_exclusion_count
        ):
            raise ValueError("exclusions_truncated must match the returned exclusion count")
        return self


class CandidateUpstreamIdentityOutput(ToolOutput):
    """Exact upstream regime, theme, leader, and stock identities."""

    regime_transition_version: Version
    regime_transition_hash: Sha256
    mainline_model_version: Version
    mainline_result_hash: Sha256
    leader_feature_version: Version
    leader_result_hash: Sha256
    stock_result_hash: Sha256
    tradeability_result_hash: Sha256
    fundamental_result_hash: Sha256
    event_result_hash: Sha256 | None
    valuation_result_hash: Sha256 | None


class CandidateComponentScoreOutput(ToolOutput):
    """One candidate score component with its upstream source label."""

    component: CandidateComponent
    score: Score
    weight: Ratio
    contribution: Score
    source: ShortText

    @model_validator(mode="after")
    def validate_contribution(self) -> Self:
        if self.contribution != self.score * self.weight:
            raise ValueError("candidate contribution must equal score times weight")
        return self


class CandidateEvidenceOutput(ToolOutput):
    """One exact supporting or opposing candidate fact."""

    feature: ShortText
    value: FiniteDecimal
    criterion: LongText
    side: CandidateEvidenceSide
    rationale: LongText


class CandidateRiskOutput(ToolOutput):
    """One applied candidate risk penalty."""

    code: CandidateRiskCode
    penalty: Annotated[Decimal, Field(gt=0, le=100, allow_inf_nan=False)]
    rationale: LongText


class CandidateRowOutput(ToolOutput):
    """One scored candidate, including scored-but-excluded rows."""

    instrument_id: Identifier
    industry_id: Identifier
    research_rank: PositiveRank
    eligible_rank: PositiveRank | None
    tier: CandidateTier
    eligible: bool
    gross_score: Score
    risk_penalty: Score
    score: Score
    components: Annotated[
        tuple[CandidateComponentScoreOutput, ...], Field(min_length=1, max_length=16)
    ]
    supporting_evidence: Annotated[tuple[CandidateEvidenceOutput, ...], Field(max_length=128)]
    counter_evidence: Annotated[tuple[CandidateEvidenceOutput, ...], Field(max_length=128)]
    risks: Annotated[tuple[CandidateRiskOutput, ...], Field(max_length=32)]
    exclusion_codes: Annotated[tuple[CandidateExclusionCode, ...], Field(max_length=16)]
    exclusion_reasons: Annotated[tuple[LongText, ...], Field(max_length=16)]
    observation_conditions: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    invalidations: Annotated[tuple[LongText, ...], Field(min_length=1, max_length=64)]
    upstream_identity: CandidateUpstreamIdentityOutput
    calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_scores_and_gate(self) -> Self:
        if tuple(item.component for item in self.components) != tuple(CandidateComponent):
            raise ValueError("candidate components must use canonical order")
        if sum((item.weight for item in self.components), Decimal(0)) != Decimal(1):
            raise ValueError("candidate component weights must sum to one")
        if sum((item.contribution for item in self.components), Decimal(0)) != self.gross_score:
            raise ValueError("candidate gross_score must equal component contributions")
        if sum((item.penalty for item in self.risks), Decimal(0)) != self.risk_penalty:
            raise ValueError("candidate risk_penalty must equal risk records")
        if self.score != max(Decimal(0), self.gross_score - self.risk_penalty):
            raise ValueError("candidate score must deduct risk_penalty")
        excluded = bool(self.exclusion_codes)
        if excluded != bool(self.exclusion_reasons):
            raise ValueError("candidate exclusion codes and reasons must appear together")
        if excluded != (self.tier is CandidateTier.EXCLUDED):
            raise ValueError("EXCLUDED tier must match hard-gate reasons")
        if self.eligible != (not excluded):
            raise ValueError("eligible must match the candidate tier")
        if self.eligible != (self.eligible_rank is not None):
            raise ValueError("eligible candidates require an eligible_rank")
        return self


class CandidateExclusionOutput(ToolOutput):
    """One upstream stock that could not receive a candidate score."""

    instrument_id: Identifier
    industry_id: Identifier | None
    code: CandidateExclusionCode
    reason: LongText
    source_hash: Sha256


class CandidateRankingOutput(_AccountDecisionBoundOutput):
    """Bounded projection of an explicitly uncalibrated candidate snapshot."""

    session_date: date
    feature_version: Version
    config_hash: Sha256
    regime: MarketRegime
    calibrated: Literal[False] = False
    calibration_version: None = None
    input_hash: Sha256
    result_hash: Sha256
    excluded_candidates_included: bool
    total_candidate_count: NonNegativeCount
    returned_candidate_count: Annotated[int, Field(ge=0, le=50)]
    candidates_truncated: bool
    total_exclusion_count: NonNegativeCount
    returned_exclusion_count: Annotated[int, Field(ge=0, le=200)]
    exclusions_truncated: bool
    candidates: Annotated[tuple[CandidateRowOutput, ...], Field(max_length=50)]
    exclusions: Annotated[tuple[CandidateExclusionOutput, ...], Field(max_length=200)]

    @model_validator(mode="after")
    def validate_result_counts(self) -> Self:
        if self.returned_candidate_count != len(self.candidates):
            raise ValueError("returned_candidate_count must equal candidates length")
        if self.returned_exclusion_count != len(self.exclusions):
            raise ValueError("returned_exclusion_count must equal exclusions length")
        if self.returned_candidate_count > self.total_candidate_count:
            raise ValueError("returned_candidate_count cannot exceed total_candidate_count")
        if self.returned_exclusion_count > self.total_exclusion_count:
            raise ValueError("returned_exclusion_count cannot exceed total_exclusion_count")
        if self.candidates_truncated != (
            self.returned_candidate_count < self.total_candidate_count
        ):
            raise ValueError("candidates_truncated must match the returned candidate count")
        if self.exclusions_truncated != (
            self.returned_exclusion_count < self.total_exclusion_count
        ):
            raise ValueError("exclusions_truncated must match the returned exclusion count")
        if not self.excluded_candidates_included and any(
            item.tier is CandidateTier.EXCLUDED for item in self.candidates
        ):
            raise ValueError("excluded candidate rows require explicit inclusion")
        if not self.excluded_candidates_included and self.exclusions:
            raise ValueError("upstream exclusions require explicit inclusion")
        return self


class CandidateExplanationOutput(_AccountDecisionBoundOutput):
    """Exact deterministic projection of one member of a candidate snapshot."""

    session_date: date
    feature_version: Version
    config_hash: Sha256
    regime: MarketRegime
    calibrated: Literal[False] = False
    calibration_version: None = None
    input_hash: Sha256
    result_hash: Sha256
    kind: Literal["SCORED_CANDIDATE", "UPSTREAM_EXCLUSION"]
    candidate: CandidateRowOutput | None
    exclusion: CandidateExclusionOutput | None

    @model_validator(mode="after")
    def validate_exact_selection(self) -> Self:
        if (self.candidate is None) == (self.exclusion is None):
            raise ValueError("exactly one candidate or exclusion must be present")
        expected_kind = "SCORED_CANDIDATE" if self.candidate is not None else "UPSTREAM_EXCLUSION"
        if self.kind != expected_kind:
            raise ValueError("explanation kind must match the selected record")
        return self


__all__ = [
    "CandidateComponentScoreOutput",
    "CandidateEvidenceOutput",
    "CandidateExclusionOutput",
    "CandidateExplanationOutput",
    "CandidateRankingOutput",
    "CandidateRiskOutput",
    "CandidateRowOutput",
    "CandidateUpstreamIdentityOutput",
    "DetectMarketRegimeArguments",
    "ExplainCandidateArguments",
    "GetMarketSnapshotArguments",
    "LeaderComponentScoreOutput",
    "LeaderEvidenceOutput",
    "LeaderExclusionOutput",
    "LeaderRankingOutput",
    "LeaderRiskOutput",
    "LeaderRowOutput",
    "LeaderUpstreamIdentityOutput",
    "MainlineEvidenceOutput",
    "MainlineInputIdentityOutput",
    "MainlinePersistenceOutput",
    "MarketDataQualityOutput",
    "MarketRegimeOutput",
    "MarketSnapshotOutput",
    "QualityIssueOutput",
    "RankCandidatesArguments",
    "RankedListArguments",
    "RegimeClassificationOutput",
    "RegimeComponentScoresOutput",
    "RegimeEvidenceOutput",
    "RegimeInputIdentityOutput",
    "RegimeTransitionEventOutput",
    "SnapshotFileOutput",
    "ThemeRankingOutput",
    "ThemeRowOutput",
    "ValidateMarketDataArguments",
]
