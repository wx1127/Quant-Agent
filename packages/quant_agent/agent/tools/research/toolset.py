"""Strict AgentTool adapters over the deterministic research pipeline."""

from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError

from quant_agent.agent.tools.contracts import AgentTool, ToolExecutionContext, ToolOutput
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import Provenance, ToolIssue, ToolResponse
from quant_agent.data.quality import QualityIssue, QualitySeverity
from quant_agent.data.snapshots import SnapshotFile, SnapshotManifest
from quant_agent.features.mainline.contracts import (
    MainlineConfig,
    MainlineEvidence,
    MainlineIndustryResult,
    MainlineInputIdentity,
    MainlineSnapshot,
    TopKPersistence,
)
from quant_agent.leaders.candidates.contracts import (
    CandidateComponentScore,
    CandidateConfig,
    CandidateEvidence,
    CandidateExclusion,
    CandidateResult,
    CandidateRisk,
    CandidateSnapshot,
    CandidateUpstreamIdentity,
)
from quant_agent.leaders.contracts import (
    LeaderComponentScore,
    LeaderConfig,
    LeaderEvidence,
    LeaderExclusion,
    LeaderResult,
    LeaderRisk,
    LeaderSnapshot,
    LeaderUpstreamIdentity,
)
from quant_agent.regime.contracts import (
    MarketRegimeResult,
    RegimeClassifierConfig,
    RegimeComponentScores,
    RegimeEvidence,
    RegimeInputIdentity,
)
from quant_agent.regime.transitions import (
    RegimeTransitionConfig,
    RegimeTransitionResult,
    TransitionAction,
    TransitionEvent,
)

from .contracts import (
    CandidateComponentScoreOutput,
    CandidateEvidenceOutput,
    CandidateExclusionOutput,
    CandidateExplanationOutput,
    CandidateRankingOutput,
    CandidateRiskOutput,
    CandidateRowOutput,
    CandidateUpstreamIdentityOutput,
    DetectMarketRegimeArguments,
    ExplainCandidateArguments,
    GetMarketSnapshotArguments,
    LeaderComponentScoreOutput,
    LeaderEvidenceOutput,
    LeaderExclusionOutput,
    LeaderRankingOutput,
    LeaderRiskOutput,
    LeaderRowOutput,
    LeaderUpstreamIdentityOutput,
    MainlineEvidenceOutput,
    MainlineInputIdentityOutput,
    MainlinePersistenceOutput,
    MarketDataQualityOutput,
    MarketRegimeOutput,
    MarketSnapshotOutput,
    QualityIssueOutput,
    RankCandidatesArguments,
    RankedListArguments,
    RegimeClassificationOutput,
    RegimeComponentScoresOutput,
    RegimeEvidenceOutput,
    RegimeInputIdentityOutput,
    RegimeTransitionEventOutput,
    SnapshotFileOutput,
    ThemeRankingOutput,
    ThemeRowOutput,
    ValidateMarketDataArguments,
)
from .inputs import (
    ResearchInputInvalid,
    ResearchInputSource,
    ResearchInputUnavailable,
)
from .service import ResearchPipeline

RESEARCH_TOOL_VERSION = "agent-research-tool-v1"
_MARKET_SERVICE = "snapshot-manifest-service"
_MARKET_VERSION = "snapshot-manifest-v1"
_QUALITY_SERVICE = "market-data-quality-service"
_QUALITY_VERSION = "market-data-quality-v1"
_REGIME_SERVICE = "market-regime-service"
_MAINLINE_SERVICE = "market-mainline-service"
_LEADER_SERVICE = "theme-leader-service"
_CANDIDATE_SERVICE = "stock-candidate-service"
_MAX_EXCLUSIONS = 200


def _success[OutputT: ToolOutput](
    context: ToolExecutionContext,
    data: OutputT,
    *,
    service: str,
    version: str,
    warnings: tuple[ToolIssue, ...] = (),
) -> ToolResponse[OutputT]:
    snapshot = context.decision_snapshot
    response = ToolResponse[object](
        ok=True,
        request_id=context.request_id,
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data=data,
        warnings=warnings,
        errors=(),
        provenance=Provenance(
            service=service,
            version=version,
            data_version=snapshot.data_version,
        ),
    ).validate_consistency()
    return cast(ToolResponse[OutputT], response)


def _failure[OutputT: ToolOutput](
    context: ToolExecutionContext,
    *,
    code: ErrorCode,
    message: str,
    service: str,
    version: str,
    data: OutputT | None = None,
    warnings: tuple[ToolIssue, ...] = (),
) -> ToolResponse[OutputT]:
    snapshot = context.decision_snapshot
    response = ToolResponse[object](
        ok=False,
        request_id=context.request_id,
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data=data,
        warnings=warnings,
        errors=(ToolIssue(code=code, message=message),),
        provenance=Provenance(
            service=service,
            version=version,
            data_version=snapshot.data_version,
        ),
    ).validate_consistency()
    return cast(ToolResponse[OutputT], response)


def _input_failure[OutputT: ToolOutput](
    context: ToolExecutionContext,
    error: ResearchInputUnavailable | ResearchInputInvalid | ValidationError,
    *,
    service: str,
    version: str,
) -> ToolResponse[OutputT]:
    if isinstance(error, ResearchInputUnavailable):
        return _failure(
            context,
            code=ErrorCode.DATA_UNAVAILABLE,
            message="required verified research input is unavailable",
            service=service,
            version=version,
        )
    return _failure(
        context,
        code=ErrorCode.DATA_INVALID,
        message="verified research input failed decision-bound validation",
        service=service,
        version=version,
    )


def _snapshot_file(value: SnapshotFile) -> SnapshotFileOutput:
    return SnapshotFileOutput(
        name=value.name,
        relative_path=value.relative_path,
        row_count=value.row_count,
        sha256=value.sha256,
    )


def _quality_issue(value: QualityIssue) -> QualityIssueOutput:
    return QualityIssueOutput(
        rule_id=value.rule_id,
        severity=value.severity,
        entity_key=value.entity_key,
        message=value.message,
    )


def _regime_components(value: RegimeComponentScores) -> RegimeComponentScoresOutput:
    return RegimeComponentScoresOutput(
        trend=value.trend,
        breadth=value.breadth,
        turnover=value.turnover,
        new_high_low=value.new_high_low,
        downside_risk=value.downside_risk,
        weighted_trend=value.weighted_trend,
        weighted_breadth=value.weighted_breadth,
        weighted_turnover=value.weighted_turnover,
        weighted_new_high_low=value.weighted_new_high_low,
        weighted_downside_risk=value.weighted_downside_risk,
        total=value.total,
    )


def _regime_evidence(value: RegimeEvidence) -> RegimeEvidenceOutput:
    return RegimeEvidenceOutput(
        feature=value.feature,
        value=value.value,
        criterion=value.criterion,
        rule_weight=value.rule_weight,
        contribution=value.contribution,
        side=value.side,
        rationale=value.rationale,
    )


def _regime_identity(value: RegimeInputIdentity) -> RegimeInputIdentityOutput:
    return RegimeInputIdentityOutput(
        trend_feature_version=value.trend_feature_version,
        trend_config_hash=value.trend_config_hash,
        trend_cache_key=value.trend_cache_key,
        breadth_feature_version=value.breadth_feature_version,
        breadth_config_hash=value.breadth_config_hash,
        breadth_cache_key=value.breadth_cache_key,
    )


def _regime_classification(value: MarketRegimeResult) -> RegimeClassificationOutput:
    return RegimeClassificationOutput(
        as_of=value.as_of,
        session_date=value.session_date,
        data_version=value.data_version,
        regime=value.regime,
        raw_score=value.score,
        confidence=value.confidence,
        confidence_meaning=value.confidence_meaning,
        confidence_definition=value.confidence_definition,
        confidence_is_probability=False,
        risk_budget_max=value.risk_budget_max,
        components=_regime_components(value.components),
        supporting_evidence=tuple(_regime_evidence(item) for item in value.evidence),
        counter_evidence=tuple(_regime_evidence(item) for item in value.counter_evidence),
        invalidations=value.invalidations,
        model_version=value.model_version,
        config_hash=value.config_hash,
        input_identity=_regime_identity(value.input_identity),
        input_hash=value.input_hash,
        result_hash=value.result_hash,
    )


def _transition_event(value: TransitionEvent) -> RegimeTransitionEventOutput:
    return RegimeTransitionEventOutput(
        session_date=value.session_date,
        as_of=value.as_of,
        raw_regime=value.raw_regime,
        previous_regime=value.previous_regime,
        final_regime=value.final_regime,
        action=value.action,
        candidate_regime=value.candidate_regime,
        candidate_count=value.candidate_count,
        required_confirmation_sessions=value.required_confirmation_sessions,
        raw_score=value.raw_score,
        raw_confidence=value.raw_confidence,
        confidence_meaning=value.confidence_meaning,
        confidence_is_probability=False,
        reason=value.reason,
        supporting_evidence=tuple(_regime_evidence(item) for item in value.supporting_evidence),
        source_result_hash=value.source_result_hash,
        transition_config_version=value.transition_config_version,
        transition_config_hash=value.transition_config_hash,
        previous_event_hash=value.previous_event_hash,
        event_hash=value.event_hash,
    )


def _regime_output(
    context: ToolExecutionContext,
    value: RegimeTransitionResult,
) -> MarketRegimeOutput:
    snapshot = context.decision_snapshot
    return MarketRegimeOutput(
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data_version=snapshot.data_version,
        session_date=value.raw_result.session_date,
        raw=_regime_classification(value.raw_result),
        effective_regime=value.final_regime,
        pending_candidate=value.pending_candidate,
        pending_count=value.pending_count,
        required_confirmation_sessions=value.required_confirmation_sessions,
        transitioned=value.event.action is TransitionAction.TRANSITION_CONFIRMED,
        confidence_is_probability=False,
        event=_transition_event(value.event),
        transition_config_version=value.transition_config_version,
        transition_config_hash=value.transition_config_hash,
        result_hash=value.result_hash,
    )


def _mainline_persistence(value: TopKPersistence) -> MainlinePersistenceOutput:
    return MainlinePersistenceOutput(
        top_k=value.top_k,
        window_sessions=value.window_sessions,
        observed_sessions=value.observed_sessions,
        top_k_hits=value.top_k_hits,
        consecutive_top_k_sessions=value.consecutive_top_k_sessions,
        recent_ranks=value.recent_ranks,
        hit_ratio=value.hit_ratio,
    )


def _mainline_evidence(value: MainlineEvidence) -> MainlineEvidenceOutput:
    return MainlineEvidenceOutput(
        feature=value.feature,
        value=value.value,
        criterion=value.criterion,
        side=value.side,
        rationale=value.rationale,
    )


def _mainline_identity(value: MainlineInputIdentity) -> MainlineInputIdentityOutput:
    return MainlineInputIdentityOutput(
        industry_feature_version=value.industry_feature_version,
        industry_config_hash=value.industry_config_hash,
        industry_cache_key=value.industry_cache_key,
        classification_version=value.classification_version,
        industry_level=value.industry_level,
        regime_model_version=value.regime_model_version,
        regime_classifier_config_hash=value.regime_classifier_config_hash,
        regime_input_hash=value.regime_input_hash,
        regime_trend_feature_version=value.regime_trend_feature_version,
        regime_trend_config_hash=value.regime_trend_config_hash,
        regime_breadth_feature_version=value.regime_breadth_feature_version,
        regime_breadth_config_hash=value.regime_breadth_config_hash,
        regime_transition_version=value.regime_transition_version,
        regime_transition_config_hash=value.regime_transition_config_hash,
        regime_transition_result_hash=value.regime_transition_result_hash,
    )


def _theme(value: MainlineIndustryResult) -> ThemeRowOutput:
    return ThemeRowOutput(
        industry_id=value.industry_id,
        state=value.state,
        previous_state=value.previous_state,
        current_rank=value.current_rank,
        current_strength_score=value.current_strength_score,
        mainline_score=value.mainline_score,
        crowding_score=value.crowding_score,
        persistence=_mainline_persistence(value.persistence),
        crowding_signals=value.crowding_signals,
        supporting_evidence=tuple(_mainline_evidence(item) for item in value.supporting_evidence),
        counter_evidence=tuple(_mainline_evidence(item) for item in value.counter_evidence),
        invalidations=value.invalidations,
        transition_reason=value.transition_reason,
    )


def _leader_identity(value: LeaderUpstreamIdentity) -> LeaderUpstreamIdentityOutput:
    return LeaderUpstreamIdentityOutput(
        mainline_model_version=value.mainline_model_version,
        mainline_result_hash=value.mainline_result_hash,
        stock_feature_version=value.stock_feature_version,
        stock_config_hash=value.stock_config_hash,
        stock_input_hash=value.stock_input_hash,
        stock_result_hash=value.stock_result_hash,
        tradeability_feature_version=value.tradeability_feature_version,
        tradeability_config_hash=value.tradeability_config_hash,
        tradeability_rule_version=value.tradeability_rule_version,
        tradeability_rule_hash=value.tradeability_rule_hash,
        tradeability_result_hash=value.tradeability_result_hash,
        fundamental_feature_version=value.fundamental_feature_version,
        fundamental_config_hash=value.fundamental_config_hash,
        fundamental_data_version=value.fundamental_data_version,
        fundamental_input_hash=value.fundamental_input_hash,
        fundamental_result_hash=value.fundamental_result_hash,
    )


def _leader_component(value: LeaderComponentScore) -> LeaderComponentScoreOutput:
    return LeaderComponentScoreOutput(
        component=value.component,
        score=value.score,
        weight=value.weight,
        contribution=value.contribution,
    )


def _leader_evidence(value: LeaderEvidence) -> LeaderEvidenceOutput:
    return LeaderEvidenceOutput(
        feature=value.feature,
        value=value.value,
        criterion=value.criterion,
        side=value.side,
        rationale=value.rationale,
    )


def _leader_risk(value: LeaderRisk) -> LeaderRiskOutput:
    return LeaderRiskOutput(
        code=value.code,
        penalty=value.penalty,
        rationale=value.rationale,
    )


def _leader(value: LeaderResult) -> LeaderRowOutput:
    return LeaderRowOutput(
        instrument_id=value.instrument_id,
        industry_id=value.industry_id,
        mainline_state=value.mainline_state,
        overall_rank=value.overall_rank,
        theme_rank=value.theme_rank,
        leader_type=value.leader_type,
        gross_score=value.gross_score,
        risk_penalty=value.risk_penalty,
        score=value.score,
        components=tuple(_leader_component(item) for item in value.components),
        supporting_evidence=tuple(_leader_evidence(item) for item in value.supporting_evidence),
        counter_evidence=tuple(_leader_evidence(item) for item in value.counter_evidence),
        risks=tuple(_leader_risk(item) for item in value.risks),
        candidate_eligible=value.candidate_eligible,
        ineligibility_reasons=value.ineligibility_reasons,
        observation_conditions=value.observation_conditions,
        invalidations=value.invalidations,
        upstream_identity=_leader_identity(value.upstream_identity),
    )


def _leader_exclusion(value: LeaderExclusion) -> LeaderExclusionOutput:
    return LeaderExclusionOutput(
        instrument_id=value.instrument_id,
        industry_id=value.industry_id,
        code=value.code,
        reason=value.reason,
        stock_result_hash=value.stock_result_hash,
    )


def _candidate_identity(
    value: CandidateUpstreamIdentity,
) -> CandidateUpstreamIdentityOutput:
    return CandidateUpstreamIdentityOutput(
        regime_transition_version=value.regime_transition_version,
        regime_transition_hash=value.regime_transition_hash,
        mainline_model_version=value.mainline_model_version,
        mainline_result_hash=value.mainline_result_hash,
        leader_feature_version=value.leader_feature_version,
        leader_result_hash=value.leader_result_hash,
        stock_result_hash=value.stock_result_hash,
        tradeability_result_hash=value.tradeability_result_hash,
        fundamental_result_hash=value.fundamental_result_hash,
        event_result_hash=value.event_result_hash,
        valuation_result_hash=value.valuation_result_hash,
    )


def _candidate_component(value: CandidateComponentScore) -> CandidateComponentScoreOutput:
    return CandidateComponentScoreOutput(
        component=value.component,
        score=value.score,
        weight=value.weight,
        contribution=value.contribution,
        source=value.source,
    )


def _candidate_evidence(value: CandidateEvidence) -> CandidateEvidenceOutput:
    return CandidateEvidenceOutput(
        feature=value.feature,
        value=value.value,
        criterion=value.criterion,
        side=value.side,
        rationale=value.rationale,
    )


def _candidate_risk(value: CandidateRisk) -> CandidateRiskOutput:
    return CandidateRiskOutput(
        code=value.code,
        penalty=value.penalty,
        rationale=value.rationale,
    )


def _candidate(value: CandidateResult) -> CandidateRowOutput:
    return CandidateRowOutput(
        instrument_id=value.instrument_id,
        industry_id=value.industry_id,
        research_rank=value.research_rank,
        eligible_rank=value.eligible_rank,
        tier=value.tier,
        eligible=value.eligible,
        gross_score=value.gross_score,
        risk_penalty=value.risk_penalty,
        score=value.score,
        components=tuple(_candidate_component(item) for item in value.components),
        supporting_evidence=tuple(_candidate_evidence(item) for item in value.supporting_evidence),
        counter_evidence=tuple(_candidate_evidence(item) for item in value.counter_evidence),
        risks=tuple(_candidate_risk(item) for item in value.risks),
        exclusion_codes=value.exclusion_codes,
        exclusion_reasons=value.exclusion_reasons,
        observation_conditions=value.observation_conditions,
        invalidations=value.invalidations,
        upstream_identity=_candidate_identity(value.upstream_identity),
        calibrated=False,
    )


def _candidate_exclusion(value: CandidateExclusion) -> CandidateExclusionOutput:
    return CandidateExclusionOutput(
        instrument_id=value.instrument_id,
        industry_id=value.industry_id,
        code=value.code,
        reason=value.reason,
        source_hash=value.source_hash,
    )


class ResearchToolset:
    """Seven read-only handlers that expose exact pipeline projections."""

    __slots__ = ("_pipeline",)

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
        self._pipeline = ResearchPipeline(
            source,
            regime_config=regime_config,
            transition_config=transition_config,
            mainline_config=mainline_config,
            leader_config=leader_config,
            candidate_config=candidate_config,
        )

    def get_market_snapshot(
        self,
        context: ToolExecutionContext,
        arguments: GetMarketSnapshotArguments,
    ) -> ToolResponse[MarketSnapshotOutput]:
        try:
            manifest = self._pipeline.load_manifest(context.decision_snapshot)
            data = self._market_snapshot_output(context, arguments, manifest)
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_MARKET_SERVICE,
                version=_MARKET_VERSION,
            )
        return _success(
            context,
            data,
            service=_MARKET_SERVICE,
            version=_MARKET_VERSION,
        )

    def validate_market_data(
        self,
        context: ToolExecutionContext,
        arguments: ValidateMarketDataArguments,
    ) -> ToolResponse[MarketDataQualityOutput]:
        try:
            report = self._pipeline.validate_market_data(context.decision_snapshot)
            issues = tuple(_quality_issue(item) for item in report.issues[: arguments.max_issues])
            blocking = sum(
                item.severity in {QualitySeverity.ERROR, QualitySeverity.CRITICAL}
                for item in report.issues
            )
            snapshot = context.decision_snapshot
            data = MarketDataQualityOutput(
                decision_id=snapshot.decision_id,
                as_of=snapshot.as_of,
                data_version=snapshot.data_version,
                data_content_hash=snapshot.data_content_hash,
                observed_at=report.observed_at,
                qualified=report.qualified,
                total_issue_count=len(report.issues),
                blocking_issue_count=blocking,
                returned_issue_count=len(issues),
                issues_truncated=len(issues) < len(report.issues),
                issues=issues,
            )
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_QUALITY_SERVICE,
                version=_QUALITY_VERSION,
            )
        warnings: tuple[ToolIssue, ...] = ()
        if data.qualified and data.total_issue_count:
            warnings = (
                ToolIssue(
                    code=ErrorCode.DATA_INVALID,
                    message="market data quality checks returned non-blocking issues",
                ),
            )
        if not data.qualified:
            return _failure(
                context,
                code=ErrorCode.DATA_INVALID,
                message="market data failed one or more blocking quality checks",
                service=_QUALITY_SERVICE,
                version=_QUALITY_VERSION,
                data=data,
                warnings=warnings,
            )
        return _success(
            context,
            data,
            service=_QUALITY_SERVICE,
            version=_QUALITY_VERSION,
            warnings=warnings,
        )

    def detect_market_regime(
        self,
        context: ToolExecutionContext,
        arguments: DetectMarketRegimeArguments,
    ) -> ToolResponse[MarketRegimeOutput]:
        del arguments
        version = self._pipeline.transition_version
        try:
            result = self._pipeline.detect_market_regime(context.decision_snapshot)
            data = _regime_output(context, result)
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_REGIME_SERVICE,
                version=version,
            )
        return _success(context, data, service=_REGIME_SERVICE, version=version)

    def rank_market_themes(
        self,
        context: ToolExecutionContext,
        arguments: RankedListArguments,
    ) -> ToolResponse[ThemeRankingOutput]:
        version = self._pipeline.mainline_version
        try:
            result = self._pipeline.rank_market_themes(context.decision_snapshot)
            data = self._theme_output(context, arguments, result)
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_MAINLINE_SERVICE,
                version=version,
            )
        return _success(context, data, service=_MAINLINE_SERVICE, version=version)

    def rank_theme_leaders(
        self,
        context: ToolExecutionContext,
        arguments: RankedListArguments,
    ) -> ToolResponse[LeaderRankingOutput]:
        version = self._pipeline.leader_version
        try:
            result = self._pipeline.rank_theme_leaders(context.decision_snapshot)
            data = self._leader_output(context, arguments, result)
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_LEADER_SERVICE,
                version=version,
            )
        return _success(context, data, service=_LEADER_SERVICE, version=version)

    def rank_stock_candidates(
        self,
        context: ToolExecutionContext,
        arguments: RankCandidatesArguments,
    ) -> ToolResponse[CandidateRankingOutput]:
        version = self._pipeline.candidate_version
        try:
            result = self._pipeline.rank_stock_candidates(context.decision_snapshot)
            data = self._candidate_output(context, arguments, result)
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_CANDIDATE_SERVICE,
                version=version,
            )
        return _success(context, data, service=_CANDIDATE_SERVICE, version=version)

    def explain_candidate(
        self,
        context: ToolExecutionContext,
        arguments: ExplainCandidateArguments,
    ) -> ToolResponse[CandidateExplanationOutput]:
        version = self._pipeline.candidate_version
        try:
            result = self._pipeline.rank_stock_candidates(context.decision_snapshot)
            candidate = next(
                (
                    item
                    for item in result.candidates
                    if item.instrument_id == arguments.instrument_id
                ),
                None,
            )
            exclusion = next(
                (
                    item
                    for item in result.exclusions
                    if item.instrument_id == arguments.instrument_id
                ),
                None,
            )
            if candidate is None and exclusion is None:
                return _failure(
                    context,
                    code=ErrorCode.NOT_FOUND,
                    message="instrument is absent from the decision-bound candidate snapshot",
                    service=_CANDIDATE_SERVICE,
                    version=version,
                )
            data = self._candidate_explanation_output(
                context,
                result,
                candidate=candidate,
                exclusion=exclusion,
            )
        except (ResearchInputUnavailable, ResearchInputInvalid, ValidationError) as error:
            return _input_failure(
                context,
                error,
                service=_CANDIDATE_SERVICE,
                version=version,
            )
        return _success(context, data, service=_CANDIDATE_SERVICE, version=version)

    @staticmethod
    def _market_snapshot_output(
        context: ToolExecutionContext,
        arguments: GetMarketSnapshotArguments,
        manifest: SnapshotManifest,
    ) -> MarketSnapshotOutput:
        files = tuple(_snapshot_file(item) for item in manifest.files[: arguments.max_files])
        snapshot = context.decision_snapshot
        return MarketSnapshotOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            market="CN_A",
            data_content_hash=snapshot.data_content_hash,
            created_at=manifest.created_at,
            total_file_count=len(manifest.files),
            returned_file_count=len(files),
            total_row_count=sum(item.row_count for item in manifest.files),
            files_truncated=len(files) < len(manifest.files),
            files=files,
        )

    @staticmethod
    def _theme_output(
        context: ToolExecutionContext,
        arguments: RankedListArguments,
        result: MainlineSnapshot,
    ) -> ThemeRankingOutput:
        rows = tuple(_theme(item) for item in result.industries[: arguments.limit])
        snapshot = context.decision_snapshot
        return ThemeRankingOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            session_date=result.session_date,
            classification_version=result.classification_version,
            industry_level=result.industry_level,
            market_regime=result.market_regime,
            model_version=result.model_version,
            config_hash=result.config_hash,
            input_identity=_mainline_identity(result.input_identity),
            input_hash=result.input_hash,
            previous_result_hash=result.previous_result_hash,
            result_hash=result.result_hash,
            total_theme_count=len(result.industries),
            returned_theme_count=len(rows),
            themes_truncated=len(rows) < len(result.industries),
            themes=rows,
        )

    @staticmethod
    def _leader_output(
        context: ToolExecutionContext,
        arguments: RankedListArguments,
        result: LeaderSnapshot,
    ) -> LeaderRankingOutput:
        leaders = tuple(_leader(item) for item in result.leaders[: arguments.limit])
        exclusions = tuple(_leader_exclusion(item) for item in result.exclusions[:_MAX_EXCLUSIONS])
        snapshot = context.decision_snapshot
        return LeaderRankingOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            session_date=result.session_date,
            classification_version=result.classification_version,
            industry_level=result.industry_level,
            feature_version=result.feature_version,
            config_hash=result.config_hash,
            mainline_result_hash=result.mainline_result_hash,
            input_hash=result.input_hash,
            result_hash=result.result_hash,
            total_leader_count=len(result.leaders),
            returned_leader_count=len(leaders),
            leaders_truncated=len(leaders) < len(result.leaders),
            total_exclusion_count=len(result.exclusions),
            returned_exclusion_count=len(exclusions),
            exclusions_truncated=len(exclusions) < len(result.exclusions),
            leaders=leaders,
            exclusions=exclusions,
        )

    @staticmethod
    def _candidate_output(
        context: ToolExecutionContext,
        arguments: RankCandidatesArguments,
        result: CandidateSnapshot,
    ) -> CandidateRankingOutput:
        visible_candidates = (
            result.candidates
            if arguments.include_excluded
            else tuple(item for item in result.candidates if item.eligible)
        )
        candidates = tuple(_candidate(item) for item in visible_candidates[: arguments.limit])
        visible_exclusions = result.exclusions if arguments.include_excluded else ()
        exclusions = tuple(
            _candidate_exclusion(item) for item in visible_exclusions[:_MAX_EXCLUSIONS]
        )
        snapshot = context.decision_snapshot
        return CandidateRankingOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            session_date=result.session_date,
            feature_version=result.feature_version,
            config_hash=result.config_hash,
            regime=result.regime,
            calibrated=False,
            calibration_version=None,
            input_hash=result.input_hash,
            result_hash=result.result_hash,
            excluded_candidates_included=arguments.include_excluded,
            total_candidate_count=len(result.candidates),
            returned_candidate_count=len(candidates),
            candidates_truncated=len(candidates) < len(result.candidates),
            total_exclusion_count=len(result.exclusions),
            returned_exclusion_count=len(exclusions),
            exclusions_truncated=len(exclusions) < len(result.exclusions),
            candidates=candidates,
            exclusions=exclusions,
        )

    @staticmethod
    def _candidate_explanation_output(
        context: ToolExecutionContext,
        result: CandidateSnapshot,
        *,
        candidate: CandidateResult | None,
        exclusion: CandidateExclusion | None,
    ) -> CandidateExplanationOutput:
        snapshot = context.decision_snapshot
        return CandidateExplanationOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            session_date=result.session_date,
            feature_version=result.feature_version,
            config_hash=result.config_hash,
            regime=result.regime,
            calibrated=False,
            calibration_version=None,
            input_hash=result.input_hash,
            result_hash=result.result_hash,
            kind="SCORED_CANDIDATE" if candidate is not None else "UPSTREAM_EXCLUSION",
            candidate=_candidate(candidate) if candidate is not None else None,
            exclusion=_candidate_exclusion(exclusion) if exclusion is not None else None,
        )


def build_research_tools(
    source: ResearchInputSource,
    *,
    regime_config: RegimeClassifierConfig | None = None,
    transition_config: RegimeTransitionConfig | None = None,
    mainline_config: MainlineConfig | None = None,
    leader_config: LeaderConfig | None = None,
    candidate_config: CandidateConfig | None = None,
) -> tuple[AgentTool[Any, Any], ...]:
    """Build all seven tools only after every trusted dependency is available."""

    toolset = ResearchToolset(
        source,
        regime_config=regime_config,
        transition_config=transition_config,
        mainline_config=mainline_config,
        leader_config=leader_config,
        candidate_config=candidate_config,
    )
    return (
        AgentTool(
            name="get_market_snapshot",
            version=RESEARCH_TOOL_VERSION,
            description="Return a bounded summary of the verified decision market snapshot.",
            arguments_type=GetMarketSnapshotArguments,
            output_type=MarketSnapshotOutput,
            handler=toolset.get_market_snapshot,
        ),
        AgentTool(
            name="validate_market_data",
            version=RESEARCH_TOOL_VERSION,
            description="Run deterministic data-quality rules at the decision boundary.",
            arguments_type=ValidateMarketDataArguments,
            output_type=MarketDataQualityOutput,
            handler=toolset.validate_market_data,
        ),
        AgentTool(
            name="detect_market_regime",
            version=RESEARCH_TOOL_VERSION,
            description="Classify and stabilize the market regime from complete PIT history.",
            arguments_type=DetectMarketRegimeArguments,
            output_type=MarketRegimeOutput,
            handler=toolset.detect_market_regime,
        ),
        AgentTool(
            name="rank_market_themes",
            version=RESEARCH_TOOL_VERSION,
            description="Rank current industry mainlines using complete PIT state history.",
            arguments_type=RankedListArguments,
            output_type=ThemeRankingOutput,
            handler=toolset.rank_market_themes,
        ),
        AgentTool(
            name="rank_theme_leaders",
            version=RESEARCH_TOOL_VERSION,
            description="Rank account-sized, tradeable leaders of confirmed mainlines.",
            arguments_type=RankedListArguments,
            output_type=LeaderRankingOutput,
            handler=toolset.rank_theme_leaders,
        ),
        AgentTool(
            name="rank_stock_candidates",
            version=RESEARCH_TOOL_VERSION,
            description="Rank uncalibrated stock candidates from exact upstream engine results.",
            arguments_type=RankCandidatesArguments,
            output_type=CandidateRankingOutput,
            handler=toolset.rank_stock_candidates,
        ),
        AgentTool(
            name="explain_candidate",
            version=RESEARCH_TOOL_VERSION,
            description="Project one exact scored candidate or upstream exclusion without advice.",
            arguments_type=ExplainCandidateArguments,
            output_type=CandidateExplanationOutput,
            handler=toolset.explain_candidate,
        ),
    )


__all__ = ["RESEARCH_TOOL_VERSION", "ResearchToolset", "build_research_tools"]
