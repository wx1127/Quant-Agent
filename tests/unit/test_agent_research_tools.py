"""Strict Agent research-tool projections, failures, and registry integration."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.unit import test_agent_research_service as research_fixtures
from tests.unit import test_leaders as leader_fixtures
from tests.unit import test_mainline as mainline_fixtures

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.agent.tools.contracts import (
    ToolArgumentsRejected,
    ToolCallContext,
    ToolCapability,
    ToolDenialReason,
    ToolExecutionContext,
    ToolPermissionDenied,
)
from quant_agent.agent.tools.registry import AgentToolRegistry, AgentToolRegistryBuilder
from quant_agent.agent.tools.research.contracts import (
    CandidateExplanationOutput,
    CandidateRankingOutput,
    LeaderRankingOutput,
    MarketDataQualityOutput,
    MarketRegimeOutput,
    MarketSnapshotOutput,
    ThemeRankingOutput,
)
from quant_agent.agent.tools.research.inputs import DataBoundQualityContext
from quant_agent.agent.tools.research.service import ResearchPipeline
from quant_agent.agent.tools.research.toolset import (
    RESEARCH_TOOL_VERSION,
    build_research_tools,
)
from quant_agent.core.errors import ErrorCode
from quant_agent.data.quality import BarQualityRecord, QualityContext
from quant_agent.leaders.candidates import CandidateTier
from quant_agent.observability.audit import AuditEvent

EXPECTED_NAMES = {
    "detect_market_regime",
    "explain_candidate",
    "get_market_snapshot",
    "rank_market_themes",
    "rank_stock_candidates",
    "rank_theme_leaders",
    "validate_market_data",
}
CAPABILITIES = frozenset(
    {
        ToolCapability.MARKET_DATA_READ,
        ToolCapability.MARKET_DATA_VALIDATE,
        ToolCapability.RESEARCH_ANALYSIS,
    }
)


class _Resolver:
    def __init__(self, snapshot: DecisionSnapshot) -> None:
        self.snapshot = snapshot

    def resolve(self, decision_id: str) -> DecisionSnapshot:
        assert decision_id == self.snapshot.decision_id
        return self.snapshot


class _AuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


def _registry(
    source: research_fixtures._ResearchSource,
    *,
    grant_account: bool = True,
) -> tuple[AgentToolRegistry, _AuditSink]:
    decision = research_fixtures._decision()
    audit = _AuditSink()
    builder = AgentToolRegistryBuilder(
        runtime_mode=decision.mode,
        principal_id="research-agent",
        granted_capabilities=CAPABILITIES,
        granted_account_ids=(frozenset({decision.account_id}) if grant_account else frozenset()),
        decision_resolver=_Resolver(decision),
        audit_sink=audit,
    )
    for tool in build_research_tools(source):
        builder.register(tool)
    return builder.build(), audit


def _call_context(request_id: str = "req-research-1") -> ToolCallContext:
    decision = research_fixtures._decision()
    return ToolCallContext(
        request_id=request_id,
        decision_id=decision.decision_id,
        requested_at=decision.as_of + timedelta(seconds=1),
    )


def _execution_context() -> ToolExecutionContext:
    decision = research_fixtures._decision()
    return ToolExecutionContext(
        request_id="req-direct-1",
        principal_id="research-agent",
        requested_at=decision.as_of + timedelta(seconds=1),
        decision_snapshot=decision,
    )


def _source_with_upstream_exclusion() -> research_fixtures._ResearchSource:
    source = research_fixtures._source()
    instrument_id = "OUTSIDE"
    stock = leader_fixtures._stock(
        instrument_id,
        industry_id=mainline_fixtures.INDUSTRIES[-1],
    )
    tradeability = leader_fixtures._tradeability(instrument_id)
    fundamental = replace(
        leader_fixtures._fundamental(instrument_id),
        data_version=research_fixtures.DATA_VERSION,
    )
    inputs = source.leader_inputs
    source.leader_inputs = replace(
        inputs,
        stock_strength=(*inputs.stock_strength, stock),
        tradeability=(*inputs.tradeability, tradeability),
        fundamentals=(*inputs.fundamentals, fundamental),
    )
    return source


def test_all_seven_tools_build_register_and_return_standard_bound_responses() -> None:
    source = research_fixtures._source()
    registry, audit = _registry(source)
    context = _call_context()
    decision = research_fixtures._decision()
    descriptors = registry.catalog(context)

    assert {item.name for item in descriptors} == EXPECTED_NAMES
    assert all(item.version == RESEARCH_TOOL_VERSION for item in descriptors)

    calls = (
        ("get_market_snapshot", {}, MarketSnapshotOutput),
        ("validate_market_data", {}, MarketDataQualityOutput),
        ("detect_market_regime", {}, MarketRegimeOutput),
        ("rank_market_themes", {"limit": 10}, ThemeRankingOutput),
        ("rank_theme_leaders", {"limit": 10}, LeaderRankingOutput),
        (
            "rank_stock_candidates",
            {"limit": 20, "include_excluded": True},
            CandidateRankingOutput,
        ),
        (
            "explain_candidate",
            {"instrument_id": "CORE"},
            CandidateExplanationOutput,
        ),
    )
    for name, arguments, expected_type in calls:
        response = registry.invoke(name, arguments, context)
        assert response.ok is True
        assert type(response.data) is expected_type
        assert response.request_id == context.request_id
        assert response.decision_id == decision.decision_id
        assert response.as_of == decision.as_of
        assert response.provenance.data_version == decision.data_version
        assert response.errors == ()

    allowed = [event for event in audit.events if event.result == "ALLOWED"]
    assert len(allowed) == len(calls)


def test_projected_numbers_and_hashes_equal_the_underlying_domain_engines() -> None:
    source = research_fixtures._source()
    registry, _ = _registry(source)
    context = _call_context()
    pipeline = ResearchPipeline(source)
    decision = research_fixtures._decision()

    regime_domain = pipeline.detect_market_regime(decision)
    regime = registry.invoke("detect_market_regime", {}, context).data
    assert type(regime) is MarketRegimeOutput
    assert regime.raw.raw_score == regime_domain.raw_result.score
    assert regime.raw.components.total == regime_domain.raw_result.components.total
    assert regime.result_hash == regime_domain.result_hash

    theme_domain = pipeline.rank_market_themes(decision)
    themes = registry.invoke("rank_market_themes", {"limit": 50}, context).data
    assert type(themes) is ThemeRankingOutput
    assert tuple(item.mainline_score for item in themes.themes) == tuple(
        item.mainline_score for item in theme_domain.industries
    )
    assert themes.result_hash == theme_domain.result_hash

    leader_domain = pipeline.rank_theme_leaders(decision)
    leaders = registry.invoke("rank_theme_leaders", {"limit": 50}, context).data
    assert type(leaders) is LeaderRankingOutput
    assert tuple(item.score for item in leaders.leaders) == tuple(
        item.score for item in leader_domain.leaders
    )
    assert leaders.result_hash == leader_domain.result_hash

    candidate_domain = pipeline.rank_stock_candidates(decision)
    candidates = registry.invoke(
        "rank_stock_candidates",
        {"limit": 50, "include_excluded": True},
        context,
    ).data
    assert type(candidates) is CandidateRankingOutput
    assert tuple(item.score for item in candidates.candidates) == tuple(
        item.score for item in candidate_domain.candidates
    )
    assert candidates.result_hash == candidate_domain.result_hash
    assert candidates.calibrated is False


def test_repeated_calls_are_data_deterministic() -> None:
    source = research_fixtures._source()
    registry, _ = _registry(source)

    first = registry.invoke(
        "rank_stock_candidates",
        {"limit": 20, "include_excluded": True},
        _call_context("req-repeat-1"),
    )
    repeated = registry.invoke(
        "rank_stock_candidates",
        {"limit": 20, "include_excluded": True},
        _call_context("req-repeat-2"),
    )

    assert first.data == repeated.data
    assert first.provenance == repeated.provenance


def test_argument_schemas_exclude_context_and_reject_extra_or_out_of_bounds_values() -> None:
    tools = build_research_tools(research_fixtures._source())
    forbidden = {
        "account_id",
        "as_of",
        "data_version",
        "decision_id",
        "mode",
        "request_id",
    }

    for tool in tools:
        properties = set(tool.argument_schema().get("properties", {}))
        assert properties.isdisjoint(forbidden)

    by_name = {tool.name: tool for tool in tools}
    with pytest.raises(ValidationError):
        by_name["get_market_snapshot"].validate_arguments('{"max_files":0}')
    with pytest.raises(ValidationError):
        by_name["rank_market_themes"].validate_arguments('{"limit":51}')
    with pytest.raises(ValidationError):
        by_name["explain_candidate"].validate_arguments('{"instrument_id":" CORE "}')
    with pytest.raises(ValidationError):
        by_name["detect_market_regime"].validate_arguments('{"decision_id":"untrusted"}')


def test_registry_rejects_model_attempts_to_supply_hidden_decision_context() -> None:
    registry, _ = _registry(research_fixtures._source())

    with pytest.raises(ToolArgumentsRejected) as raised:
        registry.invoke(
            "detect_market_regime",
            {"decision_id": research_fixtures._decision().decision_id},
            _call_context(),
        )

    assert raised.value.reason is ToolDenialReason.INVALID_ARGUMENTS


@pytest.mark.parametrize("failure", ["unavailable", "invalid"])
def test_input_failures_are_standard_errors_not_empty_successes(failure: str) -> None:
    source = research_fixtures._source()
    if failure == "unavailable":
        source.unavailable_method = "manifest"
        expected = ErrorCode.DATA_UNAVAILABLE
    else:
        source.manifest = source.manifest.model_copy(update={"content_hash": "f" * 64})
        expected = ErrorCode.DATA_INVALID
    registry, _ = _registry(source)

    response = registry.invoke("get_market_snapshot", {}, _call_context())

    assert response.ok is False
    assert response.data is None
    assert response.errors
    assert response.errors[0].code is expected
    assert response.provenance.data_version == research_fixtures.DATA_VERSION


def test_blocking_quality_failure_keeps_typed_evidence_and_is_not_empty() -> None:
    source = research_fixtures._source()
    session_date = leader_fixtures.SESSION
    invalid = BarQualityRecord(
        instrument_id="CN.SZ.000001",
        trade_date=session_date,
        open=Decimal("10"),
        high=Decimal("9"),
        low=Decimal("11"),
        close=Decimal("10"),
        volume=Decimal("-1"),
        turnover=Decimal("1"),
    )
    source.quality = DataBoundQualityContext(
        data_version=research_fixtures.DATA_VERSION,
        data_content_hash=research_fixtures.DATA_CONTENT_HASH,
        context=QualityContext(
            bars=(invalid,),
            expected_instrument_ids=(invalid.instrument_id,),
            expected_open_dates=(session_date,),
        ),
    )
    registry, _ = _registry(source)

    response = registry.invoke("validate_market_data", {}, _call_context())

    assert response.ok is False
    assert type(response.data) is MarketDataQualityOutput
    assert response.data.qualified is False
    assert response.data.blocking_issue_count > 0
    assert response.data.issues
    assert response.errors[0].code is ErrorCode.DATA_INVALID


def test_candidate_exclusion_rows_require_explicit_inclusion() -> None:
    registry, _ = _registry(research_fixtures._source())
    context = _call_context()

    hidden = registry.invoke(
        "rank_stock_candidates",
        {"limit": 20, "include_excluded": False},
        context,
    ).data
    visible = registry.invoke(
        "rank_stock_candidates",
        {"limit": 20, "include_excluded": True},
        context,
    ).data

    assert type(hidden) is CandidateRankingOutput
    assert type(visible) is CandidateRankingOutput
    assert all(item.tier is not CandidateTier.EXCLUDED for item in hidden.candidates)
    assert any(item.tier is CandidateTier.EXCLUDED for item in visible.candidates)
    assert hidden.total_candidate_count == visible.total_candidate_count
    assert hidden.returned_candidate_count < visible.returned_candidate_count
    assert hidden.candidates_truncated is True


def test_explain_candidate_distinguishes_scored_excluded_upstream_and_unknown() -> None:
    registry, _ = _registry(_source_with_upstream_exclusion())
    context = _call_context()

    normal = registry.invoke("explain_candidate", {"instrument_id": "CORE"}, context).data
    scored_excluded = registry.invoke(
        "explain_candidate", {"instrument_id": "BLOCKED"}, context
    ).data
    upstream = registry.invoke("explain_candidate", {"instrument_id": "OUTSIDE"}, context).data
    unknown = registry.invoke("explain_candidate", {"instrument_id": "UNKNOWN"}, context)

    assert type(normal) is CandidateExplanationOutput
    assert normal.kind == "SCORED_CANDIDATE"
    assert normal.candidate is not None
    assert normal.candidate.tier is not CandidateTier.EXCLUDED
    assert normal.exclusion is None

    assert type(scored_excluded) is CandidateExplanationOutput
    assert scored_excluded.kind == "SCORED_CANDIDATE"
    assert scored_excluded.candidate is not None
    assert scored_excluded.candidate.tier is CandidateTier.EXCLUDED
    assert scored_excluded.exclusion is None

    assert type(upstream) is CandidateExplanationOutput
    assert upstream.kind == "UPSTREAM_EXCLUSION"
    assert upstream.candidate is None
    assert upstream.exclusion is not None

    assert unknown.ok is False
    assert unknown.data is None
    assert unknown.errors[0].code is ErrorCode.NOT_FOUND


def test_account_scoped_research_rankings_are_hidden_and_denied_without_grant() -> None:
    registry, audit = _registry(research_fixtures._source(), grant_account=False)
    context = _call_context()

    visible = {item.name for item in registry.catalog(context)}
    assert {
        "rank_theme_leaders",
        "rank_stock_candidates",
        "explain_candidate",
    }.isdisjoint(visible)
    assert {
        "get_market_snapshot",
        "validate_market_data",
        "detect_market_regime",
        "rank_market_themes",
    }.issubset(visible)

    with pytest.raises(ToolPermissionDenied) as raised:
        registry.invoke("rank_stock_candidates", {}, context)

    assert raised.value.reason is ToolDenialReason.ACCOUNT_SCOPE_NOT_GRANTED
    assert audit.events[-1].result == "DENIED"
