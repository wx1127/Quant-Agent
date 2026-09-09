"""Adversarial validation for the closed Agent research output schemas."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError
from tests.unit import test_agent_research_tools as tool_fixtures

from quant_agent.agent.tools.contracts import ToolOutput
from quant_agent.agent.tools.research.contracts import (
    CandidateExplanationOutput,
    CandidateRankingOutput,
    DetectMarketRegimeArguments,
    ExplainCandidateArguments,
    GetMarketSnapshotArguments,
    RankCandidatesArguments,
    RankedListArguments,
    ValidateMarketDataArguments,
)
from quant_agent.agent.tools.research.toolset import ResearchToolset
from quant_agent.leaders.candidates import CandidateExclusionCode, CandidateTier
from quant_agent.regime import MarketRegime


@pytest.fixture(scope="module")
def outputs() -> dict[str, ToolOutput]:
    context = tool_fixtures._execution_context()
    toolset = ResearchToolset(tool_fixtures._source_with_upstream_exclusion())
    market = toolset.get_market_snapshot(context, GetMarketSnapshotArguments()).data
    quality = toolset.validate_market_data(context, ValidateMarketDataArguments()).data
    regime = toolset.detect_market_regime(context, DetectMarketRegimeArguments()).data
    themes = toolset.rank_market_themes(context, RankedListArguments(limit=50)).data
    leaders = toolset.rank_theme_leaders(context, RankedListArguments(limit=50)).data
    candidates = toolset.rank_stock_candidates(
        context,
        RankCandidatesArguments(limit=50, include_excluded=True),
    ).data
    explanation = toolset.explain_candidate(
        context,
        ExplainCandidateArguments(instrument_id="CORE"),
    ).data
    assert all(
        value is not None
        for value in (market, quality, regime, themes, leaders, candidates, explanation)
    )
    return {
        "market": market,
        "quality": quality,
        "regime": regime,
        "themes": themes,
        "leaders": leaders,
        "candidates": candidates,
        "explanation": explanation,
    }


def _reject(value: ToolOutput, **changes: object) -> None:
    payload = value.model_dump()
    payload.update(changes)
    with pytest.raises(ValidationError):
        type(value).model_validate(payload)


def _reject_payload(model: type[ToolOutput], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_manifest_and_quality_count_invariants_reject_contradictory_outputs(
    outputs: dict[str, ToolOutput],
) -> None:
    market = outputs["market"]
    first_file = market.files[0]  # type: ignore[attr-defined]
    _reject(market, returned_file_count=0)
    _reject(market, returned_file_count=2, files=(first_file, first_file))
    _reject(market, files_truncated=True)

    quality = outputs["quality"]
    _reject(quality, observed_at=quality.as_of + timedelta(seconds=1))  # type: ignore[attr-defined]
    _reject(quality, returned_issue_count=1)
    _reject(quality, blocking_issue_count=1, qualified=True)
    _reject(quality, issues_truncated=True)


def test_regime_projection_rejects_score_identity_transition_and_pending_drift(
    outputs: dict[str, ToolOutput],
) -> None:
    regime = outputs["regime"]
    payload = regime.model_dump()
    payload["raw"]["components"]["total"] += Decimal("1")
    _reject_payload(type(regime), payload)

    payload = regime.model_dump()
    payload["raw"]["raw_score"] += Decimal("1")
    _reject_payload(type(regime), payload)

    _reject(
        regime,
        raw=regime.raw.model_copy(  # type: ignore[attr-defined]
            update={"as_of": regime.as_of - timedelta(seconds=1)}  # type: ignore[attr-defined]
        ),
    )
    _reject(
        regime,
        raw=regime.raw.model_copy(  # type: ignore[attr-defined]
            update={"session_date": regime.session_date - timedelta(days=1)}  # type: ignore[attr-defined]
        ),
    )
    _reject(
        regime,
        event=regime.event.model_copy(  # type: ignore[attr-defined]
            update={"as_of": regime.as_of - timedelta(seconds=1)}  # type: ignore[attr-defined]
        ),
    )
    _reject(
        regime,
        event=regime.event.model_copy(  # type: ignore[attr-defined]
            update={"raw_regime": MarketRegime.DOWNTREND}
        ),
    )
    _reject(
        regime,
        event=regime.event.model_copy(  # type: ignore[attr-defined]
            update={"final_regime": MarketRegime.DOWNTREND}
        ),
    )
    _reject(
        regime,
        event=regime.event.model_copy(  # type: ignore[attr-defined]
            update={"source_result_hash": "f" * 64}
        ),
    )
    _reject(regime, transition_config_version="different-v1")
    _reject(regime, transition_config_hash="f" * 64)
    _reject(regime, transitioned=True)
    _reject(regime, pending_count=1)
    _reject(
        regime,
        pending_candidate=MarketRegime.DOWNTREND,
        pending_count=0,
        required_confirmation_sessions=2,
    )
    _reject(
        regime,
        pending_candidate=MarketRegime.DOWNTREND,
        pending_count=2,
        required_confirmation_sessions=2,
    )


def test_mainline_persistence_and_ranking_counts_cannot_be_forged(
    outputs: dict[str, ToolOutput],
) -> None:
    themes = outputs["themes"]
    persistence = themes.themes[0].persistence  # type: ignore[attr-defined]
    _reject(persistence, observed_sessions=persistence.observed_sessions + 1)
    _reject(persistence, window_sessions=1)
    _reject(persistence, top_k_hits=persistence.top_k_hits + 1)
    _reject(
        persistence,
        consecutive_top_k_sessions=persistence.consecutive_top_k_sessions + 1,
    )
    _reject(persistence, hit_ratio=persistence.hit_ratio + Decimal("0.01"))

    _reject(themes, returned_theme_count=0)
    _reject(themes, total_theme_count=0)
    _reject(themes, themes_truncated=not themes.themes_truncated)  # type: ignore[attr-defined]


def test_leader_score_gate_and_ranking_invariants_reject_mutation(
    outputs: dict[str, ToolOutput],
) -> None:
    ranking = outputs["leaders"]
    leader = ranking.leaders[0]  # type: ignore[attr-defined]
    component = leader.components[0]
    _reject(component, contribution=component.contribution + Decimal("1"))
    _reject(leader, components=tuple(reversed(leader.components)))
    changed_component = component.model_copy(
        update={"weight": Decimal(0), "contribution": Decimal(0)}
    )
    _reject(leader, components=(changed_component, *leader.components[1:]))
    _reject(
        leader,
        gross_score=leader.gross_score + Decimal("1"),
        score=leader.score + Decimal("1"),
    )
    _reject(
        leader,
        risk_penalty=Decimal("1"),
        score=leader.gross_score - Decimal("1"),
    )
    _reject(leader, score=leader.score - Decimal("0.1"))
    _reject(leader, ineligibility_reasons=("contradictory gate",))

    _reject(ranking, returned_leader_count=0)
    _reject(ranking, total_leader_count=0)
    _reject(ranking, leaders_truncated=not ranking.leaders_truncated)  # type: ignore[attr-defined]
    _reject(ranking, returned_exclusion_count=0)
    _reject(ranking, total_exclusion_count=0)
    _reject(ranking, exclusions_truncated=not ranking.exclusions_truncated)  # type: ignore[attr-defined]


def test_candidate_score_gate_and_ranking_invariants_reject_mutation(
    outputs: dict[str, ToolOutput],
) -> None:
    ranking = outputs["candidates"]
    candidate = next(  # type: ignore[attr-defined]
        item for item in ranking.candidates if item.tier is not CandidateTier.EXCLUDED
    )
    component = candidate.components[0]
    _reject(component, contribution=component.contribution + Decimal("1"))
    _reject(candidate, components=tuple(reversed(candidate.components)))
    changed_component = component.model_copy(
        update={"weight": Decimal(0), "contribution": Decimal(0)}
    )
    _reject(candidate, components=(changed_component, *candidate.components[1:]))
    _reject(
        candidate,
        gross_score=candidate.gross_score + Decimal("1"),
        score=candidate.score + Decimal("1"),
    )
    _reject(
        candidate,
        risk_penalty=Decimal("1"),
        score=candidate.gross_score - Decimal("1"),
    )
    _reject(candidate, score=candidate.score - Decimal("0.1"))
    _reject(
        candidate,
        exclusion_codes=(CandidateExclusionCode.MARKET_DOWNTREND,),
        exclusion_reasons=(),
    )
    _reject(
        candidate,
        tier=CandidateTier.A,
        exclusion_codes=(CandidateExclusionCode.MARKET_DOWNTREND,),
        exclusion_reasons=("contradictory hard gate",),
    )
    _reject(candidate, eligible=False)
    _reject(candidate, eligible_rank=None)

    _reject(ranking, returned_candidate_count=0)
    _reject(ranking, total_candidate_count=0)
    _reject(ranking, candidates_truncated=not ranking.candidates_truncated)  # type: ignore[attr-defined]
    _reject(ranking, returned_exclusion_count=0)
    _reject(ranking, total_exclusion_count=0)
    _reject(ranking, exclusions_truncated=not ranking.exclusions_truncated)  # type: ignore[attr-defined]
    _reject(ranking, excluded_candidates_included=False)


def test_candidate_explanation_requires_one_record_and_matching_kind(
    outputs: dict[str, ToolOutput],
) -> None:
    explanation = outputs["explanation"]
    ranking = outputs["candidates"]
    exclusion = ranking.exclusions[0]  # type: ignore[attr-defined]

    _reject(explanation, candidate=None, exclusion=None)
    _reject(explanation, exclusion=exclusion)
    _reject(explanation, kind="UPSTREAM_EXCLUSION")


def test_closed_output_models_are_frozen_and_reject_extra_fields(
    outputs: dict[str, ToolOutput],
) -> None:
    ranking = outputs["candidates"]
    payload = ranking.model_dump()
    payload["untrusted_probability"] = Decimal("0.99")

    _reject_payload(CandidateRankingOutput, payload)
    with pytest.raises(ValidationError):
        CandidateExplanationOutput.model_validate(
            {
                **outputs["explanation"].model_dump(),
                "kind": "UNKNOWN",
            }
        )
