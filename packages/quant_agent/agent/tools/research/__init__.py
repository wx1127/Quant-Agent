"""Deterministic, decision-bound research tools for the Agent."""

from quant_agent.agent.tools.research.contracts import (
    CandidateExplanationOutput,
    CandidateRankingOutput,
    DetectMarketRegimeArguments,
    ExplainCandidateArguments,
    GetMarketSnapshotArguments,
    LeaderRankingOutput,
    MarketDataQualityOutput,
    MarketRegimeOutput,
    MarketSnapshotOutput,
    RankCandidatesArguments,
    RankedListArguments,
    ThemeRankingOutput,
    ValidateMarketDataArguments,
)
from quant_agent.agent.tools.research.inputs import (
    AccountBoundLeaderInputs,
    DataBoundQualityContext,
    RegimeFeaturePair,
    ResearchInputError,
    ResearchInputInvalid,
    ResearchInputSource,
    ResearchInputUnavailable,
)
from quant_agent.agent.tools.research.service import ResearchPipeline
from quant_agent.agent.tools.research.toolset import (
    RESEARCH_TOOL_VERSION,
    ResearchToolset,
    build_research_tools,
)

__all__ = [
    "RESEARCH_TOOL_VERSION",
    "AccountBoundLeaderInputs",
    "CandidateExplanationOutput",
    "CandidateRankingOutput",
    "DataBoundQualityContext",
    "DetectMarketRegimeArguments",
    "ExplainCandidateArguments",
    "GetMarketSnapshotArguments",
    "LeaderRankingOutput",
    "MarketDataQualityOutput",
    "MarketRegimeOutput",
    "MarketSnapshotOutput",
    "RankCandidatesArguments",
    "RankedListArguments",
    "RegimeFeaturePair",
    "ResearchInputError",
    "ResearchInputInvalid",
    "ResearchInputSource",
    "ResearchInputUnavailable",
    "ResearchPipeline",
    "ResearchToolset",
    "ThemeRankingOutput",
    "ValidateMarketDataArguments",
    "build_research_tools",
]
