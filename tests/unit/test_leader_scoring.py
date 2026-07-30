from datetime import UTC, datetime

import pytest

from quant_agent.features.tradeability import TradeabilityResult
from quant_agent.leaders.candidates import (
    CandidateEngine,
    CandidateInput,
    CandidateTier,
)
from quant_agent.leaders.scoring import (
    LeaderConfig,
    LeaderInput,
    LeaderScoringEngine,
    LeaderType,
)
from quant_agent.themes.scoring import ThemeState

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def leader_input(
    instrument: str,
    *,
    trend: float,
    liquidity: float,
    state: ThemeState = ThemeState.CONFIRMED,
) -> LeaderInput:
    return LeaderInput(
        NOW,
        instrument,
        f"Name {instrument}",
        "TECH",
        state,
        70,
        trend,
        liquidity,
        70,
        60,
        70,
        65,
        0,
        "snap",
    )


def tradeability(instrument: str, *, tradable: bool = True) -> TradeabilityResult:
    reasons = () if tradable else ("suspended",)
    return TradeabilityResult(
        instrument,
        tradable,
        reasons,
        1000,
        100_000_000,
        0.8,
        0.7,
        5_000_000,
        1_000_000,
        True,
        "snap",
        "trade-v1",
    )


def test_leaders_only_rank_confirmed_theme_and_not_by_return_alone() -> None:
    engine = LeaderScoringEngine()
    results = engine.rank(
        [
            leader_input("TREND", trend=75, liquidity=60),
            leader_input("CAP", trend=50, liquidity=85),
        ]
    )
    assert {item.leader_type for item in results} == {
        LeaderType.TREND,
        LeaderType.CAPACITY,
    }
    assert all(item.trade_candidate_type for item in results)
    with pytest.raises(ValueError, match="confirmed"):
        engine.rank([leader_input("X", trend=80, liquidity=80, state=ThemeState.EMERGING)])
    with pytest.raises(ValueError):
        LeaderConfig(within_theme_weight=0.9)


def test_candidate_ranking_preserves_exclusions_and_has_no_probability() -> None:
    leaders = LeaderScoringEngine().rank(
        [
            leader_input("TREND", trend=75, liquidity=60),
            leader_input("CAP", trend=50, liquidity=85),
        ]
    )
    items = []
    for leader in leaders:
        items.append(
            CandidateInput(
                NOW,
                leader,
                tradeability(leader.instrument_id, tradable=leader.instrument_id == "TREND"),
                90,
                90,
                90,
                90,
                90,
                90,
                0,
                () if leader.instrument_id == "TREND" else ("regulatory inquiry",),
            )
        )
    results = CandidateEngine().rank(items)
    accepted = next(item for item in results if item.instrument_id == "TREND")
    excluded = next(item for item in results if item.instrument_id == "CAP")
    assert accepted.tradable and accepted.tier is CandidateTier.A
    assert excluded.tier is CandidateTier.EXCLUDED
    assert excluded.exclusion_reasons
    assert not hasattr(accepted, "probability")
    assert accepted.invalidations
    with pytest.raises(ValueError, match="as_of"):
        CandidateInput(
            NOW.replace(day=2),
            leaders[0],
            tradeability("X"),
            50,
            50,
            50,
            50,
            50,
            50,
            0,
            (),
        )
    assert CandidateEngine().rank([]) == ()
