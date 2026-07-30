from datetime import UTC, datetime

import pytest

from quant_agent.leaders.candidates import CandidateResult, CandidateTier
from quant_agent.regime.models import MarketRegime
from quant_agent.strategies.etf_rotation import (
    EtfRotationStrategy,
    EtfSignalInput,
)
from quant_agent.strategies.mainline_leader import (
    MainlineCandidateInput,
    MainlineLeaderStrategy,
)
from quant_agent.themes.scoring import ThemeState

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def etf(
    instrument: str,
    momentum: float,
    *,
    equity: bool = True,
    eligible: bool = True,
) -> EtfSignalInput:
    return EtfSignalInput(
        NOW,
        instrument,
        momentum,
        momentum,
        momentum,
        0.10,
        eligible,
        1000,
        100_000_000,
        0.01,
        True,
        equity,
        "snap",
    )


def candidate(instrument: str, leader_type: str = "TREND") -> CandidateResult:
    return CandidateResult(
        NOW,
        instrument,
        instrument,
        "TECH",
        leader_type,
        80,
        80,
        1,
        CandidateTier.A,
        True,
        (),
        ("support",),
        (),
        (),
        ("observe",),
        ("invalidate",),
        "snap",
        "candidate-v1",
    )


def test_etf_rotation_historical_filter_risk_weight_and_cash_fallback() -> None:
    strategy = EtfRotationStrategy()
    portfolio = strategy.generate(
        [etf("EQUITY", 0.2), etf("BOND", 0.1, equity=False), etf("BAD", 1, eligible=False)],
        parameter_version="p1",
    )
    weights = {item.instrument_id: item.weight for item in portfolio.targets}
    assert weights["BOND"] == pytest.approx(0.2)
    assert weights["EQUITY"] == pytest.approx(0.8)
    assert portfolio.cash_weight == 0
    assert portfolio.content_hash == portfolio.content_hash
    assert strategy.should_rebalance(portfolio, current_weights={}, last_rebalanced_at=None)
    cash = strategy.generate([etf("BAD", -0.1, eligible=False)], parameter_version="p1")
    assert cash.targets == () and cash.cash_weight == 1


def test_mainline_strategy_filters_types_and_scales_turnover() -> None:
    strategy = MainlineLeaderStrategy()
    portfolio = strategy.generate(
        [
            MainlineCandidateInput(candidate("TREND"), ThemeState.CONFIRMED),
            MainlineCandidateInput(candidate("CORE", "CORE"), ThemeState.CONFIRMED),
            MainlineCandidateInput(candidate("FADING"), ThemeState.FADING),
        ],
        as_of=NOW,
        market_regime=MarketRegime.UPTREND,
        current_weights={},
        parameter_version="p1",
        data_version="snap",
    )
    assert [item.instrument_id for item in portfolio.targets] == ["TREND"]
    assert portfolio.targets[0].weight == 0.2
    assert portfolio.cash_weight == 0.8
    down = strategy.generate(
        [MainlineCandidateInput(candidate("TREND"), ThemeState.CONFIRMED)],
        as_of=NOW,
        market_regime=MarketRegime.DOWNTREND,
        current_weights={"OLD": 0.4},
        parameter_version="p1",
        data_version="snap",
    )
    assert down.cash_weight >= 0.6
    assert down.targets == ()
    constrained = strategy.generate(
        [MainlineCandidateInput(candidate("TREND"), ThemeState.CONFIRMED)],
        as_of=NOW,
        market_regime=MarketRegime.UPTREND,
        current_weights={"OLD": 0.8},
        parameter_version="p1",
        data_version="snap",
    )
    assert constrained.warnings == ("target transition scaled by turnover constraint",)
    assert {item.instrument_id for item in constrained.targets} == {"OLD", "TREND"}
