from datetime import UTC, datetime

import pytest

from quant_agent.backtest.contracts import AssetType
from quant_agent.portfolio.builder import (
    PortfolioBuilder,
    StrategyAllocation,
)
from quant_agent.portfolio.snapshots import AccountSnapshot, HoldingSnapshot
from quant_agent.strategies.contracts import TargetPortfolio, TargetWeight

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def account() -> AccountSnapshot:
    return AccountSnapshot(
        "account-snapshot-1",
        "paper-1",
        NOW,
        90_000,
        0,
        (
            HoldingSnapshot(
                "OLD",
                AssetType.STOCK,
                "BANK",
                1_000,
                800,
                200,
                8,
                10,
            ),
        ),
        "paper",
        "v1",
    )


def target(strategy: str, instrument: str, weight: float) -> TargetPortfolio:
    return TargetPortfolio(
        NOW,
        strategy,
        (TargetWeight(instrument, weight, f"{strategy} signal"),),
        1 - weight,
        "snap",
        "strategy-v1",
        "params-v1",
    )


def test_account_snapshot_hash_roundtrip_and_separate_frozen_quantity() -> None:
    snapshot = account()
    restored = AccountSnapshot.from_json(snapshot.to_json())
    assert restored == snapshot
    assert restored.content_hash == snapshot.content_hash
    assert restored.total_equity == 100_000
    assert restored.holdings[0].available_quantity == 800
    with pytest.raises(ValueError, match="equal"):
        HoldingSnapshot("X", AssetType.STOCK, None, 100, 100, 100, 1, 1)


def test_builder_isolates_strategy_budgets_caps_weights_and_explains_exit() -> None:
    allocations = [
        StrategyAllocation(
            "ETF_ROTATION",
            0.60,
            target("ETF_ROTATION", "ETF1", 0.50),
            {"ETF1": AssetType.ETF},
            {"ETF1": "BROAD"},
        ),
        StrategyAllocation(
            "MAINLINE_LEADER",
            0.20,
            target("MAINLINE_LEADER", "STOCK1", 0.50),
            {"STOCK1": AssetType.STOCK},
            {"STOCK1": "TECH"},
        ),
    ]
    result = PortfolioBuilder().build(
        allocations,
        account=account(),
        decision_id="decision-1",
        data_version="snap",
    )
    lines = {item.instrument_id: item for item in result.lines}
    assert lines["ETF1"].target_weight == pytest.approx(0.25)
    assert lines["STOCK1"].target_weight == pytest.approx(0.05)
    assert lines["STOCK1"].strategy_sources == ("MAINLINE_LEADER",)
    assert lines["OLD"].target_weight == 0
    assert lines["OLD"].reason == "position exit"
    assert result.strategy_budgets == {
        "ETF_ROTATION": 0.60,
        "MAINLINE_LEADER": 0.20,
    }
    assert result.cash_target_weight == pytest.approx(0.70)
    assert result.warnings
    assert not hasattr(result, "orders")
