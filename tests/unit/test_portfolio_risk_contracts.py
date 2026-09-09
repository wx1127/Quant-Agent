"""Policy, context, configuration, and request-binding tests for portfolio risk."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit import test_portfolio_builder as fixtures

from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import (
    PortfolioRiskContext,
    PortfolioRiskInputError,
    PortfolioRiskPolicy,
    build_portfolio_risk_request,
    load_portfolio_risk_policy,
)


def _policy() -> PortfolioRiskPolicy:
    return PortfolioRiskPolicy(
        version="portfolio-risk-test-v1",
        maximum_total_weight=Decimal("0.80"),
        maximum_etf_weight=Decimal("0.50"),
        maximum_stock_weight=Decimal("0.30"),
        maximum_single_etf_weight=Decimal("0.30"),
        maximum_single_stock_weight=Decimal("0.15"),
        maximum_stock_industry_weight=Decimal("0.25"),
        turnover_warning_threshold=Decimal("0.40"),
        maximum_one_way_turnover=Decimal("0.50"),
        drawdown_warning_threshold=Decimal("0.10"),
        drawdown_stop_new_risk_threshold=Decimal("0.20"),
    )


def _context(
    *,
    account_hash: str | None = None,
    equity: Decimal = Decimal("10000"),
    high_watermark: Decimal = Decimal("10000"),
    regime_budget: Decimal = Decimal("0.80"),
) -> PortfolioRiskContext:
    account = fixtures._account()
    return PortfolioRiskContext.build(
        account_snapshot_hash=account_hash or account.content_hash,
        as_of=account.as_of,
        valuation_version="account-mark-v1",
        current_equity=equity,
        equity_high_watermark=high_watermark,
        high_watermark_at=account.as_of - timedelta(days=1),
        drawdown_source_hash=stable_hash({"drawdown": "account-1"}),
        regime_risk_budget=regime_budget,
        regime_result_hash=stable_hash({"regime": "uptrend"}),
    )


def test_checked_in_toml_matches_code_defaults_and_uses_exact_decimal_strings() -> None:
    path = Path(__file__).parents[2] / "configs" / "risk" / "portfolio-v1.toml"
    loaded = load_portfolio_risk_policy(path)

    assert loaded == PortfolioRiskPolicy()
    assert len(loaded.policy_hash) == 64
    assert loaded.policy_hash == PortfolioRiskPolicy().policy_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"maximum_total_weight": 0.8}, "exact Decimal"),
        ({"maximum_total_weight": Decimal("1.01")}, "within 0..1"),
        ({"maximum_single_etf_weight": Decimal(0)}, "hard.*positive"),
        ({"maximum_single_etf_weight": Decimal("0.51")}, "ETF limits"),
        ({"maximum_stock_industry_weight": Decimal("0.31")}, "stock limits"),
        ({"turnover_warning_threshold": Decimal("0.50")}, "turnover warning"),
        ({"drawdown_warning_threshold": Decimal("0.20")}, "drawdown warning"),
    ],
)
def test_policy_rejects_lossy_unordered_or_non_monotone_thresholds(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_policy(), **changes)


def test_context_recomputes_drawdown_normalizes_timezones_and_rejects_tampering() -> None:
    local = _context(high_watermark=Decimal("12500"))
    utc = PortfolioRiskContext.build(
        account_snapshot_hash=local.account_snapshot_hash,
        as_of=local.as_of.astimezone(UTC),
        valuation_version=local.valuation_version,
        current_equity=local.current_equity,
        equity_high_watermark=local.equity_high_watermark,
        high_watermark_at=local.high_watermark_at.astimezone(UTC),
        drawdown_source_hash=local.drawdown_source_hash,
        regime_risk_budget=local.regime_risk_budget,
        regime_result_hash=local.regime_result_hash,
    )

    assert local.current_drawdown == Decimal("0.20")
    assert utc.context_hash == local.context_hash
    with pytest.raises(ValueError, match="context_hash"):
        replace(local, regime_risk_budget=Decimal("0.70"))
    with pytest.raises(FrozenInstanceError):
        local.current_equity = Decimal(1)  # type: ignore[misc]


def test_context_rejects_future_peak_invalid_equity_budget_and_hashes() -> None:
    context = _context()

    with pytest.raises(ValueError, match="future"):
        PortfolioRiskContext.build(
            account_snapshot_hash=context.account_snapshot_hash,
            as_of=context.as_of,
            valuation_version=context.valuation_version,
            current_equity=context.current_equity,
            equity_high_watermark=context.equity_high_watermark,
            high_watermark_at=context.as_of + timedelta(microseconds=1),
            drawdown_source_hash=context.drawdown_source_hash,
            regime_risk_budget=context.regime_risk_budget,
            regime_result_hash=context.regime_result_hash,
        )
    with pytest.raises(ValueError, match="below current equity"):
        _context(high_watermark=Decimal("9999"))
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        _context(regime_budget=Decimal("1.01"))
    with pytest.raises(ValueError, match="SHA-256"):
        _context(account_hash="not-a-hash")


def test_request_hash_binds_drawdown_and_regime_context() -> None:
    account = fixtures._account()
    proposal = fixtures._build(account=account)
    policy = _policy()
    first_context = _context()
    second_context = _context(regime_budget=Decimal("0.70"))

    first = build_portfolio_risk_request(
        account=account,
        proposal=proposal,
        context=first_context,
        policy=policy,
    )
    second = build_portfolio_risk_request(
        account=account,
        proposal=proposal,
        context=second_context,
        policy=policy,
    )

    assert first.risk_context_hash == first_context.context_hash
    assert second.risk_context_hash == second_context.context_hash
    assert first.request_hash != second.request_hash


def test_request_adapter_rejects_context_not_bound_to_account() -> None:
    account = fixtures._account()
    proposal = fixtures._build(account=account)

    with pytest.raises(PortfolioRiskInputError, match="context account hash"):
        build_portfolio_risk_request(
            account=account,
            proposal=proposal,
            context=_context(account_hash="f" * 64),
            policy=_policy(),
        )
