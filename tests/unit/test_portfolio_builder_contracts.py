"""Adversarial validation tests for target-portfolio immutable contracts."""

from dataclasses import replace
from datetime import UTC, timedelta
from decimal import Decimal

import pytest
from tests.unit import test_portfolio_builder as fixtures

from quant_agent.backtest import TradableInstrumentType
from quant_agent.portfolio import (
    PITIndustryClassification,
    PortfolioBuilderConfig,
    PortfolioBuildRequest,
    PortfolioConstraintAdjustment,
    PortfolioTargetLine,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    StrategyWeightContribution,
)


def _rebuild_sleeve(
    base: StrategySleeve,
    targets: tuple[SleeveTarget, ...],
) -> StrategySleeve:
    return StrategySleeve.build(
        sleeve_id=base.sleeve_id,
        kind=base.kind,
        as_of=base.as_of,
        data_version=base.data_version,
        strategy_name=base.strategy_name,
        strategy_version=base.strategy_version,
        strategy_config_hash=base.strategy_config_hash,
        source_result_hash=base.source_result_hash,
        targets=targets,
    )


def _contribution(
    *,
    sleeve_id: str,
    sleeve_hash: str,
    kind: StrategySleeveKind,
    local: str,
    budget: str,
    proposed: str,
    applied: str,
) -> StrategyWeightContribution:
    return StrategyWeightContribution(
        sleeve_id=sleeve_id,
        sleeve_hash=sleeve_hash,
        sleeve_kind=kind,
        source_result_hash="c" * 64,
        local_target_weight=Decimal(local),
        sleeve_budget=Decimal(budget),
        proposed_portfolio_weight=Decimal(proposed),
        applied_portfolio_weight=Decimal(applied),
        reason="fixed contract test contribution",
    )


def test_config_rejects_empty_lossy_out_of_range_caps_and_oversubscribed_budgets() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PortfolioBuilderConfig(version=" ")
    with pytest.raises(ValueError, match="exact Decimal"):
        PortfolioBuilderConfig(etf_core_budget=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        PortfolioBuilderConfig(etf_core_budget=Decimal("-0.01"))
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        PortfolioBuilderConfig(etf_core_budget=Decimal("1.01"))
    with pytest.raises(ValueError, match="caps must be positive"):
        PortfolioBuilderConfig(maximum_instrument_weight=Decimal(0))
    with pytest.raises(ValueError, match="preserve the minimum cash"):
        PortfolioBuilderConfig(
            etf_core_budget=Decimal("0.70"),
            stock_enhancement_budget=Decimal("0.30"),
            minimum_cash_weight=Decimal("0.10"),
        )


def test_request_normalizes_identity_exposes_fingerprint_and_rejects_bad_fields() -> None:
    account = fixtures._account()
    request = PortfolioBuildRequest(
        decision_id="  decision  ",
        as_of=fixtures.AS_OF,
        data_version="  snapshot-v1  ",
        account_snapshot_id="  account-snapshot-20260828  ",
        account_snapshot_hash=account.content_hash,
    )

    assert request.decision_id == "decision"
    assert request.fingerprint_payload() == {
        "account_snapshot_hash": account.content_hash,
        "account_snapshot_id": "account-snapshot-20260828",
        "as_of": fixtures.AS_OF.astimezone(UTC).isoformat(timespec="microseconds"),
        "data_version": fixtures.DATA_VERSION,
        "decision_id": "decision",
    }
    with pytest.raises(ValueError, match="non-empty"):
        replace(request, decision_id=" ")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(request, account_snapshot_hash="A" * 64)


def test_sleeve_target_rejects_empty_enum_lossy_and_out_of_range_values() -> None:
    target = fixtures._sleeve(StrategySleeveKind.ETF_CORE).targets[0]

    with pytest.raises(ValueError, match="instrument_id must be non-empty"):
        replace(target, instrument_id=" ")
    with pytest.raises(ValueError, match="target reason must be non-empty"):
        replace(target, reason=" ")
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(target, instrument_type="ETF")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(target, local_target_weight=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(target, local_target_weight=Decimal("-0.01"))
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(target, local_target_weight=Decimal("1.01"))


def test_strategy_sleeve_rejects_bad_kind_fields_hash_order_type_and_total_weight() -> None:
    sleeve = fixtures._sleeve(StrategySleeveKind.ETF_CORE)

    with pytest.raises(ValueError, match="StrategySleeveKind"):
        replace(sleeve, kind="ETF_CORE")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        replace(sleeve, strategy_name=" ")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(sleeve, strategy_config_hash="short")
    with pytest.raises(ValueError, match="unique and sorted"):
        _rebuild_sleeve(sleeve, tuple(reversed(sleeve.targets)))

    stock_target = SleeveTarget(
        instrument_id="CN.SSE.600999",
        instrument_type=TradableInstrumentType.STOCK,
        local_target_weight=Decimal("0.20"),
        reason="wrong instrument type for ETF sleeve",
    )
    with pytest.raises(ValueError, match="sleeve instrument type"):
        _rebuild_sleeve(sleeve, (stock_target,))

    overweight_targets = (
        replace(sleeve.targets[0], local_target_weight=Decimal("0.60")),
        replace(sleeve.targets[1], local_target_weight=Decimal("0.60")),
    )
    with pytest.raises(ValueError, match="cannot exceed one"):
        _rebuild_sleeve(sleeve, overweight_targets)
    with pytest.raises(ValueError, match="sleeve_hash"):
        replace(sleeve, strategy_version="tampered-v2")


def test_pit_classification_rejects_empty_future_session_and_bad_source_identity() -> None:
    classification = fixtures._classification(fixtures.STOCK_A)

    assert len(classification.classification_hash) == 64
    with pytest.raises(ValueError, match="industry_id must be non-empty"):
        replace(classification, industry_id=" ")
    with pytest.raises(ValueError, match="before its session"):
        replace(
            classification,
            available_at=fixtures.AS_OF - timedelta(days=1),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(classification, source_hash="g" * 64)


def test_contribution_rejects_identity_enum_decimal_range_and_arithmetic_tampering() -> None:
    contribution = fixtures._build().lines[0].contributions[0]

    with pytest.raises(ValueError, match="sleeve_id must be non-empty"):
        replace(contribution, sleeve_id=" ")
    with pytest.raises(ValueError, match="contribution reason must be non-empty"):
        replace(contribution, reason=" ")
    with pytest.raises(ValueError, match="StrategySleeveKind"):
        replace(contribution, sleeve_kind="ETF_CORE")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SHA-256"):
        replace(contribution, sleeve_hash="short")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(contribution, source_result_hash="G" * 64)
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(contribution, sleeve_budget=0.60)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(contribution, applied_portfolio_weight=Decimal("-0.01"))
    with pytest.raises(ValueError, match="local target times sleeve budget"):
        replace(contribution, proposed_portfolio_weight=Decimal("0.29"))
    with pytest.raises(ValueError, match="cannot increase"):
        replace(contribution, applied_portfolio_weight=Decimal("0.31"))


def test_constraint_adjustment_rejects_empty_decimal_range_and_non_reducing_values() -> None:
    adjustment = fixtures._build().adjustments[0]

    with pytest.raises(ValueError, match="adjustment scope must be non-empty"):
        replace(adjustment, scope_id=" ")
    with pytest.raises(ValueError, match="adjustment rationale must be non-empty"):
        replace(adjustment, rationale=" ")
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(adjustment, before_weight=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(adjustment, after_weight=Decimal("-0.01"))
    with pytest.raises(ValueError, match="must reduce"):
        replace(adjustment, after_weight=adjustment.before_weight)


def test_target_line_rejects_scalar_arithmetic_weight_and_industry_tampering() -> None:
    result = fixtures._build()
    etf = result.lines[0]
    stock = result.lines[2]

    with pytest.raises(ValueError, match="instrument_id must be non-empty"):
        replace(etf, instrument_id=" ")
    with pytest.raises(ValueError, match="industry_id must be non-empty"):
        replace(stock, industry_id=" ")
    with pytest.raises(ValueError, match="target rationale must be non-empty"):
        replace(etf, rationale=" ")
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(etf, current_market_value=2000)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(etf, current_market_value=Decimal("-1"))
    with pytest.raises(ValueError, match="weight_delta"):
        replace(etf, weight_delta=Decimal(0))
    with pytest.raises(ValueError, match="absolute_weight_delta"):
        replace(etf, absolute_weight_delta=Decimal(0))
    with pytest.raises(ValueError, match="cannot exceed proposed"):
        replace(
            etf,
            target_weight=Decimal("0.31"),
            weight_delta=Decimal("0.11"),
            absolute_weight_delta=Decimal("0.11"),
        )
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(
            etf,
            current_weight=Decimal("1.01"),
            weight_delta=etf.target_weight - Decimal("1.01"),
            absolute_weight_delta=abs(etf.target_weight - Decimal("1.01")),
        )
    with pytest.raises(ValueError, match="point-in-time industry identity"):
        replace(stock, industry_id=None, industry_classification_hash=None)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(stock, industry_classification_hash="short")
    with pytest.raises(ValueError, match="ETF target lines cannot carry"):
        replace(
            etf,
            industry_id=fixtures.INDUSTRY,
            industry_classification_hash="d" * 64,
        )


def test_target_line_rejects_contribution_sums_order_and_wrong_sleeve_kind() -> None:
    result = fixtures._build()
    etf = result.lines[0]

    with pytest.raises(ValueError, match="proposed weight must equal"):
        replace(etf, proposed_target_weight=Decimal("0.31"))
    with pytest.raises(ValueError, match="applied weight must equal"):
        replace(
            etf,
            target_weight=Decimal("0.24"),
            weight_delta=Decimal("0.04"),
            absolute_weight_delta=Decimal("0.04"),
        )

    first = _contribution(
        sleeve_id="sleeve-a",
        sleeve_hash="a" * 64,
        kind=StrategySleeveKind.ETF_CORE,
        local="0.25",
        budget="0.60",
        proposed="0.15",
        applied="0.125",
    )
    second = replace(first, sleeve_id="sleeve-b", sleeve_hash="b" * 64)
    with pytest.raises(ValueError, match="unique and deterministically sorted"):
        replace(etf, contributions=(second, first))

    wrong_kind = _contribution(
        sleeve_id="stock-sleeve",
        sleeve_hash="e" * 64,
        kind=StrategySleeveKind.STOCK_ENHANCEMENT,
        local="0.50",
        budget="0.30",
        proposed="0.15",
        applied="0.125",
    )
    with pytest.raises(ValueError, match="kind does not match"):
        replace(
            etf,
            proposed_target_weight=Decimal("0.15"),
            target_weight=Decimal("0.125"),
            weight_delta=Decimal("-0.075"),
            absolute_weight_delta=Decimal("0.075"),
            contributions=(wrong_kind,),
        )


def test_root_rejects_sleeve_classification_and_adjustment_identity_sets() -> None:
    result = fixtures._build()

    with pytest.raises(ValueError, match="sleeve hashes must be"):
        replace(result, sleeve_hashes=result.sleeve_hashes[:1])
    with pytest.raises(ValueError, match="SHA-256"):
        replace(
            result,
            sleeve_hashes=tuple(sorted((result.sleeve_hashes[0], "z" * 64))),
        )
    with pytest.raises(ValueError, match="classification hashes must be unique"):
        replace(result, classification_hashes=tuple(reversed(result.classification_hashes)))
    with pytest.raises(ValueError, match="exactly cover"):
        replace(result, classification_hashes=result.classification_hashes[:1])
    with pytest.raises(ValueError, match="classification_input_hash"):
        replace(result, classification_input_hash="f" * 64)
    with pytest.raises(ValueError, match="adjustments must be unique"):
        replace(result, adjustments=tuple(reversed(result.adjustments)))


def test_root_rejects_metric_config_and_portfolio_decomposition_tampering() -> None:
    result = fixtures._build()

    with pytest.raises(ValueError, match="exact Decimal"):
        replace(result, one_way_turnover=0.33)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metrics cannot be negative"):
        replace(result, total_equity=Decimal("-1"))
    with pytest.raises(ValueError, match="total_equity must be positive"):
        replace(result, total_equity=Decimal(0))
    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        replace(result, etf_core_budget=Decimal("1.01"))
    with pytest.raises(ValueError, match="caps must be positive"):
        replace(result, maximum_instrument_weight=Decimal(0))
    with pytest.raises(ValueError, match="preserve the minimum cash"):
        replace(
            result,
            etf_core_budget=Decimal("0.70"),
            stock_enhancement_budget=Decimal("0.30"),
        )
    with pytest.raises(ValueError, match="config_hash"):
        replace(result, config_hash="f" * 64)
    with pytest.raises(ValueError, match="current gross and cash"):
        replace(result, current_gross_weight=Decimal("0.31"))
    with pytest.raises(ValueError, match="target gross and cash"):
        replace(result, target_gross_weight=Decimal("0.64"))
    with pytest.raises(ValueError, match="ETF plus stock"):
        replace(result, etf_target_weight=Decimal("0.42"))
    with pytest.raises(ValueError, match="minimum cash"):
        replace(
            result,
            target_gross_weight=Decimal("0.95"),
            target_cash_weight=Decimal("0.05"),
            etf_target_weight=Decimal("0.60"),
            stock_target_weight=Decimal("0.35"),
        )
    with pytest.raises(ValueError, match="ETF target weight exceeds"):
        replace(
            result,
            etf_target_weight=Decimal("0.61"),
            stock_target_weight=Decimal("0.02"),
        )
    with pytest.raises(ValueError, match="stock target weight exceeds"):
        replace(
            result,
            etf_target_weight=Decimal("0.32"),
            stock_target_weight=Decimal("0.31"),
        )


def test_root_rejects_line_totals_order_budget_binding_and_turnover_tampering() -> None:
    result = fixtures._build()

    with pytest.raises(ValueError, match="lines must be unique and sorted"):
        replace(result, lines=tuple(reversed(result.lines)))
    with pytest.raises(ValueError, match="current gross weight"):
        replace(
            result,
            current_gross_weight=Decimal("0.31"),
            current_cash_weight=Decimal("0.69"),
        )
    with pytest.raises(ValueError, match="target gross weight must equal target-line"):
        replace(
            result,
            target_gross_weight=Decimal("0.64"),
            target_cash_weight=Decimal("0.36"),
            etf_target_weight=Decimal("0.44"),
        )
    with pytest.raises(ValueError, match="ETF target weight must equal"):
        replace(
            result,
            etf_target_weight=Decimal("0.42"),
            stock_target_weight=Decimal("0.21"),
        )

    etf = result.lines[0]
    contribution = etf.contributions[0]
    wrong_budget_contribution = replace(
        contribution,
        local_target_weight=Decimal("0.60"),
        sleeve_budget=Decimal("0.50"),
    )
    wrong_budget_line = replace(etf, contributions=(wrong_budget_contribution,))
    with pytest.raises(ValueError, match="budget does not match"):
        replace(result, lines=(wrong_budget_line, *result.lines[1:]))

    with pytest.raises(ValueError, match="gross instrument turnover"):
        replace(result, gross_instrument_turnover=Decimal("0.34"))
    with pytest.raises(ValueError, match="one-way turnover"):
        replace(result, one_way_turnover=Decimal("0.34"))


def test_root_rejects_isolated_sleeve_overrun_and_constraint_adjustment_projection() -> None:
    result = fixtures._build()
    first_etf = result.lines[0]
    second_etf = result.lines[1]
    first_contribution = replace(
        first_etf.contributions[0],
        local_target_weight=Decimal("0.60"),
        proposed_portfolio_weight=Decimal("0.36"),
    )
    second_contribution = replace(
        second_etf.contributions[0],
        local_target_weight=Decimal("0.50"),
        proposed_portfolio_weight=Decimal("0.30"),
        applied_portfolio_weight=Decimal("0.25"),
    )
    expanded_first = replace(
        first_etf,
        proposed_target_weight=Decimal("0.36"),
        contributions=(first_contribution,),
    )
    expanded_second = replace(
        second_etf,
        proposed_target_weight=Decimal("0.30"),
        target_weight=Decimal("0.25"),
        target_market_value=Decimal(2500),
        weight_delta=Decimal("0.25"),
        absolute_weight_delta=Decimal("0.25"),
        contributions=(second_contribution,),
    )
    with pytest.raises(ValueError, match="exceed an isolated sleeve budget"):
        replace(
            result,
            lines=(expanded_first, expanded_second, *result.lines[2:]),
            target_gross_weight=Decimal("0.70"),
            target_cash_weight=Decimal("0.30"),
            etf_target_weight=Decimal("0.50"),
            gross_instrument_turnover=Decimal("0.40"),
            one_way_turnover=Decimal("0.40"),
        )

    changed_adjustment = replace(result.adjustments[0], after_weight=Decimal("0.19"))
    with pytest.raises(ValueError, match="adjustments do not match"):
        replace(result, adjustments=(changed_adjustment, *result.adjustments[1:]))

    identity = result.identity_payload()
    assert identity["account_snapshot_hash"] == result.account_snapshot_hash
    assert identity["input_hash"] == result.input_hash
    assert identity["result_hash"] == result.result_hash


def test_classification_hash_is_bound_to_every_pit_identity_field() -> None:
    classification = fixtures._classification(fixtures.STOCK_A)
    changed: tuple[PITIndustryClassification, ...] = (
        replace(classification, industry_id="INSURANCE"),
        replace(classification, classification_version="sw-v2"),
        replace(classification, source_hash="e" * 64),
    )

    assert all(item.classification_hash != classification.classification_hash for item in changed)


def test_adjustment_and_target_contracts_reject_invalid_runtime_enums() -> None:
    result = fixtures._build()
    adjustment: PortfolioConstraintAdjustment = result.adjustments[0]
    line: PortfolioTargetLine = result.lines[0]

    with pytest.raises(ValueError, match="PortfolioAdjustmentCode"):
        replace(adjustment, code="INDUSTRY_CAP")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(line, instrument_type="ETF")  # type: ignore[arg-type]
