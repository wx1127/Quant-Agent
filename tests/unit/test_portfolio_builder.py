"""Isolated sleeve aggregation, reducing-only constraints, and audit hashes."""

from collections.abc import Iterable
from dataclasses import fields, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from inspect import signature

import pytest
from tests.unit import test_etf_rotation as etf_fixtures
from tests.unit import test_mainline_leader_strategy as stock_fixtures

from quant_agent.backtest import TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.portfolio import (
    AccountSnapshot,
    CashSnapshot,
    PITIndustryClassification,
    PortfolioAdjustmentCode,
    PortfolioBuilderConfig,
    PortfolioBuildInputError,
    PortfolioBuildRequest,
    PortfolioPositionSnapshot,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    TargetPortfolio,
    TargetPortfolioBuilder,
    sleeve_from_etf_decision,
    sleeve_from_mainline_decision,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.etf_rotation import (
    ETFRotationDecisionStatus,
    ETFRotationStrategy,
)

AS_OF = datetime.combine(date(2026, 8, 28), time(18), tzinfo=SHANGHAI_TZ)
VALUATION_AT = datetime.combine(date(2026, 8, 28), time(15), tzinfo=SHANGHAI_TZ)
DATA_VERSION = "snapshot-v1"
ETF_A = "CN.SSE.510300"
ETF_B = "CN.SSE.510500"
STOCK_A = "CN.SSE.600000"
STOCK_B = "CN.SSE.600001"
EXIT_STOCK = "CN.SSE.600002"
INDUSTRY = "BANK"
MARK_POLICY_HASH = stable_hash({"policy": "position-mark-v1"})


def _position(
    instrument_id: str,
    instrument_type: TradableInstrumentType,
    market_value: str,
) -> PortfolioPositionSnapshot:
    quantity = Decimal(100)
    price = Decimal(market_value) / quantity
    return PortfolioPositionSnapshot.build(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        available_quantity=quantity,
        frozen_quantity=Decimal(0),
        unsettled_quantity=Decimal(0),
        average_cost=price,
        valuation_price=price,
        price_observed_at=VALUATION_AT,
        price_available_at=VALUATION_AT,
        position_as_of=AS_OF,
        price_data_version=DATA_VERSION,
        price_source_hash=stable_hash({"instrument_id": instrument_id, "price": price}),
        mark_policy_hash=MARK_POLICY_HASH,
    )


def _account(*, include_exit: bool = False) -> AccountSnapshot:
    positions = [
        _position(ETF_A, TradableInstrumentType.ETF, "2000"),
        _position(STOCK_A, TradableInstrumentType.STOCK, "1000"),
    ]
    if include_exit:
        positions.append(_position(EXIT_STOCK, TradableInstrumentType.STOCK, "1000"))
    cash_total = Decimal(6000 if include_exit else 7000)
    cash = CashSnapshot(
        as_of=AS_OF,
        currency="CNY",
        total_cash=cash_total,
        available_cash=cash_total,
        frozen_cash=Decimal(0),
    )
    return AccountSnapshot.build(
        snapshot_id="account-snapshot-20260828",
        account_id="paper-account-1",
        runtime_mode=RuntimeMode.PAPER,
        as_of=AS_OF,
        valuation_at=VALUATION_AT,
        data_version=DATA_VERSION,
        currency="CNY",
        cash=cash,
        positions=tuple(positions),
        previous_snapshot_hash="a" * 64,
        source_event_log_hash="b" * 64,
    )


def _request(account: AccountSnapshot) -> PortfolioBuildRequest:
    return PortfolioBuildRequest(
        decision_id="portfolio-decision-20260828",
        as_of=AS_OF,
        data_version=DATA_VERSION,
        account_snapshot_id=account.snapshot_id,
        account_snapshot_hash=account.content_hash,
    )


def _sleeve(
    kind: StrategySleeveKind,
    *,
    as_of: datetime = AS_OF,
    data_version: str = DATA_VERSION,
) -> StrategySleeve:
    if kind is StrategySleeveKind.ETF_CORE:
        sleeve_id = "etf-core"
        targets = (
            SleeveTarget(
                instrument_id=ETF_A,
                instrument_type=TradableInstrumentType.ETF,
                local_target_weight=Decimal("0.50"),
                reason="large-cap ETF ranks first",
            ),
            SleeveTarget(
                instrument_id=ETF_B,
                instrument_type=TradableInstrumentType.ETF,
                local_target_weight=Decimal("0.30"),
                reason="mid-cap ETF ranks second",
            ),
        )
    else:
        sleeve_id = "stock-enhancement"
        targets = (
            SleeveTarget(
                instrument_id=STOCK_A,
                instrument_type=TradableInstrumentType.STOCK,
                local_target_weight=Decimal("0.50"),
                reason="primary leader",
            ),
            SleeveTarget(
                instrument_id=STOCK_B,
                instrument_type=TradableInstrumentType.STOCK,
                local_target_weight=Decimal("0.30"),
                reason="secondary leader",
            ),
        )
    return StrategySleeve.build(
        sleeve_id=sleeve_id,
        kind=kind,
        as_of=as_of,
        data_version=data_version,
        strategy_name=sleeve_id,
        strategy_version="strategy-v1",
        strategy_config_hash=stable_hash({"strategy": sleeve_id, "config": "v1"}),
        source_result_hash=stable_hash({"strategy": sleeve_id, "result": "fixed"}),
        targets=targets,
    )


def _default_sleeves() -> tuple[StrategySleeve, StrategySleeve]:
    return (
        _sleeve(StrategySleeveKind.ETF_CORE),
        _sleeve(StrategySleeveKind.STOCK_ENHANCEMENT),
    )


def _classification(
    instrument_id: str,
    *,
    industry_id: str = INDUSTRY,
    session_date: date | None = None,
    available_at: datetime | None = None,
    data_version: str = DATA_VERSION,
    classification_version: str = "sw-v1",
) -> PITIndustryClassification:
    return PITIndustryClassification(
        instrument_id=instrument_id,
        industry_id=industry_id,
        session_date=session_date or AS_OF.date(),
        available_at=available_at or AS_OF - timedelta(hours=1),
        data_version=data_version,
        classification_version=classification_version,
        source_hash=stable_hash(
            {
                "classification_version": classification_version,
                "industry_id": industry_id,
                "instrument_id": instrument_id,
            }
        ),
    )


def _default_classifications() -> tuple[PITIndustryClassification, ...]:
    return (_classification(STOCK_A), _classification(STOCK_B))


def _config(
    *,
    maximum_instrument_weight: str = "0.25",
    maximum_stock_industry_weight: str = "0.20",
) -> PortfolioBuilderConfig:
    return PortfolioBuilderConfig(
        version="portfolio-builder-test-v1",
        etf_core_budget=Decimal("0.60"),
        stock_enhancement_budget=Decimal("0.30"),
        minimum_cash_weight=Decimal("0.10"),
        maximum_instrument_weight=Decimal(maximum_instrument_weight),
        maximum_stock_industry_weight=Decimal(maximum_stock_industry_weight),
    )


def _build(
    *,
    account: AccountSnapshot | None = None,
    sleeves: Iterable[StrategySleeve] | None = None,
    classifications: Iterable[PITIndustryClassification] | None = None,
    config: PortfolioBuilderConfig | None = None,
) -> TargetPortfolio:
    bound_account = account or _account()
    return TargetPortfolioBuilder(config or _config()).build(
        request=_request(bound_account),
        account=bound_account,
        sleeves=sleeves if sleeves is not None else _default_sleeves(),
        classifications=(
            classifications if classifications is not None else _default_classifications()
        ),
    )


def test_isolated_budgets_aggregate_then_caps_reduce_instrument_and_industry_risk() -> None:
    result = _build()
    lines = {line.instrument_id: line for line in result.lines}

    assert lines[ETF_A].current_weight == Decimal("0.20")
    assert lines[ETF_A].proposed_target_weight == Decimal("0.30")
    assert lines[ETF_A].target_weight == Decimal("0.25")
    assert lines[ETF_B].current_weight == 0
    assert lines[ETF_B].proposed_target_weight == Decimal("0.18")
    assert lines[ETF_B].target_weight == Decimal("0.18")
    assert lines[STOCK_A].current_weight == Decimal("0.10")
    assert lines[STOCK_A].proposed_target_weight == Decimal("0.15")
    assert lines[STOCK_A].target_weight == Decimal("0.125")
    assert lines[STOCK_B].proposed_target_weight == Decimal("0.09")
    assert lines[STOCK_B].target_weight == Decimal("0.075")

    assert lines[ETF_A].contributions[0].sleeve_budget == Decimal("0.60")
    assert lines[ETF_A].contributions[0].local_target_weight == Decimal("0.50")
    assert lines[ETF_A].contributions[0].proposed_portfolio_weight == Decimal("0.30")
    assert lines[ETF_A].contributions[0].applied_portfolio_weight == Decimal("0.25")
    assert lines[STOCK_A].industry_id == INDUSTRY
    assert lines[STOCK_B].industry_id == INDUSTRY
    assert lines[ETF_A].industry_id is None
    assert lines[STOCK_A].target_market_value == Decimal(1250)

    adjustments = {(item.code, item.scope_id): item for item in result.adjustments}
    instrument_cap = adjustments[(PortfolioAdjustmentCode.INSTRUMENT_CAP, ETF_A)]
    industry_cap = adjustments[(PortfolioAdjustmentCode.INDUSTRY_CAP, INDUSTRY)]
    assert (instrument_cap.before_weight, instrument_cap.after_weight) == (
        Decimal("0.30"),
        Decimal("0.25"),
    )
    assert (industry_cap.before_weight, industry_cap.after_weight) == (
        Decimal("0.24"),
        Decimal("0.20"),
    )
    assert all(
        contribution.applied_portfolio_weight <= contribution.proposed_portfolio_weight
        for line in result.lines
        for contribution in line.contributions
    )


def test_current_target_weights_and_one_way_turnover_include_cash_leg() -> None:
    result = _build()

    assert result.total_equity == Decimal(10000)
    assert result.current_gross_weight == Decimal("0.30")
    assert result.current_cash_weight == Decimal("0.70")
    assert result.etf_target_weight == Decimal("0.43")
    assert result.stock_target_weight == Decimal("0.20")
    assert result.target_gross_weight == Decimal("0.63")
    assert result.target_cash_weight == Decimal("0.37")
    assert result.gross_instrument_turnover == Decimal("0.33")
    assert result.one_way_turnover == Decimal("0.33")
    assert result.one_way_turnover == (
        result.gross_instrument_turnover
        + abs(result.target_cash_weight - result.current_cash_weight)
    ) / Decimal(2)
    assert sum((line.current_weight for line in result.lines), Decimal(0)) == Decimal("0.30")
    assert sum((line.target_weight for line in result.lines), Decimal(0)) == Decimal("0.63")


def test_current_holding_without_strategy_target_is_an_explicit_exit() -> None:
    account = _account(include_exit=True)
    result = _build(
        account=account,
        classifications=(*_default_classifications(), _classification(EXIT_STOCK)),
    )
    exit_line = next(line for line in result.lines if line.instrument_id == EXIT_STOCK)

    assert exit_line.current_market_value == Decimal(1000)
    assert exit_line.current_weight == Decimal("0.10")
    assert exit_line.proposed_target_weight == 0
    assert exit_line.target_weight == 0
    assert exit_line.weight_delta == Decimal("-0.10")
    assert exit_line.absolute_weight_delta == Decimal("0.10")
    assert not exit_line.contributions
    assert "reduce the existing position to cash" in exit_line.rationale
    assert result.current_gross_weight == Decimal("0.40")
    assert result.current_cash_weight == Decimal("0.60")
    assert result.gross_instrument_turnover == Decimal("0.43")
    assert result.one_way_turnover == Decimal("0.33")


def test_sleeve_and_classification_input_order_cannot_change_result_or_hashes() -> None:
    sleeves = _default_sleeves()
    classifications = _default_classifications()

    ordered = _build(sleeves=sleeves, classifications=classifications)
    reversed_inputs = _build(
        sleeves=reversed(sleeves),
        classifications=reversed(classifications),
    )

    assert reversed_inputs == ordered
    assert reversed_inputs.input_hash == ordered.input_hash
    assert reversed_inputs.result_hash == ordered.result_hash
    assert tuple(line.instrument_id for line in ordered.lines) == tuple(
        sorted(line.instrument_id for line in ordered.lines)
    )


def test_account_request_id_hash_time_and_data_version_mismatches_fail_closed() -> None:
    account = _account()
    request = _request(account)
    builder = TargetPortfolioBuilder(_config())
    sleeves = _default_sleeves()
    classifications = _default_classifications()

    with pytest.raises(PortfolioBuildInputError, match="snapshot id"):
        builder.build(
            request=replace(request, account_snapshot_id="wrong"),
            account=account,
            sleeves=sleeves,
            classifications=classifications,
        )
    with pytest.raises(PortfolioBuildInputError, match="snapshot hash"):
        builder.build(
            request=replace(request, account_snapshot_hash="f" * 64),
            account=account,
            sleeves=sleeves,
            classifications=classifications,
        )
    with pytest.raises(PortfolioBuildInputError, match="as_of"):
        builder.build(
            request=replace(request, as_of=AS_OF + timedelta(seconds=1)),
            account=account,
            sleeves=sleeves,
            classifications=classifications,
        )
    with pytest.raises(PortfolioBuildInputError, match="data versions"):
        builder.build(
            request=replace(request, data_version="future-v2"),
            account=account,
            sleeves=sleeves,
            classifications=classifications,
        )
    with pytest.raises(ValueError, match="content_hash"):
        replace(account, content_hash="f" * 64)


def test_sleeve_time_version_and_hash_mismatches_fail_closed() -> None:
    account = _account()
    request = _request(account)
    builder = TargetPortfolioBuilder(_config())
    stock = _sleeve(StrategySleeveKind.STOCK_ENHANCEMENT)

    with pytest.raises(PortfolioBuildInputError, match="sleeve as_of"):
        builder.build(
            request=request,
            account=account,
            sleeves=(
                _sleeve(
                    StrategySleeveKind.ETF_CORE,
                    as_of=AS_OF - timedelta(seconds=1),
                ),
                stock,
            ),
            classifications=_default_classifications(),
        )
    with pytest.raises(PortfolioBuildInputError, match="sleeve data version"):
        builder.build(
            request=request,
            account=account,
            sleeves=(
                _sleeve(StrategySleeveKind.ETF_CORE, data_version="future-v2"),
                stock,
            ),
            classifications=_default_classifications(),
        )
    with pytest.raises(ValueError, match="sleeve_hash"):
        replace(_sleeve(StrategySleeveKind.ETF_CORE), sleeve_hash="f" * 64)
    with pytest.raises(ValueError, match="sleeve_hash"):
        replace(
            _sleeve(StrategySleeveKind.ETF_CORE),
            source_result_hash="f" * 64,
        )


def test_industry_classifications_require_exact_pit_versioned_coverage() -> None:
    first, second = _default_classifications()

    with pytest.raises(PortfolioBuildInputError, match="exactly cover"):
        _build(classifications=(first,))
    with pytest.raises(PortfolioBuildInputError, match="exactly cover"):
        _build(classifications=(first, second, _classification("CN.SSE.600099")))
    with pytest.raises(PortfolioBuildInputError, match="unique"):
        _build(classifications=(first, first, second))
    with pytest.raises(PortfolioBuildInputError, match="session"):
        _build(
            classifications=(
                replace(first, session_date=AS_OF.date() - timedelta(days=1)),
                second,
            )
        )
    with pytest.raises(PortfolioBuildInputError, match="future"):
        _build(
            classifications=(
                replace(first, available_at=AS_OF + timedelta(seconds=1)),
                second,
            )
        )
    with pytest.raises(PortfolioBuildInputError, match="data version"):
        _build(classifications=(replace(first, data_version="future-v2"), second))
    with pytest.raises(PortfolioBuildInputError, match="versions cannot be mixed"):
        _build(
            classifications=(
                first,
                _classification(STOCK_B, classification_version="sw-v2"),
            )
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(first, source_hash="not-a-hash")


def test_tightening_constraints_never_increases_any_target_or_total_risk() -> None:
    loose = _build(
        config=_config(
            maximum_instrument_weight="0.40",
            maximum_stock_industry_weight="0.30",
        )
    )
    tight = _build()
    loose_by_id = {line.instrument_id: line for line in loose.lines}

    assert loose.target_gross_weight == Decimal("0.72")
    assert tight.target_gross_weight == Decimal("0.63")
    assert tight.target_cash_weight >= loose.target_cash_weight
    assert all(
        line.target_weight <= loose_by_id[line.instrument_id].target_weight for line in tight.lines
    )
    assert max(line.target_weight for line in tight.lines) <= Decimal("0.25")
    assert sum(
        (
            line.target_weight
            for line in tight.lines
            if line.instrument_type is TradableInstrumentType.STOCK and line.industry_id == INDUSTRY
        ),
        Decimal(0),
    ) == Decimal("0.20")


def test_result_and_nested_economic_content_hash_tampering_is_rejected() -> None:
    result = _build()

    with pytest.raises(ValueError, match="result_hash"):
        replace(result, result_hash="f" * 64)
    changed_line = replace(result.lines[0], rationale="tampered audit rationale")
    with pytest.raises(ValueError, match="result_hash"):
        replace(result, lines=(changed_line, *result.lines[1:]))
    with pytest.raises(ValueError, match="input_hash"):
        replace(result, input_hash="f" * 64)


def test_root_contract_recomputes_config_constraints_and_hash_chains() -> None:
    result = _build()
    request = _request(_account())

    assert result.account_snapshot_hash == request.account_snapshot_hash
    assert "account_snapshot_hash" not in signature(TargetPortfolio.build).parameters
    assert result.etf_core_budget == Decimal("0.60")
    assert result.stock_enhancement_budget == Decimal("0.30")
    assert result.minimum_cash_weight == Decimal("0.10")
    assert set(result.classification_hashes) == {
        line.industry_classification_hash
        for line in result.lines
        if line.instrument_type is TradableInstrumentType.STOCK
    }
    assert all(
        contribution.sleeve_hash in result.sleeve_hashes
        for line in result.lines
        for contribution in line.contributions
    )

    first_line = result.lines[0]
    bad_market_value = replace(first_line, target_market_value=Decimal(999))
    with pytest.raises(ValueError, match="target market values"):
        replace(result, lines=(bad_market_value, *result.lines[1:]))

    first_contribution = first_line.contributions[0]
    unbound_contribution = replace(first_contribution, sleeve_hash="f" * 64)
    unbound_line = replace(first_line, contributions=(unbound_contribution,))
    with pytest.raises(ValueError, match="declared sleeve hash"):
        replace(result, lines=(unbound_line, *result.lines[1:]))

    tighter_cap = Decimal("0.10")
    tighter_config_hash = stable_hash(
        {
            "version": result.config_version,
            "etf_core_budget": result.etf_core_budget,
            "stock_enhancement_budget": result.stock_enhancement_budget,
            "minimum_cash_weight": result.minimum_cash_weight,
            "maximum_instrument_weight": tighter_cap,
            "maximum_stock_industry_weight": result.maximum_stock_industry_weight,
        }
    )
    with pytest.raises(ValueError, match="constraint projection"):
        replace(
            result,
            maximum_instrument_weight=tighter_cap,
            config_hash=tighter_config_hash,
        )


def test_contracts_reject_runtime_enum_and_lossy_decimal_injection() -> None:
    result = _build()

    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(result.lines[0], instrument_type="BOND")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PortfolioAdjustmentCode"):
        replace(result.adjustments[0], code="BOGUS")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="StrategySleeveKind"):
        StrategySleeve.build(
            sleeve_id="bad-kind",
            kind="ETF_CORE",  # type: ignore[arg-type]
            as_of=AS_OF,
            data_version=DATA_VERSION,
            strategy_name="bad-kind",
            strategy_version="v1",
            strategy_config_hash="a" * 64,
            source_result_hash="b" * 64,
            targets=(),
        )
    with pytest.raises(ValueError, match="exact Decimal"):
        PortfolioBuilderConfig(etf_core_budget=0.60)  # type: ignore[arg-type]


def test_target_portfolio_has_no_executable_order_contract() -> None:
    result = _build()
    portfolio_fields = {item.name for item in fields(result)}
    line_fields = {item.name for item in fields(result.lines[0])}

    assert not hasattr(result, "orders")
    assert not hasattr(result, "order_type")
    assert not hasattr(result, "side")
    assert not hasattr(result, "quantity")
    assert not any("order" in name for name in portfolio_fields | line_fields)
    assert "target_quantity" not in line_fields


def test_strategy_decision_adapters_preserve_complete_targets_and_reject_no_rebalance() -> None:
    etf_strategy = ETFRotationStrategy(etf_fixtures._config())
    etf_decision = etf_strategy.decide(
        request=etf_fixtures._request(),
        universe_revisions=etf_fixtures._universe(),
        prices=etf_fixtures._prices(),
        regime=etf_fixtures._regime(),
    )
    etf_sleeve = sleeve_from_etf_decision(etf_decision)

    assert etf_sleeve.kind is StrategySleeveKind.ETF_CORE
    assert etf_sleeve.source_result_hash == etf_decision.result_hash
    assert {item.instrument_id: item.local_target_weight for item in etf_sleeve.targets} == {
        item.instrument_id: item.target_weight for item in etf_decision.targets
    }

    stock_decision = stock_fixtures._decide()
    stock_sleeve = sleeve_from_mainline_decision(stock_decision)
    assert stock_sleeve.kind is StrategySleeveKind.STOCK_ENHANCEMENT
    assert stock_sleeve.source_result_hash == stock_decision.result_hash
    assert {item.instrument_id: item.local_target_weight for item in stock_sleeve.targets} == {
        item.instrument_id: item.target_weight for item in stock_decision.targets
    }

    no_rebalance = etf_strategy.decide(
        request=etf_fixtures._request(session_index=1),
        universe_revisions=etf_fixtures._universe(),
        prices=(),
        regime=etf_fixtures._regime(),
    )
    assert no_rebalance.status is ETFRotationDecisionStatus.NO_REBALANCE
    with pytest.raises(PortfolioBuildInputError, match="NO_REBALANCE"):
        sleeve_from_etf_decision(no_rebalance)
