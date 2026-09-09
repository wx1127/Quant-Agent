"""Real-chain tests for atomic, credential-free paper execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_agent.backtest import FeeRule, FeeRuleBook, Side, TradableInstrumentType
from quant_agent.backtest.cn_market import (
    CNMarketRule,
    CNSlippageModel,
    CNSlippageModelBook,
    MarketSessionState,
)
from quant_agent.config import RuntimeMode
from quant_agent.execution.order_drafts import (
    OrderDraftBatch,
    OrderDraftGeneratorConfig,
    OrderDraftInputError,
    OrderDraftLine,
)
from quant_agent.execution.paper import (
    PaperAccountState,
    PaperExecutionConfig,
    PaperExecutionEngine,
    PaperExecutionInputError,
    PaperExecutionRequest,
    PaperNoFillReason,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionLot,
)
from quant_agent.features.tradeability import MarketTradeState
from quant_agent.portfolio import AccountSnapshot, CashSnapshot, PortfolioPositionSnapshot
from quant_agent.regime.contracts import stable_hash

from . import test_order_draft_generator as draft_fixtures

NEXT_TRADING_DAY = date(2026, 8, 31)
SUBMITTED_AT = draft_fixtures.DRAFT_AS_OF + timedelta(minutes=1)

BUY_ONLY_TARGETS = {
    draft_fixtures.BUY_A: Decimal("0.32"),
    draft_fixtures.ODD_LOT: Decimal("0.01625"),
    draft_fixtures.PARTIAL_SELL: Decimal("0.3125"),
}
ODD_SELL_ONLY_TARGETS = {
    draft_fixtures.ODD_LOT: Decimal(0),
    draft_fixtures.PARTIAL_SELL: Decimal("0.3125"),
}


@dataclass(frozen=True, slots=True)
class _StateChange:
    available_quantity: Decimal
    market_state: MarketTradeState = MarketTradeState.NORMAL
    buy_allowed: bool = True
    sell_allowed: bool = True
    data_version: str | None = None


@dataclass(frozen=True, slots=True)
class _ExecutionFixture:
    snapshot: AccountSnapshot
    draft: OrderDraftBatch
    account: PaperAccountState
    rule: CNMarketRule
    fee_rule_book: FeeRuleBook
    slippage_model: CNSlippageModel
    engine: PaperExecutionEngine
    request: PaperExecutionRequest
    draft_states: tuple[MarketSessionState, ...]


def _execution_state(
    source: MarketSessionState,
    change: _StateChange,
) -> MarketSessionState:
    observed_at = draft_fixtures.DRAFT_AS_OF + timedelta(seconds=20)
    available_at = observed_at + timedelta(seconds=10)
    return MarketSessionState(
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        trading_day=source.trading_day,
        observed_at=observed_at,
        available_at=available_at,
        revision=f"{source.revision}-execution",
        data_version=change.data_version or source.data_version,
        state=change.market_state,
        buy_allowed=change.buy_allowed,
        sell_allowed=change.sell_allowed,
        reference_price=source.reference_price,
        lower_limit_price=source.lower_limit_price,
        upper_limit_price=source.upper_limit_price,
        available_quantity=change.available_quantity,
    )


def _market_rule(states: tuple[MarketSessionState, ...]) -> CNMarketRule:
    return replace(
        draft_fixtures._market_rule(states),
        trading_calendar=(draft_fixtures.TRADING_DAY, NEXT_TRADING_DAY),
    )


def _engine(
    *,
    rule: CNMarketRule,
    fee_rule_book: FeeRuleBook,
    slippage_model: CNSlippageModel,
    config: PaperExecutionConfig | None = None,
) -> PaperExecutionEngine:
    return PaperExecutionEngine(
        market_rules=(rule,),
        fee_rule_book=fee_rule_book,
        slippage_model_book=CNSlippageModelBook((slippage_model,)),
        config=config,
    )


def _request(
    *,
    account: PaperAccountState,
    draft: OrderDraftBatch,
    config: PaperExecutionConfig,
    submitted_at: datetime = SUBMITTED_AT,
    suffix: str = "base",
) -> PaperExecutionRequest:
    return PaperExecutionRequest.build(
        request_id=f"paper-request-{suffix}",
        idempotency_key=f"paper-idempotency-{suffix}",
        account_id=account.account_id,
        expected_account_state_hash=account.state_hash,
        expected_account_snapshot_hash=account.source_snapshot_hash,
        batch_hash=draft.batch_hash,
        submitted_at=submitted_at,
        config=config,
    )


def _execution_fixture(
    *,
    snapshot: AccountSnapshot | None = None,
    local_targets: dict[str, Decimal] | None = None,
    state_changes: dict[str, _StateChange] | None = None,
    config: PaperExecutionConfig | None = None,
) -> _ExecutionFixture:
    source = snapshot or draft_fixtures._account()
    chain = draft_fixtures._chain(account=source, local_targets=local_targets)
    draft_states = draft_fixtures._market_states(chain.proposal)
    changes = state_changes or {}
    execution_states = tuple(
        _execution_state(state, changes[state.instrument_id])
        for state in draft_states
        if state.instrument_id in changes
    )
    rule = _market_rule((*draft_states, *execution_states))
    fee_rule_book = draft_fixtures._fee_rule_book()
    slippage_model = draft_fixtures._slippage_model()
    draft = draft_fixtures._generator(
        market_rules=(rule,),
        fee_rule_book=fee_rule_book,
        slippage_model=slippage_model,
    ).generate(
        account_snapshot=chain.account,
        target_portfolio=chain.proposal,
        risk_result=chain.risk_result,
        market_states=draft_states,
        draft_as_of=draft_fixtures.DRAFT_AS_OF,
    )
    execution_config = config or PaperExecutionConfig(version="paper-engine-test-v1")
    engine = _engine(
        rule=rule,
        fee_rule_book=fee_rule_book,
        slippage_model=slippage_model,
        config=execution_config,
    )
    account = engine.open_account(snapshot=source)
    request = _request(
        account=account,
        draft=draft,
        config=execution_config,
    )
    return _ExecutionFixture(
        snapshot=source,
        draft=draft,
        account=account,
        rule=rule,
        fee_rule_book=fee_rule_book,
        slippage_model=slippage_model,
        engine=engine,
        request=request,
        draft_states=draft_states,
    )


def _segmented_snapshot() -> AccountSnapshot:
    source = draft_fixtures._account()
    position = PortfolioPositionSnapshot.build(
        instrument_id=draft_fixtures.ODD_LOT,
        instrument_type=source.positions[0].instrument_type,
        available_quantity=Decimal(70),
        frozen_quantity=Decimal(10),
        unsettled_quantity=Decimal(20),
        average_cost=Decimal(10),
        valuation_price=Decimal(10),
        price_observed_at=source.valuation_at,
        price_available_at=source.valuation_at,
        position_as_of=source.as_of,
        price_data_version=source.data_version,
        price_source_hash=stable_hash({"position": "segmented"}),
        mark_policy_hash=stable_hash({"policy": "position-mark-v1"}),
    )
    return AccountSnapshot.build(
        snapshot_id="account-segmented-inventory",
        account_id=source.account_id,
        runtime_mode=RuntimeMode.PAPER,
        as_of=source.as_of,
        valuation_at=source.valuation_at,
        data_version=source.data_version,
        currency=source.currency,
        cash=source.cash,
        positions=(position,),
        previous_snapshot_hash=source.previous_snapshot_hash,
        source_event_log_hash=source.source_event_log_hash,
    )


def _opening_lots(snapshot: AccountSnapshot) -> tuple[PaperPositionLot, ...]:
    instrument = snapshot.positions[0]
    return (
        PaperPositionLot.build(
            lot_id="opening-available",
            instrument_id=instrument.instrument_id,
            instrument_type=instrument.instrument_type,
            quantity=Decimal(70),
            cost_basis=Decimal(700),
            acquired_on=date.min,
            sellable_on=date.min,
            source_id=snapshot.snapshot_id,
        ),
        PaperPositionLot.build(
            lot_id="opening-frozen",
            instrument_id=instrument.instrument_id,
            instrument_type=instrument.instrument_type,
            quantity=Decimal(10),
            cost_basis=Decimal(100),
            acquired_on=date.min,
            sellable_on=date.min,
            externally_frozen=Decimal(10),
            source_id=snapshot.snapshot_id,
        ),
        PaperPositionLot.build(
            lot_id="opening-unsettled",
            instrument_id=instrument.instrument_id,
            instrument_type=instrument.instrument_type,
            quantity=Decimal(20),
            cost_basis=Decimal(200),
            acquired_on=draft_fixtures.TRADING_DAY,
            sellable_on=NEXT_TRADING_DAY,
            source_id=snapshot.snapshot_id,
        ),
    )


def _refresh_snapshot(
    account: PaperAccountState,
    *,
    as_of: datetime,
    snapshot_id: str = "paper-refresh-snapshot-1",
    previous_snapshot_hash: str | None = None,
    source_event_log_hash: str | None = None,
    cash_delta: Decimal = Decimal(0),
) -> AccountSnapshot:
    assert all(as_of >= order.expires_at for order in account.orders)
    projected_positions = tuple(
        PaperPosition.build(
            as_of=as_of,
            instrument_id=position.instrument_id,
            instrument_type=position.instrument_type,
            lots=position.lots,
        )
        for position in account.positions
    )
    snapshot_positions = tuple(
        PortfolioPositionSnapshot.build(
            instrument_id=position.instrument_id,
            instrument_type=position.instrument_type,
            available_quantity=position.available_quantity,
            frozen_quantity=position.frozen_quantity,
            unsettled_quantity=position.unsettled_quantity,
            average_cost=position.average_cost,
            valuation_price=Decimal(10),
            price_observed_at=as_of,
            price_available_at=as_of,
            position_as_of=as_of,
            price_data_version="cn-market-20260831-v1",
            price_source_hash=stable_hash({"paper-refresh-position": position.instrument_id}),
            mark_policy_hash=stable_hash({"paper-refresh-mark": "v1"}),
        )
        for position in projected_positions
    )
    cash = CashSnapshot(
        as_of=as_of,
        currency=account.currency,
        total_cash=account.total_cash + cash_delta,
        available_cash=account.total_cash - account.external_frozen_cash + cash_delta,
        frozen_cash=account.external_frozen_cash,
    )
    return AccountSnapshot.build(
        snapshot_id=snapshot_id,
        account_id=account.account_id,
        runtime_mode=RuntimeMode.PAPER,
        as_of=as_of,
        valuation_at=as_of,
        data_version="cn-market-20260831-v1",
        currency=account.currency,
        cash=cash,
        positions=snapshot_positions,
        previous_snapshot_hash=previous_snapshot_hash or account.source_snapshot_hash,
        source_event_log_hash=source_event_log_hash or account.event_log_hash,
    )


def _rebuild_account(
    account: PaperAccountState,
    *,
    event_log_hash: str,
    external_frozen_cash: Decimal | None = None,
) -> PaperAccountState:
    return PaperAccountState.build(
        account_id=account.account_id,
        as_of=account.as_of,
        data_version=account.data_version,
        currency=account.currency,
        source_snapshot_id=account.source_snapshot_id,
        source_snapshot_hash=account.source_snapshot_hash,
        source_snapshot_as_of=account.source_snapshot_as_of,
        total_cash=account.total_cash,
        external_frozen_cash=(
            account.external_frozen_cash if external_frozen_cash is None else external_frozen_cash
        ),
        positions=account.positions,
        orders=account.orders,
        processed_batch_hashes=account.processed_batch_hashes,
        previous_state_hash=account.previous_state_hash,
        event_log_hash=event_log_hash,
    )


def _forged_sell_line(
    fixture: _ExecutionFixture,
    *,
    current_quantity: Decimal,
    target_quantity: Decimal,
    quantity: Decimal,
    is_full_liquidation: bool,
) -> OrderDraftLine:
    source = next(
        line for line in fixture.draft.lines if line.instrument_id == draft_fixtures.PARTIAL_SELL
    )
    state = next(
        item for item in fixture.draft_states if item.instrument_id == source.instrument_id
    )
    fee_rule = fixture.fee_rule_book.select(
        instrument_type=source.instrument_type,
        side=Side.SELL,
        trading_day=fixture.draft.trading_day,
        version=source.fee_rule_version,
    )
    fees = fee_rule.assess(quantity * source.estimated_execution_price)
    return OrderDraftLine.build(
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        side=Side.SELL,
        current_quantity=current_quantity,
        target_quantity=target_quantity,
        quantity=quantity,
        is_full_liquidation=is_full_liquidation,
        reference_price=source.reference_price,
        estimated_execution_price=source.estimated_execution_price,
        price_observed_at=source.price_observed_at,
        price_available_at=source.price_available_at,
        data_version=source.data_version,
        state_revision=source.state_revision,
        state_hash=source.state_hash,
        participation_rate=quantity / state.available_quantity,
        commission=fees.commission,
        stamp_duty=fees.stamp_duty,
        transfer_fee=fees.transfer_fee,
        other_fee=fees.other_fee,
        market_rule_version=source.market_rule_version,
        market_rule_hash=source.market_rule_hash,
        fee_rule_version=source.fee_rule_version,
        fee_rule_hash=source.fee_rule_hash,
        slippage_model_version=source.slippage_model_version,
        slippage_model_hash=source.slippage_model_hash,
        rationale="hash-consistent adversarial paper draft",
    )


def _single_line_draft(
    fixture: _ExecutionFixture,
    *,
    line: OrderDraftLine,
) -> OrderDraftBatch:
    source = fixture.draft
    config = OrderDraftGeneratorConfig(
        version=source.generator_config_version,
        validity_seconds=source.validity_seconds,
        funding_policy=source.funding_policy,
    )
    return OrderDraftBatch.build(
        decision_id=source.decision_id,
        draft_as_of=source.draft_as_of,
        data_version=source.data_version,
        currency=source.currency,
        account_snapshot_id=source.account_snapshot_id,
        account_snapshot_hash=source.account_snapshot_hash,
        account_snapshot_as_of=source.account_snapshot_as_of,
        runtime_mode=source.runtime_mode,
        portfolio_proposal_hash=source.portfolio_proposal_hash,
        risk_request_hash=source.risk_request_hash,
        risk_result_hash=source.risk_result_hash,
        risk_status=source.risk_status,
        risk_checked_at=source.risk_checked_at,
        risk_engine_version=source.risk_engine_version,
        risk_policy_version=source.risk_policy_version,
        risk_policy_hash=source.risk_policy_hash,
        config=config,
        available_cash=source.available_cash,
        lines=(line,),
    )


def _account_with_retyped_position(
    account: PaperAccountState,
    *,
    instrument_id: str,
    instrument_type: TradableInstrumentType,
) -> PaperAccountState:
    positions: list[PaperPosition] = []
    for position in account.positions:
        if position.instrument_id != instrument_id:
            positions.append(position)
            continue
        lots = tuple(
            PaperPositionLot.build(
                lot_id=lot.lot_id,
                instrument_id=lot.instrument_id,
                instrument_type=instrument_type,
                quantity=lot.quantity,
                cost_basis=lot.cost_basis,
                acquired_on=lot.acquired_on,
                sellable_on=lot.sellable_on,
                externally_frozen=lot.externally_frozen,
                source_id=lot.source_id,
            )
            for lot in position.lots
        )
        positions.append(
            PaperPosition.build(
                as_of=position.as_of,
                instrument_id=position.instrument_id,
                instrument_type=instrument_type,
                lots=lots,
                reserved_sell_quantity=position.reserved_sell_quantity,
            )
        )
    return PaperAccountState.build(
        account_id=account.account_id,
        as_of=account.as_of,
        data_version=account.data_version,
        currency=account.currency,
        source_snapshot_id=account.source_snapshot_id,
        source_snapshot_hash=account.source_snapshot_hash,
        source_snapshot_as_of=account.source_snapshot_as_of,
        total_cash=account.total_cash,
        external_frozen_cash=account.external_frozen_cash,
        positions=tuple(positions),
        orders=account.orders,
        processed_batch_hashes=account.processed_batch_hashes,
        previous_state_hash=account.previous_state_hash,
        event_log_hash=stable_hash({"event-log": "retyped-paper-position"}),
    )


def test_open_account_derives_deterministic_sellable_lots_for_fully_available_inventory() -> None:
    fixture = _execution_fixture()

    reopened = fixture.engine.open_account(snapshot=fixture.snapshot)

    assert reopened == fixture.account
    assert reopened.available_cash == fixture.snapshot.cash.available_cash
    assert reopened.frozen_cash == fixture.snapshot.cash.frozen_cash
    for paper_position, source_position in zip(
        reopened.positions,
        fixture.snapshot.positions,
        strict=True,
    ):
        assert len(paper_position.lots) == 1
        lot = paper_position.lots[0]
        assert lot.quantity == source_position.total_quantity
        assert lot.cost_basis == source_position.cost_basis
        assert lot.sellable_on == date.min
        assert paper_position.available_quantity == source_position.available_quantity
        assert paper_position.frozen_quantity == 0
        assert paper_position.unsettled_quantity == 0


def test_open_account_requires_and_reconciles_explicit_frozen_and_unsettled_lots() -> None:
    fixture = _execution_fixture()
    snapshot = _segmented_snapshot()

    with pytest.raises(PaperExecutionInputError, match="requires explicit sellable lots"):
        fixture.engine.open_account(snapshot=snapshot)

    account = fixture.engine.open_account(
        snapshot=snapshot,
        opening_lots=_opening_lots(snapshot),
    )
    position = account.positions[0]
    assert (
        position.total_quantity,
        position.available_quantity,
        position.frozen_quantity,
        position.unsettled_quantity,
        position.cost_basis,
    ) == (
        Decimal(100),
        Decimal(70),
        Decimal(10),
        Decimal(20),
        Decimal(1000),
    )

    bad_lots = (*_opening_lots(snapshot)[:-1],)
    with pytest.raises(PaperExecutionInputError, match="do not reconcile"):
        fixture.engine.open_account(snapshot=snapshot, opening_lots=bad_lots)


def test_mixed_batch_books_real_slippage_fees_cash_positions_and_buy_t_plus_one() -> None:
    fixture = _execution_fixture()

    receipt = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )

    assert len(receipt.orders) == len(fixture.draft.lines) == 4
    assert len(receipt.attempts) == len(receipt.fills) == 4
    assert all(order.status is PaperOrderStatus.FILLED for order in receipt.orders)
    assert all(order.cash_reserved == 0 for order in receipt.orders)
    assert all(order.sell_reserved_quantity == 0 for order in receipt.orders)

    lines_by_id = {line.instrument_id: line for line in fixture.draft.lines}
    states_by_id = {state.instrument_id: state for state in fixture.draft_states}
    for fill in receipt.fills:
        line = lines_by_id[fill.instrument_id]
        state = states_by_id[fill.instrument_id]
        expected_price = fixture.slippage_model.execution_price(
            side=line.side,
            reference_price=state.reference_price,
            participation_rate=line.quantity / state.available_quantity,
            price_tick=fixture.rule.price_tick,
        )
        expected_fee = fixture.fee_rule_book.select(
            instrument_type=line.instrument_type,
            side=line.side,
            trading_day=draft_fixtures.TRADING_DAY,
        ).assess(line.quantity * expected_price)
        assert fill.price == expected_price
        assert fill.quantity == line.quantity
        assert fill.commission == expected_fee.commission
        assert fill.stamp_duty == expected_fee.stamp_duty
        assert fill.transfer_fee == expected_fee.transfer_fee
        assert fill.other_fee == expected_fee.other_fee
        assert fill.total_fee == expected_fee.total_amount

    assert receipt.account_after.total_cash == fixture.account.total_cash + sum(
        (fill.cash_change for fill in receipt.fills),
        Decimal(0),
    )
    assert receipt.account_after.available_cash == receipt.account_after.total_cash
    before_quantities = {
        position.instrument_id: position.total_quantity for position in fixture.account.positions
    }
    after_quantities = {
        position.instrument_id: position.total_quantity
        for position in receipt.account_after.positions
    }
    for instrument_id, line in lines_by_id.items():
        signed_fill = next(fill for fill in receipt.fills if fill.instrument_id == instrument_id)
        delta = signed_fill.quantity if line.side is Side.BUY else -signed_fill.quantity
        assert after_quantities.get(instrument_id, Decimal(0)) == (
            before_quantities.get(instrument_id, Decimal(0)) + delta
        )

    for instrument_id in (draft_fixtures.BUY_A, draft_fixtures.BUY_B):
        position = next(
            item for item in receipt.account_after.positions if item.instrument_id == instrument_id
        )
        assert position.available_quantity == 0
        assert position.unsettled_quantity == position.total_quantity
        assert {lot.sellable_on for lot in position.lots} == {NEXT_TRADING_DAY}
    assert draft_fixtures.ODD_LOT not in after_quantities


def test_later_execution_capacity_can_partially_fill_and_retain_exact_holds() -> None:
    fixture = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(available_quantity=Decimal(1000)),
        },
    )

    receipt = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )

    assert len(receipt.orders) == len(receipt.attempts) == len(receipt.fills) == 1
    order = receipt.orders[0]
    attempt = receipt.attempts[0]
    fill = receipt.fills[0]
    assert order.status is PaperOrderStatus.PARTIALLY_FILLED
    assert (order.quantity, order.filled_quantity, order.remaining_quantity) == (
        Decimal(200),
        Decimal(100),
        Decimal(100),
    )
    assert attempt.state_revision.endswith("-execution")
    assert attempt.participation_rate == Decimal("0.1")
    assert fill.quantity == Decimal(100)
    assert order.cash_reserved == fixture.draft.lines[0].reserved_cash + fill.cash_change
    assert receipt.account_after.frozen_cash == order.cash_reserved
    assert receipt.account_after.available_cash == (
        receipt.account_after.total_cash - order.cash_reserved
    )
    position = next(
        item
        for item in receipt.account_after.positions
        if item.instrument_id == draft_fixtures.BUY_A
    )
    assert position.unsettled_quantity == Decimal(100)


def test_refresh_expires_holds_settles_t_plus_one_and_allows_a_second_batch() -> None:
    fixture = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(available_quantity=Decimal(1000)),
        },
    )
    first = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )
    assert first.orders[0].status is PaperOrderStatus.PARTIALLY_FILLED
    assert first.account_after.frozen_cash > first.account_after.external_frozen_cash

    refresh_at = first.processed_at + timedelta(days=3)
    snapshot = _refresh_snapshot(first.account_after, as_of=refresh_at)
    refreshed = fixture.engine.refresh_account(
        account=first.account_after,
        snapshot=snapshot,
    )

    assert refreshed.previous_state_hash == first.account_after.state_hash
    assert refreshed.source_snapshot_hash == snapshot.content_hash
    assert refreshed.processed_batch_hashes == first.account_after.processed_batch_hashes
    assert refreshed.orders[0].status is PaperOrderStatus.EXPIRED
    assert refreshed.orders[0].cash_reserved == 0
    assert refreshed.frozen_cash == refreshed.external_frozen_cash
    bought = next(
        position
        for position in refreshed.positions
        if position.instrument_id == draft_fixtures.BUY_A
    )
    assert bought.unsettled_quantity == 0
    assert bought.available_quantity == bought.total_quantity

    chain = draft_fixtures._chain(
        account=snapshot,
        local_targets=BUY_ONLY_TARGETS,
        decision_id="portfolio-decision-second-batch",
    )
    next_day_states = tuple(
        replace(
            state,
            trading_day=NEXT_TRADING_DAY,
            observed_at=refresh_at,
            available_at=refresh_at,
            revision=f"{state.revision}-next-day",
            data_version=snapshot.data_version,
        )
        for state in draft_fixtures._market_states(chain.proposal)
    )
    next_rule = replace(
        _market_rule(next_day_states),
        trading_calendar=(
            draft_fixtures.TRADING_DAY,
            NEXT_TRADING_DAY,
            NEXT_TRADING_DAY + timedelta(days=1),
        ),
    )
    second_draft_at = refresh_at + timedelta(minutes=2)
    second_draft = draft_fixtures._generator(
        market_rules=(next_rule,),
        fee_rule_book=fixture.fee_rule_book,
        slippage_model=fixture.slippage_model,
    ).generate(
        account_snapshot=snapshot,
        target_portfolio=chain.proposal,
        risk_result=chain.risk_result,
        market_states=next_day_states,
        draft_as_of=second_draft_at,
    )
    second_engine = _engine(
        rule=next_rule,
        fee_rule_book=fixture.fee_rule_book,
        slippage_model=fixture.slippage_model,
        config=fixture.engine.config,
    )
    second_request = _request(
        account=refreshed,
        draft=second_draft,
        config=second_engine.config,
        submitted_at=second_draft_at + timedelta(minutes=1),
        suffix="second-batch",
    )
    second = second_engine.execute(
        request=second_request,
        draft=second_draft,
        account=refreshed,
    )

    assert set(second.account_after.processed_batch_hashes) == {
        fixture.draft.batch_hash,
        second_draft.batch_hash,
    }
    assert set(order.order_id for order in first.account_after.orders).issubset(
        {order.order_id for order in second.account_after.orders}
    )
    assert all(order.batch_hash == second_draft.batch_hash for order in second.orders)


def test_refresh_fails_closed_on_time_lineage_event_log_and_cash_mismatches() -> None:
    fixture = _execution_fixture()
    first = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )
    refresh_at = first.processed_at + timedelta(days=3)

    with pytest.raises(PaperExecutionInputError, match="snapshot must be newer"):
        fixture.engine.refresh_account(
            account=first.account_after,
            snapshot=fixture.snapshot,
        )
    wrong_lineage = _refresh_snapshot(
        first.account_after,
        as_of=refresh_at,
        snapshot_id="paper-refresh-wrong-lineage",
        previous_snapshot_hash=stable_hash({"wrong": "snapshot-lineage"}),
    )
    with pytest.raises(PaperExecutionInputError, match="lineage is stale"):
        fixture.engine.refresh_account(account=first.account_after, snapshot=wrong_lineage)
    wrong_event_log = _refresh_snapshot(
        first.account_after,
        as_of=refresh_at,
        snapshot_id="paper-refresh-wrong-event-log",
        source_event_log_hash=stable_hash({"wrong": "event-log"}),
    )
    with pytest.raises(PaperExecutionInputError, match="event-log boundary is stale"):
        fixture.engine.refresh_account(account=first.account_after, snapshot=wrong_event_log)
    wrong_cash = _refresh_snapshot(
        first.account_after,
        as_of=refresh_at,
        snapshot_id="paper-refresh-wrong-cash",
        cash_delta=Decimal(1),
    )
    with pytest.raises(PaperExecutionInputError, match="cash does not reconcile"):
        fixture.engine.refresh_account(account=first.account_after, snapshot=wrong_cash)


@pytest.mark.parametrize(
    ("local_targets", "instrument_id", "change", "expected_reason", "expected_side"),
    (
        (
            BUY_ONLY_TARGETS,
            draft_fixtures.BUY_A,
            _StateChange(available_quantity=Decimal(0)),
            PaperNoFillReason.ZERO_CAPACITY,
            Side.BUY,
        ),
        (
            ODD_SELL_ONLY_TARGETS,
            draft_fixtures.ODD_LOT,
            _StateChange(
                available_quantity=Decimal(10000),
                market_state=MarketTradeState.SUSPENDED,
                buy_allowed=False,
                sell_allowed=False,
            ),
            PaperNoFillReason.SUSPENDED,
            Side.SELL,
        ),
        (
            ODD_SELL_ONLY_TARGETS,
            draft_fixtures.ODD_LOT,
            _StateChange(
                available_quantity=Decimal(10000),
                sell_allowed=False,
            ),
            PaperNoFillReason.SIDE_BLOCKED,
            Side.SELL,
        ),
    ),
)
def test_zero_capacity_suspension_and_side_block_remain_accepted_with_live_holds(
    local_targets: dict[str, Decimal],
    instrument_id: str,
    change: _StateChange,
    expected_reason: PaperNoFillReason,
    expected_side: Side,
) -> None:
    fixture = _execution_fixture(
        local_targets=local_targets,
        state_changes={instrument_id: change},
    )

    receipt = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )

    assert not receipt.fills
    assert len(receipt.orders) == len(receipt.attempts) == 1
    order = receipt.orders[0]
    attempt = receipt.attempts[0]
    assert order.side is expected_side
    assert order.status is PaperOrderStatus.ACCEPTED
    assert order.filled_quantity == 0
    assert attempt.status is PaperOrderStatus.ACCEPTED
    assert attempt.no_fill_reason is expected_reason
    assert receipt.account_after.total_cash == fixture.account.total_cash
    if expected_side is Side.BUY:
        assert order.cash_reserved == fixture.draft.lines[0].reserved_cash
        assert receipt.account_after.frozen_cash == order.cash_reserved
        assert not receipt.account_after.positions or all(
            position.instrument_id != instrument_id for position in receipt.account_after.positions
        )
    else:
        assert order.sell_reserved_quantity == order.quantity
        position = next(
            item for item in receipt.account_after.positions if item.instrument_id == instrument_id
        )
        assert position.available_quantity == 0
        assert position.frozen_quantity == order.quantity


def test_partial_fill_disabled_keeps_the_complete_buy_reservation_without_a_fill() -> None:
    config = PaperExecutionConfig(version="paper-no-partial-v1", allow_partial=False)
    fixture = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(available_quantity=Decimal(1000)),
        },
        config=config,
    )

    receipt = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )

    assert not receipt.fills
    assert receipt.orders[0].status is PaperOrderStatus.ACCEPTED
    assert receipt.attempts[0].no_fill_reason is PaperNoFillReason.PARTIAL_FILL_DISABLED
    assert receipt.orders[0].cash_reserved == fixture.draft.lines[0].reserved_cash
    assert receipt.account_after.total_cash == fixture.account.total_cash
    assert receipt.account_after.frozen_cash == fixture.draft.lines[0].reserved_cash


def test_full_odd_lot_liquidation_is_authorized_from_the_bound_draft_line() -> None:
    fixture = _execution_fixture(local_targets=ODD_SELL_ONLY_TARGETS)

    assert not fixture.rule.odd_lot_liquidation_order_ids
    assert fixture.draft.lines[0].quantity == Decimal(13)
    assert fixture.draft.lines[0].is_full_liquidation
    receipt = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )

    assert receipt.orders[0].status is PaperOrderStatus.FILLED
    assert receipt.fills[0].quantity == Decimal(13)
    assert all(
        position.instrument_id != draft_fixtures.ODD_LOT
        for position in receipt.account_after.positions
    )


def test_hash_consistent_draft_current_quantity_must_match_the_paper_position() -> None:
    fixture = _execution_fixture()
    line = _forged_sell_line(
        fixture,
        current_quantity=Decimal(200),
        target_quantity=Decimal(100),
        quantity=Decimal(100),
        is_full_liquidation=False,
    )
    draft = _single_line_draft(fixture, line=line)
    request = _request(
        account=fixture.account,
        draft=draft,
        config=fixture.engine.config,
        suffix="forged-current-quantity",
    )

    with pytest.raises(PaperExecutionInputError, match="current_quantity"):
        fixture.engine.execute(request=request, draft=draft, account=fixture.account)

    assert fixture.account.state_hash == request.expected_account_state_hash
    assert not fixture.account.orders
    assert not fixture.account.processed_batch_hashes


def test_hash_consistent_odd_lot_flag_cannot_fake_a_full_position_liquidation() -> None:
    fixture = _execution_fixture()
    line = _forged_sell_line(
        fixture,
        current_quantity=Decimal(13),
        target_quantity=Decimal(0),
        quantity=Decimal(13),
        is_full_liquidation=True,
    )
    draft = _single_line_draft(fixture, line=line)
    request = _request(
        account=fixture.account,
        draft=draft,
        config=fixture.engine.config,
        suffix="forged-full-liquidation",
    )

    with pytest.raises(PaperExecutionInputError, match="full-liquidation quantity"):
        fixture.engine.execute(request=request, draft=draft, account=fixture.account)

    position = next(
        item
        for item in fixture.account.positions
        if item.instrument_id == draft_fixtures.PARTIAL_SELL
    )
    assert position.total_quantity == Decimal(250)
    assert not fixture.account.orders
    assert not fixture.account.processed_batch_hashes


def test_draft_instrument_type_must_match_the_content_addressed_paper_position() -> None:
    fixture = _execution_fixture()
    changed_account = _account_with_retyped_position(
        fixture.account,
        instrument_id=draft_fixtures.PARTIAL_SELL,
        instrument_type=TradableInstrumentType.STOCK,
    )
    request = _request(
        account=changed_account,
        draft=fixture.draft,
        config=fixture.engine.config,
        suffix="position-type-mismatch",
    )

    with pytest.raises(PaperExecutionInputError, match="instrument type"):
        fixture.engine.execute(
            request=request,
            draft=fixture.draft,
            account=changed_account,
        )

    assert changed_account.state_hash == request.expected_account_state_hash
    assert not changed_account.orders
    assert not changed_account.processed_batch_hashes


def test_side_blocked_state_cannot_accept_an_invalid_sell_lot_quantity() -> None:
    fixture = _execution_fixture(
        state_changes={
            draft_fixtures.PARTIAL_SELL: _StateChange(
                available_quantity=Decimal(10000),
                sell_allowed=False,
            )
        }
    )
    line = _forged_sell_line(
        fixture,
        current_quantity=Decimal(250),
        target_quantity=Decimal(200),
        quantity=Decimal(50),
        is_full_liquidation=False,
    )
    draft = _single_line_draft(fixture, line=line)
    request = _request(
        account=fixture.account,
        draft=draft,
        config=fixture.engine.config,
        suffix="blocked-invalid-lot",
    )

    with pytest.raises(PaperExecutionInputError, match="integer multiple"):
        fixture.engine.execute(request=request, draft=draft, account=fixture.account)

    assert fixture.account.state_hash == request.expected_account_state_hash
    assert not fixture.account.orders
    assert not fixture.account.processed_batch_hashes


def test_live_assisted_snapshot_cannot_open_a_paper_account() -> None:
    fixture = _execution_fixture()
    live_snapshot = draft_fixtures._account(runtime_mode=RuntimeMode.LIVE_ASSISTED)

    with pytest.raises(PaperExecutionInputError, match="PAPER source snapshot"):
        fixture.engine.open_account(snapshot=live_snapshot)


def test_expired_stale_account_and_wrong_execution_data_version_fail_atomically() -> None:
    fixture = _execution_fixture()
    original_hash = fixture.account.state_hash
    expired = _request(
        account=fixture.account,
        draft=fixture.draft,
        config=fixture.engine.config,
        submitted_at=fixture.draft.expires_at,
        suffix="expired",
    )
    with pytest.raises(PaperExecutionInputError, match="expired"):
        fixture.engine.execute(request=expired, draft=fixture.draft, account=fixture.account)
    assert fixture.account.state_hash == original_hash

    stale_request = fixture.request
    changed_account = _rebuild_account(
        fixture.account,
        event_log_hash=stable_hash({"event-log": "concurrent-change"}),
    )
    with pytest.raises(PaperExecutionInputError, match="state changed"):
        fixture.engine.execute(
            request=stale_request,
            draft=fixture.draft,
            account=changed_account,
        )
    assert changed_account.state_hash != original_hash

    wrong_data = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(
                available_quantity=Decimal(10000),
                data_version="unexpected-execution-data-v2",
            ),
        },
    )
    with pytest.raises(PaperExecutionInputError, match="data version does not match"):
        wrong_data.engine.execute(
            request=wrong_data.request,
            draft=wrong_data.draft,
            account=wrong_data.account,
        )
    assert wrong_data.account.state_hash == wrong_data.request.expected_account_state_hash


def test_configured_market_state_age_fails_closed_before_ledger_change() -> None:
    config = PaperExecutionConfig(
        version="paper-tight-market-freshness-v1",
        max_market_state_age_seconds=60,
    )
    fixture = _execution_fixture(config=config)

    with pytest.raises(PaperExecutionInputError, match="market state is stale"):
        fixture.engine.execute(
            request=fixture.request,
            draft=fixture.draft,
            account=fixture.account,
        )
    assert fixture.account.state_hash == fixture.request.expected_account_state_hash
    assert not fixture.account.orders
    assert not fixture.account.processed_batch_hashes


@pytest.mark.parametrize("changed_component", ("rule", "fee", "slippage"))
def test_pinned_rule_fee_and_slippage_hash_mismatches_fail_before_any_ledger_change(
    changed_component: str,
) -> None:
    fixture = _execution_fixture(local_targets=BUY_ONLY_TARGETS)
    rule = fixture.rule
    fee_rule_book = fixture.fee_rule_book
    slippage_model = fixture.slippage_model
    expected_message = changed_component
    if changed_component == "rule":
        rule = replace(rule, max_participation_rate=Decimal("0.20"))
        expected_message = "market rule hash"
    elif changed_component == "fee":
        buy_rule = fee_rule_book.select(
            instrument_type=fixture.draft.lines[0].instrument_type,
            side=Side.BUY,
            trading_day=draft_fixtures.TRADING_DAY,
        )
        sell_rule = fee_rule_book.select(
            instrument_type=fixture.draft.lines[0].instrument_type,
            side=Side.SELL,
            trading_day=draft_fixtures.TRADING_DAY,
        )
        changed_buy_rule = FeeRule.model_validate(
            {
                **buy_rule.model_dump(mode="python"),
                "commission_rate": Decimal("0.002"),
            }
        )
        fee_rule_book = FeeRuleBook((changed_buy_rule, sell_rule))
        expected_message = "fee rule hash"
    else:
        slippage_model = replace(slippage_model, base_rate=Decimal("0.002"))
        expected_message = "slippage model hash"
    changed_engine = _engine(
        rule=rule,
        fee_rule_book=fee_rule_book,
        slippage_model=slippage_model,
        config=fixture.engine.config,
    )

    with pytest.raises(PaperExecutionInputError, match=expected_message):
        changed_engine.execute(
            request=fixture.request,
            draft=fixture.draft,
            account=fixture.account,
        )
    assert fixture.account.state_hash == fixture.request.expected_account_state_hash
    assert not fixture.account.orders
    assert not fixture.account.processed_batch_hashes


def test_current_cash_boundary_is_checked_before_same_batch_sell_proceeds() -> None:
    fixture = _execution_fixture()
    constrained = _rebuild_account(
        fixture.account,
        event_log_hash=stable_hash({"event-log": "cash-constrained"}),
        external_frozen_cash=Decimal(6000),
    )
    request = _request(
        account=constrained,
        draft=fixture.draft,
        config=fixture.engine.config,
        suffix="cash-constrained",
    )

    assert fixture.draft.estimated_sell_cash_proceeds > 0
    assert constrained.available_cash < fixture.draft.reserved_cash_required
    with pytest.raises(PaperExecutionInputError, match="draft available cash is stale"):
        fixture.engine.execute(
            request=request,
            draft=fixture.draft,
            account=constrained,
        )
    assert constrained.total_cash == fixture.account.total_cash
    assert not constrained.orders
    assert not constrained.processed_batch_hashes


def test_upstream_draft_contract_also_rejects_using_expected_sales_to_fund_buys() -> None:
    cash_constrained_snapshot = draft_fixtures._account(available_cash=Decimal(1000))
    chain = draft_fixtures._chain(account=cash_constrained_snapshot)
    states = draft_fixtures._market_states(chain.proposal)
    rule = _market_rule(states)

    with pytest.raises(OrderDraftInputError, match="reserved cash requirements exceed"):
        draft_fixtures._generator(market_rules=(rule,)).generate(
            account_snapshot=chain.account,
            target_portfolio=chain.proposal,
            risk_result=chain.risk_result,
            market_states=states,
            draft_as_of=draft_fixtures.DRAFT_AS_OF,
        )
