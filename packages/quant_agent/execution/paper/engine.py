"""Pure, credential-free paper matching and ledger transitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

from quant_agent.backtest import (
    FeeRule,
    FeeRuleBook,
    OrderEvent,
    OrderStatus,
    OrderType,
    Side,
    TradableInstrumentType,
)
from quant_agent.backtest.cn_market import (
    CNMarketRule,
    CNSlippageModel,
    CNSlippageModelBook,
    MarketSessionState,
    MatchStatus,
    NoFillReason,
)
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.execution.order_drafts import OrderDraftBatch, OrderDraftLine
from quant_agent.features.tradeability import MarketTradeState
from quant_agent.portfolio import AccountSnapshot
from quant_agent.regime.contracts import stable_hash

from .contracts import (
    PAPER_EXECUTION_ENGINE_VERSION,
    PaperAccountState,
    PaperExecutionConfig,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperFill,
    PaperMatchAttempt,
    PaperNoFillReason,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionLot,
)
from .transition import replay_account_refresh, replay_execution_transition

_ZERO = Decimal(0)


class _MarketRuleResolver:
    def __init__(self, rules: Iterable[CNMarketRule]) -> None:
        self._rules = tuple(rules)
        if not self._rules:
            raise PaperExecutionInputError("at least one CN market rule is required")
        if any(not isinstance(rule, CNMarketRule) for rule in self._rules):
            raise PaperExecutionInputError("market rules must contain only CNMarketRule values")
        boundaries: set[tuple[TradableInstrumentType, date]] = set()
        versions: set[tuple[TradableInstrumentType, str, str]] = set()
        for rule in self._rules:
            boundary = (rule.instrument_type, rule.effective_from)
            version = (rule.instrument_type, rule.rule_id, rule.version)
            if boundary in boundaries:
                raise PaperExecutionInputError(
                    "market rules are ambiguous at an instrument effective boundary"
                )
            if version in versions:
                raise PaperExecutionInputError("market rule identities and versions must be unique")
            boundaries.add(boundary)
            versions.add(version)

    def select(
        self,
        *,
        instrument_type: TradableInstrumentType,
        trading_day: date,
    ) -> CNMarketRule:
        candidates = tuple(
            rule
            for rule in self._rules
            if rule.instrument_type is instrument_type and rule.applies_on(trading_day)
        )
        if not candidates:
            raise PaperExecutionInputError(
                f"no effective {instrument_type.value} market rule on {trading_day.isoformat()}"
            )
        return max(candidates, key=lambda rule: rule.effective_from)


class PaperExecutionEngine:
    """Plan one atomic paper submission without IO, credentials, or broker authority."""

    def __init__(
        self,
        *,
        market_rules: Iterable[CNMarketRule],
        fee_rule_book: FeeRuleBook,
        slippage_model_book: CNSlippageModelBook,
        config: PaperExecutionConfig | None = None,
    ) -> None:
        if not isinstance(fee_rule_book, FeeRuleBook):
            raise PaperExecutionInputError("fee_rule_book must be a FeeRuleBook")
        if not isinstance(slippage_model_book, CNSlippageModelBook):
            raise PaperExecutionInputError("slippage_model_book must be a CNSlippageModelBook")
        if config is not None and not isinstance(config, PaperExecutionConfig):
            raise PaperExecutionInputError("config must be a PaperExecutionConfig")
        self._market_rules = _MarketRuleResolver(market_rules)
        self._fee_rule_book = fee_rule_book
        self._slippage_model_book = slippage_model_book
        self._config = config or PaperExecutionConfig()

    @property
    def config(self) -> PaperExecutionConfig:
        return self._config

    def open_account(
        self,
        *,
        snapshot: AccountSnapshot,
        opening_lots: tuple[PaperPositionLot, ...] = (),
    ) -> PaperAccountState:
        """Create a paper lot ledger from one exact PAPER account snapshot."""

        try:
            snapshot = AccountSnapshot.from_json(snapshot.to_json())
            return self._open_account(snapshot=snapshot, opening_lots=opening_lots)
        except PaperExecutionInputError:
            raise
        except (TypeError, ValueError) as error:
            raise PaperExecutionInputError(str(error)) from error

    def _open_account(
        self,
        *,
        snapshot: AccountSnapshot,
        opening_lots: tuple[PaperPositionLot, ...],
    ) -> PaperAccountState:
        if snapshot.runtime_mode.value != "PAPER":
            raise PaperExecutionInputError("paper accounts require a PAPER source snapshot")
        if any(not isinstance(lot, PaperPositionLot) for lot in opening_lots):
            raise PaperExecutionInputError("opening_lots must contain only PaperPositionLot values")
        supplied_by_instrument: dict[str, list[PaperPositionLot]] = {}
        for lot in opening_lots:
            supplied_by_instrument.setdefault(lot.instrument_id, []).append(replace(lot))
        snapshot_ids = {position.instrument_id for position in snapshot.positions}
        if set(supplied_by_instrument).difference(snapshot_ids):
            raise PaperExecutionInputError("opening lots contain an unknown account position")
        positions: list[PaperPosition] = []
        for source in snapshot.positions:
            supplied = tuple(supplied_by_instrument.get(source.instrument_id, ()))
            if not supplied:
                if source.frozen_quantity or source.unsettled_quantity:
                    raise PaperExecutionInputError(
                        f"{source.instrument_id} frozen or unsettled opening inventory "
                        "requires explicit sellable lots"
                    )
                opening_lot_hash = stable_hash(
                    {
                        "snapshot": snapshot.content_hash,
                        "instrument": source.instrument_id,
                    }
                )
                supplied = (
                    PaperPositionLot.build(
                        lot_id=f"paper-opening:{opening_lot_hash}",
                        instrument_id=source.instrument_id,
                        instrument_type=source.instrument_type,
                        quantity=source.total_quantity,
                        cost_basis=source.cost_basis,
                        acquired_on=date.min,
                        sellable_on=date.min,
                        source_id=snapshot.snapshot_id,
                    ),
                )
            position = PaperPosition.build(
                as_of=snapshot.as_of,
                instrument_id=source.instrument_id,
                instrument_type=source.instrument_type,
                lots=supplied,
            )
            actual = (
                position.total_quantity,
                position.available_quantity,
                position.frozen_quantity,
                position.unsettled_quantity,
                position.cost_basis,
            )
            expected = (
                source.total_quantity,
                source.available_quantity,
                source.frozen_quantity,
                source.unsettled_quantity,
                source.cost_basis,
            )
            if actual != expected:
                raise PaperExecutionInputError(
                    f"{source.instrument_id} opening lots do not reconcile to the snapshot"
                )
            positions.append(position)
        opening_event_log_hash = stable_hash(
            {
                "engine_version": PAPER_EXECUTION_ENGINE_VERSION,
                "config_hash": self._config.config_hash,
                "source_snapshot_hash": snapshot.content_hash,
                "lot_hashes": sorted(
                    lot.lot_hash for position in positions for lot in position.lots
                ),
            }
        )
        state = PaperAccountState.build(
            account_id=snapshot.account_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            currency=snapshot.currency,
            source_snapshot_id=snapshot.snapshot_id,
            source_snapshot_hash=snapshot.content_hash,
            source_snapshot_as_of=snapshot.as_of,
            total_cash=snapshot.cash.total_cash,
            external_frozen_cash=snapshot.cash.frozen_cash,
            positions=tuple(positions),
            orders=(),
            processed_batch_hashes=(),
            event_log_hash=opening_event_log_hash,
        )
        if (
            state.total_cash != snapshot.cash.total_cash
            or state.available_cash != snapshot.cash.available_cash
            or state.frozen_cash != snapshot.cash.frozen_cash
        ):
            raise PaperExecutionInputError("paper opening cash does not match the source snapshot")
        return state

    def refresh_account(
        self,
        *,
        account: PaperAccountState,
        snapshot: AccountSnapshot,
    ) -> PaperAccountState:
        """Advance order expiry and settlement to a new hash-linked snapshot boundary."""

        try:
            return replay_account_refresh(
                account_before=replace(account),
                snapshot=snapshot,
            )
        except PaperExecutionInputError:
            raise
        except (TypeError, ValueError) as error:
            raise PaperExecutionInputError(str(error)) from error

    def execute(
        self,
        *,
        request: PaperExecutionRequest,
        draft: OrderDraftBatch,
        account: PaperAccountState,
    ) -> PaperExecutionReceipt:
        """Return an immutable execution receipt; callers commit it transactionally."""

        try:
            trusted_request = replace(request)
            trusted_lines = tuple(replace(line) for line in draft.lines)
            trusted_draft = replace(draft, lines=trusted_lines)
            trusted_account = replace(account)
            return self._execute(
                request=trusted_request,
                draft=trusted_draft,
                account=trusted_account,
            )
        except PaperExecutionInputError:
            raise
        except (LookupError, TypeError, ValueError) as error:
            raise PaperExecutionInputError(str(error)) from error

    def _execute(
        self,
        *,
        request: PaperExecutionRequest,
        draft: OrderDraftBatch,
        account: PaperAccountState,
    ) -> PaperExecutionReceipt:
        self._validate_submission(request=request, draft=draft, account=account)
        trading_day = draft.trading_day
        selected: dict[str, tuple[CNMarketRule, CNSlippageModel, FeeRule]] = {}
        orders: list[PaperOrder] = []
        cash_required = _ZERO
        positions_by_id = {position.instrument_id: position for position in account.positions}
        sell_required: dict[str, Decimal] = {}
        for line in draft.lines:
            rule = self._market_rules.select(
                instrument_type=line.instrument_type,
                trading_day=trading_day,
            )
            model = self._slippage_model_book.select(trading_day)
            fee_rule = self._fee_rule_book.select(
                instrument_type=line.instrument_type,
                side=line.side,
                trading_day=trading_day,
                version=line.fee_rule_version,
            )
            self._validate_pinned_rules(line=line, rule=rule, model=model, fee_rule=fee_rule)
            position = positions_by_id.get(line.instrument_id)
            if position is not None and position.instrument_type is not line.instrument_type:
                raise PaperExecutionInputError(
                    f"{line.instrument_id} draft instrument type does not match paper position"
                )
            account_quantity = position.total_quantity if position is not None else _ZERO
            if line.is_full_liquidation and (position is None or line.quantity != account_quantity):
                raise PaperExecutionInputError(
                    f"{line.instrument_id} full-liquidation quantity does not match "
                    "the paper position"
                )
            if line.current_quantity != account_quantity:
                raise PaperExecutionInputError(
                    f"{line.instrument_id} draft current_quantity does not match paper position"
                )
            rule.validate_quantity(
                side=line.side,
                quantity=line.quantity,
                is_full_liquidation=line.is_full_liquidation,
            )
            selected[line.instrument_id] = (rule, model, fee_rule)
            cash_required += line.reserved_cash
            if line.side is Side.SELL:
                sell_required[line.instrument_id] = (
                    sell_required.get(line.instrument_id, _ZERO) + line.quantity
                )
            order_id = self._order_id(account=account, draft=draft, line=line)
            orders.append(
                PaperOrder.build(
                    order_id=order_id,
                    batch_hash=draft.batch_hash,
                    line_hash=line.line_hash,
                    decision_id=draft.decision_id,
                    instrument_id=line.instrument_id,
                    instrument_type=line.instrument_type,
                    side=line.side,
                    quantity=line.quantity,
                    filled_quantity=_ZERO,
                    is_full_liquidation=line.is_full_liquidation,
                    status=PaperOrderStatus.ACCEPTED,
                    created_at=request.submitted_at,
                    expires_at=draft.expires_at,
                    cash_reserved=line.reserved_cash,
                    sell_reserved_quantity=(line.quantity if line.side is Side.SELL else _ZERO),
                    market_rule_version=rule.version,
                    market_rule_hash=rule.rule_hash,
                    fee_rule_version=fee_rule.version,
                    fee_rule_hash=self._fee_rule_hash(fee_rule),
                    slippage_model_version=model.version,
                    slippage_model_hash=model.model_hash,
                )
            )
        if cash_required > account.available_cash:
            raise PaperExecutionInputError(
                "paper order reservations exceed current account available cash"
            )
        for instrument_id, quantity in sell_required.items():
            position = positions_by_id.get(instrument_id)
            if position is None or quantity > position.available_quantity:
                raise PaperExecutionInputError(
                    f"{instrument_id} paper sell reservations exceed available quantity"
                )

        final_orders: list[PaperOrder] = []
        attempts: list[PaperMatchAttempt] = []
        fills: list[PaperFill] = []
        total_cash = account.total_cash
        lines_by_hash = {line.line_hash: line for line in draft.lines}
        for sequence, accepted in enumerate(orders):
            line = lines_by_hash[accepted.line_hash]
            rule, model, fee_rule = selected[line.instrument_id]
            state = rule.state_for(
                instrument_id=line.instrument_id,
                trading_day=trading_day,
                as_of=request.submitted_at,
            )
            if state.data_version != draft.data_version:
                raise PaperExecutionInputError(
                    f"{line.instrument_id} execution state data version does not match draft"
                )
            if request.submitted_at - state.observed_at > timedelta(
                seconds=request.max_market_state_age_seconds
            ):
                raise PaperExecutionInputError(
                    f"{line.instrument_id} execution market state is stale"
                )
            attempt_id = self._attempt_id(
                order=accepted,
                request=request,
                state=state,
            )
            blocked_reason = self._blocked_reason(state=state, side=line.side)
            if blocked_reason is not None:
                attempt = self._no_fill_attempt(
                    attempt_id=attempt_id,
                    order=accepted,
                    at=request.submitted_at,
                    state=state,
                    rule=rule,
                    model=model,
                    reason=blocked_reason,
                )
                final_order = self._order_after_no_fill(
                    order=accepted,
                    attempt_id=attempt_id,
                )
                attempts.append(attempt)
                final_orders.append(final_order)
                continue
            instruction = OrderEvent(
                event_id=f"paper-instruction:{accepted.order_id}",
                run_id=request.request_id,
                sequence=sequence,
                event_time=request.submitted_at,
                trading_day=trading_day,
                instrument_id=line.instrument_id,
                instrument_type=line.instrument_type,
                order_id=accepted.order_id,
                signal_event_id=line.line_hash,
                signal_time=draft.draft_as_of,
                side=line.side,
                order_type=OrderType.MARKET,
                quantity=line.quantity,
                limit_price=None,
                status=OrderStatus.CREATED,
                previous_status=None,
                filled_quantity=_ZERO,
                reference_price=state.reference_price,
                price_observed_at=state.observed_at,
                price_available_at=state.available_at,
            )
            matched = rule.match(
                instruction,
                slippage_model=model,
                allow_partial=request.allow_partial,
                is_full_liquidation=line.is_full_liquidation,
            )
            if matched.state_hash != state.state_hash:
                raise PaperExecutionInputError("paper matcher used an unexpected market state")
            if matched.status is MatchStatus.NO_FILL:
                reason = self._match_no_fill_reason(matched.reason)
                attempt = self._no_fill_attempt(
                    attempt_id=attempt_id,
                    order=accepted,
                    at=request.submitted_at,
                    state=state,
                    rule=rule,
                    model=model,
                    reason=reason,
                )
                final_order = self._order_after_no_fill(
                    order=accepted,
                    attempt_id=attempt_id,
                )
                attempts.append(attempt)
                final_orders.append(final_order)
                continue
            assert matched.execution_price is not None
            actual_fees = fee_rule.assess(matched.gross_amount)
            actual_cash_change = (
                -(matched.gross_amount + actual_fees.total_amount)
                if line.side is Side.BUY
                else matched.gross_amount - actual_fees.total_amount
            )
            cash_debit = max(-actual_cash_change, _ZERO)
            remaining_cash_reserve = accepted.cash_reserved - cash_debit
            if remaining_cash_reserve < 0 or (
                matched.unfilled_quantity > 0
                and line.side is Side.BUY
                and remaining_cash_reserve == 0
            ):
                attempt = self._no_fill_attempt(
                    attempt_id=attempt_id,
                    order=accepted,
                    at=request.submitted_at,
                    state=state,
                    rule=rule,
                    model=model,
                    reason=PaperNoFillReason.INSUFFICIENT_CASH,
                    status=PaperOrderStatus.REJECTED,
                )
                final_order = self._terminal_rejected_order(
                    order=accepted,
                    attempt_id=attempt_id,
                )
                attempts.append(attempt)
                final_orders.append(final_order)
                continue
            result_status = (
                PaperOrderStatus.FILLED
                if matched.unfilled_quantity == 0
                else PaperOrderStatus.PARTIALLY_FILLED
            )
            attempt = PaperMatchAttempt.build(
                attempt_id=attempt_id,
                order_id=accepted.order_id,
                attempted_at=request.submitted_at,
                status=result_status,
                requested_quantity=matched.requested_quantity,
                filled_quantity=matched.filled_quantity,
                reference_price=matched.reference_price,
                execution_price=matched.execution_price,
                participation_rate=matched.participation_rate,
                state_revision=state.revision,
                data_version=state.data_version,
                state_hash=state.state_hash,
                market_rule_version=rule.version,
                market_rule_hash=rule.rule_hash,
                slippage_model_version=model.version,
                slippage_model_hash=model.model_hash,
            )
            sellable_on = (
                rule.sellable_on(acquired_on=trading_day) if line.side is Side.BUY else None
            )
            fill_identity = stable_hash(
                {
                    "attempt": attempt.attempt_hash,
                    "fee_rule": self._fee_rule_hash(fee_rule),
                }
            )
            fill_id = f"paper-fill:{fill_identity}"
            fill = PaperFill.build(
                fill_id=fill_id,
                order_id=accepted.order_id,
                attempt_id=attempt_id,
                instrument_id=line.instrument_id,
                instrument_type=line.instrument_type,
                filled_at=request.submitted_at,
                side=line.side,
                quantity=matched.filled_quantity,
                price=matched.execution_price,
                commission=actual_fees.commission,
                stamp_duty=actual_fees.stamp_duty,
                transfer_fee=actual_fees.transfer_fee,
                other_fee=actual_fees.other_fee,
                sellable_on=sellable_on,
                fee_rule_version=fee_rule.version,
                fee_rule_hash=self._fee_rule_hash(fee_rule),
            )
            final_order = PaperOrder.build(
                order_id=accepted.order_id,
                batch_hash=accepted.batch_hash,
                line_hash=accepted.line_hash,
                decision_id=accepted.decision_id,
                instrument_id=accepted.instrument_id,
                instrument_type=accepted.instrument_type,
                side=accepted.side,
                quantity=accepted.quantity,
                filled_quantity=matched.filled_quantity,
                is_full_liquidation=accepted.is_full_liquidation,
                status=result_status,
                created_at=accepted.created_at,
                expires_at=accepted.expires_at,
                cash_reserved=(remaining_cash_reserve if matched.unfilled_quantity else _ZERO),
                sell_reserved_quantity=(
                    matched.unfilled_quantity
                    if line.side is Side.SELL and matched.unfilled_quantity
                    else _ZERO
                ),
                market_rule_version=accepted.market_rule_version,
                market_rule_hash=accepted.market_rule_hash,
                fee_rule_version=accepted.fee_rule_version,
                fee_rule_hash=accepted.fee_rule_hash,
                slippage_model_version=accepted.slippage_model_version,
                slippage_model_hash=accepted.slippage_model_hash,
                last_attempt_id=attempt_id,
            )
            total_cash += fill.cash_change
            if total_cash < 0:
                raise PaperExecutionInputError("paper fill would create negative total cash")
            attempts.append(attempt)
            fills.append(fill)
            final_orders.append(final_order)

        ordered_orders = tuple(sorted(final_orders, key=lambda order: order.order_id))
        ordered_attempts = tuple(sorted(attempts, key=lambda attempt: attempt.attempt_id))
        ordered_fills = tuple(sorted(fills, key=lambda fill: fill.fill_id))
        event_log_hash = PaperExecutionReceipt.event_log_hash_for(
            previous_event_log_hash=account.event_log_hash,
            request_hash=request.request_hash,
            batch_hash=draft.batch_hash,
            orders=ordered_orders,
            attempts=ordered_attempts,
            fills=ordered_fills,
        )
        account_after = replay_execution_transition(
            account_before=account,
            batch_hash=draft.batch_hash,
            processed_at=request.submitted_at,
            event_log_hash=event_log_hash,
            orders=ordered_orders,
            fills=ordered_fills,
        )
        receipt_identity = stable_hash(
            {
                "request": request.request_hash,
                "account_after": account_after.state_hash,
            }
        )
        receipt_id = f"paper-receipt:{receipt_identity}"
        return PaperExecutionReceipt.build(
            receipt_id=receipt_id,
            request=request,
            account_before=account,
            account_after=account_after,
            processed_at=request.submitted_at,
            orders=ordered_orders,
            attempts=ordered_attempts,
            fills=ordered_fills,
        )

    def _validate_submission(
        self,
        *,
        request: PaperExecutionRequest,
        draft: OrderDraftBatch,
        account: PaperAccountState,
    ) -> None:
        if (
            request.config_hash != self._config.config_hash
            or request.config_version != self._config.version
            or request.allow_partial != self._config.allow_partial
            or request.order_policy is not self._config.order_policy
            or request.max_market_state_age_seconds != self._config.max_market_state_age_seconds
            or request.environment is not self._config.environment
        ):
            raise PaperExecutionInputError("request does not match paper execution config")
        if request.engine_version != PAPER_EXECUTION_ENGINE_VERSION:
            raise PaperExecutionInputError("request paper engine version does not match")
        if request.account_id != account.account_id:
            raise PaperExecutionInputError("request account_id does not match paper account")
        if request.expected_account_state_hash != account.state_hash:
            raise PaperExecutionInputError("paper account state changed before submission")
        if request.expected_account_snapshot_hash != account.source_snapshot_hash:
            raise PaperExecutionInputError("paper account snapshot changed before submission")
        if request.batch_hash != draft.batch_hash:
            raise PaperExecutionInputError("request batch hash does not match order draft")
        if draft.runtime_mode.value != "PAPER" or account.runtime_mode.value != "PAPER":
            raise PaperExecutionInputError("paper execution accepts only PAPER inputs")
        if draft.account_snapshot_id != account.source_snapshot_id:
            raise PaperExecutionInputError("draft account snapshot id is stale")
        if draft.account_snapshot_hash != account.source_snapshot_hash:
            raise PaperExecutionInputError("draft account snapshot hash is stale")
        if draft.account_snapshot_as_of != account.source_snapshot_as_of:
            raise PaperExecutionInputError("draft account snapshot boundary is stale")
        if draft.data_version != account.data_version or draft.currency != account.currency:
            raise PaperExecutionInputError("draft data version or currency does not match account")
        if draft.available_cash != account.available_cash:
            raise PaperExecutionInputError("draft available cash is stale")
        if draft.batch_hash in account.processed_batch_hashes:
            raise PaperExecutionInputError("paper draft batch was already processed")
        if request.submitted_at < account.as_of:
            raise PaperExecutionInputError("paper submission cannot precede account state")
        if any(
            order.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.PARTIALLY_FILLED}
            and request.submitted_at >= order.expires_at
            for order in account.orders
        ):
            raise PaperExecutionInputError(
                "expired active paper orders require an account refresh before execution"
            )
        if request.submitted_at < draft.draft_as_of:
            raise PaperExecutionInputError("paper submission cannot precede draft creation")
        if request.submitted_at >= draft.expires_at:
            raise PaperExecutionInputError("paper order draft is expired")
        if request.submitted_at.astimezone(SHANGHAI_TZ).date() != draft.trading_day:
            raise PaperExecutionInputError("paper submission must remain on the draft trading day")

    @staticmethod
    def _validate_pinned_rules(
        *,
        line: OrderDraftLine,
        rule: CNMarketRule,
        model: CNSlippageModel,
        fee_rule: FeeRule,
    ) -> None:
        fee_hash = PaperExecutionEngine._fee_rule_hash(fee_rule)
        checks = (
            (rule.version, line.market_rule_version, "market rule version"),
            (rule.rule_hash, line.market_rule_hash, "market rule hash"),
            (model.version, line.slippage_model_version, "slippage model version"),
            (model.model_hash, line.slippage_model_hash, "slippage model hash"),
            (fee_rule.version, line.fee_rule_version, "fee rule version"),
            (fee_hash, line.fee_rule_hash, "fee rule hash"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise PaperExecutionInputError(f"draft {label} is not executable")

    @staticmethod
    def _blocked_reason(
        *,
        state: MarketSessionState,
        side: Side,
    ) -> PaperNoFillReason | None:
        if state.state is MarketTradeState.SUSPENDED:
            return PaperNoFillReason.SUSPENDED
        allowed = state.buy_allowed if side is Side.BUY else state.sell_allowed
        return None if allowed else PaperNoFillReason.SIDE_BLOCKED

    @staticmethod
    def _no_fill_attempt(
        *,
        attempt_id: str,
        order: PaperOrder,
        at: datetime,
        state: MarketSessionState,
        rule: CNMarketRule,
        model: CNSlippageModel,
        reason: PaperNoFillReason,
        status: PaperOrderStatus = PaperOrderStatus.ACCEPTED,
    ) -> PaperMatchAttempt:
        return PaperMatchAttempt.build(
            attempt_id=attempt_id,
            order_id=order.order_id,
            attempted_at=at,
            status=status,
            requested_quantity=order.remaining_quantity,
            filled_quantity=_ZERO,
            reference_price=state.reference_price,
            execution_price=None,
            participation_rate=_ZERO,
            state_revision=state.revision,
            data_version=state.data_version,
            state_hash=state.state_hash,
            market_rule_version=rule.version,
            market_rule_hash=rule.rule_hash,
            slippage_model_version=model.version,
            slippage_model_hash=model.model_hash,
            no_fill_reason=reason,
        )

    @staticmethod
    def _order_after_no_fill(*, order: PaperOrder, attempt_id: str) -> PaperOrder:
        return PaperOrder.build(
            order_id=order.order_id,
            batch_hash=order.batch_hash,
            line_hash=order.line_hash,
            decision_id=order.decision_id,
            instrument_id=order.instrument_id,
            instrument_type=order.instrument_type,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=_ZERO,
            is_full_liquidation=order.is_full_liquidation,
            status=PaperOrderStatus.ACCEPTED,
            created_at=order.created_at,
            expires_at=order.expires_at,
            cash_reserved=order.cash_reserved,
            sell_reserved_quantity=order.sell_reserved_quantity,
            market_rule_version=order.market_rule_version,
            market_rule_hash=order.market_rule_hash,
            fee_rule_version=order.fee_rule_version,
            fee_rule_hash=order.fee_rule_hash,
            slippage_model_version=order.slippage_model_version,
            slippage_model_hash=order.slippage_model_hash,
            last_attempt_id=attempt_id,
        )

    @staticmethod
    def _terminal_rejected_order(*, order: PaperOrder, attempt_id: str) -> PaperOrder:
        return PaperOrder.build(
            order_id=order.order_id,
            batch_hash=order.batch_hash,
            line_hash=order.line_hash,
            decision_id=order.decision_id,
            instrument_id=order.instrument_id,
            instrument_type=order.instrument_type,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=_ZERO,
            is_full_liquidation=order.is_full_liquidation,
            status=PaperOrderStatus.REJECTED,
            created_at=order.created_at,
            expires_at=order.expires_at,
            cash_reserved=_ZERO,
            sell_reserved_quantity=_ZERO,
            market_rule_version=order.market_rule_version,
            market_rule_hash=order.market_rule_hash,
            fee_rule_version=order.fee_rule_version,
            fee_rule_hash=order.fee_rule_hash,
            slippage_model_version=order.slippage_model_version,
            slippage_model_hash=order.slippage_model_hash,
            last_attempt_id=attempt_id,
        )

    @staticmethod
    def _match_no_fill_reason(reason: NoFillReason | None) -> PaperNoFillReason:
        mapping = {
            NoFillReason.ZERO_CAPACITY: PaperNoFillReason.ZERO_CAPACITY,
            NoFillReason.PARTIAL_FILL_DISABLED: PaperNoFillReason.PARTIAL_FILL_DISABLED,
            NoFillReason.LIMIT_NOT_MARKETABLE: PaperNoFillReason.LIMIT_NOT_MARKETABLE,
        }
        if reason not in mapping:
            raise PaperExecutionInputError("paper matcher returned an unknown no-fill reason")
        return mapping[reason]

    @staticmethod
    def _fee_rule_hash(rule: FeeRule) -> str:
        return stable_hash(rule.model_dump(mode="python"))

    @staticmethod
    def _order_id(
        *,
        account: PaperAccountState,
        draft: OrderDraftBatch,
        line: OrderDraftLine,
    ) -> str:
        identity = stable_hash(
            {
                "account": account.account_id,
                "batch": draft.batch_hash,
                "line": line.line_hash,
            }
        )
        return f"paper-order:{identity}"

    @staticmethod
    def _attempt_id(
        *,
        order: PaperOrder,
        request: PaperExecutionRequest,
        state: MarketSessionState,
    ) -> str:
        identity = stable_hash(
            {
                "order": order.order_id,
                "request": request.request_hash,
                "state": state.state_hash,
            }
        )
        return f"paper-attempt:{identity}"


__all__ = ["PaperExecutionEngine"]
