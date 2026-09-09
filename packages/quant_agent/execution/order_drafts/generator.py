"""Deterministic conversion from a risk-checked target portfolio to review-only drafts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction

from quant_agent.backtest import FeeRule, FeeRuleBook, Side
from quant_agent.backtest.cn_market import (
    CNMarketRule,
    CNSlippageModel,
    CNSlippageModelBook,
    MarketSessionState,
)
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.portfolio import AccountSnapshot, PortfolioTargetLine, TargetPortfolio
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import RiskCheckResult, RiskCheckStatus

from .contracts import (
    OrderDraftBatch,
    OrderDraftGeneratorConfig,
    OrderDraftInputError,
    OrderDraftLine,
)

_ZERO = Decimal(0)


def _is_integral(value: Decimal) -> bool:
    return value == value.to_integral_value()


def _floor_to_unit(value: Fraction, unit: Decimal) -> Decimal:
    """Floor a non-negative exact quantity without Decimal-context rounding."""

    unit_as_integer = int(unit)
    if unit != Decimal(unit_as_integer) or unit_as_integer <= 0:
        raise OrderDraftInputError("draft lot size must be a positive whole number")
    lot_count = value // unit_as_integer
    return Decimal(lot_count * unit_as_integer)


class OrderDraftGenerator:
    """Build immutable order estimates without creating an executable order object."""

    def __init__(
        self,
        *,
        market_rules: Iterable[CNMarketRule],
        fee_rule_book: FeeRuleBook,
        slippage_model_book: CNSlippageModelBook,
        config: OrderDraftGeneratorConfig | None = None,
    ) -> None:
        rules = tuple(market_rules)
        if not rules:
            raise OrderDraftInputError("at least one market rule is required")
        if any(not isinstance(rule, CNMarketRule) for rule in rules):
            raise OrderDraftInputError("market_rules must contain only CNMarketRule values")
        boundaries: set[tuple[object, date]] = set()
        versions: set[tuple[object, str, str]] = set()
        for rule in rules:
            boundary = (rule.instrument_type, rule.effective_from)
            identity = (rule.instrument_type, rule.rule_id, rule.version)
            if boundary in boundaries:
                raise OrderDraftInputError(
                    "market rules are ambiguous at an instrument effective boundary"
                )
            if identity in versions:
                raise OrderDraftInputError("market rule identities and versions must be unique")
            boundaries.add(boundary)
            versions.add(identity)
        if not isinstance(fee_rule_book, FeeRuleBook):
            raise OrderDraftInputError("fee_rule_book must be a FeeRuleBook")
        if not isinstance(slippage_model_book, CNSlippageModelBook):
            raise OrderDraftInputError("slippage_model_book must be a CNSlippageModelBook")
        if config is not None and not isinstance(config, OrderDraftGeneratorConfig):
            raise OrderDraftInputError("config must be an OrderDraftGeneratorConfig")
        self._market_rules = rules
        self._fee_rule_book = fee_rule_book
        self._slippage_model_book = slippage_model_book
        self._config = config or OrderDraftGeneratorConfig()

    @property
    def config(self) -> OrderDraftGeneratorConfig:
        """Return the immutable configuration fixed for this generator instance."""

        return self._config

    def generate(
        self,
        *,
        account_snapshot: AccountSnapshot,
        target_portfolio: TargetPortfolio,
        risk_result: RiskCheckResult,
        market_states: Iterable[MarketSessionState],
        draft_as_of: datetime,
    ) -> OrderDraftBatch:
        """Generate one deterministic batch or fail the complete proposal closed."""

        try:
            return self._generate(
                account_snapshot=account_snapshot,
                target_portfolio=target_portfolio,
                risk_result=risk_result,
                market_states=tuple(market_states),
                draft_as_of=draft_as_of,
            )
        except OrderDraftInputError:
            raise
        except (LookupError, TypeError, ValueError) as error:
            raise OrderDraftInputError(str(error)) from error

    def _generate(
        self,
        *,
        account_snapshot: AccountSnapshot,
        target_portfolio: TargetPortfolio,
        risk_result: RiskCheckResult,
        market_states: tuple[MarketSessionState, ...],
        draft_as_of: datetime,
    ) -> OrderDraftBatch:
        if not isinstance(account_snapshot, AccountSnapshot):
            raise OrderDraftInputError("account_snapshot must be an AccountSnapshot")
        if not isinstance(target_portfolio, TargetPortfolio):
            raise OrderDraftInputError("target_portfolio must be a TargetPortfolio")
        if not isinstance(risk_result, RiskCheckResult):
            raise OrderDraftInputError("risk_result must be a RiskCheckResult")
        ensure_aware(draft_as_of)
        trading_day = draft_as_of.astimezone(SHANGHAI_TZ).date()
        self._validate_account_and_proposal(
            account=account_snapshot,
            proposal=target_portfolio,
            trading_day=trading_day,
            draft_as_of=draft_as_of,
        )
        self._validate_risk_result(
            account=account_snapshot,
            proposal=target_portfolio,
            risk_result=risk_result,
            draft_as_of=draft_as_of,
        )
        states_by_instrument = self._validate_market_states(
            proposal=target_portfolio,
            market_states=market_states,
            trading_day=trading_day,
            draft_as_of=draft_as_of,
        )
        positions_by_instrument = {
            position.instrument_id: position for position in account_snapshot.positions
        }
        slippage_model = self._slippage_model_book.select(trading_day)
        lines: list[OrderDraftLine] = []
        sizing_state_hashes: set[str] = set()
        sizing_market_rule_hashes: set[str] = set()
        for target_line in target_portfolio.lines:
            state = states_by_instrument[target_line.instrument_id]
            rule = self._select_market_rule(
                instrument_type=target_line.instrument_type,
                trading_day=trading_day,
            )
            self._validate_rule_state(
                rule=rule,
                state=state,
                draft_as_of=draft_as_of,
            )
            sizing_state_hashes.add(state.state_hash)
            sizing_market_rule_hashes.add(rule.rule_hash)
            position = positions_by_instrument.get(target_line.instrument_id)
            current_quantity = _ZERO if position is None else position.total_quantity
            if not _is_integral(current_quantity):
                raise OrderDraftInputError(
                    f"{target_line.instrument_id} account quantity must be a whole number"
                )
            sized = self._size_line(
                target_line=target_line,
                current_quantity=current_quantity,
                buy_lot_size=rule.buy_lot_size,
                sell_lot_size=rule.sell_lot_size,
                reference_price=state.reference_price,
            )
            if sized is None:
                continue
            side, quantity, target_quantity, is_full_liquidation = sized
            if side is Side.SELL:
                if position is None:
                    raise OrderDraftInputError(
                        f"{target_line.instrument_id} cannot be sold without an account position"
                    )
                if not _is_integral(position.available_quantity):
                    raise OrderDraftInputError(
                        f"{target_line.instrument_id} available quantity must be a whole number"
                    )
                if quantity > position.available_quantity:
                    raise OrderDraftInputError(
                        f"{target_line.instrument_id} draft sell quantity exceeds account "
                        "available quantity"
                    )
                if is_full_liquidation and position.available_quantity != current_quantity:
                    raise OrderDraftInputError(
                        f"{target_line.instrument_id} full liquidation requires the complete "
                        "position to be available"
                    )
            self._validate_side_and_capacity(
                rule=rule,
                state=state,
                side=side,
                quantity=quantity,
                is_full_liquidation=is_full_liquidation,
            )
            participation_rate = self._participation_rate(
                quantity=quantity,
                market_capacity=state.available_quantity,
                maximum=rule.max_participation_rate,
                instrument_id=target_line.instrument_id,
            )
            estimated_price = self._execution_price(
                model=slippage_model,
                rule=rule,
                state=state,
                side=side,
                participation_rate=participation_rate,
            )
            fee_rule = self._fee_rule_book.select(
                instrument_type=target_line.instrument_type,
                side=side,
                trading_day=trading_day,
            )
            estimated_fees = fee_rule.assess(quantity * estimated_price)
            reservation_price = self._reservation_price(
                side=side,
                estimated_price=estimated_price,
                state=state,
            )
            reservation_fee = (
                fee_rule.assess(quantity * reservation_price).total_amount
                if side is Side.BUY
                else estimated_fees.total_amount
            )
            lines.append(
                OrderDraftLine.build(
                    instrument_id=target_line.instrument_id,
                    instrument_type=target_line.instrument_type,
                    side=side,
                    current_quantity=current_quantity,
                    target_quantity=target_quantity,
                    quantity=quantity,
                    is_full_liquidation=is_full_liquidation,
                    reference_price=state.reference_price,
                    estimated_execution_price=estimated_price,
                    price_observed_at=state.observed_at,
                    price_available_at=state.available_at,
                    data_version=state.data_version,
                    state_revision=state.revision,
                    state_hash=state.state_hash,
                    participation_rate=participation_rate,
                    commission=estimated_fees.commission,
                    stamp_duty=estimated_fees.stamp_duty,
                    transfer_fee=estimated_fees.transfer_fee,
                    other_fee=estimated_fees.other_fee,
                    market_rule_version=rule.version,
                    market_rule_hash=rule.rule_hash,
                    fee_rule_version=fee_rule.version,
                    fee_rule_hash=self._fee_rule_hash(fee_rule),
                    slippage_model_version=slippage_model.version,
                    slippage_model_hash=slippage_model.model_hash,
                    rationale=target_line.rationale,
                    cash_reservation_price=reservation_price,
                    cash_reservation_fee=reservation_fee,
                )
            )
        if not lines:
            raise OrderDraftInputError("target portfolio produces no non-zero order draft lines")
        request = risk_result.request
        return OrderDraftBatch.build(
            decision_id=target_portfolio.decision_id,
            draft_as_of=draft_as_of,
            data_version=target_portfolio.data_version,
            currency=account_snapshot.currency,
            account_snapshot_id=account_snapshot.snapshot_id,
            account_snapshot_hash=account_snapshot.content_hash,
            account_snapshot_as_of=account_snapshot.as_of,
            runtime_mode=account_snapshot.runtime_mode,
            portfolio_proposal_hash=target_portfolio.result_hash,
            risk_request_hash=request.request_hash,
            risk_result_hash=risk_result.result_hash,
            risk_status=risk_result.status,
            risk_checked_at=risk_result.evaluated_at,
            risk_engine_version=risk_result.risk_engine_version,
            risk_policy_version=request.policy_version,
            risk_policy_hash=request.policy_hash,
            config=self._config,
            available_cash=account_snapshot.cash.available_cash,
            lines=tuple(lines),
            sizing_state_hashes=tuple(sorted(sizing_state_hashes)),
            sizing_market_rule_hashes=tuple(sorted(sizing_market_rule_hashes)),
        )

    @staticmethod
    def _validate_account_and_proposal(
        *,
        account: AccountSnapshot,
        proposal: TargetPortfolio,
        trading_day: date,
        draft_as_of: datetime,
    ) -> None:
        if account.runtime_mode not in {RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}:
            raise OrderDraftInputError(
                "order drafts require a PAPER or LIVE_ASSISTED account snapshot"
            )
        if account.as_of.astimezone(SHANGHAI_TZ).date() != trading_day:
            raise OrderDraftInputError("overnight order drafts are not supported in version 1")
        if account.as_of > draft_as_of:
            raise OrderDraftInputError("account snapshot cannot be after draft_as_of")
        checks = (
            (proposal.account_snapshot_id, account.snapshot_id, "proposal account id"),
            (proposal.account_snapshot_hash, account.content_hash, "proposal account hash"),
            (proposal.as_of, account.as_of, "proposal account boundary"),
            (proposal.data_version, account.data_version, "proposal data version"),
            (proposal.total_equity, account.total_equity, "proposal total equity"),
            (
                proposal.current_gross_weight,
                account.gross_exposure,
                "proposal current exposure",
            ),
            (proposal.current_cash_weight, account.cash_weight, "proposal cash weight"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise OrderDraftInputError(f"{label} does not match the account snapshot")
        positions = {item.instrument_id: item for item in account.positions}
        proposal_ids = {line.instrument_id for line in proposal.lines}
        missing_positions = set(positions).difference(proposal_ids)
        if missing_positions:
            missing = ", ".join(sorted(missing_positions))
            raise OrderDraftInputError(f"proposal does not cover account positions: {missing}")
        for line in proposal.lines:
            position = positions.get(line.instrument_id)
            expected_market_value = _ZERO if position is None else position.market_value
            if line.current_market_value != expected_market_value:
                raise OrderDraftInputError(
                    f"{line.instrument_id} proposal current value does not match the account"
                )
            if position is not None and line.instrument_type is not position.instrument_type:
                raise OrderDraftInputError(
                    f"{line.instrument_id} proposal instrument type does not match the account"
                )

    @staticmethod
    def _validate_risk_result(
        *,
        account: AccountSnapshot,
        proposal: TargetPortfolio,
        risk_result: RiskCheckResult,
        draft_as_of: datetime,
    ) -> None:
        if risk_result.status not in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}:
            raise OrderDraftInputError(
                f"risk result {risk_result.status.value} cannot create an order draft"
            )
        if risk_result.hard_violations or risk_result.error_code is not None:
            raise OrderDraftInputError("hard risk violations and risk errors fail closed")
        if risk_result.evaluated_at > draft_as_of:
            raise OrderDraftInputError("risk result cannot be evaluated after draft_as_of")
        request = risk_result.request
        checks = (
            (request.decision_id, proposal.decision_id, "risk decision id"),
            (request.account_snapshot_id, account.snapshot_id, "risk account id"),
            (request.account_snapshot_hash, account.content_hash, "risk account hash"),
            (request.account_snapshot_as_of, account.as_of, "risk account boundary"),
            (
                request.portfolio_proposal_hash,
                proposal.result_hash,
                "risk proposal hash",
            ),
            (request.data_version, account.data_version, "risk account data version"),
            (request.data_version, proposal.data_version, "risk proposal data version"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise OrderDraftInputError(f"{label} does not match the bound input")

    @staticmethod
    def _validate_market_states(
        *,
        proposal: TargetPortfolio,
        market_states: tuple[MarketSessionState, ...],
        trading_day: date,
        draft_as_of: datetime,
    ) -> dict[str, MarketSessionState]:
        if any(not isinstance(state, MarketSessionState) for state in market_states):
            raise OrderDraftInputError("market_states must contain only MarketSessionState values")
        state_ids = tuple(state.instrument_id for state in market_states)
        if len(set(state_ids)) != len(state_ids):
            raise OrderDraftInputError("market states must be unique per instrument")
        expected_ids = tuple(line.instrument_id for line in proposal.lines)
        if set(state_ids) != set(expected_ids):
            raise OrderDraftInputError("market states must exactly cover target portfolio lines")
        states = {state.instrument_id: state for state in market_states}
        for line in proposal.lines:
            state = states[line.instrument_id]
            if state.instrument_type is not line.instrument_type:
                raise OrderDraftInputError(
                    f"{line.instrument_id} market state instrument type does not match"
                )
            if state.trading_day != trading_day:
                raise OrderDraftInputError(
                    f"{line.instrument_id} market state trading day does not match"
                )
            if state.data_version != proposal.data_version:
                raise OrderDraftInputError(
                    f"{line.instrument_id} market state data version does not match"
                )
            if state.available_at > draft_as_of:
                raise OrderDraftInputError(
                    f"{line.instrument_id} future market state is not point-in-time safe"
                )
        return states

    def _select_market_rule(
        self,
        *,
        instrument_type: object,
        trading_day: date,
    ) -> CNMarketRule:
        candidates = tuple(
            rule
            for rule in self._market_rules
            if rule.instrument_type is instrument_type and rule.applies_on(trading_day)
        )
        if not candidates:
            raise OrderDraftInputError(
                f"no effective market rule for {instrument_type!s} on {trading_day.isoformat()}"
            )
        return max(candidates, key=lambda rule: rule.effective_from)

    @staticmethod
    def _validate_rule_state(
        *,
        rule: CNMarketRule,
        state: MarketSessionState,
        draft_as_of: datetime,
    ) -> None:
        known = tuple(
            candidate
            for candidate in rule.session_states
            if candidate.instrument_id == state.instrument_id
            and candidate.trading_day == state.trading_day
            and candidate.available_at <= draft_as_of
        )
        if not known:
            raise OrderDraftInputError(
                f"{state.instrument_id} has no market-rule state available at draft time"
            )
        latest_available_at = max(candidate.available_at for candidate in known)
        latest = tuple(
            candidate for candidate in known if candidate.available_at == latest_available_at
        )
        if not any(candidate == state for candidate in latest):
            raise OrderDraftInputError(
                f"{state.instrument_id} market state is not the latest bound revision"
            )
        prices = (
            state.reference_price,
            state.lower_limit_price,
            state.upper_limit_price,
        )
        if any(price is not None and price % rule.price_tick for price in prices):
            raise OrderDraftInputError(
                f"{state.instrument_id} market state prices are not tick aligned"
            )

    @staticmethod
    def _size_line(
        *,
        target_line: PortfolioTargetLine,
        current_quantity: Decimal,
        buy_lot_size: Decimal,
        sell_lot_size: Decimal,
        reference_price: Decimal,
    ) -> tuple[Side, Decimal, Decimal, bool] | None:
        if target_line.target_weight == 0:
            if current_quantity == 0:
                return None
            return Side.SELL, current_quantity, _ZERO, True
        raw_target_quantity = Fraction(target_line.target_market_value) / Fraction(reference_price)
        exact_current_quantity = Fraction(current_quantity)
        if raw_target_quantity > exact_current_quantity:
            quantity = _floor_to_unit(
                raw_target_quantity - exact_current_quantity,
                buy_lot_size,
            )
            if quantity == 0:
                return None
            target_quantity = Decimal(int(current_quantity) + int(quantity))
            return Side.BUY, quantity, target_quantity, False
        if raw_target_quantity < exact_current_quantity:
            quantity = _floor_to_unit(
                exact_current_quantity - raw_target_quantity,
                sell_lot_size,
            )
            if quantity == 0:
                return None
            target_quantity = Decimal(int(current_quantity) - int(quantity))
            return Side.SELL, quantity, target_quantity, False
        return None

    @staticmethod
    def _validate_side_and_capacity(
        *,
        rule: CNMarketRule,
        state: MarketSessionState,
        side: Side,
        quantity: Decimal,
        is_full_liquidation: bool,
    ) -> None:
        allowed = state.buy_allowed if side is Side.BUY else state.sell_allowed
        if not allowed:
            raise OrderDraftInputError(
                f"{state.instrument_id} market state blocks {side.value} drafts"
            )
        rule.validate_quantity(
            side=side,
            quantity=quantity,
            is_full_liquidation=is_full_liquidation,
        )

    @staticmethod
    def _participation_rate(
        *,
        quantity: Decimal,
        market_capacity: Decimal,
        maximum: Decimal,
        instrument_id: str,
    ) -> Decimal:
        if market_capacity <= 0:
            raise OrderDraftInputError(f"{instrument_id} has no market draft capacity")
        with localcontext() as context:
            context.prec = 50
            participation = quantity / market_capacity
        if participation > maximum:
            raise OrderDraftInputError(
                f"{instrument_id} draft exceeds the market participation limit"
            )
        return participation

    @staticmethod
    def _execution_price(
        *,
        model: CNSlippageModel,
        rule: CNMarketRule,
        state: MarketSessionState,
        side: Side,
        participation_rate: Decimal,
    ) -> Decimal:
        result = model.execution_price(
            side=side,
            reference_price=state.reference_price,
            participation_rate=participation_rate,
            price_tick=rule.price_tick,
        )
        if state.upper_limit_price is not None:
            result = min(result, state.upper_limit_price)
        if state.lower_limit_price is not None:
            result = max(result, state.lower_limit_price)
        if result <= 0 or result % rule.price_tick:
            raise OrderDraftInputError(
                f"{state.instrument_id} estimated execution price is invalid"
            )
        return result

    @staticmethod
    def _reservation_price(
        *,
        side: Side,
        estimated_price: Decimal,
        state: MarketSessionState,
    ) -> Decimal:
        if side is Side.BUY and state.upper_limit_price is not None:
            return max(estimated_price, state.upper_limit_price)
        return estimated_price

    @staticmethod
    def _fee_rule_hash(rule: FeeRule) -> str:
        return stable_hash(rule.model_dump(mode="python"))


__all__ = ["OrderDraftGenerator"]
