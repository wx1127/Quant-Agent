"""Point-in-time liquidity, market-state, and account-capacity analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from quant_agent.data.domain import InstrumentStatus
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.features.tradeability.contracts import (
    CapacityEstimate,
    InstrumentTradeabilityRule,
    InstrumentTradeabilityState,
    InsufficientTradeabilityData,
    MarketTradeState,
    TradeabilityBar,
    TradeabilityCalendarSession,
    TradeabilityConfig,
    TradeabilityContribution,
    TradeabilityContributionCode,
    TradeabilityEligibility,
    TradeabilityInputError,
    TradeabilityReason,
    TradeabilityReasonCode,
    TradeabilityRequest,
    TradeabilitySnapshot,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    units = (value / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return units * tick


class TradeabilityAnalyzer:
    """Evaluate practical daily tradeability without embedding instrument strategy rules."""

    def __init__(self, config: TradeabilityConfig | None = None) -> None:
        self._config = config or TradeabilityConfig()

    @property
    def config(self) -> TradeabilityConfig:
        """Expose the immutable configuration used for this analyzer."""

        return self._config

    def analyze(
        self,
        *,
        request: TradeabilityRequest,
        instrument_states: Iterable[InstrumentTradeabilityState],
        calendar: Iterable[TradeabilityCalendarSession],
        bars: Iterable[TradeabilityBar],
        rules: Iterable[InstrumentTradeabilityRule],
    ) -> TradeabilitySnapshot:
        """Build one strict PIT snapshot for an instrument/account/side combination."""

        master = self._select_master(request, tuple(instrument_states))
        rule = self._select_rule(request, tuple(rules))
        window_sessions = self._select_calendar(request, tuple(calendar))
        long_dates = tuple(item.trade_date for item in window_sessions)
        short_dates = long_dates[-self._config.short_turnover_window :]
        selected_bars = self._select_bars(
            request,
            tuple(bars),
            window_sessions=window_sessions,
            master=master,
        )
        bars_by_date = {item.trade_date: item for item in selected_bars}
        current = bars_by_date.get(request.session_date)
        if current is None:
            raise InsufficientTradeabilityData(
                "current session requires a point-in-time tradeability bar"
            )

        average_short = self._average_turnover(
            bars_by_date,
            short_dates,
            listed_on=master.listed_on,
        )
        average_long = self._average_turnover(
            bars_by_date,
            long_dates,
            listed_on=master.listed_on,
        )
        turnover_percentile, turnover_observations = self._turnover_rate_percentile(
            bars_by_date,
            long_dates,
        )
        listing_days = max((request.session_date - master.listed_on).days + 1, 0)
        market_state, upper_limit, lower_limit = self._market_state(current, master, rule)
        capacity = self._capacity(request, average_short, average_long)
        contributions, numeric_reasons = self._numeric_gates(
            listing_days=listing_days,
            average_short=average_short,
            average_long=average_long,
            turnover_percentile=turnover_percentile,
            capacity=capacity,
        )
        structural_reasons = self._structural_reasons(
            request=request,
            master=master,
            rule=rule,
            market_state=market_state,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
        )
        reasons = (*structural_reasons, *numeric_reasons)
        eligibility = (
            TradeabilityEligibility.INELIGIBLE
            if any(reason.blocking for reason in reasons)
            else TradeabilityEligibility.ELIGIBLE
        )
        input_hash = self._input_hash(
            request=request,
            master=master,
            rule=rule,
            calendar=window_sessions,
            bars=selected_bars,
        )
        result_hash = self._result_hash(
            request=request,
            rule=rule,
            market_state=market_state,
            eligibility=eligibility,
            reasons=reasons,
            contributions=contributions,
            listing_days=listing_days,
            average_short=average_short,
            average_long=average_long,
            turnover_percentile=turnover_percentile,
            turnover_observations=turnover_observations,
            short_dates=short_dates,
            long_dates=long_dates,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            capacity=capacity,
            input_hash=input_hash,
        )
        return TradeabilitySnapshot(
            request=request,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            rule_id=rule.rule_id,
            rule_version=rule.version,
            rule_hash=rule.rule_hash,
            market_state=market_state,
            eligibility=eligibility,
            reasons=reasons,
            contributions=contributions,
            listing_days=listing_days,
            average_turnover_short=average_short,
            average_turnover_long=average_long,
            turnover_rate_percentile=turnover_percentile,
            turnover_rate_observation_count=turnover_observations,
            short_window_dates=short_dates,
            long_window_dates=long_dates,
            upper_limit_price=upper_limit,
            lower_limit_price=lower_limit,
            capacity=capacity,
            input_hash=input_hash,
            result_hash=result_hash,
        )

    @staticmethod
    def _select_master(
        request: TradeabilityRequest,
        states: tuple[InstrumentTradeabilityState, ...],
    ) -> InstrumentTradeabilityState:
        candidates = [
            item
            for item in states
            if item.instrument_id == request.instrument_id
            and item.applies_on(request.session_date)
            and item.available_at <= request.as_of
        ]
        if not candidates:
            raise InsufficientTradeabilityData(
                "no point-in-time master/status record covers the requested session"
            )
        latest_available = max(item.available_at for item in candidates)
        latest = [item for item in candidates if item.available_at == latest_available]
        if len(latest) != 1:
            if any(item != latest[0] for item in latest[1:]):
                raise TradeabilityInputError("ambiguous master revisions at the same available_at")
            master = latest[0]
        else:
            master = latest[0]
        if master.instrument_type is not request.instrument_type:
            raise TradeabilityInputError("master instrument_type does not match request")
        if master.data_version != request.data_version:
            raise TradeabilityInputError("master data_version does not match request")
        return master

    @staticmethod
    def _select_rule(
        request: TradeabilityRequest,
        rules: tuple[InstrumentTradeabilityRule, ...],
    ) -> InstrumentTradeabilityRule:
        candidates = [
            item
            for item in rules
            if item.instrument_type is request.instrument_type
            and item.applies_on(request.session_date)
        ]
        if not candidates:
            raise InsufficientTradeabilityData(
                "no instrument tradeability rule covers the requested session"
            )
        latest_effective = max(item.effective_from for item in candidates)
        latest = [item for item in candidates if item.effective_from == latest_effective]
        if len(latest) != 1:
            raise TradeabilityInputError("ambiguous tradeability rules at an effective boundary")
        return latest[0]

    def _select_calendar(
        self,
        request: TradeabilityRequest,
        sessions: tuple[TradeabilityCalendarSession, ...],
    ) -> tuple[TradeabilityCalendarSession, ...]:
        selected: dict[date, TradeabilityCalendarSession] = {}
        for item in sessions:
            if item.market != request.market or item.trade_date > request.session_date:
                continue
            if item.available_at > request.as_of:
                continue
            current = selected.get(item.trade_date)
            if current is None or item.available_at > current.available_at:
                selected[item.trade_date] = item
            elif item.available_at == current.available_at and item != current:
                raise TradeabilityInputError("ambiguous calendar revisions")
        current_session = selected.get(request.session_date)
        if current_session is None or not current_session.is_open:
            raise InsufficientTradeabilityData(
                "session_date is missing or not an open calendar day"
            )
        open_sessions = tuple(item for _, item in sorted(selected.items()) if item.is_open)
        if len(open_sessions) < self._config.long_turnover_window:
            raise InsufficientTradeabilityData(
                f"requires {self._config.long_turnover_window} open calendar sessions; "
                f"received {len(open_sessions)}"
            )
        window = open_sessions[-self._config.long_turnover_window :]
        if any(item.data_version != request.data_version for item in window):
            raise TradeabilityInputError("calendar data_version does not match request")
        return window

    @staticmethod
    def _select_bars(
        request: TradeabilityRequest,
        bars: tuple[TradeabilityBar, ...],
        *,
        window_sessions: tuple[TradeabilityCalendarSession, ...],
        master: InstrumentTradeabilityState,
    ) -> tuple[TradeabilityBar, ...]:
        open_dates = {item.trade_date for item in window_sessions}
        required_dates = {value for value in open_dates if value >= master.listed_on}
        first_date = window_sessions[0].trade_date
        selected: dict[date, TradeabilityBar] = {}
        for item in bars:
            if (
                item.instrument_id != request.instrument_id
                or item.trade_date > request.session_date
            ):
                continue
            if item.observed_at > request.as_of or item.available_at > request.as_of:
                continue
            if item.trade_date < first_date:
                continue
            if item.trade_date not in open_dates or item.trade_date < master.listed_on:
                raise TradeabilityInputError(
                    f"bar date {item.trade_date} is absent from the open-session window"
                )
            current = selected.get(item.trade_date)
            if current is None or item.available_at > current.available_at:
                selected[item.trade_date] = item
            elif item.available_at == current.available_at and item != current:
                raise TradeabilityInputError("ambiguous bar revisions")
        missing = sorted(required_dates - set(selected))
        if missing:
            raise InsufficientTradeabilityData(
                f"tradeability bar window is incomplete; first missing session {missing[0]}"
            )
        ordered = tuple(selected[trade_date] for trade_date in sorted(required_dates))
        if any(item.data_version != request.data_version for item in ordered):
            raise TradeabilityInputError("bar data_version does not match request")
        return ordered

    @staticmethod
    def _average_turnover(
        bars: dict[date, TradeabilityBar],
        dates: tuple[date, ...],
        *,
        listed_on: date,
    ) -> Decimal:
        amounts = (bars[value].turnover_amount if value >= listed_on else _ZERO for value in dates)
        return sum(amounts, _ZERO) / len(dates)

    def _turnover_rate_percentile(
        self,
        bars: dict[date, TradeabilityBar],
        long_dates: tuple[date, ...],
    ) -> tuple[Decimal | None, int]:
        dates = long_dates[-self._config.turnover_rate_percentile_window :]
        current_rate = bars[long_dates[-1]].turnover_rate
        rates = tuple(
            bars[value].turnover_rate
            for value in dates
            if value in bars and bars[value].turnover_rate is not None
        )
        present = tuple(value for value in rates if value is not None)
        sufficient = (
            current_rate is not None
            and len(present) >= self._config.minimum_turnover_rate_observations
        )
        if not sufficient:
            if self._config.minimum_turnover_rate_percentile is not None:
                raise InsufficientTradeabilityData(
                    "configured turnover-rate percentile gate lacks enough PIT observations"
                )
            return None, len(present)
        assert current_rate is not None
        percentile = Decimal(sum(value <= current_rate for value in present)) / len(present)
        return percentile, len(present)

    def _market_state(
        self,
        current: TradeabilityBar,
        master: InstrumentTradeabilityState,
        rule: InstrumentTradeabilityRule,
    ) -> tuple[MarketTradeState, Decimal | None, Decimal | None]:
        if master.status is InstrumentStatus.SUSPENDED or current.is_suspended:
            return MarketTradeState.SUSPENDED, None, None
        if rule.price_limit_ratio is None:
            return MarketTradeState.NORMAL, None, None
        upper = _round_to_tick(
            current.previous_close * (_ONE + rule.price_limit_ratio),
            rule.price_tick,
        )
        lower = _round_to_tick(
            current.previous_close * (_ONE - rule.price_limit_ratio),
            rule.price_tick,
        )
        tolerance = rule.boundary_tolerance
        prices = (current.open, current.high, current.low, current.close)
        if max(prices) > upper + tolerance or min(prices) < lower - tolerance:
            raise TradeabilityInputError("OHLC prices exceed the supplied price-limit rule")
        at_upper = current.close >= upper - tolerance
        at_lower = current.close <= lower + tolerance
        if at_upper and at_lower:
            raise TradeabilityInputError("price cannot be at both upper and lower limits")
        if at_upper:
            one_price = all(abs(value - upper) <= tolerance for value in prices)
            return (
                MarketTradeState.ONE_PRICE_LIMIT_UP if one_price else MarketTradeState.LIMIT_UP,
                upper,
                lower,
            )
        if at_lower:
            one_price = all(abs(value - lower) <= tolerance for value in prices)
            return (
                MarketTradeState.ONE_PRICE_LIMIT_DOWN if one_price else MarketTradeState.LIMIT_DOWN,
                upper,
                lower,
            )
        return MarketTradeState.NORMAL, upper, lower

    def _capacity(
        self,
        request: TradeabilityRequest,
        average_short: Decimal,
        average_long: Decimal,
    ) -> CapacityEstimate:
        short_capacity = average_short * self._config.participation_rate
        long_capacity = average_long * self._config.participation_rate
        daily_capacity = min(short_capacity, long_capacity)
        capacity_ratio = daily_capacity / request.target_notional
        maximum_weight = min(daily_capacity / request.account_value, _ONE)
        estimated_days = request.target_notional / daily_capacity if daily_capacity > 0 else None
        return CapacityEstimate(
            account_value=request.account_value,
            target_position_weight=request.target_position_weight,
            target_notional=request.target_notional,
            average_turnover_short=average_short,
            average_turnover_long=average_long,
            participation_rate=self._config.participation_rate,
            short_window_capacity=short_capacity,
            long_window_capacity=long_capacity,
            daily_capacity=daily_capacity,
            capacity_ratio=capacity_ratio,
            maximum_account_weight=maximum_weight,
            estimated_trade_days=estimated_days,
        )

    def _numeric_gates(
        self,
        *,
        listing_days: int,
        average_short: Decimal,
        average_long: Decimal,
        turnover_percentile: Decimal | None,
        capacity: CapacityEstimate,
    ) -> tuple[tuple[TradeabilityContribution, ...], tuple[TradeabilityReason, ...]]:
        contributions: list[TradeabilityContribution] = []
        reasons: list[TradeabilityReason] = []

        def gate(
            *,
            code: TradeabilityContributionCode,
            value: Decimal,
            threshold: Decimal,
            description: str,
            reason_code: TradeabilityReasonCode,
            reason_message: str,
        ) -> None:
            passed = value >= threshold
            contributions.append(
                TradeabilityContribution(
                    code=code,
                    value=value,
                    threshold=threshold,
                    passed=passed,
                    blocking=not passed,
                    description=description,
                )
            )
            if not passed:
                reasons.append(
                    TradeabilityReason(
                        code=reason_code,
                        blocking=True,
                        message=reason_message,
                        value=value,
                        threshold=threshold,
                    )
                )

        gate(
            code=TradeabilityContributionCode.LISTING_DAYS,
            value=Decimal(listing_days),
            threshold=Decimal(self._config.minimum_listing_days),
            description="calendar-day listing age gate",
            reason_code=TradeabilityReasonCode.NEWLY_LISTED,
            reason_message="listing age is below the configured minimum",
        )
        gate(
            code=TradeabilityContributionCode.AVERAGE_TURNOVER_SHORT,
            value=average_short,
            threshold=self._config.minimum_average_turnover_short,
            description=f"{self._config.short_turnover_window}-session average traded amount",
            reason_code=TradeabilityReasonCode.LOW_AVERAGE_TURNOVER_20,
            reason_message="short-window average traded amount is below its floor",
        )
        gate(
            code=TradeabilityContributionCode.AVERAGE_TURNOVER_LONG,
            value=average_long,
            threshold=self._config.minimum_average_turnover_long,
            description=f"{self._config.long_turnover_window}-session average traded amount",
            reason_code=TradeabilityReasonCode.LOW_AVERAGE_TURNOVER_60,
            reason_message="long-window average traded amount is below its floor",
        )
        percentile_threshold = self._config.minimum_turnover_rate_percentile
        if turnover_percentile is not None:
            passed = (
                turnover_percentile >= percentile_threshold
                if percentile_threshold is not None
                else None
            )
            blocking = passed is False
            contributions.append(
                TradeabilityContribution(
                    code=TradeabilityContributionCode.TURNOVER_RATE_PERCENTILE,
                    value=turnover_percentile,
                    threshold=percentile_threshold,
                    passed=passed,
                    blocking=blocking,
                    description="current turnover-rate percentile within the PIT history",
                )
            )
            if blocking:
                assert percentile_threshold is not None
                reasons.append(
                    TradeabilityReason(
                        code=TradeabilityReasonCode.LOW_TURNOVER_RATE_PERCENTILE,
                        blocking=True,
                        message="turnover-rate percentile is below its configured floor",
                        value=turnover_percentile,
                        threshold=percentile_threshold,
                    )
                )
        gate(
            code=TradeabilityContributionCode.CAPACITY_RATIO,
            value=capacity.capacity_ratio,
            threshold=self._config.minimum_capacity_ratio,
            description="daily participation capacity divided by requested target notional",
            reason_code=TradeabilityReasonCode.INSUFFICIENT_CAPACITY,
            reason_message="estimated daily capacity is insufficient for the account target",
        )
        return tuple(contributions), tuple(reasons)

    @staticmethod
    def _structural_reasons(
        *,
        request: TradeabilityRequest,
        master: InstrumentTradeabilityState,
        rule: InstrumentTradeabilityRule,
        market_state: MarketTradeState,
        upper_limit: Decimal | None,
        lower_limit: Decimal | None,
    ) -> tuple[TradeabilityReason, ...]:
        reasons: list[TradeabilityReason] = []
        if request.session_date < master.listed_on:
            reasons.append(
                TradeabilityReason(
                    code=TradeabilityReasonCode.NOT_YET_LISTED,
                    blocking=True,
                    message="instrument was not listed on the requested session",
                )
            )
        if master.status is InstrumentStatus.DELISTED or (
            master.delisted_on is not None and request.session_date >= master.delisted_on
        ):
            reasons.append(
                TradeabilityReason(
                    code=TradeabilityReasonCode.DELISTED,
                    blocking=True,
                    message="instrument was delisted on the requested session",
                )
            )
        if market_state is MarketTradeState.SUSPENDED:
            reasons.append(
                TradeabilityReason(
                    code=TradeabilityReasonCode.SUSPENDED,
                    blocking=True,
                    message="instrument is suspended and cannot trade",
                )
            )
        elif market_state is not MarketTradeState.NORMAL:
            reason_code = TradeabilityReasonCode(market_state.value)
            blocking = rule.blocks(request.side, market_state)
            boundary = (
                upper_limit
                if market_state in (MarketTradeState.ONE_PRICE_LIMIT_UP, MarketTradeState.LIMIT_UP)
                else lower_limit
            )
            reasons.append(
                TradeabilityReason(
                    code=reason_code,
                    blocking=blocking,
                    message=(
                        f"current market state is {market_state.value}; "
                        f"rule {'blocks' if blocking else 'permits'} {request.side.value}"
                    ),
                    value=boundary,
                )
            )
        return tuple(reasons)

    def _input_hash(
        self,
        *,
        request: TradeabilityRequest,
        master: InstrumentTradeabilityState,
        rule: InstrumentTradeabilityRule,
        calendar: tuple[TradeabilityCalendarSession, ...],
        bars: tuple[TradeabilityBar, ...],
    ) -> str:
        return _hash(
            {
                "bars": [item.fingerprint_payload() for item in bars],
                "calendar": [item.fingerprint_payload() for item in calendar],
                "config_hash": self._config.config_hash,
                "master": master.fingerprint_payload(),
                "request": request.fingerprint_payload(),
                "rule_hash": rule.rule_hash,
            }
        )

    def _result_hash(
        self,
        *,
        request: TradeabilityRequest,
        rule: InstrumentTradeabilityRule,
        market_state: MarketTradeState,
        eligibility: TradeabilityEligibility,
        reasons: tuple[TradeabilityReason, ...],
        contributions: tuple[TradeabilityContribution, ...],
        listing_days: int,
        average_short: Decimal,
        average_long: Decimal,
        turnover_percentile: Decimal | None,
        turnover_observations: int,
        short_dates: tuple[date, ...],
        long_dates: tuple[date, ...],
        upper_limit: Decimal | None,
        lower_limit: Decimal | None,
        capacity: CapacityEstimate,
        input_hash: str,
    ) -> str:
        return _hash(
            {
                "average_turnover_long": canonical_decimal(average_long),
                "average_turnover_short": canonical_decimal(average_short),
                "capacity": _capacity_payload(capacity),
                "config_hash": self._config.config_hash,
                "contributions": [_contribution_payload(item) for item in contributions],
                "eligibility": eligibility.value,
                "feature_version": self._config.version,
                "input_hash": input_hash,
                "listing_days": listing_days,
                "long_window_dates": [item.isoformat() for item in long_dates],
                "lower_limit_price": (
                    canonical_decimal(lower_limit) if lower_limit is not None else None
                ),
                "market_state": market_state.value,
                "reasons": [_reason_payload(item) for item in reasons],
                "request": request.fingerprint_payload(),
                "rule_hash": rule.rule_hash,
                "short_window_dates": [item.isoformat() for item in short_dates],
                "turnover_rate_observation_count": turnover_observations,
                "turnover_rate_percentile": (
                    canonical_decimal(turnover_percentile)
                    if turnover_percentile is not None
                    else None
                ),
                "upper_limit_price": (
                    canonical_decimal(upper_limit) if upper_limit is not None else None
                ),
            }
        )


def _capacity_payload(value: CapacityEstimate) -> dict[str, str | None]:
    return {
        "account_value": canonical_decimal(value.account_value),
        "average_turnover_long": canonical_decimal(value.average_turnover_long),
        "average_turnover_short": canonical_decimal(value.average_turnover_short),
        "capacity_ratio": canonical_decimal(value.capacity_ratio),
        "daily_capacity": canonical_decimal(value.daily_capacity),
        "estimated_trade_days": (
            canonical_decimal(value.estimated_trade_days)
            if value.estimated_trade_days is not None
            else None
        ),
        "long_window_capacity": canonical_decimal(value.long_window_capacity),
        "maximum_account_weight": canonical_decimal(value.maximum_account_weight),
        "participation_rate": canonical_decimal(value.participation_rate),
        "short_window_capacity": canonical_decimal(value.short_window_capacity),
        "target_notional": canonical_decimal(value.target_notional),
        "target_position_weight": canonical_decimal(value.target_position_weight),
    }


def _contribution_payload(
    value: TradeabilityContribution,
) -> dict[str, str | bool | None]:
    return {
        "blocking": value.blocking,
        "code": value.code.value,
        "description": value.description,
        "passed": value.passed,
        "threshold": (canonical_decimal(value.threshold) if value.threshold is not None else None),
        "value": canonical_decimal(value.value),
    }


def _reason_payload(value: TradeabilityReason) -> dict[str, str | bool | None]:
    return {
        "blocking": value.blocking,
        "code": value.code.value,
        "message": value.message,
        "threshold": (canonical_decimal(value.threshold) if value.threshold is not None else None),
        "value": canonical_decimal(value.value) if value.value is not None else None,
    }


__all__ = ["TradeabilityAnalyzer"]
