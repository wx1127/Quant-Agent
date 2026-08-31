"""Deterministic ETF target adapters for vectorized and event-driven backtests."""

from datetime import date, datetime
from decimal import Decimal

from quant_agent.backtest import (
    SignalDirection,
    SignalEvent,
    TargetEvent,
    WeightSignal,
)
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.etf_rotation.contracts import (
    ETFEventAdaptation,
    ETFRotationDecision,
    ETFRotationDecisionStatus,
)


def to_weight_signals(decision: ETFRotationDecision) -> tuple[WeightSignal, ...]:
    """Adapt a complete ETF target snapshot for next-session vectorized execution."""

    if decision.status is ETFRotationDecisionStatus.NO_REBALANCE:
        return ()
    return tuple(
        WeightSignal(
            instrument_id=item.instrument_id,
            instrument_type=item.instrument_type,
            signal_date=decision.signal_date,
            as_of=decision.as_of,
            target_weight=item.target_weight,
            data_version=decision.data_version,
            strategy_version=decision.strategy_version,
        )
        for item in decision.targets
    )


def _event_id(
    *,
    decision: ETFRotationDecision,
    run_id: str,
    kind: str,
    instrument_id: str,
) -> str:
    digest = stable_hash(
        {
            "decision_result_hash": decision.result_hash,
            "instrument_id": instrument_id,
            "kind": kind,
            "run_id": run_id,
        }
    )
    return f"etf-rotation-{kind.lower()}-{digest}"


def to_event_targets(
    decision: ETFRotationDecision,
    *,
    run_id: str,
    next_trading_day: date,
    execution_time: datetime,
    starting_sequence: int = 0,
) -> ETFEventAdaptation:
    """Create signal events now and weight targets at the next trading session."""

    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    ensure_aware(execution_time)
    if execution_time.astimezone(SHANGHAI_TZ).date() != next_trading_day:
        raise ValueError("execution_time must fall on next_trading_day")
    if next_trading_day <= decision.signal_date or execution_time <= decision.as_of:
        raise ValueError("ETF event targets must execute after the signal session")
    if (
        not isinstance(starting_sequence, int)
        or isinstance(starting_sequence, bool)
        or starting_sequence < 0
    ):
        raise ValueError("starting_sequence must be a non-negative integer")
    if decision.status is ETFRotationDecisionStatus.NO_REBALANCE:
        return ETFEventAdaptation(signal_events=(), target_events=())

    signal_events: list[SignalEvent] = []
    target_events: list[TargetEvent] = []
    target_count = len(decision.targets)
    for index, target in enumerate(decision.targets):
        direction = SignalDirection.LONG if target.target_weight > 0 else SignalDirection.FLAT
        signal_id = _event_id(
            decision=decision,
            run_id=run_id,
            kind="SIGNAL",
            instrument_id=target.instrument_id,
        )
        signal = SignalEvent(
            event_id=signal_id,
            run_id=run_id,
            sequence=starting_sequence + index,
            event_time=decision.as_of,
            trading_day=decision.signal_date,
            instrument_id=target.instrument_id,
            instrument_type=target.instrument_type,
            direction=direction,
            strength=(target.target_weight if direction is SignalDirection.LONG else Decimal(0)),
            strategy_version=decision.strategy_version,
            data_version=decision.data_version,
        )
        target_event = TargetEvent(
            event_id=_event_id(
                decision=decision,
                run_id=run_id,
                kind="TARGET",
                instrument_id=target.instrument_id,
            ),
            run_id=run_id,
            sequence=starting_sequence + target_count + index,
            event_time=execution_time,
            trading_day=next_trading_day,
            instrument_id=target.instrument_id,
            instrument_type=target.instrument_type,
            signal_event_id=signal.event_id,
            signal_time=signal.event_time,
            direction=direction,
            target_weight=target.target_weight,
        )
        signal_events.append(signal)
        target_events.append(target_event)
    return ETFEventAdaptation(
        signal_events=tuple(signal_events),
        target_events=tuple(target_events),
    )


__all__ = ["to_event_targets", "to_weight_signals"]
