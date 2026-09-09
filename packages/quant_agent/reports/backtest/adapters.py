"""Adapters from backtest-engine outputs to normalized report inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from decimal import Decimal, localcontext
from enum import Enum
from typing import Any

from quant_agent.backtest import VectorizedBacktestResult
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.reports.backtest.contracts import (
    BacktestDailyInput,
    BacktestReportInputError,
    SessionRegimeAttribution,
)


def _vector_canonical(value: Any) -> Any:
    """Mirror the vector engine's exact high-precision hash encoding."""

    if isinstance(value, datetime):
        ensure_aware(value)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("cannot hash a non-finite Decimal")
        if value == 0:
            return "0"
        rendered = format(value, "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _vector_canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_vector_canonical(item) for item in value]
    return value


def _vector_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _vector_canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vectorized_daily_inputs(
    result: VectorizedBacktestResult,
    regimes: Mapping[date, SessionRegimeAttribution],
) -> tuple[BacktestDailyInput, ...]:
    """Add daily fees back before compounding to produce a gross counterfactual path."""

    expected_result_hash = _vector_hash(
        {
            "benchmark_id": result.benchmark_id,
            "input_hash": result.input_hash,
            "metrics": asdict(result.metrics),
            "points": [asdict(item) for item in result.points],
            "request": asdict(result.request),
            "trades": [asdict(item) for item in result.trades],
        }
    )
    if result.result_hash != expected_result_hash:
        raise BacktestReportInputError("vectorized result_hash does not match its economic output")
    expected_dates = tuple(point.trading_day for point in result.points)
    if set(regimes) != set(expected_dates):
        raise BacktestReportInputError(
            "vectorized report regimes must cover the result calendar exactly"
        )
    previous_net_nav = result.request.initial_nav
    gross_nav = result.request.initial_nav
    normalized: list[BacktestDailyInput] = []
    for point in result.points:
        with localcontext() as context:
            context.prec = 50
            gross_daily_growth = (point.nav + point.total_fees) / previous_net_nav
            gross_nav *= gross_daily_growth
        normalized.append(
            BacktestDailyInput(
                trading_day=point.trading_day,
                available_at=datetime.combine(
                    point.trading_day,
                    time(16),
                    tzinfo=SHANGHAI_TZ,
                ),
                gross_nav=gross_nav,
                net_nav=point.nav,
                benchmark_nav=point.benchmark_nav,
                turnover=point.turnover,
                transaction_cost=point.total_fees,
                regime_attribution=regimes[point.trading_day],
            )
        )
        previous_net_nav = point.nav
    return tuple(normalized)


__all__ = ["vectorized_daily_inputs"]
