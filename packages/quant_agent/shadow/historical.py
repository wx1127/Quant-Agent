"""Strict point-in-time historical shadow analysis over unadjusted daily bars."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import cast
from zoneinfo import ZoneInfo

from quant_agent.data.domain import DailyBar
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.shadow.models import ShadowDayEvidence, ShadowRunMode

_SHANGHAI = ZoneInfo("Asia/Shanghai")
CandidateExclusion = Callable[[DailyBar, list[DailyBar]], str | None]
SegmentResolver = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class HistoricalReplayResult:
    evidence: ShadowDayEvidence
    details: dict[str, object]


class HistoricalShadowEngine:
    """Replay one day using only records available by its virtual close-time clock."""

    def __init__(
        self,
        *,
        minimum_daily_instruments: int = 1000,
        candidate_exclusion: CandidateExclusion | None = None,
        segment_resolver: SegmentResolver | None = None,
    ) -> None:
        if minimum_daily_instruments < 1:
            raise ValueError("minimum daily instruments must be positive")
        self._minimum_daily_instruments = minimum_daily_instruments
        self._candidate_exclusion = candidate_exclusion
        self._segment_resolver = segment_resolver

    def run_day(self, trading_date: date, bars: tuple[DailyBar, ...]) -> HistoricalReplayResult:
        observed_at = datetime.combine(trading_date, time(16, 5), tzinfo=_SHANGHAI)
        eligible = tuple(
            sorted(
                (
                    item
                    for item in bars
                    if item.trade_date <= trading_date and item.available_at <= observed_at
                ),
                key=lambda item: (item.trade_date, item.instrument_id),
            )
        )
        current = tuple(item for item in eligible if item.trade_date == trading_date)
        current_ids = [item.instrument_id for item in current]
        duplicates = len(current_ids) - len(set(current_ids))
        if not current:
            raise ValueError(f"no point-in-time daily bars for {trading_date}")

        history: dict[str, list[DailyBar]] = {}
        for item in eligible:
            history.setdefault(item.instrument_id, []).append(item)
        regime, regime_details = _market_regime(current, history)
        mainlines, segment_details = _mainlines(current, history, self._segment_resolver)
        candidates, candidate_details, candidate_exclusions = _candidates(
            current,
            history,
            set(mainlines),
            self._candidate_exclusion,
            self._segment_resolver,
        )

        input_hash = _bars_hash(eligible)
        account = AccountSnapshot(
            snapshot_id=f"historical-shadow-{trading_date}",
            account_id="shadow-virtual-1",
            as_of=observed_at,
            available_cash=1_000_000.0,
            frozen_cash=0.0,
            holdings=(),
            source="historical-shadow",
            version="historical-shadow-account-v1",
        )
        data_complete = (
            len(current) >= self._minimum_daily_instruments
            and duplicates == 0
            and all(item.available_at <= observed_at for item in eligible)
        )
        excluded_future_rows = len(bars) - len(eligible)
        data_version = f"tushare-pit-{trading_date}-{input_hash[:12]}"
        segment_note = (
            "mainlines use point-in-time supplied industries"
            if self._segment_resolver is not None
            else "mainlines are stable exchange-board segments"
        )
        evidence = ShadowDayEvidence(
            trading_date=trading_date,
            observed_at=observed_at,
            market_data_as_of=max(item.available_at for item in eligible),
            input_snapshot_hash=input_hash,
            virtual_account_snapshot_hash=account.content_hash,
            data_version=data_version,
            regime=regime,
            mainline_ids=mainlines,
            candidate_ids=candidates,
            data_complete=data_complete,
            report_generated=True,
            pipeline_succeeded=data_complete,
            reconciliation_matched=True,
            future_data_violations=0,
            executable_orders_emitted=0,
            manual_intervention_minutes=0,
            notes=(
                "historical point-in-time replay; virtual clock 16:05 Asia/Shanghai; "
                f"unadjusted daily bars only; {segment_note}; "
                f"future rows excluded by query={excluded_future_rows}"
            ),
            run_mode=ShadowRunMode.HISTORICAL_POINT_IN_TIME,
        )
        return HistoricalReplayResult(
            evidence=evidence,
            details={
                "trading_date": trading_date.isoformat(),
                "virtual_clock": observed_at.isoformat(),
                "market_data_as_of": evidence.market_data_as_of.isoformat(),
                "eligible_history_rows": len(eligible),
                "daily_instruments": len(current),
                "duplicate_daily_instruments": duplicates,
                "excluded_future_rows": excluded_future_rows,
                "input_snapshot_hash": input_hash,
                "virtual_account_snapshot_hash": account.content_hash,
                "regime": regime_details,
                "mainlines": segment_details,
                "candidates": candidate_details,
                "candidate_exclusions": candidate_exclusions,
                "limitations": [
                    (
                        "historical replay does not validate real-time provider latency "
                        "or scheduler uptime"
                    ),
                    (
                        "industry history is unavailable for this credential; board segments "
                        "prevent membership look-ahead"
                    ),
                    "provider historical revisions cannot be reconstructed from the daily endpoint",
                ],
            },
        )


def _bars_hash(bars: tuple[DailyBar, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in bars]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _market_regime(
    current: tuple[DailyBar, ...],
    history: dict[str, list[DailyBar]],
) -> tuple[str, dict[str, object]]:
    returns: list[float] = []
    above_ma20 = 0
    ma20_count = 0
    for item in current:
        series = history[item.instrument_id]
        if len(series) >= 2:
            returns.append(float(series[-1].close / series[-2].close - 1))
        if len(series) >= 20:
            closes = [float(value.close) for value in series[-20:]]
            above_ma20 += closes[-1] > statistics.fmean(closes)
            ma20_count += 1
    advancing_ratio = sum(value > 0 for value in returns) / len(returns) if returns else 0.0
    above_ratio = above_ma20 / ma20_count if ma20_count else 0.0
    score = 100 * (0.55 * advancing_ratio + 0.45 * above_ratio)
    if score >= 65:
        regime = "UPTREND"
    elif score >= 55:
        regime = "RANGE_STRONG"
    elif score >= 45:
        regime = "DIVERGENT"
    else:
        regime = "DOWNTREND"
    return regime, {
        "name": regime,
        "score": round(score, 6),
        "advancing_ratio": round(advancing_ratio, 6),
        "above_ma20_ratio": round(above_ratio, 6),
        "return_observations": len(returns),
        "ma20_observations": ma20_count,
    }


def _segment(instrument_id: str) -> str:
    _country, exchange, symbol = instrument_id.split(".", maxsplit=2)
    if exchange == "SH" and symbol.startswith("688"):
        return "BOARD.STAR"
    if exchange == "SZ" and symbol.startswith(("300", "301")):
        return "BOARD.CHINEXT"
    if exchange == "SH":
        return "BOARD.SH_MAIN"
    if exchange == "SZ":
        return "BOARD.SZ_MAIN"
    return f"BOARD.{exchange}"


def _five_day_return(series: list[DailyBar]) -> float | None:
    if len(series) < 6:
        return None
    return float(series[-1].close / series[-6].close - 1)


def _mainlines(
    current: tuple[DailyBar, ...],
    history: dict[str, list[DailyBar]],
    segment_resolver: SegmentResolver | None = None,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for item in current:
        return_5d = _five_day_return(history[item.instrument_id])
        segment = (
            segment_resolver(item.instrument_id)
            if segment_resolver is not None
            else _segment(item.instrument_id)
        )
        if return_5d is not None and segment is not None:
            grouped.setdefault(segment, []).append(
                (return_5d, float(item.turnover))
            )
    rows: list[dict[str, object]] = []
    for segment, values in grouped.items():
        returns = [item[0] for item in values]
        turnover = sum(item[1] for item in values)
        rows.append(
            {
                "segment": segment,
                "median_return_5d": statistics.median(returns),
                "turnover": turnover,
                "members": len(values),
            }
        )
    rows.sort(
        key=lambda item: (
            cast(float, item["median_return_5d"]),
            cast(float, item["turnover"]),
            str(item["segment"]),
        ),
        reverse=True,
    )
    selected = tuple(str(item["segment"]) for item in rows[:2])
    return selected, rows


def _candidates(
    current: tuple[DailyBar, ...],
    history: dict[str, list[DailyBar]],
    mainlines: set[str],
    exclusion: CandidateExclusion | None = None,
    segment_resolver: SegmentResolver | None = None,
) -> tuple[tuple[str, ...], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    max_turnover = max((float(item.turnover) for item in current), default=1.0)
    for item in current:
        series = history[item.instrument_id]
        segment = (
            segment_resolver(item.instrument_id)
            if segment_resolver is not None
            else _segment(item.instrument_id)
        )
        if segment not in mainlines or len(series) < 21:
            continue
        reason = exclusion(item, series) if exclusion is not None else None
        if reason is not None:
            excluded.append({"instrument_id": item.instrument_id, "reason": reason})
            continue
        return_5d = float(series[-1].close / series[-6].close - 1)
        return_20d = float(series[-1].close / series[-21].close - 1)
        turnover_score = float(item.turnover) / max_turnover if max_turnover else 0.0
        score = 0.6 * return_20d + 0.3 * return_5d + 0.1 * turnover_score
        rows.append(
            {
                "instrument_id": item.instrument_id,
                "segment": segment,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "turnover": float(item.turnover),
                "score": score,
            }
        )
    rows.sort(
        key=lambda item: (cast(float, item["score"]), str(item["instrument_id"])),
        reverse=True,
    )
    selected_rows = rows[:10]
    excluded.sort(key=lambda item: str(item["instrument_id"]))
    return (
        tuple(str(item["instrument_id"]) for item in selected_rows),
        selected_rows,
        excluded,
    )
