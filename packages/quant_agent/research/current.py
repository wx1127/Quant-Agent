"""Build a current industry-mainline snapshot from available stock data."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from quant_agent.core.time import ensure_aware
from quant_agent.data.domain import DailyBar


@dataclass(frozen=True, slots=True)
class StockProfile:
    """Current stock name and industry classification with an explicit availability time."""

    instrument_id: str
    name: str
    industry: str
    listed_on: date
    source: str
    available_at: datetime
    version: str

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if not self.instrument_id or not self.name or not self.industry:
            raise ValueError("stock profile id, name and industry are required")


class CurrentIndustryResearchEngine:
    """Rank real industries and their stocks without pretending they are probabilities."""

    def __init__(self, *, minimum_industry_members: int = 10, theme_limit: int = 8) -> None:
        if minimum_industry_members < 2:
            raise ValueError("minimum industry members must be at least two")
        if theme_limit < 1:
            raise ValueError("theme limit must be positive")
        self._minimum_industry_members = minimum_industry_members
        self._theme_limit = theme_limit

    def build(
        self,
        *,
        market_date: date,
        generated_at: datetime,
        bars: tuple[DailyBar, ...],
        profiles: tuple[StockProfile, ...],
    ) -> dict[str, Any]:
        ensure_aware(generated_at)
        eligible_bars = tuple(
            item
            for item in bars
            if item.trade_date <= market_date and item.available_at <= generated_at
        )
        eligible_profiles = tuple(item for item in profiles if item.available_at <= generated_at)
        if not eligible_bars or not eligible_profiles:
            raise ValueError("current research requires available bars and stock profiles")
        history: dict[str, list[DailyBar]] = {}
        for item in sorted(eligible_bars, key=lambda row: (row.instrument_id, row.trade_date)):
            history.setdefault(item.instrument_id, []).append(item)
        profile_by_id = {item.instrument_id: item for item in eligible_profiles}
        current = {
            instrument_id: series[-1]
            for instrument_id, series in history.items()
            if series[-1].trade_date == market_date and instrument_id in profile_by_id
        }
        if not current:
            raise ValueError("no classified stock bars exist for the selected market date")

        regime = _regime(current, history)
        theme_inputs = _theme_inputs(current, history, profile_by_id)
        eligible_themes = [
            item
            for item in theme_inputs
            if int(item["member_count"]) >= self._minimum_industry_members
        ]
        _score_themes(eligible_themes)
        eligible_themes.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
        themes = eligible_themes[: self._theme_limit]
        theme_members: list[dict[str, Any]] = []
        leaders: list[dict[str, Any]] = []
        for theme_rank, theme in enumerate(themes, start=1):
            theme["rank"] = theme_rank
            theme["state"] = "CONFIRMED" if theme_rank <= 3 else "WATCH"
            members = _stock_scores(
                str(theme["id"]),
                str(theme["name"]),
                theme_rank,
                tuple(theme.pop("instrument_ids")),
                current,
                history,
                profile_by_id,
            )
            theme["scored_member_count"] = len(members)
            theme_members.extend(members)
            leaders.extend(members[:3])

        candidates = sorted(
            (member for member in theme_members if int(member["theme_rank"]) <= 3),
            key=lambda member: (-float(member["score"]), str(member["instrument_id"])),
        )[:20]
        candidate_ids = {str(member["instrument_id"]) for member in candidates}
        for member in theme_members:
            member["selected_candidate"] = str(member["instrument_id"]) in candidate_ids
        evidence = [_stock_evidence(member) for member in theme_members]
        stock_details = stock_details_from_members(theme_members, history)
        canonical_input = {
            "market_date": market_date.isoformat(),
            "bar_versions": sorted({bar.version for bar in eligible_bars}),
            "profile_version": sorted({profile.version for profile in eligible_profiles}),
            "themes": [(theme["id"], theme["score"]) for theme in themes],
        }
        data_hash = hashlib.sha256(
            json.dumps(canonical_input, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "schema_version": "current-industry-research-v1",
            "as_of": generated_at.isoformat(),
            "market_as_of": market_date.isoformat(),
            "data_version": f"current-industry-{market_date}-{data_hash[:12]}",
            "classification": "TUSHARE_STOCK_BASIC_INDUSTRY",
            "regime": regime,
            "themes": themes,
            "theme_members": theme_members,
            "leaders": leaders,
            "candidates": candidates,
            "evidence": evidence,
            "stock_details": stock_details,
        }


def stock_details_from_members(
    theme_members: list[dict[str, Any]],
    history: dict[str, list[DailyBar]],
    *,
    bar_limit: int = 25,
) -> list[dict[str, Any]]:
    """Create compact, serializable stock details from point-in-time daily bars."""
    if bar_limit < 2:
        raise ValueError("stock detail requires at least two bars")
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for member in theme_members:
        instrument_id = str(member["instrument_id"])
        if instrument_id in seen or instrument_id not in history:
            continue
        seen.add(instrument_id)
        series = history[instrument_id][-bar_limit:]
        if not series:
            continue
        latest = series[-1]
        previous_close = series[-2].close if len(series) >= 2 else latest.close
        change = latest.close - previous_close
        change_pct = float(change / previous_close) if previous_close else 0.0
        details.append(
            {
                "instrument_id": instrument_id,
                "symbol": member.get("symbol", instrument_id.rsplit(".", maxsplit=1)[-1]),
                "name": member.get("name", instrument_id),
                "theme_id": member.get("theme_id"),
                "theme_name": member.get("theme_name"),
                "trade_date": latest.trade_date.isoformat(),
                "latest_price": float(latest.close),
                "previous_close": float(previous_close),
                "change": round(float(change), 4),
                "change_pct": round(change_pct, 6),
                "open": float(latest.open),
                "high": float(latest.high),
                "low": float(latest.low),
                "close": float(latest.close),
                "volume": float(latest.volume),
                "turnover": float(latest.turnover),
                "bars": [
                    {
                        "trade_date": bar.trade_date.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "turnover": float(bar.turnover),
                    }
                    for bar in series
                ],
            }
        )
    return details


def _returns(series: list[DailyBar]) -> tuple[float, float] | None:
    if len(series) < 21:
        return None
    return (
        float(series[-1].close / series[-6].close - 1),
        float(series[-1].close / series[-21].close - 1),
    )


def _regime(current: dict[str, DailyBar], history: dict[str, list[DailyBar]]) -> dict[str, Any]:
    daily_returns: list[float] = []
    above_ma20 = 0
    ma20_count = 0
    for instrument_id in current:
        series = history[instrument_id]
        if len(series) >= 2:
            daily_returns.append(float(series[-1].close / series[-2].close - 1))
        if len(series) >= 20:
            closes = [float(item.close) for item in series[-20:]]
            above_ma20 += closes[-1] > statistics.fmean(closes)
            ma20_count += 1
    advancing = sum(item > 0 for item in daily_returns) / len(daily_returns)
    above = above_ma20 / ma20_count if ma20_count else 0.0
    score = 100 * (0.55 * advancing + 0.45 * above)
    name = (
        "UPTREND"
        if score >= 65
        else "RANGE_STRONG"
        if score >= 55
        else "DIVERGENT"
        if score >= 45
        else "DOWNTREND"
    )
    support: list[str] = []
    counter: list[str] = []
    (support if advancing >= 0.5 else counter).append(
        "上涨家数占比高于一半" if advancing >= 0.5 else "上涨家数占比不足一半"
    )
    (support if above >= 0.5 else counter).append(
        "多数股票位于20日均线上方" if above >= 0.5 else "多数股票位于20日均线下方"
    )
    return {
        "name": name,
        "score": round(score, 2),
        "advancing_ratio": round(advancing, 4),
        "above_ma20_ratio": round(above, 4),
        "support_evidence": support,
        "counter_evidence": counter,
    }


def _theme_inputs(
    current: dict[str, DailyBar],
    history: dict[str, list[DailyBar]],
    profiles: dict[str, StockProfile],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, float, float, float, bool]]] = {}
    for instrument_id, bar in current.items():
        return_pair = _returns(history[instrument_id])
        if return_pair is None:
            continue
        return_5d, return_20d = return_pair
        series = history[instrument_id]
        advancing = series[-1].close > series[-2].close
        grouped.setdefault(profiles[instrument_id].industry, []).append(
            (instrument_id, return_5d, return_20d, float(bar.turnover), advancing)
        )
    rows: list[dict[str, Any]] = []
    for industry, industry_members in grouped.items():
        industry_id = "industry-" + hashlib.sha256(industry.encode()).hexdigest()[:12]
        rows.append(
            {
                "id": industry_id,
                "name": industry,
                "kind": "INDUSTRY",
                "member_count": len(industry_members),
                "median_return_5d": statistics.median(row[1] for row in industry_members),
                "median_return_20d": statistics.median(row[2] for row in industry_members),
                "advancing_ratio": sum(row[4] for row in industry_members) / len(industry_members),
                "turnover": sum(row[3] for row in industry_members),
                "instrument_ids": [row[0] for row in industry_members],
                "persistence_days": 1,
            }
        )
    return rows


def _normalize(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(item[field]) for item in rows]
    low, high = min(values), max(values)
    if high == low:
        return {str(item["id"]): 50.0 for item in rows}
    return {str(item["id"]): 100 * (float(item[field]) - low) / (high - low) for item in rows}


def _score_themes(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no industry has enough scored members")
    return_5d = _normalize(rows, "median_return_5d")
    return_20d = _normalize(rows, "median_return_20d")
    turnover = _normalize(rows, "turnover")
    for item in rows:
        theme_id = str(item["id"])
        item["score"] = round(
            0.35 * return_5d[theme_id]
            + 0.25 * return_20d[theme_id]
            + 25 * float(item["advancing_ratio"])
            + 0.15 * turnover[theme_id],
            2,
        )


def _stock_scores(
    theme_id: str,
    theme_name: str,
    theme_rank: int,
    instrument_ids: tuple[str, ...],
    current: dict[str, DailyBar],
    history: dict[str, list[DailyBar]],
    profiles: dict[str, StockProfile],
) -> list[dict[str, Any]]:
    raw = []
    max_turnover = max(float(current[item].turnover) for item in instrument_ids)
    for instrument_id in instrument_ids:
        return_5d, return_20d = _returns(history[instrument_id]) or (0.0, 0.0)
        turnover = float(current[instrument_id].turnover)
        raw_score = 0.55 * return_20d + 0.30 * return_5d + 0.15 * turnover / max_turnover
        raw.append((instrument_id, return_5d, return_20d, turnover, raw_score))
    raw.sort(key=lambda item: (-item[4], item[0]))
    scores = [item[4] for item in raw]
    low, high = min(scores), max(scores)
    rows = []
    for rank, (instrument_id, return_5d, return_20d, turnover, raw_score) in enumerate(
        raw, start=1
    ):
        score = 50.0 if high == low else 100 * (raw_score - low) / (high - low)
        rows.append(
            {
                "instrument_id": instrument_id,
                "symbol": instrument_id.rsplit(".", maxsplit=1)[-1],
                "name": profiles[instrument_id].name,
                "theme_id": theme_id,
                "theme_name": theme_name,
                "theme_rank": theme_rank,
                "rank": rank,
                "role": "LEADER" if rank <= 3 else "CANDIDATE",
                "leader_type": "核心龙头" if rank == 1 else "趋势龙头" if rank <= 3 else "候选",
                "score": round(score, 2),
                "return_5d": round(return_5d, 6),
                "return_20d": round(return_20d, 6),
                "turnover": round(turnover, 2),
                "tradable": True,
            }
        )
    return rows


def _stock_evidence(item: dict[str, Any]) -> dict[str, Any]:
    support = [
        f"行业内综合排名第 {item['rank']} 名",
        f"5日收益 {float(item['return_5d']):.2%}",
        f"20日收益 {float(item['return_20d']):.2%}",
    ]
    counter = []
    if float(item["return_5d"]) < 0:
        counter.append("5日动量已经转弱")
    if not counter:
        counter.append("量化排名不代表未来收益概率")
    return {
        "instrument_id": item["instrument_id"],
        "name": item["name"],
        "theme_id": item["theme_id"],
        "theme_name": item["theme_name"],
        "score": item["score"],
        "support_evidence": support,
        "counter_evidence": counter,
        "invalidations": [
            "所属行业退出领先主线",
            "5日或20日动量跌破策略阈值",
            "停牌、涨跌停或数据完整性门禁失败",
        ],
    }
