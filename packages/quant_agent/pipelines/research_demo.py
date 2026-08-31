"""Deterministic market-regime and industry-ranking research demonstration."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from quant_agent.data.domain import IndustryMembership
from quant_agent.features.core import FeatureObservation, canonical_decimal
from quant_agent.features.industry_strength import (
    IndustryStrengthAnalyzer,
    IndustryStrengthStatus,
    PointInTimeIndustryMembership,
)
from quant_agent.features.market_breadth import (
    BreadthObservation,
    MarketBreadthAnalyzer,
)
from quant_agent.features.market_trend import MarketTrendAnalyzer
from quant_agent.regime import MarketRegimeClassifier

_TZ = ZoneInfo("Asia/Shanghai")
_START = date(2026, 1, 5)
_SESSION_COUNT = 130
_INDUSTRY_UP = "DEMO:L3:GROWTH"
_INDUSTRY_DEFENSIVE = "DEMO:L3:DEFENSIVE"


@dataclass(frozen=True, slots=True)
class ResearchDemoResult:
    """Compact, JSON-ready proof that deterministic quantitative analysis is wired."""

    session_date: date
    data_version: str
    market_regime: str
    environment_score: Decimal
    max_risk_budget: Decimal
    confidence: Decimal
    confidence_meaning: str
    trend_score: Decimal
    advance_decline_ratio: Decimal
    above_average_ratio: Decimal
    industry_ranking: tuple[tuple[int, str, Decimal], ...]
    result_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "advance_decline_ratio": canonical_decimal(self.advance_decline_ratio),
            "above_average_ratio": canonical_decimal(self.above_average_ratio),
            "confidence": canonical_decimal(self.confidence),
            "confidence_meaning": self.confidence_meaning,
            "data_version": self.data_version,
            "environment_score": canonical_decimal(self.environment_score),
            "industry_ranking": [
                {
                    "industry_id": industry_id,
                    "rank": rank,
                    "score": canonical_decimal(score),
                }
                for rank, industry_id, score in self.industry_ranking
            ],
            "market_regime": self.market_regime,
            "max_risk_budget": canonical_decimal(self.max_risk_budget),
            "result_hash": self.result_hash,
            "session_date": self.session_date.isoformat(),
            "trend_score": canonical_decimal(self.trend_score),
        }


def _session_time(session_date: date, value: time) -> datetime:
    return datetime.combine(session_date, value, tzinfo=_TZ)


def _index_series(
    index_id: str,
    *,
    start: Decimal,
    daily_step: Decimal,
) -> list[FeatureObservation]:
    return [
        FeatureObservation(
            observation_key=f"{index_id}:{session_date.isoformat()}",
            observed_at=_session_time(session_date, time(15, 0)),
            available_at=_session_time(session_date, time(15, 30)),
            value=start + daily_step * day,
            revision="demo-v1",
        )
        for day in range(_SESSION_COUNT)
        for session_date in (_START + timedelta(days=day),)
    ]


def _stock_series() -> tuple[
    tuple[str, ...],
    list[BreadthObservation],
    list[PointInTimeIndustryMembership],
]:
    specifications = (
        ("DEMO.UP.1", _INDUSTRY_UP, Decimal("20"), Decimal("0.12")),
        ("DEMO.UP.2", _INDUSTRY_UP, Decimal("25"), Decimal("0.10")),
        ("DEMO.UP.3", _INDUSTRY_UP, Decimal("30"), Decimal("0.08")),
        ("DEMO.DEF.1", _INDUSTRY_DEFENSIVE, Decimal("30"), Decimal("0.03")),
        ("DEMO.DEF.2", _INDUSTRY_DEFENSIVE, Decimal("35"), Decimal("0.02")),
        ("DEMO.DEF.3", _INDUSTRY_DEFENSIVE, Decimal("40"), Decimal("-0.01")),
    )
    observations: list[BreadthObservation] = []
    memberships: list[PointInTimeIndustryMembership] = []
    for instrument_id, industry_id, start, step in specifications:
        memberships.append(
            PointInTimeIndustryMembership(
                membership=IndustryMembership(
                    instrument_id=instrument_id,
                    industry_id=industry_id,
                    effective_from=_START,
                    source="demo",
                    version="DEMO-v1",
                ),
                level=3,
                available_at=_session_time(_START, time(18, 0)),
                revision="demo-membership-v1",
            )
        )
        for day in range(_SESSION_COUNT):
            session_date = _START + timedelta(days=day)
            observations.append(
                BreadthObservation(
                    instrument_id=instrument_id,
                    trade_date=session_date,
                    observed_at=_session_time(session_date, time(15, 0)),
                    available_at=_session_time(session_date, time(15, 30)),
                    close=start + step * day,
                    turnover=Decimal("100000000")
                    + Decimal(day * 500000)
                    + Decimal(len(instrument_id) * 10000),
                    is_active=True,
                    is_suspended=False,
                    revision="demo-v1",
                )
            )
    return tuple(item[0] for item in specifications), observations, memberships


def run_research_demo(*, data_version: str = "research_demo_v1") -> ResearchDemoResult:
    """Run a fully deterministic PIT-safe regime and industry analysis."""

    if not data_version.strip():
        raise ValueError("data_version must be non-empty")
    session_date = _START + timedelta(days=_SESSION_COUNT - 1)
    as_of = _session_time(session_date, time(18, 0))
    benchmark = _index_series("DEMO.BENCHMARK", start=Decimal(100), daily_step=Decimal("0.08"))
    index_observations = {
        "DEMO.LARGE": _index_series("DEMO.LARGE", start=Decimal(100), daily_step=Decimal("0.08")),
        "DEMO.MID": _index_series("DEMO.MID", start=Decimal(120), daily_step=Decimal("0.07")),
        "DEMO.SMALL": _index_series("DEMO.SMALL", start=Decimal(80), daily_step=Decimal("0.05")),
    }
    trend = MarketTrendAnalyzer().analyze(
        session_date=session_date,
        required_indices=tuple(index_observations),
        observations=index_observations,
        as_of=as_of,
        data_version=data_version,
    )
    instruments, stocks, memberships = _stock_series()
    expected_by_date = {_START + timedelta(days=day): instruments for day in range(_SESSION_COUNT)}
    breadth = MarketBreadthAnalyzer().analyze(
        session_date=session_date,
        as_of=as_of,
        data_version=data_version,
        expected_active_by_date=expected_by_date,
        observations=stocks,
    )
    industries = IndustryStrengthAnalyzer().analyze(
        session_date=session_date,
        as_of=as_of,
        data_version=data_version,
        classification_version="DEMO-v1",
        industry_level=3,
        industry_ids=(_INDUSTRY_UP, _INDUSTRY_DEFENSIVE),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )
    regime = MarketRegimeClassifier().classify(
        trend_snapshot=trend,
        breadth_snapshot=breadth,
    )
    advance_decline = breadth.advance_decline_ratio
    if advance_decline is None or breadth.above_average_ratio is None:
        raise AssertionError("research demo unexpectedly produced incomplete breadth")
    ranking = tuple(
        (item.rank, item.industry_id, item.score)
        for item in sorted(
            industries.industries,
            key=lambda value: value.rank or 10_000,
        )
        if item.status is IndustryStrengthStatus.READY
        and item.rank is not None
        and item.score is not None
    )
    identity_payload = {
        "data_version": data_version,
        "industry_cache_key": industries.cache_key,
        "regime_result_hash": regime.result_hash,
        "session_date": session_date.isoformat(),
    }
    result_hash = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResearchDemoResult(
        session_date=session_date,
        data_version=data_version,
        market_regime=regime.regime.value,
        environment_score=regime.score,
        max_risk_budget=regime.max_risk_budget,
        confidence=regime.confidence,
        confidence_meaning=regime.confidence_meaning.value,
        trend_score=trend.aggregate_score,
        advance_decline_ratio=advance_decline,
        above_average_ratio=breadth.above_average_ratio,
        industry_ranking=ranking,
        result_hash=result_hash,
    )
