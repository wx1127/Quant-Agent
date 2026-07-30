"""Point-in-time fundamental quality and risk features."""

import statistics
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import clamp_score


@dataclass(frozen=True, slots=True)
class FundamentalMetric:
    instrument_id: str
    report_period: date
    metric_name: str
    metric_value: Decimal
    announced_at: datetime
    available_at: datetime
    source: str
    revision: str

    def __post_init__(self) -> None:
        ensure_aware(self.announced_at)
        ensure_aware(self.available_at)
        if self.available_at < self.announced_at:
            raise ValueError("available_at cannot precede announced_at")


@dataclass(frozen=True, slots=True)
class FundamentalFeatureResult:
    as_of: datetime
    instrument_id: str
    quality_score: float | None
    risk_penalty: float
    missing_metrics: tuple[str, ...]
    risk_flags: tuple[str, ...]
    source_records: tuple[str, ...]
    data_version: str
    feature_version: str


class FundamentalFeatureEngine:
    feature_version = "fundamental_quality_v1"
    required_metrics = (
        "roe",
        "revenue_growth",
        "operating_cash_flow_ratio",
        "debt_ratio",
        "profit_stability",
    )

    def calculate(
        self,
        metrics: list[FundamentalMetric],
        *,
        instrument_id: str,
        as_of: datetime,
        data_version: str,
    ) -> FundamentalFeatureResult:
        ensure_aware(as_of)
        available = [
            item
            for item in metrics
            if item.instrument_id == instrument_id and item.available_at <= as_of
        ]
        latest: dict[str, FundamentalMetric] = {}
        for item in sorted(
            available,
            key=lambda metric: (
                metric.report_period,
                metric.available_at,
                metric.revision,
            ),
        ):
            latest[item.metric_name] = item
        missing = tuple(name for name in self.required_metrics if name not in latest)
        scores: list[float] = []
        if "roe" in latest:
            scores.append(clamp_score(float(latest["roe"].metric_value), 0.0, 0.20))
        if "revenue_growth" in latest:
            scores.append(clamp_score(float(latest["revenue_growth"].metric_value), -0.20, 0.30))
        if "operating_cash_flow_ratio" in latest:
            scores.append(
                clamp_score(float(latest["operating_cash_flow_ratio"].metric_value), 0.0, 1.5)
            )
        if "debt_ratio" in latest:
            scores.append(100 - clamp_score(float(latest["debt_ratio"].metric_value), 0.30, 0.85))
        if "profit_stability" in latest:
            scores.append(clamp_score(float(latest["profit_stability"].metric_value), 0.0, 1.0))
        flags: list[str] = []
        if "roe" in latest and latest["roe"].metric_value < 0:
            flags.append("negative return on equity")
        if (
            "operating_cash_flow_ratio" in latest
            and latest["operating_cash_flow_ratio"].metric_value < 0
        ):
            flags.append("negative operating cash flow")
        if "debt_ratio" in latest and latest["debt_ratio"].metric_value > Decimal("0.85"):
            flags.append("high leverage")
        risk_penalty = min(40.0, len(flags) * 15.0 + len(missing) * 4.0)
        return FundamentalFeatureResult(
            as_of=as_of,
            instrument_id=instrument_id,
            quality_score=statistics.fmean(scores) if not missing else None,
            risk_penalty=risk_penalty,
            missing_metrics=missing,
            risk_flags=tuple(flags),
            source_records=tuple(
                f"{item.source}:{item.report_period}:{item.metric_name}:{item.revision}"
                for item in sorted(latest.values(), key=lambda metric: metric.metric_name)
            ),
            data_version=data_version,
            feature_version=self.feature_version,
        )
