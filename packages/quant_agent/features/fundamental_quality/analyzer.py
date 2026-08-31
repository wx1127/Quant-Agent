"""Point-in-time fundamental revision selection, quality scoring, and risk flags."""

from collections.abc import Iterable
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal, localcontext

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.data.domain import FundamentalPoint
from quant_agent.features.fundamental_quality.contracts import (
    FundamentalComponent,
    FundamentalComponentResult,
    FundamentalEvidence,
    FundamentalEvidenceSide,
    FundamentalFlag,
    FundamentalFlagCode,
    FundamentalFlagKind,
    FundamentalQualityConfig,
    FundamentalQualityInputError,
    FundamentalQualityRequest,
    FundamentalQualitySnapshot,
    FundamentalQualityStatus,
    FundamentalSourceReference,
    stable_fundamental_hash,
)


def _clamp(value: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    return max(minimum, min(maximum, value))


def _divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def _higher_is_better(value: Decimal, zero: Decimal, full: Decimal) -> Decimal:
    return _clamp(_divide(value - zero, full - zero) * Decimal(100), Decimal(0), Decimal(100))


def _lower_is_better(value: Decimal, full: Decimal, zero: Decimal) -> Decimal:
    return _clamp(_divide(zero - value, zero - full) * Decimal(100), Decimal(0), Decimal(100))


def _source_reference(point: FundamentalPoint) -> FundamentalSourceReference:
    return FundamentalSourceReference(
        metric_name=point.metric_name,
        metric_value=point.metric_value,
        report_period=point.report_period,
        announced_at=point.announced_at,
        available_at=point.available_at,
        provider_revision=point.provider_revision,
        source=point.source,
    )


def _point_fingerprint(point: FundamentalPoint) -> dict[str, str]:
    return _source_reference(point).fingerprint_payload()


def _source_sort_key(
    source: FundamentalSourceReference,
) -> tuple[date, str, datetime, str, str]:
    return (
        source.report_period,
        source.metric_name,
        source.available_at,
        source.source,
        source.provider_revision,
    )


class FundamentalQualityAnalyzer:
    """Score one instrument from revisions legally available at a decision time."""

    def __init__(self, config: FundamentalQualityConfig | None = None) -> None:
        self._config = config or FundamentalQualityConfig()

    @property
    def config(self) -> FundamentalQualityConfig:
        """Return the immutable metric mapping and scoring policy."""

        return self._config

    def analyze(
        self,
        *,
        instrument_id: str,
        as_of: datetime,
        data_version: str,
        points: Iterable[FundamentalPoint],
    ) -> FundamentalQualitySnapshot:
        """Evaluate a direct instrument/as-of request using immutable PIT revisions."""

        return self.evaluate(
            request=FundamentalQualityRequest(
                instrument_id=instrument_id,
                as_of=as_of,
                data_version=data_version,
            ),
            points=points,
        )

    def evaluate(
        self,
        *,
        request: FundamentalQualityRequest,
        points: Iterable[FundamentalPoint],
    ) -> FundamentalQualitySnapshot:
        """Select latest eligible revisions and return an auditable quality snapshot."""

        revisions = tuple(points)
        self._validate_points(revisions, request)
        latest = self._latest_revisions(revisions, request)
        current_report_period = max(
            (report_period for report_period, _metric_name in latest),
            default=None,
        )

        profitability = self._latest_semantic(
            latest,
            self._config.metric_mapping.profitability,
            report_period=current_report_period,
        )
        cash_flow = self._latest_semantic(
            latest,
            self._config.metric_mapping.operating_cash_flow,
            report_period=current_report_period,
        )
        leverage = self._latest_semantic(
            latest,
            self._config.metric_mapping.leverage,
            report_period=current_report_period,
        )
        growth_points = self._growth_history(latest)

        components = self._components(
            profitability=profitability,
            cash_flow=cash_flow,
            leverage=leverage,
            growth_points=growth_points,
        )
        supporting, opposing = self._evidence(components)
        flags = self._flags(
            components=components,
            profitability=profitability,
            cash_flow=cash_flow,
            leverage=leverage,
            growth_points=growth_points,
        )
        quality_score = sum((item.contribution for item in components), Decimal(0))
        risk_penalty = min(
            self._config.maximum_risk_penalty,
            sum((item.penalty for item in flags), Decimal(0)),
        )
        adjusted_score = max(Decimal(0), quality_score - risk_penalty)
        missing_count = sum(item.missing for item in components)
        status = (
            FundamentalQualityStatus.READY
            if missing_count == 0
            else (
                FundamentalQualityStatus.INSUFFICIENT_DATA
                if missing_count == len(components)
                else FundamentalQualityStatus.PARTIAL
            )
        )
        selected_sources = self._selected_sources(components)
        input_hash = stable_fundamental_hash(
            {
                "instrument_id": request.instrument_id,
                "selected_sources": [item.fingerprint_payload() for item in selected_sources],
            }
        )
        cache_key = stable_fundamental_hash(
            {
                "config_hash": self._config.config_hash,
                "input_hash": input_hash,
                "request": request.fingerprint_payload(),
            }
        )
        result_hash = stable_fundamental_hash(
            {
                "adjusted_quality_score": adjusted_score,
                "cache_key": cache_key,
                "components": [asdict(item) for item in components],
                "counter_evidence": [asdict(item) for item in opposing],
                "flags": [asdict(item) for item in flags],
                "quality_score": quality_score,
                "risk_penalty": risk_penalty,
                "status": status,
                "supporting_evidence": [asdict(item) for item in supporting],
            }
        )
        mapping = self._config.metric_mapping
        return FundamentalQualitySnapshot(
            instrument_id=request.instrument_id,
            as_of=request.as_of,
            data_version=request.data_version,
            feature_version=self._config.version,
            metric_mapping_version=mapping.version,
            metric_mapping_hash=mapping.mapping_hash,
            config_hash=self._config.config_hash,
            status=status,
            quality_score=quality_score,
            risk_penalty=risk_penalty,
            adjusted_quality_score=adjusted_score,
            components=components,
            supporting_evidence=supporting,
            counter_evidence=opposing,
            flags=flags,
            selected_sources=selected_sources,
            input_hash=input_hash,
            cache_key=cache_key,
            result_hash=result_hash,
        )

    @staticmethod
    def _validate_points(
        points: tuple[FundamentalPoint, ...],
        request: FundamentalQualityRequest,
    ) -> None:
        ensure_aware(request.as_of)
        for point in points:
            if not isinstance(point, FundamentalPoint):
                raise FundamentalQualityInputError(
                    "fundamental inputs must be immutable FundamentalPoint values"
                )
            if point.instrument_id != request.instrument_id:
                raise FundamentalQualityInputError(
                    "fundamental inputs cannot mix instruments or differ from the request"
                )
            if any(
                not value.strip()
                for value in (
                    point.instrument_id,
                    point.metric_name,
                    point.provider_revision,
                    point.source,
                )
            ):
                raise FundamentalQualityInputError(
                    "fundamental identifiers, metric, revision, and source must be non-empty"
                )
            ensure_aware(point.announced_at)
            ensure_aware(point.available_at)
            if not point.metric_value.is_finite():
                raise FundamentalQualityInputError("fundamental metric values must be finite")
            if point.available_at < point.announced_at:
                raise FundamentalQualityInputError(
                    "fundamental available_at cannot precede announced_at"
                )
            announcement_date = point.announced_at.astimezone(SHANGHAI_TZ).date()
            if point.report_period > announcement_date:
                raise FundamentalQualityInputError(
                    "fundamental report_period cannot be after its announcement date"
                )

    def _latest_revisions(
        self,
        points: tuple[FundamentalPoint, ...],
        request: FundamentalQualityRequest,
    ) -> dict[tuple[date, str], FundamentalPoint]:
        mapped_names = set(self._config.metric_mapping.all_metric_names)
        selected: dict[tuple[date, str], FundamentalPoint] = {}
        for point in points:
            if point.metric_name not in mapped_names or point.available_at > request.as_of:
                continue
            if point.announced_at > request.as_of:
                raise FundamentalQualityInputError(
                    "a selected fundamental revision was announced after as_of"
                )
            key = (point.report_period, point.metric_name)
            current = selected.get(key)
            if current is None or point.available_at > current.available_at:
                selected[key] = point
                continue
            if point.available_at == current.available_at and _point_fingerprint(
                point
            ) != _point_fingerprint(current):
                raise FundamentalQualityInputError(
                    "ambiguous fundamental revisions share report, metric, and available_at"
                )
        return selected

    @staticmethod
    def _latest_semantic(
        latest: dict[tuple[date, str], FundamentalPoint],
        metric_names: tuple[str, ...],
        *,
        report_period: date | None,
    ) -> FundamentalPoint | None:
        if report_period is None:
            return None
        for metric_name in metric_names:
            point = latest.get((report_period, metric_name))
            if point is not None:
                return point
        return None

    def _growth_history(
        self,
        latest: dict[tuple[date, str], FundamentalPoint],
    ) -> tuple[FundamentalPoint, ...]:
        mapping = self._config.metric_mapping.growth
        periods = sorted(
            {report_period for report_period, metric_name in latest if metric_name in mapping},
            reverse=True,
        )[: self._config.growth_window_periods]
        selected: list[FundamentalPoint] = []
        for report_period in periods:
            for metric_name in mapping:
                point = latest.get((report_period, metric_name))
                if point is not None:
                    selected.append(point)
                    break
        return tuple(sorted(selected, key=lambda item: item.report_period))

    def _components(
        self,
        *,
        profitability: FundamentalPoint | None,
        cash_flow: FundamentalPoint | None,
        leverage: FundamentalPoint | None,
        growth_points: tuple[FundamentalPoint, ...],
    ) -> tuple[FundamentalComponentResult, ...]:
        weights = self._config.component_weights
        profitability_score = (
            _higher_is_better(
                profitability.metric_value,
                self._config.profitability_zero_score,
                self._config.profitability_full_score,
            )
            if profitability is not None
            else None
        )
        cash_flow_score = (
            _higher_is_better(
                cash_flow.metric_value,
                self._config.cash_flow_zero_score,
                self._config.cash_flow_full_score,
            )
            if cash_flow is not None
            else None
        )
        leverage_score = (
            _lower_is_better(
                leverage.metric_value,
                self._config.leverage_full_score_max,
                self._config.leverage_zero_score_min,
            )
            if leverage is not None
            else None
        )
        growth_range = (
            max(item.metric_value for item in growth_points)
            - min(item.metric_value for item in growth_points)
            if growth_points
            else None
        )
        growth_score = (
            _lower_is_better(
                growth_range,
                self._config.growth_range_full_score_max,
                self._config.growth_range_zero_score_min,
            )
            if growth_range is not None
            and len(growth_points) >= self._config.minimum_growth_periods
            else None
        )

        return (
            self._component(
                component=FundamentalComponent.PROFITABILITY,
                raw_value=profitability.metric_value if profitability else None,
                normalized_score=profitability_score,
                weight=weights[FundamentalComponent.PROFITABILITY],
                points=(profitability,) if profitability else (),
                missing_reason="no mapped profitability metric was available as_of",
            ),
            self._component(
                component=FundamentalComponent.OPERATING_CASH_FLOW,
                raw_value=cash_flow.metric_value if cash_flow else None,
                normalized_score=cash_flow_score,
                weight=weights[FundamentalComponent.OPERATING_CASH_FLOW],
                points=(cash_flow,) if cash_flow else (),
                missing_reason="no mapped operating cash-flow metric was available as_of",
            ),
            self._component(
                component=FundamentalComponent.LEVERAGE,
                raw_value=leverage.metric_value if leverage else None,
                normalized_score=leverage_score,
                weight=weights[FundamentalComponent.LEVERAGE],
                points=(leverage,) if leverage else (),
                missing_reason="no mapped leverage metric was available as_of",
            ),
            self._component(
                component=FundamentalComponent.GROWTH_STABILITY,
                raw_value=growth_range,
                normalized_score=growth_score,
                weight=weights[FundamentalComponent.GROWTH_STABILITY],
                points=growth_points,
                missing_reason=(
                    "growth history has "
                    f"{len(growth_points)} periods; requires "
                    f"{self._config.minimum_growth_periods}"
                ),
            ),
        )

    @staticmethod
    def _component(
        *,
        component: FundamentalComponent,
        raw_value: Decimal | None,
        normalized_score: Decimal | None,
        weight: Decimal,
        points: tuple[FundamentalPoint, ...],
        missing_reason: str,
    ) -> FundamentalComponentResult:
        sources = tuple(_source_reference(point) for point in points)
        return FundamentalComponentResult(
            component=component,
            raw_value=raw_value,
            normalized_score=normalized_score,
            configured_weight=weight,
            contribution=(
                normalized_score * weight if normalized_score is not None else Decimal(0)
            ),
            sources=sources,
            missing_reason=missing_reason if normalized_score is None else None,
        )

    def _evidence(
        self,
        components: tuple[FundamentalComponentResult, ...],
    ) -> tuple[tuple[FundamentalEvidence, ...], tuple[FundamentalEvidence, ...]]:
        supporting: list[FundamentalEvidence] = []
        opposing: list[FundamentalEvidence] = []
        criteria = {
            FundamentalComponent.PROFITABILITY: (
                f"score >= {self._config.evidence_support_score_min}; "
                f"full at {self._config.profitability_full_score}"
            ),
            FundamentalComponent.OPERATING_CASH_FLOW: (
                f"score >= {self._config.evidence_support_score_min}; "
                f"full at {self._config.cash_flow_full_score}"
            ),
            FundamentalComponent.LEVERAGE: (
                f"score >= {self._config.evidence_support_score_min}; "
                f"full at <= {self._config.leverage_full_score_max}"
            ),
            FundamentalComponent.GROWTH_STABILITY: (
                f"score >= {self._config.evidence_support_score_min}; "
                f"full range <= {self._config.growth_range_full_score_max}"
            ),
        }
        for component in components:
            is_supporting = (
                component.normalized_score is not None
                and component.normalized_score >= self._config.evidence_support_score_min
            )
            side = (
                FundamentalEvidenceSide.SUPPORTING
                if is_supporting
                else FundamentalEvidenceSide.OPPOSING
            )
            rationale = (
                f"{component.component.value.lower()} meets the configured quality threshold"
                if is_supporting
                else (
                    component.missing_reason
                    or f"{component.component.value.lower()} is below the quality threshold"
                )
            )
            evidence = FundamentalEvidence(
                component=component.component,
                side=side,
                observed_value=component.raw_value,
                normalized_score=component.normalized_score,
                criterion=criteria[component.component],
                rationale=rationale,
                sources=component.sources,
            )
            (supporting if is_supporting else opposing).append(evidence)
        return tuple(supporting), tuple(opposing)

    def _flags(
        self,
        *,
        components: tuple[FundamentalComponentResult, ...],
        profitability: FundamentalPoint | None,
        cash_flow: FundamentalPoint | None,
        leverage: FundamentalPoint | None,
        growth_points: tuple[FundamentalPoint, ...],
    ) -> tuple[FundamentalFlag, ...]:
        flags: list[FundamentalFlag] = []
        missing_codes = {
            FundamentalComponent.PROFITABILITY: FundamentalFlagCode.MISSING_PROFITABILITY,
            FundamentalComponent.OPERATING_CASH_FLOW: (
                FundamentalFlagCode.MISSING_OPERATING_CASH_FLOW
            ),
            FundamentalComponent.LEVERAGE: FundamentalFlagCode.MISSING_LEVERAGE,
            FundamentalComponent.GROWTH_STABILITY: FundamentalFlagCode.MISSING_GROWTH_HISTORY,
        }
        for component in components:
            if component.missing:
                flags.append(
                    FundamentalFlag(
                        code=missing_codes[component.component],
                        kind=FundamentalFlagKind.MISSING,
                        component=component.component,
                        penalty=self._config.missing_component_penalty,
                        rationale=component.missing_reason or "required component is missing",
                        sources=component.sources,
                    )
                )

        if profitability is not None and profitability.metric_value < 0:
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.NEGATIVE_PROFITABILITY,
                    component=FundamentalComponent.PROFITABILITY,
                    penalty=self._config.negative_profitability_penalty,
                    rationale="latest profitability metric is negative",
                    points=(profitability,),
                )
            )
        if cash_flow is not None and cash_flow.metric_value < 0:
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.NEGATIVE_OPERATING_CASH_FLOW,
                    component=FundamentalComponent.OPERATING_CASH_FLOW,
                    penalty=self._config.negative_cash_flow_penalty,
                    rationale="latest operating cash-flow metric is negative",
                    points=(cash_flow,),
                )
            )
        if leverage is not None and leverage.metric_value >= self._config.leverage_zero_score_min:
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.HIGH_LEVERAGE,
                    component=FundamentalComponent.LEVERAGE,
                    penalty=self._config.high_leverage_penalty,
                    rationale="latest leverage is at or above the zero-score threshold",
                    points=(leverage,),
                )
            )

        growth_range = (
            max(item.metric_value for item in growth_points)
            - min(item.metric_value for item in growth_points)
            if len(growth_points) >= self._config.minimum_growth_periods
            else None
        )
        if growth_range is not None and growth_range >= self._config.growth_range_zero_score_min:
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.UNSTABLE_GROWTH,
                    component=FundamentalComponent.GROWTH_STABILITY,
                    penalty=self._config.unstable_growth_penalty,
                    rationale="growth range is at or above the zero-score threshold",
                    points=growth_points,
                )
            )
        if growth_points and any(
            abs(item.metric_value) >= self._config.extreme_growth_absolute_threshold
            for item in growth_points
        ):
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.EXTREME_GROWTH,
                    component=FundamentalComponent.GROWTH_STABILITY,
                    penalty=self._config.extreme_growth_penalty,
                    rationale="at least one growth observation exceeds the absolute threshold",
                    points=growth_points,
                )
            )
        if (
            profitability is not None
            and cash_flow is not None
            and profitability.metric_value > 0
            and cash_flow.metric_value <= 0
        ):
            flags.append(
                self._anomaly_flag(
                    code=FundamentalFlagCode.CASH_EARNINGS_MISMATCH,
                    component=None,
                    penalty=self._config.cash_earnings_mismatch_penalty,
                    rationale="positive profitability is not supported by operating cash flow",
                    points=(profitability, cash_flow),
                )
            )
        return tuple(flags)

    @staticmethod
    def _anomaly_flag(
        *,
        code: FundamentalFlagCode,
        component: FundamentalComponent | None,
        penalty: Decimal,
        rationale: str,
        points: tuple[FundamentalPoint, ...],
    ) -> FundamentalFlag:
        return FundamentalFlag(
            code=code,
            kind=FundamentalFlagKind.ANOMALY,
            component=component,
            penalty=penalty,
            rationale=rationale,
            sources=tuple(_source_reference(point) for point in points),
        )

    @staticmethod
    def _selected_sources(
        components: tuple[FundamentalComponentResult, ...],
    ) -> tuple[FundamentalSourceReference, ...]:
        by_fingerprint: dict[tuple[tuple[str, str], ...], FundamentalSourceReference] = {}
        for component in components:
            for source in component.sources:
                key = tuple(sorted(source.fingerprint_payload().items()))
                by_fingerprint[key] = source
        return tuple(sorted(by_fingerprint.values(), key=_source_sort_key))


__all__ = ["FundamentalQualityAnalyzer"]
