"""Deterministic baseline market-regime classifier."""

from dataclasses import asdict
from decimal import Decimal

from quant_agent.features.market_breadth import MarketBreadthSnapshot
from quant_agent.features.market_trend import MarketTrendSnapshot
from quant_agent.regime.contracts import (
    EvidenceSide,
    InsufficientRegimeData,
    MarketRegime,
    MarketRegimeResult,
    RegimeClassifierConfig,
    RegimeComponentScores,
    RegimeEvidence,
    RegimeInputIdentity,
    RegimeInputMismatch,
    stable_hash,
)


def _clamp(value: Decimal, lower: Decimal = Decimal(0), upper: Decimal = Decimal(100)) -> Decimal:
    return min(upper, max(lower, value))


class MarketRegimeClassifier:
    """Classify aligned trend and breadth snapshots using transparent rules."""

    def __init__(self, config: RegimeClassifierConfig | None = None) -> None:
        self._config = config or RegimeClassifierConfig()

    def classify(
        self,
        *,
        trend_snapshot: MarketTrendSnapshot,
        breadth_snapshot: MarketBreadthSnapshot,
    ) -> MarketRegimeResult:
        """Return a reproducible regime result, failing closed on unsafe breadth."""

        self._validate_inputs(trend_snapshot, breadth_snapshot)
        metrics = self._critical_breadth_metrics(breadth_snapshot)
        advance_decline = metrics["advance_decline_ratio"]
        above_average = metrics["above_average_ratio"]
        new_high = metrics["new_high_ratio"]
        new_low = metrics["new_low_ratio"]
        turnover = metrics["turnover_percentile"]
        downside_volatility = metrics["downside_volatility"]

        components = self._component_scores(
            aggregate_trend=trend_snapshot.aggregate_score,
            advance_decline=advance_decline,
            above_average=above_average,
            new_high=new_high,
            new_low=new_low,
            turnover=turnover,
            downside_volatility=downside_volatility,
        )
        short_trend = self._window_score(
            trend_snapshot,
            self._config.bottom_short_window,
        )
        long_trend = self._window_score(
            trend_snapshot,
            self._config.bottom_long_window,
        )
        bottom_checks = self._bottom_checks(
            short_trend=short_trend,
            long_trend=long_trend,
            advance_decline=advance_decline,
            above_average=above_average,
            new_low=new_low,
        )
        regime = self._select_regime(
            score=components.total,
            aggregate_trend=trend_snapshot.aggregate_score,
            above_average=above_average,
            high_low_spread=new_high - new_low,
            bottom_checks=bottom_checks,
        )
        rules = self._evidence_rules(
            regime=regime,
            components=components,
            aggregate_trend=trend_snapshot.aggregate_score,
            advance_decline=advance_decline,
            above_average=above_average,
            new_high=new_high,
            new_low=new_low,
            short_trend=short_trend,
            long_trend=long_trend,
        )
        supporting, opposing = self._partition_evidence(rules)
        total_rule_weight = sum((item.rule_weight for item in rules), Decimal(0))
        confidence = (
            sum(
                (item.rule_weight for item in supporting),
                Decimal(0),
            )
            / total_rule_weight
        )

        input_identity = RegimeInputIdentity(
            trend_feature_version=trend_snapshot.feature_version,
            trend_config_hash=trend_snapshot.config_hash,
            trend_cache_key=trend_snapshot.cache_key,
            breadth_feature_version=breadth_snapshot.feature_version,
            breadth_config_hash=breadth_snapshot.config_hash,
            breadth_cache_key=breadth_snapshot.cache_key,
        )
        input_hash = stable_hash(
            {
                "as_of": trend_snapshot.as_of,
                "breadth_session_date": breadth_snapshot.session_date,
                "data_version": trend_snapshot.data_version,
                "input_identity": asdict(input_identity),
            }
        )
        invalidations = self._invalidations(regime)
        result_hash = stable_hash(
            {
                "components": asdict(components),
                "confidence": confidence,
                "config_hash": self._config.config_hash,
                "data_version": trend_snapshot.data_version,
                "evidence": [asdict(item) for item in supporting],
                "counter_evidence": [asdict(item) for item in opposing],
                "input_hash": input_hash,
                "invalidations": invalidations,
                "model_version": self._config.version,
                "regime": regime.value,
                "risk_budget_max": self._config.risk_budget_for(regime),
                "score": components.total,
            }
        )
        return MarketRegimeResult(
            as_of=trend_snapshot.as_of,
            session_date=breadth_snapshot.session_date,
            data_version=trend_snapshot.data_version,
            regime=regime,
            score=components.total,
            confidence=confidence,
            risk_budget_max=self._config.risk_budget_for(regime),
            components=components,
            evidence=supporting,
            counter_evidence=opposing,
            invalidations=invalidations,
            model_version=self._config.version,
            config_hash=self._config.config_hash,
            input_identity=input_identity,
            input_hash=input_hash,
            result_hash=result_hash,
        )

    def _validate_inputs(
        self,
        trend: MarketTrendSnapshot,
        breadth: MarketBreadthSnapshot,
    ) -> None:
        if trend.as_of != breadth.as_of:
            raise RegimeInputMismatch("trend and breadth snapshots must share one as_of instant")
        if trend.session_date != breadth.session_date:
            raise RegimeInputMismatch("trend and breadth snapshots must share one session_date")
        if trend.data_version != breadth.data_version:
            raise RegimeInputMismatch("trend and breadth snapshots must share one data_version")
        if breadth.current_coverage < self._config.minimum_breadth_coverage:
            raise InsufficientRegimeData(
                f"breadth coverage {breadth.current_coverage} is below regime floor "
                f"{self._config.minimum_breadth_coverage}"
            )
        if breadth.minimum_historical_coverage_observed < self._config.minimum_breadth_coverage:
            raise InsufficientRegimeData(
                "historical breadth coverage is below the regime safety floor"
            )
        critical_counts = {
            "return_denominator": (
                breadth.return_denominator,
                self._config.minimum_return_denominator,
            ),
            "moving_average_denominator": (
                breadth.moving_average_denominator,
                self._config.minimum_moving_average_denominator,
            ),
            "high_low_denominator": (
                breadth.high_low_denominator,
                self._config.minimum_high_low_denominator,
            ),
            "turnover_history_count": (
                breadth.turnover_history_count,
                self._config.minimum_turnover_history_count,
            ),
            "downside_observation_count": (
                breadth.downside_observation_count,
                self._config.minimum_downside_observation_count,
            ),
        }
        insufficient = tuple(
            name for name, (actual, minimum) in critical_counts.items() if actual < minimum
        )
        if insufficient:
            raise InsufficientRegimeData(
                "critical breadth history unavailable: " + ", ".join(insufficient)
            )

    @staticmethod
    def _critical_breadth_metrics(snapshot: MarketBreadthSnapshot) -> dict[str, Decimal]:
        values = {
            "advance_decline_ratio": snapshot.advance_decline_ratio,
            "above_average_ratio": snapshot.above_average_ratio,
            "new_high_ratio": snapshot.new_high_ratio,
            "new_low_ratio": snapshot.new_low_ratio,
            "turnover_percentile": snapshot.turnover_percentile,
            "downside_volatility": snapshot.downside_volatility,
        }
        missing = tuple(name for name, value in values.items() if value is None)
        if missing:
            raise InsufficientRegimeData(
                "critical breadth metrics unavailable: " + ", ".join(missing)
            )
        return {name: value for name, value in values.items() if value is not None}

    def _component_scores(
        self,
        *,
        aggregate_trend: Decimal,
        advance_decline: Decimal,
        above_average: Decimal,
        new_high: Decimal,
        new_low: Decimal,
        turnover: Decimal,
        downside_volatility: Decimal,
    ) -> RegimeComponentScores:
        trend = _clamp((aggregate_trend + Decimal(100)) / 2)
        breadth = _clamp((advance_decline + above_average) * Decimal(50))
        turnover_score = _clamp(turnover * Decimal(100))
        new_high_low = _clamp(Decimal(50) + (new_high - new_low) * Decimal(50))
        downside_risk = _clamp(
            Decimal(100)
            * (Decimal(1) - downside_volatility / self._config.downside_volatility_ceiling)
        )
        return RegimeComponentScores(
            trend=trend,
            breadth=breadth,
            turnover=turnover_score,
            new_high_low=new_high_low,
            downside_risk=downside_risk,
            weighted_trend=trend * self._config.trend_weight,
            weighted_breadth=breadth * self._config.breadth_weight,
            weighted_turnover=turnover_score * self._config.turnover_weight,
            weighted_new_high_low=new_high_low * self._config.new_high_low_weight,
            weighted_downside_risk=downside_risk * self._config.downside_risk_weight,
        )

    @staticmethod
    def _window_score(snapshot: MarketTrendSnapshot, window: int) -> Decimal | None:
        scores = tuple(
            metric.score
            for index in snapshot.indices
            for metric in index.windows
            if metric.window == window
        )
        if not scores:
            return None
        return sum(scores, Decimal(0)) / len(scores)

    def _bottom_checks(
        self,
        *,
        short_trend: Decimal | None,
        long_trend: Decimal | None,
        advance_decline: Decimal,
        above_average: Decimal,
        new_low: Decimal,
    ) -> tuple[bool, ...]:
        return (
            long_trend is not None and long_trend <= self._config.bottom_long_trend_max,
            short_trend is not None and short_trend >= self._config.bottom_short_trend_min,
            advance_decline >= self._config.bottom_advance_decline_min,
            above_average >= self._config.bottom_above_average_min,
            new_low <= self._config.bottom_new_low_max,
        )

    def _select_regime(
        self,
        *,
        score: Decimal,
        aggregate_trend: Decimal,
        above_average: Decimal,
        high_low_spread: Decimal,
        bottom_checks: tuple[bool, ...],
    ) -> MarketRegime:
        if sum(bottom_checks) >= self._config.bottom_min_confirming_signals:
            return MarketRegime.BOTTOM_RECOVERY
        weak_signals = sum(
            (
                above_average <= self._config.divergence_above_average_max,
                high_low_spread <= self._config.divergence_high_low_spread_max,
            )
        )
        if (
            score >= self._config.divergent_min_score
            and aggregate_trend >= self._config.divergence_trend_min
            and weak_signals >= self._config.divergence_min_weak_signals
        ):
            return MarketRegime.DIVERGENT
        if score >= self._config.uptrend_min_score:
            return MarketRegime.UPTREND
        if score >= self._config.range_strong_min_score:
            return MarketRegime.RANGE_STRONG
        if score >= self._config.divergent_min_score:
            return MarketRegime.DIVERGENT
        return MarketRegime.DOWNTREND

    def _evidence_rules(
        self,
        *,
        regime: MarketRegime,
        components: RegimeComponentScores,
        aggregate_trend: Decimal,
        advance_decline: Decimal,
        above_average: Decimal,
        new_high: Decimal,
        new_low: Decimal,
        short_trend: Decimal | None,
        long_trend: Decimal | None,
    ) -> tuple[RegimeEvidence, ...]:
        neutral = self._config.neutral_component_score
        weights = self._config.evidence_weights_for(regime)
        if regime is MarketRegime.UPTREND:
            checks = (
                (
                    "environment_score",
                    components.total,
                    f">={self._config.uptrend_min_score}",
                    components.total >= self._config.uptrend_min_score,
                    weights[0],
                    "环境分进入上升区间",
                ),
                (
                    "trend_score",
                    components.trend,
                    f">={neutral}",
                    components.trend >= neutral,
                    weights[1],
                    "主要指数趋势为正向",
                ),
                (
                    "breadth_score",
                    components.breadth,
                    f">={neutral}",
                    components.breadth >= neutral,
                    weights[2],
                    "上涨与均线宽度扩散",
                ),
                (
                    "new_high_low_score",
                    components.new_high_low,
                    f">={neutral}",
                    components.new_high_low >= neutral,
                    weights[3],
                    "新高相对新低占优",
                ),
                (
                    "downside_risk_score",
                    components.downside_risk,
                    f">={neutral}",
                    components.downside_risk >= neutral,
                    weights[4],
                    "下行波动受控",
                ),
            )
        elif regime is MarketRegime.RANGE_STRONG:
            checks = (
                (
                    "environment_score",
                    components.total,
                    f"[{self._config.range_strong_min_score},{self._config.uptrend_min_score})",
                    self._config.range_strong_min_score
                    <= components.total
                    < self._config.uptrend_min_score,
                    weights[0],
                    "环境分位于震荡偏强区间",
                ),
                (
                    "trend_score",
                    components.trend,
                    f">={neutral}",
                    components.trend >= neutral,
                    weights[1],
                    "指数趋势仍有正向支撑",
                ),
                (
                    "breadth_score",
                    components.breadth,
                    f">={neutral}",
                    components.breadth >= neutral,
                    weights[2],
                    "市场宽度保持过半",
                ),
                (
                    "turnover_score",
                    components.turnover,
                    f">={neutral}",
                    components.turnover >= neutral,
                    weights[3],
                    "成交活跃度不弱",
                ),
                (
                    "downside_risk_score",
                    components.downside_risk,
                    f">={neutral}",
                    components.downside_risk >= neutral,
                    weights[4],
                    "下行风险仍可控",
                ),
            )
        elif regime is MarketRegime.DIVERGENT:
            spread = new_high - new_low
            checks = (
                (
                    "environment_score",
                    components.total,
                    f">={self._config.divergent_min_score}",
                    components.total >= self._config.divergent_min_score,
                    weights[0],
                    "环境分尚未进入全面下跌区间",
                ),
                (
                    "aggregate_trend",
                    aggregate_trend,
                    f">={self._config.divergence_trend_min}",
                    aggregate_trend >= self._config.divergence_trend_min,
                    weights[1],
                    "指数仍处相对高位或保持趋势",
                ),
                (
                    "above_average_ratio",
                    above_average,
                    f"<={self._config.divergence_above_average_max}",
                    above_average <= self._config.divergence_above_average_max,
                    weights[2],
                    "均线以上股票比例走弱",
                ),
                (
                    "new_high_low_spread",
                    spread,
                    f"<={self._config.divergence_high_low_spread_max}",
                    spread <= self._config.divergence_high_low_spread_max,
                    weights[3],
                    "新高新低结构与指数背离",
                ),
                (
                    "downside_risk_score",
                    components.downside_risk,
                    f"<{neutral}",
                    components.downside_risk < neutral,
                    weights[4],
                    "下行波动风险升高",
                ),
            )
        elif regime is MarketRegime.DOWNTREND:
            checks = (
                (
                    "environment_score",
                    components.total,
                    f"<{self._config.divergent_min_score}",
                    components.total < self._config.divergent_min_score,
                    weights[0],
                    "环境分进入下跌区间",
                ),
                (
                    "trend_score",
                    components.trend,
                    f"<{neutral}",
                    components.trend < neutral,
                    weights[1],
                    "主要指数趋势为负向",
                ),
                (
                    "breadth_score",
                    components.breadth,
                    f"<{neutral}",
                    components.breadth < neutral,
                    weights[2],
                    "上涨与均线宽度收缩",
                ),
                (
                    "new_high_low_score",
                    components.new_high_low,
                    f"<{neutral}",
                    components.new_high_low < neutral,
                    weights[3],
                    "新低相对新高占优",
                ),
                (
                    "downside_risk_score",
                    components.downside_risk,
                    f"<{neutral}",
                    components.downside_risk < neutral,
                    weights[4],
                    "下行波动风险偏高",
                ),
            )
        else:
            short_value = short_trend if short_trend is not None else Decimal(-100)
            long_value = long_trend if long_trend is not None else Decimal(100)
            checks = (
                (
                    "long_trend_score",
                    long_value,
                    f"<={self._config.bottom_long_trend_max}",
                    long_trend is not None and long_trend <= self._config.bottom_long_trend_max,
                    weights[0],
                    "长周期仍保留前期显著下跌痕迹",
                ),
                (
                    "short_trend_score",
                    short_value,
                    f">={self._config.bottom_short_trend_min}",
                    short_trend is not None and short_trend >= self._config.bottom_short_trend_min,
                    weights[1],
                    "短周期趋势率先企稳转强",
                ),
                (
                    "advance_decline_ratio",
                    advance_decline,
                    f">={self._config.bottom_advance_decline_min}",
                    advance_decline >= self._config.bottom_advance_decline_min,
                    weights[2],
                    "上涨家数占比开始修复",
                ),
                (
                    "above_average_ratio",
                    above_average,
                    f">={self._config.bottom_above_average_min}",
                    above_average >= self._config.bottom_above_average_min,
                    weights[3],
                    "均线宽度开始修复",
                ),
                (
                    "new_low_ratio",
                    new_low,
                    f"<={self._config.bottom_new_low_max}",
                    new_low <= self._config.bottom_new_low_max,
                    weights[4],
                    "当前新低比例已受控",
                ),
            )
        return tuple(
            RegimeEvidence(
                feature=feature,
                value=value,
                criterion=criterion,
                rule_weight=weight,
                contribution=weight * Decimal(100),
                side=EvidenceSide.SUPPORTING if passed else EvidenceSide.OPPOSING,
                rationale=rationale,
            )
            for feature, value, criterion, passed, weight, rationale in checks
        )

    @staticmethod
    def _partition_evidence(
        rules: tuple[RegimeEvidence, ...],
    ) -> tuple[tuple[RegimeEvidence, ...], tuple[RegimeEvidence, ...]]:
        supporting = tuple(item for item in rules if item.side is EvidenceSide.SUPPORTING)
        opposing = tuple(item for item in rules if item.side is EvidenceSide.OPPOSING)
        return supporting, opposing

    def _invalidations(self, regime: MarketRegime) -> tuple[str, ...]:
        if regime is MarketRegime.UPTREND:
            return (
                f"环境分跌破 {self._config.uptrend_min_score}",
                f"均线以上比例降至 {self._config.divergence_above_average_max} 或以下并形成分化",
            )
        if regime is MarketRegime.RANGE_STRONG:
            return (
                f"环境分跌破 {self._config.range_strong_min_score}",
                f"环境分升至 {self._config.uptrend_min_score} 或以上并保持广度确认",
            )
        if regime is MarketRegime.DIVERGENT:
            return (
                "均线以上比例恢复至 "
                f"{self._config.divergence_above_average_max} 以上且新高重新占优",
                f"环境分跌破 {self._config.divergent_min_score}, 转为全面下跌",
            )
        if regime is MarketRegime.DOWNTREND:
            return (
                f"环境分恢复至 {self._config.divergent_min_score} 或以上",
                "长弱短强、宽度修复和新低受控条件同时成立",
            )
        return (
            f"短周期趋势跌破 {self._config.bottom_short_trend_min}",
            f"上涨家数占比跌破 {self._config.bottom_advance_decline_min}",
            f"新低比例升破 {self._config.bottom_new_low_max}",
        )
