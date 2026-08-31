"""Cross-sectional leader scoring over aligned point-in-time feature snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from decimal import Decimal

from quant_agent.features.fundamental_quality import FundamentalQualitySnapshot
from quant_agent.features.mainline import MainlineIndustryResult, MainlineSnapshot, MainlineState
from quant_agent.features.stock_strength import (
    StockStrengthSnapshot,
    StockStrengthStatus,
)
from quant_agent.features.tradeability import TradeabilitySnapshot
from quant_agent.leaders.contracts import (
    LeaderComponent,
    LeaderComponentScore,
    LeaderConfig,
    LeaderEvidence,
    LeaderEvidenceSide,
    LeaderExclusion,
    LeaderExclusionCode,
    LeaderInputError,
    LeaderResult,
    LeaderRisk,
    LeaderRiskCode,
    LeaderSnapshot,
    LeaderType,
    LeaderUpstreamIdentity,
    stable_leader_hash,
)


def _clamp(value: Decimal) -> Decimal:
    return max(Decimal(0), min(Decimal(100), value))


def _signed_score(value: Decimal) -> Decimal:
    return _clamp((value + Decimal(100)) / 2)


@dataclass(frozen=True, slots=True)
class _AlignedInputs:
    stock: StockStrengthSnapshot
    tradeability: TradeabilitySnapshot
    fundamental: FundamentalQualitySnapshot
    mainline: MainlineIndustryResult
    mainline_snapshot: MainlineSnapshot


@dataclass(frozen=True, slots=True)
class _ProvisionalLeader:
    instrument_id: str
    industry_id: str
    mainline_state: MainlineState
    leader_type: LeaderType
    gross_score: Decimal
    risk_penalty: Decimal
    score: Decimal
    components: tuple[LeaderComponentScore, ...]
    supporting_evidence: tuple[LeaderEvidence, ...]
    counter_evidence: tuple[LeaderEvidence, ...]
    risks: tuple[LeaderRisk, ...]
    candidate_eligible: bool
    ineligibility_reasons: tuple[str, ...]
    observation_conditions: tuple[str, ...]
    invalidations: tuple[str, ...]
    upstream_identity: LeaderUpstreamIdentity


class LeaderEngine:
    """Rank only PIT-valid members of CONFIRMED or CROWDED mainlines."""

    def __init__(self, config: LeaderConfig | None = None) -> None:
        self._config = config or LeaderConfig()

    def rank(
        self,
        *,
        mainline: MainlineSnapshot,
        stock_strength: Iterable[StockStrengthSnapshot],
        tradeability: Iterable[TradeabilitySnapshot],
        fundamentals: Iterable[FundamentalQualitySnapshot],
    ) -> LeaderSnapshot:
        """Align upstream identities, score active members, and retain exclusions."""

        stocks = self._stock_map(stock_strength)
        trades = self._tradeability_map(tradeability)
        fundamental_map = self._fundamental_map(fundamentals)
        if set(stocks) != set(trades) or set(stocks) != set(fundamental_map):
            raise LeaderInputError(
                "stock strength, tradeability, and fundamental snapshots must cover identical IDs"
            )
        active_mainlines = {
            item.industry_id: item
            for item in mainline.industries
            if item.state in {MainlineState.CONFIRMED, MainlineState.CROWDED}
        }
        aligned: list[_AlignedInputs] = []
        exclusions: list[LeaderExclusion] = []
        for instrument_id in sorted(stocks):
            stock = stocks[instrument_id]
            trade = trades[instrument_id]
            fundamental = fundamental_map[instrument_id]
            self._validate_alignment(mainline, stock, trade, fundamental)
            industry_id = stock.current_industry_id
            if stock.status not in {StockStrengthStatus.READY, StockStrengthStatus.DEGRADED}:
                exclusions.append(
                    LeaderExclusion(
                        instrument_id=instrument_id,
                        industry_id=industry_id,
                        code=LeaderExclusionCode.STOCK_STRENGTH_UNAVAILABLE,
                        reason=f"stock strength status is {stock.status.value}",
                        stock_result_hash=stock.result_hash,
                    )
                )
                continue
            if industry_id is None or industry_id not in active_mainlines:
                exclusions.append(
                    LeaderExclusion(
                        instrument_id=instrument_id,
                        industry_id=industry_id,
                        code=LeaderExclusionCode.NOT_ACTIVE_MAINLINE_MEMBER,
                        reason="stock is not a PIT member of a CONFIRMED or CROWDED mainline",
                        stock_result_hash=stock.result_hash,
                    )
                )
                continue
            aligned.append(
                _AlignedInputs(
                    stock=stock,
                    tradeability=trade,
                    fundamental=fundamental,
                    mainline=active_mainlines[industry_id],
                    mainline_snapshot=mainline,
                )
            )

        return_rank_scores = self._theme_return_rank_scores(aligned)
        provisional = tuple(
            self._score(item, return_rank_scores[item.stock.instrument_id]) for item in aligned
        )
        leaders = self._assign_ranks(provisional)
        input_hash = self._input_hash(
            mainline=mainline,
            stocks=stocks,
            trades=trades,
            fundamentals=fundamental_map,
        )
        result_hash = self._result_hash(
            input_hash=input_hash,
            leaders=leaders,
            exclusions=tuple(exclusions),
        )
        return LeaderSnapshot(
            session_date=mainline.session_date,
            as_of=mainline.as_of,
            data_version=mainline.data_version,
            classification_version=mainline.classification_version,
            industry_level=mainline.industry_level,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            mainline_result_hash=mainline.result_hash,
            leaders=leaders,
            exclusions=tuple(sorted(exclusions, key=lambda item: item.instrument_id)),
            input_hash=input_hash,
            result_hash=result_hash,
        )

    analyze = rank

    @staticmethod
    def _stock_map(
        snapshots: Iterable[StockStrengthSnapshot],
    ) -> dict[str, StockStrengthSnapshot]:
        result: dict[str, StockStrengthSnapshot] = {}
        for snapshot in snapshots:
            if snapshot.instrument_id in result:
                raise LeaderInputError("stock strength instrument IDs must be unique")
            result[snapshot.instrument_id] = snapshot
        return result

    @staticmethod
    def _tradeability_map(
        snapshots: Iterable[TradeabilitySnapshot],
    ) -> dict[str, TradeabilitySnapshot]:
        result: dict[str, TradeabilitySnapshot] = {}
        for snapshot in snapshots:
            instrument_id = snapshot.request.instrument_id
            if instrument_id in result:
                raise LeaderInputError("tradeability instrument IDs must be unique")
            result[instrument_id] = snapshot
        return result

    @staticmethod
    def _fundamental_map(
        snapshots: Iterable[FundamentalQualitySnapshot],
    ) -> dict[str, FundamentalQualitySnapshot]:
        result: dict[str, FundamentalQualitySnapshot] = {}
        for snapshot in snapshots:
            if snapshot.instrument_id in result:
                raise LeaderInputError("fundamental instrument IDs must be unique")
            result[snapshot.instrument_id] = snapshot
        return result

    @staticmethod
    def _validate_alignment(
        mainline: MainlineSnapshot,
        stock: StockStrengthSnapshot,
        tradeability: TradeabilitySnapshot,
        fundamental: FundamentalQualitySnapshot,
    ) -> None:
        request = tradeability.request
        if (
            stock.session_date != mainline.session_date
            or request.session_date != mainline.session_date
        ):
            raise LeaderInputError("leader upstream session_date values must match mainline")
        if (
            stock.as_of > mainline.as_of
            or request.as_of > mainline.as_of
            or fundamental.as_of > mainline.as_of
        ):
            raise LeaderInputError("leader upstream snapshot contains future information")
        if (
            stock.data_version != mainline.data_version
            or request.data_version != mainline.data_version
        ):
            raise LeaderInputError("market feature data_version values must match mainline")
        if (
            stock.classification_version != mainline.classification_version
            or stock.industry_level != mainline.industry_level
        ):
            raise LeaderInputError("stock industry identity must match mainline classification")

    @staticmethod
    def _theme_return_rank_scores(inputs: list[_AlignedInputs]) -> dict[str, Decimal]:
        grouped: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
        for item in inputs:
            shortest = min(item.stock.horizons, key=lambda row: row.horizon)
            grouped[item.mainline.industry_id].append(
                (item.stock.instrument_id, shortest.stock_return)
            )
        scores: dict[str, Decimal] = {}
        for values in grouped.values():
            count = len(values)
            for instrument_id, value in values:
                if count == 1:
                    # A one-stock cross-section cannot prove relative leadership.
                    scores[instrument_id] = Decimal(0)
                    continue
                lower_count = sum(other < value for _, other in values)
                equal_peers = sum(other == value for _, other in values) - 1
                average_rank_below = Decimal(lower_count) + Decimal(equal_peers) / 2
                scores[instrument_id] = Decimal(100) * average_rank_below / (count - 1)
        return scores

    def _score(self, inputs: _AlignedInputs, theme_return_rank: Decimal) -> _ProvisionalLeader:
        stock = inputs.stock
        tradeability = inputs.tradeability
        fundamental = inputs.fundamental
        mainline = inputs.mainline
        within_theme = self._stock_component(stock, "industry_relative")
        trend = _signed_score(stock.trend_quality.score) if stock.trend_quality else Decimal(0)
        percentile = (
            tradeability.turnover_rate_percentile * Decimal(100)
            if tradeability.turnover_rate_percentile is not None
            else Decimal(0)
        )
        capacity = _clamp(
            tradeability.capacity.capacity_ratio
            / self._config.capacity_ratio_full_score
            * Decimal(100)
        )
        liquidity = (
            percentile * self._config.liquidity_percentile_weight
            + capacity * self._config.liquidity_capacity_weight
        )
        volume_price = (
            _signed_score(stock.volume_confirmation.score)
            if stock.volume_confirmation
            else Decimal(0)
        )
        theme_leadership = (
            theme_return_rank * self._config.theme_return_rank_weight
            + volume_price * self._config.theme_volume_price_weight
        )
        relative_resilience = self._stock_component(stock, "benchmark_relative")
        pullback = stock.trend_quality.pullback_depth if stock.trend_quality else Decimal(1)
        pullback_resilience = _clamp(
            Decimal(100) - pullback / self._config.pullback_tolerance * Decimal(100)
        )
        downside = (
            relative_resilience * self._config.downside_relative_weight
            + pullback_resilience * self._config.downside_pullback_weight
        )
        logic = mainline.mainline_score
        component_values = {
            LeaderComponent.WITHIN_THEME_STRENGTH: within_theme,
            LeaderComponent.TREND_QUALITY: trend,
            LeaderComponent.LIQUIDITY: liquidity,
            LeaderComponent.THEME_LEADERSHIP: theme_leadership,
            LeaderComponent.DOWNSIDE_RESILIENCE: downside,
            LeaderComponent.LOGIC_RELEVANCE: logic,
            LeaderComponent.FUNDAMENTAL_QUALITY: fundamental.quality_score,
        }
        components = tuple(
            LeaderComponentScore(
                component=component,
                score=component_values[component],
                weight=self._config.component_weights[component],
                contribution=(
                    component_values[component] * self._config.component_weights[component]
                ),
            )
            for component in LeaderComponent
        )
        gross_score = sum((item.contribution for item in components), Decimal(0))
        risks = self._risks(stock, mainline, fundamental)
        risk_penalty = sum((item.penalty for item in risks), Decimal(0))
        score = max(Decimal(0), gross_score - risk_penalty)
        leader_type = self._classify(
            within_theme=within_theme,
            trend=trend,
            liquidity=liquidity,
            theme_leadership=theme_leadership,
            downside=downside,
        )
        supporting, opposing = self._evidence(components, tradeability, fundamental)
        ineligibility = self._ineligibility_reasons(
            leader_type=leader_type,
            score=score,
            tradeability=tradeability,
        )
        candidate_eligible = not ineligibility
        observation_conditions, invalidations = self._conditions(leader_type)
        return _ProvisionalLeader(
            instrument_id=stock.instrument_id,
            industry_id=mainline.industry_id,
            mainline_state=mainline.state,
            leader_type=leader_type,
            gross_score=gross_score,
            risk_penalty=risk_penalty,
            score=score,
            components=components,
            supporting_evidence=supporting,
            counter_evidence=opposing,
            risks=risks,
            candidate_eligible=candidate_eligible,
            ineligibility_reasons=ineligibility,
            observation_conditions=observation_conditions,
            invalidations=invalidations,
            upstream_identity=self._upstream_identity(inputs),
        )

    @staticmethod
    def _stock_component(snapshot: StockStrengthSnapshot, name: str) -> Decimal:
        item = next((value for value in snapshot.contributions if value.component == name), None)
        return _signed_score(item.normalized_score) if item is not None else Decimal(0)

    def _risks(
        self,
        stock: StockStrengthSnapshot,
        mainline: MainlineIndustryResult,
        fundamental: FundamentalQualitySnapshot,
    ) -> tuple[LeaderRisk, ...]:
        candidates: list[tuple[LeaderRiskCode, Decimal, str]] = []
        if fundamental.risk_penalty:
            candidates.append(
                (
                    LeaderRiskCode.FUNDAMENTAL,
                    fundamental.risk_penalty * self._config.fundamental_risk_multiplier,
                    "fundamental anomaly and missing-data penalties",
                )
            )
        if mainline.state is MainlineState.CROWDED and mainline.crowding_score:
            candidates.append(
                (
                    LeaderRiskCode.MAINLINE_CROWDING,
                    mainline.crowding_score * self._config.crowded_penalty_rate,
                    "crowded mainline raises reversal and execution risk",
                )
            )
        if stock.status is StockStrengthStatus.DEGRADED:
            candidates.append(
                (
                    LeaderRiskCode.DEGRADED_STRENGTH,
                    self._config.degraded_strength_penalty,
                    "stock strength was computed with degraded upstream coverage",
                )
            )
        remaining = self._config.maximum_risk_penalty
        applied: list[LeaderRisk] = []
        for code, penalty, rationale in candidates:
            value = min(remaining, penalty)
            if value <= 0:
                continue
            applied.append(LeaderRisk(code=code, penalty=value, rationale=rationale))
            remaining -= value
            if remaining <= 0:
                break
        return tuple(applied)

    def _classify(
        self,
        *,
        within_theme: Decimal,
        trend: Decimal,
        liquidity: Decimal,
        theme_leadership: Decimal,
        downside: Decimal,
    ) -> LeaderType:
        if (
            liquidity >= self._config.capacity_liquidity_min
            and theme_leadership >= self._config.capacity_leadership_min
            and trend >= self._config.capacity_trend_min
        ):
            return LeaderType.CAPACITY
        if (
            trend >= self._config.trend_quality_min
            and within_theme >= self._config.trend_within_strength_min
            and downside >= self._config.trend_downside_min
        ):
            return LeaderType.TREND
        if max(within_theme, theme_leadership) >= self._config.elasticity_min:
            return LeaderType.ELASTICITY
        return LeaderType.OBSERVATION

    @staticmethod
    def _evidence(
        components: tuple[LeaderComponentScore, ...],
        tradeability: TradeabilitySnapshot,
        fundamental: FundamentalQualitySnapshot,
    ) -> tuple[tuple[LeaderEvidence, ...], tuple[LeaderEvidence, ...]]:
        evidence = [
            LeaderEvidence(
                feature=item.component.value,
                value=item.score,
                criterion=">= 50",
                side=(
                    LeaderEvidenceSide.SUPPORTING
                    if item.score >= 50
                    else LeaderEvidenceSide.OPPOSING
                ),
                rationale=(f"{item.component.value.lower()} normalized leadership contribution"),
            )
            for item in components
        ]
        evidence.append(
            LeaderEvidence(
                feature="TRADEABILITY",
                value=Decimal(100) if tradeability.eligible else Decimal(0),
                criterion="ELIGIBLE",
                side=(
                    LeaderEvidenceSide.SUPPORTING
                    if tradeability.eligible
                    else LeaderEvidenceSide.OPPOSING
                ),
                rationale="account-sized tradeability gate",
            )
        )
        evidence.extend(
            LeaderEvidence(
                feature=f"FUNDAMENTAL_FLAG:{flag.code.value}",
                value=flag.penalty,
                criterion="penalty = 0",
                side=LeaderEvidenceSide.OPPOSING,
                rationale=flag.rationale,
            )
            for flag in fundamental.flags
        )
        supporting = tuple(item for item in evidence if item.side is LeaderEvidenceSide.SUPPORTING)
        opposing = tuple(item for item in evidence if item.side is LeaderEvidenceSide.OPPOSING)
        return supporting, opposing

    def _ineligibility_reasons(
        self,
        *,
        leader_type: LeaderType,
        score: Decimal,
        tradeability: TradeabilitySnapshot,
    ) -> tuple[str, ...]:
        reasons = [
            f"{item.code.value}: {item.message}" for item in tradeability.reasons if item.blocking
        ]
        if not tradeability.eligible and not reasons:
            reasons.append("TRADEABILITY_INELIGIBLE: upstream eligibility gate failed")
        if leader_type not in {LeaderType.TREND, LeaderType.CAPACITY}:
            reasons.append(f"LEADER_TYPE_NOT_CANDIDATE: {leader_type.value}")
        if score < self._config.minimum_candidate_score:
            reasons.append(
                f"LEADER_SCORE_BELOW_MINIMUM: {score} < {self._config.minimum_candidate_score}"
            )
        return tuple(reasons)

    @staticmethod
    def _conditions(leader_type: LeaderType) -> tuple[tuple[str, ...], tuple[str, ...]]:
        observations = {
            LeaderType.TREND: (
                "trend quality and within-theme strength remain above classification thresholds",
                "pullbacks remain controlled while benchmark-relative strength persists",
            ),
            LeaderType.CAPACITY: (
                "account-sized capacity and turnover percentile remain sufficient",
                "theme leadership remains broad rather than a one-session spike",
            ),
            LeaderType.ELASTICITY: (
                "short-horizon strength develops durable trend and liquidity confirmation",
            ),
            LeaderType.OBSERVATION: (
                "at least one trend or capacity classification path becomes confirmed",
            ),
        }[leader_type]
        invalidations = (
            "industry leaves CONFIRMED or CROWDED mainline state",
            "stock ceases to be a point-in-time member of the ranked industry",
            "upstream strength, tradeability, or fundamental identity changes materially",
        )
        return observations, invalidations

    @staticmethod
    def _upstream_identity(inputs: _AlignedInputs) -> LeaderUpstreamIdentity:
        stock = inputs.stock
        trade = inputs.tradeability
        fundamental = inputs.fundamental
        mainline = inputs.mainline_snapshot
        return LeaderUpstreamIdentity(
            mainline_model_version=mainline.model_version,
            mainline_result_hash=mainline.result_hash,
            stock_feature_version=stock.feature_version,
            stock_config_hash=stock.config_hash,
            stock_input_hash=stock.input_hash,
            stock_result_hash=stock.result_hash,
            tradeability_feature_version=trade.feature_version,
            tradeability_config_hash=trade.config_hash,
            tradeability_rule_version=trade.rule_version,
            tradeability_rule_hash=trade.rule_hash,
            tradeability_result_hash=trade.result_hash,
            fundamental_feature_version=fundamental.feature_version,
            fundamental_config_hash=fundamental.config_hash,
            fundamental_data_version=fundamental.data_version,
            fundamental_input_hash=fundamental.input_hash,
            fundamental_result_hash=fundamental.result_hash,
        )

    def _assign_ranks(
        self, provisional: tuple[_ProvisionalLeader, ...]
    ) -> tuple[LeaderResult, ...]:
        global_order = sorted(provisional, key=lambda item: (-item.score, item.instrument_id))
        overall_ranks = {
            item.instrument_id: rank for rank, item in enumerate(global_order, start=1)
        }
        grouped: dict[str, list[_ProvisionalLeader]] = defaultdict(list)
        for item in provisional:
            grouped[item.industry_id].append(item)
        theme_ranks = {
            item.instrument_id: rank
            for values in grouped.values()
            for rank, item in enumerate(
                sorted(values, key=lambda value: (-value.score, value.instrument_id)),
                start=1,
            )
        }
        return tuple(
            LeaderResult(
                instrument_id=item.instrument_id,
                industry_id=item.industry_id,
                mainline_state=item.mainline_state,
                overall_rank=overall_ranks[item.instrument_id],
                theme_rank=theme_ranks[item.instrument_id],
                leader_type=item.leader_type,
                gross_score=item.gross_score,
                risk_penalty=item.risk_penalty,
                score=item.score,
                components=item.components,
                supporting_evidence=item.supporting_evidence,
                counter_evidence=item.counter_evidence,
                risks=item.risks,
                candidate_eligible=item.candidate_eligible,
                ineligibility_reasons=item.ineligibility_reasons,
                observation_conditions=item.observation_conditions,
                invalidations=item.invalidations,
                upstream_identity=item.upstream_identity,
            )
            for item in global_order
        )

    def _input_hash(
        self,
        *,
        mainline: MainlineSnapshot,
        stocks: dict[str, StockStrengthSnapshot],
        trades: dict[str, TradeabilitySnapshot],
        fundamentals: dict[str, FundamentalQualitySnapshot],
    ) -> str:
        return stable_leader_hash(
            {
                "config_hash": self._config.config_hash,
                "fundamentals": [
                    fundamentals[key].identity_payload() for key in sorted(fundamentals)
                ],
                "mainline": mainline.identity_payload(),
                "stocks": [stocks[key].identity_payload() for key in sorted(stocks)],
                "tradeability": [
                    {
                        "config_hash": trades[key].config_hash,
                        "request": trades[key].request.fingerprint_payload(),
                        "result_hash": trades[key].result_hash,
                        "rule_hash": trades[key].rule_hash,
                        "rule_version": trades[key].rule_version,
                    }
                    for key in sorted(trades)
                ],
            }
        )

    @staticmethod
    def _result_hash(
        *,
        input_hash: str,
        leaders: tuple[LeaderResult, ...],
        exclusions: tuple[LeaderExclusion, ...],
    ) -> str:
        return stable_leader_hash(
            {
                "exclusions": [asdict(item) for item in exclusions],
                "input_hash": input_hash,
                "leaders": [asdict(item) for item in leaders],
            }
        )


__all__ = ["LeaderEngine"]
