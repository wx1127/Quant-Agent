"""Cross-sectional candidate scoring over aligned point-in-time upstream snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from decimal import Decimal

from quant_agent.features.fundamental_quality import FundamentalQualitySnapshot
from quant_agent.features.mainline import MainlineIndustryResult, MainlineSnapshot
from quant_agent.features.stock_strength import StockStrengthSnapshot
from quant_agent.features.tradeability import TradeabilitySnapshot
from quant_agent.leaders import LeaderResult, LeaderSnapshot
from quant_agent.leaders.candidates.contracts import (
    CandidateComponent,
    CandidateComponentScore,
    CandidateConfig,
    CandidateEvidence,
    CandidateEvidenceSide,
    CandidateExclusion,
    CandidateExclusionCode,
    CandidateInputError,
    CandidateResult,
    CandidateRisk,
    CandidateRiskCode,
    CandidateSignalKind,
    CandidateSnapshot,
    CandidateSupplementalSignal,
    CandidateTier,
    CandidateUpstreamIdentity,
)
from quant_agent.leaders.contracts import stable_leader_hash
from quant_agent.regime import MarketRegime, RegimeTransitionResult


def _signed_score(value: Decimal) -> Decimal:
    return max(Decimal(0), min(Decimal(100), (value + Decimal(100)) / 2))


@dataclass(frozen=True, slots=True)
class _AlignedCandidate:
    leader: LeaderResult
    theme: MainlineIndustryResult
    stock: StockStrengthSnapshot
    tradeability: TradeabilitySnapshot
    fundamental: FundamentalQualitySnapshot
    event: CandidateSupplementalSignal | None
    valuation: CandidateSupplementalSignal | None


@dataclass(frozen=True, slots=True)
class _ProvisionalCandidate:
    instrument_id: str
    industry_id: str
    gross_score: Decimal
    risk_penalty: Decimal
    score: Decimal
    components: tuple[CandidateComponentScore, ...]
    supporting_evidence: tuple[CandidateEvidence, ...]
    counter_evidence: tuple[CandidateEvidence, ...]
    risks: tuple[CandidateRisk, ...]
    exclusion_codes: tuple[CandidateExclusionCode, ...]
    exclusion_reasons: tuple[str, ...]
    observation_conditions: tuple[str, ...]
    invalidations: tuple[str, ...]
    upstream_identity: CandidateUpstreamIdentity

    @property
    def eligible(self) -> bool:
        return not self.exclusion_codes


class CandidateEngine:
    """Build an uncalibrated A/B/watch pool without making price promises."""

    def __init__(self, config: CandidateConfig | None = None) -> None:
        self._config = config or CandidateConfig()

    @property
    def config(self) -> CandidateConfig:
        return self._config

    def rank(
        self,
        *,
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
        stock_strength: Iterable[StockStrengthSnapshot],
        tradeability: Iterable[TradeabilitySnapshot],
        fundamentals: Iterable[FundamentalQualitySnapshot],
        supplemental_signals: Iterable[CandidateSupplementalSignal] = (),
    ) -> CandidateSnapshot:
        """Align every upstream identity and return deterministic candidate ranks."""

        self._validate_top_level(regime, mainline, leaders)
        expected_ids = {item.instrument_id for item in leaders.leaders}
        stocks = self._unique_map(
            stock_strength,
            identity=lambda item: item.instrument_id,
            label="stock strength",
        )
        trades = self._unique_map(
            tradeability,
            identity=lambda item: item.request.instrument_id,
            label="tradeability",
        )
        fundamental_map = self._unique_map(
            fundamentals,
            identity=lambda item: item.instrument_id,
            label="fundamental",
        )
        if (
            set(stocks) != expected_ids
            or set(trades) != expected_ids
            or set(fundamental_map) != expected_ids
        ):
            raise CandidateInputError(
                "candidate feature snapshots must cover exactly the ranked leader IDs"
            )
        supplemental = self._supplemental_map(
            supplemental_signals,
            expected_ids=expected_ids,
            mainline=mainline,
        )
        themes = {item.industry_id: item for item in mainline.industries}
        aligned: list[_AlignedCandidate] = []
        for leader in leaders.leaders:
            theme = themes.get(leader.industry_id)
            if theme is None:
                raise CandidateInputError("leader industry is absent from the mainline snapshot")
            stock = stocks[leader.instrument_id]
            trade = trades[leader.instrument_id]
            fundamental = fundamental_map[leader.instrument_id]
            self._validate_candidate_alignment(
                mainline=mainline,
                leaders=leaders,
                leader=leader,
                stock=stock,
                tradeability=trade,
                fundamental=fundamental,
            )
            aligned.append(
                _AlignedCandidate(
                    leader=leader,
                    theme=theme,
                    stock=stock,
                    tradeability=trade,
                    fundamental=fundamental,
                    event=supplemental.get((leader.instrument_id, CandidateSignalKind.EVENT)),
                    valuation=supplemental.get(
                        (leader.instrument_id, CandidateSignalKind.VALUATION)
                    ),
                )
            )

        provisional = tuple(self._score(regime, leaders, item) for item in aligned)
        candidates = self._assign_ranks(provisional)
        exclusions = tuple(
            CandidateExclusion(
                instrument_id=item.instrument_id,
                industry_id=item.industry_id,
                code=CandidateExclusionCode.UPSTREAM_LEADER_EXCLUSION,
                reason=f"{item.code.value}: {item.reason}",
                source_hash=item.stock_result_hash,
            )
            for item in leaders.exclusions
        )
        input_hash = self._input_hash(
            regime=regime,
            mainline=mainline,
            leaders=leaders,
            stocks=stocks,
            trades=trades,
            fundamentals=fundamental_map,
            supplemental=supplemental,
        )
        result_hash = stable_leader_hash(
            {
                "candidates": [asdict(item) for item in candidates],
                "exclusions": [asdict(item) for item in exclusions],
                "input_hash": input_hash,
            }
        )
        return CandidateSnapshot(
            session_date=mainline.session_date,
            as_of=mainline.as_of,
            data_version=mainline.data_version,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            regime=regime.final_regime,
            candidates=candidates,
            exclusions=exclusions,
            calibrated=False,
            calibration_version=None,
            input_hash=input_hash,
            result_hash=result_hash,
        )

    analyze = rank

    @staticmethod
    def _unique_map[T](
        values: Iterable[T],
        *,
        identity: Callable[[T], str],
        label: str,
    ) -> dict[str, T]:
        result: dict[str, T] = {}
        for value in values:
            key = identity(value)
            if not isinstance(key, str) or not key:
                raise CandidateInputError(f"{label} identity must be a non-empty string")
            if key in result:
                raise CandidateInputError(f"{label} instrument IDs must be unique")
            result[key] = value
        return result

    @staticmethod
    def _supplemental_map(
        values: Iterable[CandidateSupplementalSignal],
        *,
        expected_ids: set[str],
        mainline: MainlineSnapshot,
    ) -> dict[tuple[str, CandidateSignalKind], CandidateSupplementalSignal]:
        result: dict[tuple[str, CandidateSignalKind], CandidateSupplementalSignal] = {}
        for value in values:
            if not isinstance(value, CandidateSupplementalSignal):
                raise CandidateInputError(
                    "supplemental inputs must be immutable CandidateSupplementalSignal values"
                )
            if value.instrument_id not in expected_ids:
                raise CandidateInputError("supplemental signal references an unranked instrument")
            if value.session_date != mainline.session_date or value.as_of > mainline.as_of:
                raise CandidateInputError("supplemental signal contains future or misdated data")
            if value.data_version != mainline.data_version:
                raise CandidateInputError("supplemental signal data_version must match mainline")
            key = (value.instrument_id, value.kind)
            if key in result:
                raise CandidateInputError("candidate supplemental signal kinds must be unique")
            result[key] = value
        return result

    @staticmethod
    def _validate_top_level(
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
    ) -> None:
        raw = regime.raw_result
        if (
            raw.session_date != mainline.session_date
            or regime.event.session_date != mainline.session_date
            or raw.as_of != mainline.as_of
            or regime.event.as_of != mainline.as_of
        ):
            raise CandidateInputError("regime, mainline, and leader dates must align exactly")
        if raw.data_version != mainline.data_version:
            raise CandidateInputError("regime data_version must match mainline")
        if mainline.market_regime is not regime.final_regime:
            raise CandidateInputError("mainline market_regime must match the stabilized regime")
        identity = mainline.input_identity
        if (
            identity.regime_transition_result_hash != regime.result_hash
            or identity.regime_transition_version != regime.transition_config_version
            or identity.regime_transition_config_hash != regime.transition_config_hash
        ):
            raise CandidateInputError("mainline does not bind the supplied stabilized regime")
        if (
            leaders.session_date != mainline.session_date
            or leaders.as_of != mainline.as_of
            or leaders.data_version != mainline.data_version
            or leaders.classification_version != mainline.classification_version
            or leaders.industry_level != mainline.industry_level
        ):
            raise CandidateInputError("leader snapshot must align exactly with mainline")
        if leaders.mainline_result_hash != mainline.result_hash:
            raise CandidateInputError("leader snapshot does not bind the supplied mainline")
        if regime.event.source_result_hash != raw.result_hash:
            raise CandidateInputError("regime transition does not bind its raw classification")

    @staticmethod
    def _validate_candidate_alignment(
        *,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
        leader: LeaderResult,
        stock: StockStrengthSnapshot,
        tradeability: TradeabilitySnapshot,
        fundamental: FundamentalQualitySnapshot,
    ) -> None:
        request = tradeability.request
        if (
            stock.instrument_id != leader.instrument_id
            or fundamental.instrument_id != leader.instrument_id
        ):
            raise CandidateInputError("candidate upstream instrument IDs do not match the leader")
        if stock.current_industry_id != leader.industry_id:
            raise CandidateInputError("stock PIT industry membership does not match the leader")
        if (
            stock.session_date != mainline.session_date
            or request.session_date != mainline.session_date
            or stock.as_of > mainline.as_of
            or request.as_of > mainline.as_of
            or fundamental.as_of > mainline.as_of
        ):
            raise CandidateInputError("candidate feature snapshots contain future or misdated data")
        identity = leader.upstream_identity
        if (
            identity.mainline_result_hash != mainline.result_hash
            or identity.stock_result_hash != stock.result_hash
            or identity.tradeability_result_hash != tradeability.result_hash
            or identity.fundamental_result_hash != fundamental.result_hash
        ):
            raise CandidateInputError("leader upstream identities do not match candidate inputs")
        if leaders.mainline_result_hash != identity.mainline_result_hash:
            raise CandidateInputError(
                "leader result and leader snapshot mainline identities differ"
            )

    def _score(
        self,
        regime: RegimeTransitionResult,
        leaders: LeaderSnapshot,
        inputs: _AlignedCandidate,
    ) -> _ProvisionalCandidate:
        event_score = inputs.event.score if inputs.event is not None else Decimal(0)
        fundamental_or_event = max(
            inputs.fundamental.adjusted_quality_score,
            event_score,
        )
        valuation_score = inputs.valuation.score if inputs.valuation is not None else Decimal(0)
        trend = (
            _signed_score(inputs.stock.trend_quality.score)
            if inputs.stock.trend_quality is not None
            else Decimal(0)
        )
        volume_price = (
            _signed_score(inputs.stock.volume_confirmation.score)
            if inputs.stock.volume_confirmation is not None
            else Decimal(0)
        )
        values = {
            CandidateComponent.MARKET_REGIME: self._config.regime_scores[regime.final_regime],
            CandidateComponent.THEME_SCORE: inputs.theme.mainline_score,
            CandidateComponent.LEADER_SCORE: inputs.leader.gross_score,
            CandidateComponent.PRICE_TREND: trend,
            CandidateComponent.VOLUME_PRICE_STRUCTURE: volume_price,
            CandidateComponent.FUNDAMENTAL_OR_EVENT: fundamental_or_event,
            CandidateComponent.VALUATION: valuation_score,
        }
        sources = {
            CandidateComponent.MARKET_REGIME: regime.result_hash,
            CandidateComponent.THEME_SCORE: inputs.leader.upstream_identity.mainline_result_hash,
            CandidateComponent.LEADER_SCORE: leaders.result_hash,
            CandidateComponent.PRICE_TREND: inputs.stock.result_hash,
            CandidateComponent.VOLUME_PRICE_STRUCTURE: inputs.stock.result_hash,
            CandidateComponent.FUNDAMENTAL_OR_EVENT: (
                inputs.event.result_hash
                if inputs.event is not None
                and inputs.event.score >= inputs.fundamental.adjusted_quality_score
                else inputs.fundamental.result_hash
            ),
            CandidateComponent.VALUATION: (
                inputs.valuation.result_hash
                if inputs.valuation is not None
                else "MISSING_PIT_VALUATION"
            ),
        }
        components = tuple(
            CandidateComponentScore(
                component=component,
                score=values[component],
                weight=self._config.component_weights[component],
                contribution=values[component] * self._config.component_weights[component],
                source=sources[component],
            )
            for component in CandidateComponent
        )
        gross_score = sum((item.contribution for item in components), Decimal(0))
        risks = self._risks(inputs)
        risk_penalty = sum((item.penalty for item in risks), Decimal(0))
        score = max(Decimal(0), gross_score - risk_penalty)
        exclusion_codes, exclusion_reasons = self._hard_filters(regime, inputs.leader)
        supporting, opposing = self._evidence(components, risks, exclusion_reasons)
        observations, invalidations = self._conditions(inputs, score)
        return _ProvisionalCandidate(
            instrument_id=inputs.leader.instrument_id,
            industry_id=inputs.leader.industry_id,
            gross_score=gross_score,
            risk_penalty=risk_penalty,
            score=score,
            components=components,
            supporting_evidence=supporting,
            counter_evidence=opposing,
            risks=risks,
            exclusion_codes=exclusion_codes,
            exclusion_reasons=exclusion_reasons,
            observation_conditions=observations,
            invalidations=invalidations,
            upstream_identity=CandidateUpstreamIdentity(
                regime_transition_version=regime.transition_config_version,
                regime_transition_hash=regime.result_hash,
                mainline_model_version=inputs.leader.upstream_identity.mainline_model_version,
                mainline_result_hash=inputs.leader.upstream_identity.mainline_result_hash,
                leader_feature_version=leaders.feature_version,
                leader_result_hash=leaders.result_hash,
                stock_result_hash=inputs.stock.result_hash,
                tradeability_result_hash=inputs.tradeability.result_hash,
                fundamental_result_hash=inputs.fundamental.result_hash,
                event_result_hash=inputs.event.result_hash if inputs.event else None,
                valuation_result_hash=(inputs.valuation.result_hash if inputs.valuation else None),
            ),
        )

    def _risks(self, inputs: _AlignedCandidate) -> tuple[CandidateRisk, ...]:
        candidates: list[tuple[CandidateRiskCode, Decimal, str]] = []
        if inputs.leader.risk_penalty > 0:
            candidates.append(
                (
                    CandidateRiskCode.LEADER,
                    inputs.leader.risk_penalty,
                    "versioned leader risk penalties",
                )
            )
        if inputs.event is not None and inputs.event.risk_penalty > 0:
            candidates.append(
                (
                    CandidateRiskCode.EVENT,
                    inputs.event.risk_penalty,
                    "point-in-time event risk evidence",
                )
            )
        if inputs.valuation is not None and inputs.valuation.risk_penalty > 0:
            candidates.append(
                (
                    CandidateRiskCode.VALUATION,
                    inputs.valuation.risk_penalty,
                    "point-in-time valuation risk evidence",
                )
            )
        remaining = self._config.maximum_risk_penalty
        result: list[CandidateRisk] = []
        for code, penalty, rationale in candidates:
            applied = min(remaining, penalty)
            if applied <= 0:
                continue
            result.append(CandidateRisk(code=code, penalty=applied, rationale=rationale))
            remaining -= applied
            if remaining <= 0:
                break
        return tuple(result)

    def _hard_filters(
        self,
        regime: RegimeTransitionResult,
        leader: LeaderResult,
    ) -> tuple[tuple[CandidateExclusionCode, ...], tuple[str, ...]]:
        codes: list[CandidateExclusionCode] = []
        reasons: list[str] = []
        if not leader.candidate_eligible:
            codes.append(CandidateExclusionCode.LEADER_FILTER)
            reasons.extend(leader.ineligibility_reasons)
        if (
            regime.final_regime is MarketRegime.DOWNTREND
            and not self._config.allow_downtrend_candidates
        ):
            codes.append(CandidateExclusionCode.MARKET_DOWNTREND)
            reasons.append("MARKET_DOWNTREND: configured policy blocks tradeable candidates")
        return tuple(codes), tuple(reasons)

    def _evidence(
        self,
        components: tuple[CandidateComponentScore, ...],
        risks: tuple[CandidateRisk, ...],
        exclusion_reasons: tuple[str, ...],
    ) -> tuple[tuple[CandidateEvidence, ...], tuple[CandidateEvidence, ...]]:
        values = [
            CandidateEvidence(
                feature=item.component.value,
                value=item.score,
                criterion=f">= {self._config.evidence_support_minimum}",
                side=(
                    CandidateEvidenceSide.SUPPORTING
                    if item.score >= self._config.evidence_support_minimum
                    else CandidateEvidenceSide.OPPOSING
                ),
                rationale=(f"{item.component.value.lower()} is an uncalibrated score component"),
            )
            for item in components
        ]
        values.extend(
            CandidateEvidence(
                feature=f"RISK:{risk.code.value}",
                value=risk.penalty,
                criterion="penalty = 0",
                side=CandidateEvidenceSide.OPPOSING,
                rationale=risk.rationale,
            )
            for risk in risks
        )
        values.extend(
            CandidateEvidence(
                feature="HARD_FILTER",
                value=Decimal(0),
                criterion="no blocking reason",
                side=CandidateEvidenceSide.OPPOSING,
                rationale=reason,
            )
            for reason in exclusion_reasons
        )
        return (
            tuple(item for item in values if item.side is CandidateEvidenceSide.SUPPORTING),
            tuple(item for item in values if item.side is CandidateEvidenceSide.OPPOSING),
        )

    def _conditions(
        self,
        inputs: _AlignedCandidate,
        score: Decimal,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        observations = (
            f"uncalibrated candidate score remains near or above {score}",
            "market, mainline, trend, volume-price, and risk identities remain current",
            (
                "obtain PIT valuation evidence before interpreting the valuation component"
                if inputs.valuation is None
                else "valuation score remains supported by its point-in-time source"
            ),
        )
        invalidations = tuple(
            dict.fromkeys(
                (
                    "stabilized market regime changes or loses its current risk budget",
                    "mainline industry is no longer confirmed or becomes invalid",
                    *inputs.leader.invalidations,
                    "tradeability becomes ineligible at the decision account size",
                    "new negative fundamental or event evidence changes the risk score",
                )
            )
        )
        return observations, invalidations

    def _assign_ranks(
        self,
        values: tuple[_ProvisionalCandidate, ...],
    ) -> tuple[CandidateResult, ...]:
        ordered = sorted(values, key=lambda item: (-item.score, item.instrument_id))
        eligible_ids = [item.instrument_id for item in ordered if item.eligible]
        eligible_rank = {
            instrument_id: rank for rank, instrument_id in enumerate(eligible_ids, start=1)
        }
        result: list[CandidateResult] = []
        for research_rank, item in enumerate(ordered, start=1):
            if not item.eligible:
                tier = CandidateTier.EXCLUDED
            elif item.score >= self._config.tier_a_minimum:
                tier = CandidateTier.A
            elif item.score >= self._config.tier_b_minimum:
                tier = CandidateTier.B
            else:
                tier = CandidateTier.WATCH
            result.append(
                CandidateResult(
                    instrument_id=item.instrument_id,
                    industry_id=item.industry_id,
                    research_rank=research_rank,
                    eligible_rank=eligible_rank.get(item.instrument_id),
                    tier=tier,
                    gross_score=item.gross_score,
                    risk_penalty=item.risk_penalty,
                    score=item.score,
                    components=item.components,
                    supporting_evidence=item.supporting_evidence,
                    counter_evidence=item.counter_evidence,
                    risks=item.risks,
                    exclusion_codes=item.exclusion_codes,
                    exclusion_reasons=item.exclusion_reasons,
                    observation_conditions=item.observation_conditions,
                    invalidations=item.invalidations,
                    upstream_identity=item.upstream_identity,
                )
            )
        return tuple(result)

    def _input_hash(
        self,
        *,
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
        stocks: dict[str, StockStrengthSnapshot],
        trades: dict[str, TradeabilitySnapshot],
        fundamentals: dict[str, FundamentalQualitySnapshot],
        supplemental: dict[tuple[str, CandidateSignalKind], CandidateSupplementalSignal],
    ) -> str:
        return stable_leader_hash(
            {
                "config_hash": self._config.config_hash,
                "fundamentals": [
                    fundamentals[key].identity_payload() for key in sorted(fundamentals)
                ],
                "leader_result_hash": leaders.result_hash,
                "mainline": mainline.identity_payload(),
                "regime_result_hash": regime.result_hash,
                "stocks": [stocks[key].identity_payload() for key in sorted(stocks)],
                "supplemental": [
                    supplemental[key].identity_payload()
                    for key in sorted(
                        supplemental,
                        key=lambda item: (item[0], item[1].value),
                    )
                ],
                "tradeability": [
                    {
                        "request": trades[key].request.fingerprint_payload(),
                        "result_hash": trades[key].result_hash,
                    }
                    for key in sorted(trades)
                ],
            }
        )


__all__ = ["CandidateEngine"]
