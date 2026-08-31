"""Deterministic mainline-leader target portfolio construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from decimal import Decimal, localcontext

from quant_agent.features.mainline import MainlineIndustryResult, MainlineSnapshot, MainlineState
from quant_agent.leaders import LeaderResult, LeaderSnapshot
from quant_agent.leaders.candidates import CandidateResult, CandidateSnapshot
from quant_agent.regime import MarketRegime, RegimeTransitionResult
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.mainline_leader.contracts import (
    MainlineLeaderAction,
    MainlineLeaderConfig,
    MainlineLeaderDecision,
    MainlineLeaderDecisionStatus,
    MainlineLeaderInputError,
    MainlineLeaderPortfolioState,
    MainlineLeaderReasonCode,
    MainlineLeaderRequest,
    MainlineLeaderSelection,
    MainlineLeaderTargetWeight,
    MainlineLeaderWeightingMode,
)


def _divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        raise ZeroDivisionError("mainline-leader allocation denominator cannot be zero")
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def _action(prior: Decimal, target: Decimal) -> MainlineLeaderAction:
    if prior == 0 and target == 0:
        return MainlineLeaderAction.EXCLUDE
    if prior == 0:
        return MainlineLeaderAction.ENTER
    if target == 0:
        return MainlineLeaderAction.EXIT
    if target > prior:
        return MainlineLeaderAction.INCREASE
    if target < prior:
        return MainlineLeaderAction.REDUCE
    return MainlineLeaderAction.HOLD


def _append_reason(
    reasons: dict[str, list[MainlineLeaderReasonCode]],
    instrument_id: str,
    code: MainlineLeaderReasonCode,
) -> None:
    if code not in reasons[instrument_id]:
        reasons[instrument_id].append(code)


class MainlineLeaderStrategy:
    """Convert aligned PIT research snapshots into complete stock target weights."""

    def __init__(self, config: MainlineLeaderConfig | None = None) -> None:
        self._config = config or MainlineLeaderConfig()

    @property
    def config(self) -> MainlineLeaderConfig:
        return self._config

    def decide(
        self,
        *,
        request: MainlineLeaderRequest,
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
        candidates: CandidateSnapshot,
        portfolio: MainlineLeaderPortfolioState,
    ) -> MainlineLeaderDecision:
        """Build one replayable target portfolio without creating execution orders."""

        self._validate_alignment(request, regime, mainline, leaders, candidates, portfolio)
        leader_by_id = {item.instrument_id: item for item in leaders.leaders}
        candidate_by_id = {item.instrument_id: item for item in candidates.candidates}
        mainline_by_id = {item.industry_id: item for item in mainline.industries}
        prior_by_id = {item.instrument_id: item for item in portfolio.positions}
        universe_ids = tuple(sorted(set(candidate_by_id) | set(prior_by_id)))
        reasons: dict[str, list[MainlineLeaderReasonCode]] = {
            instrument_id: [] for instrument_id in universe_ids
        }

        exposure_cap = min(
            self._config.exposure_for(regime.final_regime),
            regime.raw_result.risk_budget_max,
        )
        market_risk_off = (
            self._config.force_downtrend_cash and regime.final_regime is MarketRegime.DOWNTREND
        ) or exposure_cap == 0

        qualified: dict[str, bool] = {}
        for instrument_id in universe_ids:
            candidate = candidate_by_id.get(instrument_id)
            leader = leader_by_id.get(instrument_id)
            prior_position = prior_by_id.get(instrument_id)
            if candidate is None and leader is None and prior_position is None:
                raise AssertionError("strategy universe ID has no input source")
            if candidate is not None:
                industry_id = candidate.industry_id
            elif leader is not None:
                industry_id = leader.industry_id
            else:
                assert prior_position is not None
                industry_id = prior_position.industry_id
            state = mainline_by_id.get(industry_id)
            item_reasons = reasons[instrument_id]
            if market_risk_off:
                item_reasons.append(MainlineLeaderReasonCode.MARKET_RISK_OFF)
            if candidate is None:
                item_reasons.append(MainlineLeaderReasonCode.PRIOR_POSITION_INVALID)
            else:
                if not candidate.eligible:
                    item_reasons.append(MainlineLeaderReasonCode.CANDIDATE_INELIGIBLE)
                if candidate.tier not in self._config.eligible_tiers:
                    item_reasons.append(MainlineLeaderReasonCode.TIER_FILTER)
                if candidate.score < self._config.minimum_candidate_score:
                    item_reasons.append(MainlineLeaderReasonCode.SCORE_FILTER)
            if leader is None or not leader.candidate_eligible:
                item_reasons.append(MainlineLeaderReasonCode.LEADER_INELIGIBLE)
            elif leader.leader_type not in self._config.eligible_leader_types:
                item_reasons.append(MainlineLeaderReasonCode.LEADER_TYPE_FILTER)
            if state is None:
                item_reasons.append(MainlineLeaderReasonCode.MAINLINE_INACTIVE)
            elif state.state is MainlineState.FADING:
                item_reasons.append(MainlineLeaderReasonCode.MAINLINE_FADING)
            elif state.state not in self._config.eligible_mainline_states:
                item_reasons.append(MainlineLeaderReasonCode.MAINLINE_INACTIVE)
            qualified[instrument_id] = not item_reasons

        ranked = sorted(
            (
                candidate
                for instrument_id, candidate in candidate_by_id.items()
                if qualified[instrument_id]
            ),
            key=lambda item: (item.eligible_rank or len(candidate_by_id) + 1, item.instrument_id),
        )
        selected = tuple(ranked[: self._config.maximum_positions])
        selected_ids = {item.instrument_id for item in selected}
        for instrument_id, is_qualified in qualified.items():
            if not is_qualified:
                continue
            _append_reason(
                reasons,
                instrument_id,
                (
                    MainlineLeaderReasonCode.SELECTED
                    if instrument_id in selected_ids
                    else MainlineLeaderReasonCode.OUTSIDE_TOP_N
                ),
            )

        allocations = self._allocate(
            selected=selected,
            mainline_by_id=mainline_by_id,
            exposure=exposure_cap,
        )
        desired = {
            instrument_id: allocations.get(instrument_id, Decimal(0))
            for instrument_id in universe_ids
        }
        prior_weights = {
            instrument_id: (
                prior_by_id[instrument_id].target_weight
                if instrument_id in prior_by_id
                else Decimal(0)
            )
            for instrument_id in universe_ids
        }

        forced_ids = {
            instrument_id
            for instrument_id in prior_by_id
            if market_risk_off or not qualified.get(instrument_id, False)
        }
        base = dict(prior_weights)
        for instrument_id in forced_ids:
            base[instrument_id] = Decimal(0)

        exposure_reduced_ids: set[str] = set()
        base_gross = sum(base.values(), Decimal(0))
        if base_gross > exposure_cap:
            scale = _divide(exposure_cap, base_gross)
            for instrument_id, value in tuple(base.items()):
                if value > 0:
                    base[instrument_id] = value * scale
                    exposure_reduced_ids.add(instrument_id)
                    _append_reason(
                        reasons,
                        instrument_id,
                        MainlineLeaderReasonCode.EXPOSURE_CAP,
                    )

        scheduled = request.session_index % self._config.rebalance_frequency_sessions == 0
        if scheduled and not market_risk_off:
            raw_discretionary = sum(
                (abs(desired[item] - base[item]) for item in universe_ids),
                Decimal(0),
            )
            if raw_discretionary > self._config.maximum_discretionary_turnover:
                factor = _divide(
                    self._config.maximum_discretionary_turnover,
                    raw_discretionary,
                )
                target = {
                    instrument_id: base[instrument_id]
                    + (desired[instrument_id] - base[instrument_id]) * factor
                    for instrument_id in universe_ids
                }
                for instrument_id in universe_ids:
                    if target[instrument_id] != desired[instrument_id]:
                        _append_reason(
                            reasons,
                            instrument_id,
                            MainlineLeaderReasonCode.TURNOVER_CAP,
                        )
            else:
                target = desired
        else:
            target = base
            if not market_risk_off:
                for instrument_id in universe_ids:
                    if target[instrument_id] != desired[instrument_id]:
                        _append_reason(
                            reasons,
                            instrument_id,
                            MainlineLeaderReasonCode.REBALANCE_SCHEDULE,
                        )

        total_turnover = sum(
            (abs(target[item] - prior_weights[item]) for item in universe_ids),
            Decimal(0),
        )
        forced_exit_turnover = sum(
            (
                abs(target[item] - prior_weights[item])
                for item in universe_ids
                if item in forced_ids
                or (item in exposure_reduced_ids and target[item] < prior_weights[item])
            ),
            Decimal(0),
        )
        discretionary_turnover = total_turnover - forced_exit_turnover
        if discretionary_turnover > self._config.maximum_discretionary_turnover:
            raise AssertionError("turnover interpolation exceeded the configured cap")

        selections = tuple(
            self._selection(
                instrument_id=instrument_id,
                candidate=candidate_by_id.get(instrument_id),
                leader=leader_by_id.get(instrument_id),
                mainline_by_id=mainline_by_id,
                qualified=qualified[instrument_id],
                desired=instrument_id in selected_ids,
                prior_weight=prior_weights[instrument_id],
                unconstrained_weight=desired[instrument_id],
                target_weight=target[instrument_id],
                reason_codes=tuple(reasons[instrument_id]),
                prior_industry=(
                    prior_by_id[instrument_id].industry_id if instrument_id in prior_by_id else None
                ),
            )
            for instrument_id in universe_ids
        )
        targets = tuple(
            MainlineLeaderTargetWeight(
                instrument_id=instrument_id,
                target_weight=target[instrument_id],
            )
            for instrument_id in universe_ids
        )
        gross = sum(target.values(), Decimal(0))
        status = (
            MainlineLeaderDecisionStatus.CASH
            if gross == 0
            else (
                MainlineLeaderDecisionStatus.HOLD
                if total_turnover == 0
                else MainlineLeaderDecisionStatus.REBALANCE
            )
        )
        reason = self._decision_reason(
            status=status,
            market_risk_off=market_risk_off,
            forced_exit_turnover=forced_exit_turnover,
            discretionary_turnover=discretionary_turnover,
        )
        input_hash = stable_hash(
            {
                "candidate_result_hash": candidates.result_hash,
                "config_hash": self._config.config_hash,
                "leader_result_hash": leaders.result_hash,
                "mainline_result_hash": mainline.result_hash,
                "portfolio_input_hash": portfolio.input_hash,
                "regime_result_hash": regime.result_hash,
                "request": request.fingerprint_payload(),
            }
        )
        result_hash = stable_hash(
            {
                "cash_target_weight": Decimal(1) - gross,
                "discretionary_turnover": discretionary_turnover,
                "exposure_cap": exposure_cap,
                "forced_exit_turnover": forced_exit_turnover,
                "gross_target_weight": gross,
                "input_hash": input_hash,
                "reason": reason,
                "selections": [asdict(item) for item in selections],
                "status": status,
                "targets": [asdict(item) for item in targets],
                "total_turnover": total_turnover,
            }
        )
        return MainlineLeaderDecision(
            signal_date=request.signal_date,
            as_of=request.as_of,
            session_index=request.session_index,
            data_version=request.data_version,
            strategy_version=self._config.version,
            config_hash=self._config.config_hash,
            status=status,
            market_regime=regime.final_regime,
            exposure_cap=exposure_cap,
            gross_target_weight=gross,
            cash_target_weight=Decimal(1) - gross,
            forced_exit_turnover=forced_exit_turnover,
            discretionary_turnover=discretionary_turnover,
            total_turnover=total_turnover,
            selections=selections,
            targets=targets,
            regime_result_hash=regime.result_hash,
            mainline_result_hash=mainline.result_hash,
            leader_result_hash=leaders.result_hash,
            candidate_result_hash=candidates.result_hash,
            portfolio_input_hash=portfolio.input_hash,
            input_hash=input_hash,
            result_hash=result_hash,
            reason=reason,
        )

    def _validate_alignment(
        self,
        request: MainlineLeaderRequest,
        regime: RegimeTransitionResult,
        mainline: MainlineSnapshot,
        leaders: LeaderSnapshot,
        candidates: CandidateSnapshot,
        portfolio: MainlineLeaderPortfolioState,
    ) -> None:
        raw = regime.raw_result
        event = regime.event
        if (
            raw.session_date != request.signal_date
            or event.session_date != request.signal_date
            or raw.as_of != request.as_of
            or event.as_of != request.as_of
        ):
            raise MainlineLeaderInputError("regime date/as_of must align to the strategy request")
        if raw.data_version != request.data_version:
            raise MainlineLeaderInputError("regime data_version must match the request")
        if event.source_result_hash != raw.result_hash:
            raise MainlineLeaderInputError("regime event does not bind its raw result")
        if (
            event.transition_config_version != regime.transition_config_version
            or event.transition_config_hash != regime.transition_config_hash
        ):
            raise MainlineLeaderInputError("regime transition identities are misaligned")
        if (
            mainline.session_date != request.signal_date
            or mainline.as_of != request.as_of
            or mainline.data_version != request.data_version
            or mainline.market_regime is not regime.final_regime
        ):
            raise MainlineLeaderInputError("mainline snapshot must align exactly to the request")
        if mainline.input_identity.regime_transition_result_hash != regime.result_hash:
            raise MainlineLeaderInputError("mainline snapshot does not bind the stabilized regime")
        if (
            leaders.session_date != request.signal_date
            or leaders.as_of != request.as_of
            or leaders.data_version != request.data_version
            or leaders.mainline_result_hash != mainline.result_hash
        ):
            raise MainlineLeaderInputError("leader snapshot must bind the current mainline")
        if (
            candidates.session_date != request.signal_date
            or candidates.as_of != request.as_of
            or candidates.data_version != request.data_version
            or candidates.regime is not regime.final_regime
        ):
            raise MainlineLeaderInputError("candidate snapshot must align exactly to the request")
        candidate_ids = {item.instrument_id for item in candidates.candidates}
        leader_ids = {item.instrument_id for item in leaders.leaders}
        if candidate_ids != leader_ids:
            raise MainlineLeaderInputError("candidate rows must cover current leaders exactly")
        leader_by_id = {item.instrument_id: item for item in leaders.leaders}
        for candidate in candidates.candidates:
            leader = leader_by_id[candidate.instrument_id]
            identity = candidate.upstream_identity
            if (
                candidate.industry_id != leader.industry_id
                or identity.regime_transition_version != regime.transition_config_version
                or identity.regime_transition_hash != regime.result_hash
                or identity.mainline_model_version != mainline.model_version
                or identity.mainline_result_hash != mainline.result_hash
                or identity.leader_feature_version != leaders.feature_version
                or identity.leader_result_hash != leaders.result_hash
            ):
                raise MainlineLeaderInputError(
                    "candidate upstream identities do not bind regime, mainline, and leaders"
                )
        if portfolio.as_of > request.as_of:
            raise MainlineLeaderInputError("future prior portfolio state is not allowed")
        if portfolio.data_version != request.data_version:
            raise MainlineLeaderInputError("prior portfolio data_version must match the request")
        if any(
            item.target_weight > self._config.maximum_instrument_weight
            for item in portfolio.positions
        ):
            raise MainlineLeaderInputError(
                "prior target weight exceeds the frozen per-instrument limit"
            )

    def _allocate(
        self,
        *,
        selected: tuple[CandidateResult, ...],
        mainline_by_id: Mapping[str, MainlineIndustryResult],
        exposure: Decimal,
    ) -> dict[str, Decimal]:
        if not selected or exposure <= 0:
            return {}
        basis: dict[str, Decimal] = {}
        for candidate in selected:
            if self._config.weighting_mode is MainlineLeaderWeightingMode.EQUAL:
                value = Decimal(1)
            else:
                value = candidate.score * _divide(
                    Decimal(100) - candidate.risk_penalty,
                    Decimal(100),
                )
            basis[candidate.instrument_id] = value
        if not any(basis.values()):
            basis = {instrument_id: Decimal(1) for instrument_id in basis}
        allocations = self._capped_allocation(basis, exposure)
        for candidate in selected:
            mainline = mainline_by_id[candidate.industry_id]
            if mainline.state is MainlineState.CROWDED:
                allocations[candidate.instrument_id] *= self._config.crowded_weight_multiplier
        return allocations

    def _capped_allocation(
        self,
        basis: dict[str, Decimal],
        exposure: Decimal,
    ) -> dict[str, Decimal]:
        remaining = dict(basis)
        allocations: dict[str, Decimal] = {}
        remaining_exposure = exposure
        while remaining and remaining_exposure > 0:
            total_basis = sum(remaining.values(), Decimal(0))
            if total_basis == 0:
                proposed = {
                    instrument_id: _divide(remaining_exposure, Decimal(len(remaining)))
                    for instrument_id in remaining
                }
            else:
                proposed = {
                    instrument_id: remaining_exposure * _divide(value, total_basis)
                    for instrument_id, value in remaining.items()
                }
            capped = tuple(
                sorted(
                    instrument_id
                    for instrument_id, value in proposed.items()
                    if value > self._config.maximum_instrument_weight
                )
            )
            if not capped:
                ordered_ids = tuple(sorted(proposed))
                assigned = Decimal(0)
                for instrument_id in ordered_ids[:-1]:
                    allocations[instrument_id] = proposed[instrument_id]
                    assigned += proposed[instrument_id]
                allocations[ordered_ids[-1]] = remaining_exposure - assigned
                break
            for instrument_id in capped:
                allocations[instrument_id] = self._config.maximum_instrument_weight
                remaining_exposure -= self._config.maximum_instrument_weight
                del remaining[instrument_id]
            if remaining_exposure <= 0:
                break
        return {instrument_id: allocations[instrument_id] for instrument_id in sorted(allocations)}

    @staticmethod
    def _selection(
        *,
        instrument_id: str,
        candidate: CandidateResult | None,
        leader: LeaderResult | None,
        mainline_by_id: Mapping[str, MainlineIndustryResult],
        qualified: bool,
        desired: bool,
        prior_weight: Decimal,
        unconstrained_weight: Decimal,
        target_weight: Decimal,
        reason_codes: tuple[MainlineLeaderReasonCode, ...],
        prior_industry: str | None,
    ) -> MainlineLeaderSelection:
        industry_id = (
            candidate.industry_id
            if candidate is not None
            else (leader.industry_id if leader is not None else prior_industry)
        )
        if industry_id is None:  # pragma: no cover - prior-only IDs always retain industry.
            raise AssertionError("selection industry identity is unavailable")
        mainline = mainline_by_id.get(industry_id)
        return MainlineLeaderSelection(
            instrument_id=instrument_id,
            industry_id=industry_id,
            candidate_rank=candidate.eligible_rank if candidate is not None else None,
            candidate_tier=candidate.tier if candidate is not None else None,
            candidate_score=candidate.score if candidate is not None else None,
            candidate_risk_penalty=candidate.risk_penalty if candidate is not None else None,
            leader_type=leader.leader_type if leader is not None else None,
            mainline_state=mainline.state if mainline is not None else None,
            qualified=qualified,
            desired=desired,
            prior_weight=prior_weight,
            unconstrained_weight=unconstrained_weight,
            target_weight=target_weight,
            action=_action(prior_weight, target_weight),
            reason_codes=reason_codes,
            rationale=", ".join(code.value for code in reason_codes),
        )

    @staticmethod
    def _decision_reason(
        *,
        status: MainlineLeaderDecisionStatus,
        market_risk_off: bool,
        forced_exit_turnover: Decimal,
        discretionary_turnover: Decimal,
    ) -> str:
        if market_risk_off:
            return "market regime or risk budget requires an all-cash stock target"
        if status is MainlineLeaderDecisionStatus.CASH:
            return "no current tradeable mainline leader passed the frozen entry rules"
        if status is MainlineLeaderDecisionStatus.HOLD:
            return "frozen rebalance and turnover rules retain the prior target portfolio"
        return (
            "target portfolio changed with "
            f"forced turnover {forced_exit_turnover} and discretionary turnover "
            f"{discretionary_turnover}"
        )


__all__ = ["MainlineLeaderStrategy"]
