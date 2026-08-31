"""Deterministic point-in-time ETF momentum and volatility rotation strategy."""

from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, date
from decimal import Decimal, localcontext
from itertools import pairwise

from quant_agent.backtest import ResearchPriceBar, TradableInstrumentType
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime import MarketRegime, RegimeTransitionResult
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.etf_rotation.contracts import (
    ETFHorizonMomentum,
    ETFMomentumMetrics,
    ETFRotationCandidate,
    ETFRotationConfig,
    ETFRotationDecision,
    ETFRotationDecisionStatus,
    ETFRotationInputError,
    ETFRotationRequest,
    ETFTargetWeight,
    ETFUniverseEligibility,
    ETFUniverseRevision,
    ETFUniverseSelection,
)


def _divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return _divide(sum(values, Decimal(0)), Decimal(len(values)))


def _annualized_volatility(
    closes: tuple[Decimal, ...],
    *,
    window: int,
    annualization_sessions: int,
) -> Decimal:
    selected = closes[-(window + 1) :]
    returns = tuple(_divide(current, previous) - 1 for previous, current in pairwise(selected))
    mean_return = _mean(returns)
    variance = _mean(tuple((value - mean_return) ** 2 for value in returns))
    with localcontext() as context:
        context.prec = 50
        return variance.sqrt() * Decimal(annualization_sessions).sqrt()


def _bar_payload(bar: ResearchPriceBar) -> dict[str, str]:
    return {
        "available_at": bar.available_at.astimezone(UTC).isoformat(timespec="microseconds"),
        "close_price": canonical_decimal(bar.close_price),
        "data_version": bar.data_version,
        "instrument_id": bar.instrument_id,
        "instrument_type": bar.instrument_type.value,
        "open_price": canonical_decimal(bar.open_price),
        "revision": bar.revision,
        "trading_day": bar.trading_day.isoformat(),
    }


class ETFRotationStrategy:
    """Generate complete long-only ETF targets from PIT data and stabilized regime."""

    def __init__(self, config: ETFRotationConfig | None = None) -> None:
        self._config = config or ETFRotationConfig()

    @property
    def config(self) -> ETFRotationConfig:
        """Return the immutable strategy parameters."""

        return self._config

    def decide(
        self,
        *,
        request: ETFRotationRequest,
        universe_revisions: Iterable[ETFUniverseRevision],
        prices: Iterable[ResearchPriceBar],
        regime: RegimeTransitionResult,
    ) -> ETFRotationDecision:
        """Return one replayable target decision without using future revisions."""

        self._validate_regime(request, regime)
        universe, universe_hash = self._select_universe(request, tuple(universe_revisions))
        scheduled = request.session_index % self._config.rebalance_frequency_sessions == 0
        forced_cash = (
            self._config.force_downtrend_cash and regime.final_regime is MarketRegime.DOWNTREND
        )
        if not scheduled and not forced_cash:
            price_hash = stable_hash({"prices": []})
            return self._decision(
                request=request,
                regime=regime,
                status=ETFRotationDecisionStatus.NO_REBALANCE,
                universe=universe,
                candidates=(),
                targets=(),
                gross=None,
                cash=None,
                universe_hash=universe_hash,
                price_hash=price_hash,
                reason=(
                    f"session_index {request.session_index} is outside the configured "
                    f"{self._config.rebalance_frequency_sessions}-session rebalance schedule"
                ),
            )

        known_ids = tuple(item.instrument_id for item in universe)
        eligible_ids = {item.instrument_id for item in universe if item.eligible}
        if forced_cash:
            price_hash = stable_hash({"prices": []})
            cash_candidates = tuple(
                ETFRotationCandidate(
                    instrument_id=item.instrument_id,
                    universe_eligibility=item.eligibility,
                    metrics=None,
                    qualifies=False,
                    rank=None,
                    selected=False,
                    target_weight=Decimal(0),
                    reason="stabilized DOWNTREND forces the ETF portfolio to cash",
                )
                for item in universe
            )
            return self._cash_decision(
                request=request,
                regime=regime,
                universe=universe,
                candidates=cash_candidates,
                known_ids=known_ids,
                universe_hash=universe_hash,
                price_hash=price_hash,
                reason="stabilized DOWNTREND uses the default all-cash policy",
            )

        price_histories, price_hash = self._select_prices(
            request=request,
            prices=tuple(prices),
            eligible_ids=eligible_ids,
        )
        metrics_by_id, metric_failures = self._metrics(
            request=request,
            eligible_ids=eligible_ids,
            histories=price_histories,
        )
        ranked = sorted(
            (
                (instrument_id, metrics)
                for instrument_id, metrics in metrics_by_id.items()
                if metrics.above_long_trend
                and metrics.weighted_momentum > self._config.minimum_weighted_momentum
            ),
            key=lambda item: (-item[1].risk_adjusted_momentum, item[0]),
        )
        ranks = {instrument_id: index for index, (instrument_id, _metrics) in enumerate(ranked, 1)}
        selected = ranked[: self._config.top_n]
        configured_exposure = self._config.exposure_for(regime.final_regime)
        exposure = min(configured_exposure, regime.raw_result.risk_budget_max)
        allocations = self._allocate(selected, exposure)

        candidates: list[ETFRotationCandidate] = []
        universe_by_id = {item.instrument_id: item for item in universe}
        for instrument_id in known_ids:
            universe_item = universe_by_id[instrument_id]
            metrics = metrics_by_id.get(instrument_id)
            rank = ranks.get(instrument_id)
            qualifies = rank is not None
            target_weight = allocations.get(instrument_id, Decimal(0))
            if not universe_item.eligible:
                reason = f"PIT universe status is {universe_item.eligibility.value}"
            elif metrics is None:
                reason = metric_failures[instrument_id]
            elif not metrics.above_long_trend:
                reason = "latest close is not above the configured long-term trend average"
            elif metrics.weighted_momentum <= self._config.minimum_weighted_momentum:
                reason = "weighted momentum does not exceed the configured minimum"
            elif target_weight > 0:
                reason = f"selected at deterministic rank {rank}"
            else:
                reason = f"qualified at rank {rank} but falls outside Top-{self._config.top_n}"
            candidates.append(
                ETFRotationCandidate(
                    instrument_id=instrument_id,
                    universe_eligibility=universe_item.eligibility,
                    metrics=metrics,
                    qualifies=qualifies,
                    rank=rank,
                    selected=target_weight > 0,
                    target_weight=target_weight,
                    reason=reason,
                )
            )

        if not allocations:
            return self._cash_decision(
                request=request,
                regime=regime,
                universe=universe,
                candidates=tuple(candidates),
                known_ids=known_ids,
                universe_hash=universe_hash,
                price_hash=price_hash,
                reason="no PIT-eligible ETF passed momentum and long-trend filters",
            )

        targets = tuple(
            ETFTargetWeight(
                instrument_id=instrument_id,
                target_weight=allocations.get(instrument_id, Decimal(0)),
            )
            for instrument_id in known_ids
        )
        gross = sum((item.target_weight for item in targets), Decimal(0))
        return self._decision(
            request=request,
            regime=regime,
            status=ETFRotationDecisionStatus.REBALANCE,
            universe=universe,
            candidates=tuple(candidates),
            targets=targets,
            gross=gross,
            cash=Decimal(1) - gross,
            universe_hash=universe_hash,
            price_hash=price_hash,
            reason=(
                f"selected {len(allocations)} ETF(s) under {regime.final_regime.value} "
                "risk-budget and per-instrument caps"
            ),
        )

    @staticmethod
    def _validate_regime(
        request: ETFRotationRequest,
        regime: RegimeTransitionResult,
    ) -> None:
        raw = regime.raw_result
        event = regime.event
        if (
            raw.session_date != request.signal_date
            or event.session_date != request.signal_date
            or raw.as_of != request.as_of
            or event.as_of != request.as_of
        ):
            raise ETFRotationInputError(
                "stabilized regime date/as_of must align exactly to the strategy request"
            )
        if raw.data_version != request.data_version:
            raise ETFRotationInputError("regime data_version must match the strategy request")
        if event.source_result_hash != raw.result_hash:
            raise ETFRotationInputError("regime event does not bind its raw result")
        if (
            event.transition_config_version != regime.transition_config_version
            or event.transition_config_hash != regime.transition_config_hash
        ):
            raise ETFRotationInputError("regime transition identities are misaligned")

    @staticmethod
    def _universe_eligibility(
        revision: ETFUniverseRevision,
        signal_date: date,
    ) -> ETFUniverseEligibility:
        if signal_date < revision.listed_on:
            return ETFUniverseEligibility.NOT_LISTED
        if revision.delisted_on is not None and signal_date > revision.delisted_on:
            return ETFUniverseEligibility.DELISTED
        if not revision.enabled:
            return ETFUniverseEligibility.DISABLED
        return ETFUniverseEligibility.ELIGIBLE

    def _select_universe(
        self,
        request: ETFRotationRequest,
        revisions: tuple[ETFUniverseRevision, ...],
    ) -> tuple[tuple[ETFUniverseSelection, ...], str]:
        latest: dict[tuple[str, date], ETFUniverseRevision] = {}
        for revision in revisions:
            if not isinstance(revision, ETFUniverseRevision):
                raise ETFRotationInputError(
                    "ETF universe inputs must be immutable ETFUniverseRevision values"
                )
            if revision.available_at > request.as_of:
                continue
            if revision.data_version != request.data_version:
                raise ETFRotationInputError(
                    "eligible ETF universe revisions must match request data_version"
                )
            key = (revision.instrument_id, revision.effective_from)
            current = latest.get(key)
            if current is None or revision.available_at > current.available_at:
                latest[key] = revision
            elif revision.available_at == current.available_at and revision != current:
                raise ETFRotationInputError(
                    "ambiguous ETF universe revisions share instrument, effective_from, "
                    "and available_at"
                )

        grouped: dict[str, list[ETFUniverseRevision]] = {}
        for revision in latest.values():
            grouped.setdefault(revision.instrument_id, []).append(revision)
        selections: list[ETFUniverseSelection] = []
        for instrument_id in sorted(grouped):
            history = grouped[instrument_id]
            applicable = [item for item in history if item.applies_on(request.signal_date)]
            if len(applicable) > 1:
                raise ETFRotationInputError(
                    "ETF universe has overlapping status intervals for one instrument"
                )
            reference = (
                applicable[0]
                if applicable
                else max(history, key=lambda item: (item.effective_from, item.available_at))
            )
            eligibility = (
                self._universe_eligibility(reference, request.signal_date)
                if applicable
                else ETFUniverseEligibility.NO_EFFECTIVE_REVISION
            )
            selections.append(
                ETFUniverseSelection(
                    instrument_id=instrument_id,
                    eligibility=eligibility,
                    listed_on=reference.listed_on,
                    delisted_on=reference.delisted_on,
                    enabled=reference.enabled,
                    effective_from=reference.effective_from,
                    effective_to=reference.effective_to,
                    available_at=reference.available_at,
                    revision=reference.revision,
                )
            )
        frozen_latest = tuple(
            sorted(
                latest.values(),
                key=lambda item: (item.instrument_id, item.effective_from),
            )
        )
        universe_hash = stable_hash(
            {"revisions": [item.fingerprint_payload() for item in frozen_latest]}
        )
        return tuple(selections), universe_hash

    def _select_prices(
        self,
        *,
        request: ETFRotationRequest,
        prices: tuple[ResearchPriceBar, ...],
        eligible_ids: set[str],
    ) -> tuple[dict[str, tuple[ResearchPriceBar, ...]], str]:
        selected: dict[tuple[str, date], ResearchPriceBar] = {}
        for bar in prices:
            if not isinstance(bar, ResearchPriceBar):
                raise ETFRotationInputError(
                    "ETF rotation prices must be immutable ResearchPriceBar values"
                )
            if bar.instrument_type is not TradableInstrumentType.ETF:
                raise ETFRotationInputError("ETF rotation price inputs must have ETF type")
            if bar.available_at > request.as_of:
                continue
            if bar.data_version != request.data_version:
                raise ETFRotationInputError("ETF price data_version must match the request")
            if bar.trading_day > request.signal_date:
                raise ETFRotationInputError("future ETF price dates cannot enter a decision")
            if bar.instrument_id not in eligible_ids:
                continue
            key = (bar.instrument_id, bar.trading_day)
            current = selected.get(key)
            if current is None or bar.available_at > current.available_at:
                selected[key] = bar
            elif bar.available_at == current.available_at and bar != current:
                raise ETFRotationInputError(
                    "ambiguous ETF price revisions share instrument, date, and available_at"
                )
        histories: dict[str, tuple[ResearchPriceBar, ...]] = {}
        for instrument_id in sorted(eligible_ids):
            histories[instrument_id] = tuple(
                sorted(
                    (
                        bar
                        for (known_id, _day), bar in selected.items()
                        if known_id == instrument_id
                    ),
                    key=lambda item: item.trading_day,
                )
            )
        selected_rows = tuple(
            sorted(selected.values(), key=lambda item: (item.trading_day, item.instrument_id))
        )
        price_hash = stable_hash({"prices": [_bar_payload(item) for item in selected_rows]})
        return histories, price_hash

    def _metrics(
        self,
        *,
        request: ETFRotationRequest,
        eligible_ids: set[str],
        histories: dict[str, tuple[ResearchPriceBar, ...]],
    ) -> tuple[dict[str, ETFMomentumMetrics], dict[str, str]]:
        metrics: dict[str, ETFMomentumMetrics] = {}
        failures: dict[str, str] = {}
        expected_dates: tuple[date, ...] | None = None
        required = self._config.required_price_observations
        for instrument_id in sorted(eligible_ids):
            history = histories[instrument_id]
            if not history or history[-1].trading_day != request.signal_date:
                failures[instrument_id] = "no price known at the strategy decision session"
                continue
            if len(history) < required:
                failures[instrument_id] = (
                    f"price history has {len(history)} observations; requires {required}"
                )
                continue
            selected = history[-required:]
            dates = tuple(item.trading_day for item in selected)
            if expected_dates is None:
                expected_dates = dates
            elif dates != expected_dates:
                raise ETFRotationInputError(
                    "rankable ETF price histories are not aligned to identical sessions"
                )
            metrics[instrument_id] = self._calculate_metrics(selected)
        return metrics, failures

    def _calculate_metrics(
        self,
        history: tuple[ResearchPriceBar, ...],
    ) -> ETFMomentumMetrics:
        closes = tuple(item.close_price for item in history)
        horizons = tuple(
            ETFHorizonMomentum(
                horizon=horizon,
                return_value=_divide(closes[-1], closes[-horizon - 1]) - 1,
                configured_weight=weight,
                contribution=(_divide(closes[-1], closes[-horizon - 1]) - 1) * weight,
            )
            for horizon, weight in zip(
                self._config.momentum_horizons,
                self._config.momentum_weights,
                strict=True,
            )
        )
        weighted = sum((item.contribution for item in horizons), Decimal(0))
        volatility = _annualized_volatility(
            closes,
            window=self._config.volatility_window,
            annualization_sessions=self._config.volatility_annualization_sessions,
        )
        denominator = max(volatility, self._config.volatility_floor)
        long_average = _mean(closes[-self._config.long_trend_window :])
        return ETFMomentumMetrics(
            horizons=horizons,
            weighted_momentum=weighted,
            annualized_volatility=volatility,
            risk_adjusted_momentum=_divide(weighted, denominator),
            latest_close=closes[-1],
            long_trend_average=long_average,
            above_long_trend=closes[-1] > long_average,
            observation_dates=tuple(item.trading_day for item in history),
            price_revisions=tuple(item.revision for item in history),
        )

    def _allocate(
        self,
        selected: list[tuple[str, ETFMomentumMetrics]],
        exposure: Decimal,
    ) -> dict[str, Decimal]:
        if not selected or exposure <= 0:
            return {}
        remaining = {instrument_id: metrics for instrument_id, metrics in selected}
        allocations: dict[str, Decimal] = {}
        remaining_exposure = exposure
        while remaining and remaining_exposure > 0:
            inverse_volatility = {
                instrument_id: _divide(
                    Decimal(1),
                    max(metrics.annualized_volatility, self._config.volatility_floor),
                )
                for instrument_id, metrics in remaining.items()
            }
            if len(set(inverse_volatility.values())) == 1:
                equal_weight = _divide(remaining_exposure, Decimal(len(remaining)))
                proposed = {instrument_id: equal_weight for instrument_id in inverse_volatility}
            else:
                total_inverse = sum(inverse_volatility.values(), Decimal(0))
                proposed = {
                    instrument_id: remaining_exposure * _divide(value, total_inverse)
                    for instrument_id, value in inverse_volatility.items()
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
                    weight = proposed[instrument_id]
                    allocations[instrument_id] = weight
                    assigned += weight
                allocations[ordered_ids[-1]] = remaining_exposure - assigned
                break
            for instrument_id in capped:
                allocations[instrument_id] = self._config.maximum_instrument_weight
                remaining_exposure -= self._config.maximum_instrument_weight
                del remaining[instrument_id]
            if remaining_exposure <= 0:
                break
        return {key: allocations[key] for key in sorted(allocations)}

    def _cash_decision(
        self,
        *,
        request: ETFRotationRequest,
        regime: RegimeTransitionResult,
        universe: tuple[ETFUniverseSelection, ...],
        candidates: tuple[ETFRotationCandidate, ...],
        known_ids: tuple[str, ...],
        universe_hash: str,
        price_hash: str,
        reason: str,
    ) -> ETFRotationDecision:
        return self._decision(
            request=request,
            regime=regime,
            status=ETFRotationDecisionStatus.CASH,
            universe=universe,
            candidates=candidates,
            targets=tuple(
                ETFTargetWeight(instrument_id=instrument_id, target_weight=Decimal(0))
                for instrument_id in known_ids
            ),
            gross=Decimal(0),
            cash=Decimal(1),
            universe_hash=universe_hash,
            price_hash=price_hash,
            reason=reason,
        )

    def _decision(
        self,
        *,
        request: ETFRotationRequest,
        regime: RegimeTransitionResult,
        status: ETFRotationDecisionStatus,
        universe: tuple[ETFUniverseSelection, ...],
        candidates: tuple[ETFRotationCandidate, ...],
        targets: tuple[ETFTargetWeight, ...],
        gross: Decimal | None,
        cash: Decimal | None,
        universe_hash: str,
        price_hash: str,
        reason: str,
    ) -> ETFRotationDecision:
        configured_exposure = self._config.exposure_for(regime.final_regime)
        input_hash = stable_hash(
            {
                "config_hash": self._config.config_hash,
                "price_input_hash": price_hash,
                "regime_result_hash": regime.result_hash,
                "request": request.fingerprint_payload(),
                "universe_input_hash": universe_hash,
            }
        )
        result_hash = stable_hash(
            {
                "candidates": [asdict(item) for item in candidates],
                "cash_target_weight": cash,
                "gross_target_weight": gross,
                "input_hash": input_hash,
                "reason": reason,
                "status": status,
                "targets": [asdict(item) for item in targets],
            }
        )
        return ETFRotationDecision(
            signal_date=request.signal_date,
            as_of=request.as_of,
            data_version=request.data_version,
            strategy_version=self._config.version,
            config_hash=self._config.config_hash,
            status=status,
            market_regime=regime.final_regime,
            regime_result_hash=regime.result_hash,
            regime_risk_budget=regime.raw_result.risk_budget_max,
            configured_regime_exposure=configured_exposure,
            gross_target_weight=gross,
            cash_target_weight=cash,
            universe=universe,
            candidates=candidates,
            targets=targets,
            universe_input_hash=universe_hash,
            price_input_hash=price_hash,
            input_hash=input_hash,
            result_hash=result_hash,
            reason=reason,
        )


__all__ = ["ETFRotationStrategy"]
