"""Deterministic aggregation of isolated strategy sleeves into target weights."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, localcontext

from quant_agent.backtest import TradableInstrumentType
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.regime.contracts import stable_hash

from .contracts import (
    PITIndustryClassification,
    PortfolioAdjustmentCode,
    PortfolioBuilderConfig,
    PortfolioBuildInputError,
    PortfolioBuildRequest,
    PortfolioConstraintAdjustment,
    PortfolioTargetLine,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    StrategyWeightContribution,
    TargetPortfolio,
)

_ContributionInput = tuple[StrategySleeve, SleeveTarget, Decimal]


class TargetPortfolioBuilder:
    """Scale strategy-local targets by budgets, then apply reducing-only constraints."""

    def __init__(self, config: PortfolioBuilderConfig | None = None) -> None:
        self._config = config or PortfolioBuilderConfig()

    @property
    def config(self) -> PortfolioBuilderConfig:
        """Return the immutable, versioned builder configuration."""

        return self._config

    def build(
        self,
        *,
        request: PortfolioBuildRequest,
        account: AccountSnapshot,
        sleeves: Iterable[StrategySleeve],
        classifications: Iterable[PITIndustryClassification],
    ) -> TargetPortfolio:
        """Build a complete target portfolio without creating executable orders."""

        ordered_sleeves = tuple(sorted(sleeves, key=lambda item: (item.kind.value, item.sleeve_id)))
        ordered_classifications = tuple(
            sorted(classifications, key=lambda item: item.instrument_id)
        )
        self._validate_account_binding(request=request, account=account)
        self._validate_sleeves(request=request, sleeves=ordered_sleeves)

        contribution_inputs, instrument_types = self._scale_sleeves(ordered_sleeves)
        required_stock_ids = {
            position.instrument_id
            for position in account.positions
            if position.instrument_type is TradableInstrumentType.STOCK
        }
        required_stock_ids.update(
            instrument_id
            for instrument_id, items in contribution_inputs.items()
            if instrument_types[instrument_id] is TradableInstrumentType.STOCK
            and any(proposed > 0 for _, _, proposed in items)
        )
        classification_by_id = self._validate_classifications(
            request=request,
            classifications=ordered_classifications,
            required_stock_ids=required_stock_ids,
        )
        self._validate_account_instrument_types(
            account=account,
            instrument_types=instrument_types,
        )

        proposed_weights = {
            instrument_id: sum((item[2] for item in items), Decimal(0))
            for instrument_id, items in contribution_inputs.items()
        }
        target_weights, adjustments = self._apply_constraints(
            proposed_weights=proposed_weights,
            instrument_types=instrument_types,
            classification_by_id=classification_by_id,
        )
        lines = self._build_lines(
            account=account,
            contribution_inputs=contribution_inputs,
            instrument_types=instrument_types,
            classification_by_id=classification_by_id,
            proposed_weights=proposed_weights,
            target_weights=target_weights,
            adjustments=adjustments,
        )

        etf_target_weight = sum(
            (
                line.target_weight
                for line in lines
                if line.instrument_type is TradableInstrumentType.ETF
            ),
            Decimal(0),
        )
        stock_target_weight = sum(
            (
                line.target_weight
                for line in lines
                if line.instrument_type is TradableInstrumentType.STOCK
            ),
            Decimal(0),
        )
        target_gross_weight = etf_target_weight + stock_target_weight
        target_cash_weight = Decimal(1) - target_gross_weight
        if etf_target_weight > self._config.etf_core_budget:
            raise PortfolioBuildInputError("ETF targets exceed their isolated sleeve budget")
        if stock_target_weight > self._config.stock_enhancement_budget:
            raise PortfolioBuildInputError("stock targets exceed their isolated sleeve budget")
        if target_cash_weight < self._config.minimum_cash_weight:
            raise PortfolioBuildInputError("aggregated targets violate the minimum cash reserve")

        gross_instrument_turnover = sum(
            (line.absolute_weight_delta for line in lines),
            Decimal(0),
        )
        one_way_turnover = (
            gross_instrument_turnover + abs(target_cash_weight - account.cash_weight)
        ) / Decimal(2)
        classification_hashes = tuple(
            sorted(classification.classification_hash for classification in ordered_classifications)
        )
        classification_input_hash = stable_hash({"classification_hashes": classification_hashes})
        return TargetPortfolio.build(
            request=request,
            config=self._config,
            total_equity=account.total_equity,
            current_gross_weight=account.gross_exposure,
            current_cash_weight=account.cash_weight,
            target_gross_weight=target_gross_weight,
            target_cash_weight=target_cash_weight,
            etf_target_weight=etf_target_weight,
            stock_target_weight=stock_target_weight,
            gross_instrument_turnover=gross_instrument_turnover,
            one_way_turnover=one_way_turnover,
            lines=lines,
            adjustments=adjustments,
            sleeve_hashes=tuple(sorted(sleeve.sleeve_hash for sleeve in ordered_sleeves)),
            classification_hashes=classification_hashes,
            classification_input_hash=classification_input_hash,
        )

    @staticmethod
    def _validate_account_binding(
        *,
        request: PortfolioBuildRequest,
        account: AccountSnapshot,
    ) -> None:
        if request.account_snapshot_id != account.snapshot_id:
            raise PortfolioBuildInputError("request account snapshot id does not match account")
        if request.account_snapshot_hash != account.content_hash:
            raise PortfolioBuildInputError("request account snapshot hash does not match account")
        if request.as_of != account.as_of:
            raise PortfolioBuildInputError("request and account as_of must align exactly")
        if request.data_version != account.data_version:
            raise PortfolioBuildInputError("request and account data versions must align")

    @staticmethod
    def _validate_sleeves(
        *,
        request: PortfolioBuildRequest,
        sleeves: tuple[StrategySleeve, ...],
    ) -> None:
        required_kinds = set(StrategySleeveKind)
        actual_kinds = {sleeve.kind for sleeve in sleeves}
        if len(sleeves) != len(required_kinds) or actual_kinds != required_kinds:
            raise PortfolioBuildInputError(
                "exactly one ETF and one stock strategy sleeve are required"
            )
        sleeve_ids = {sleeve.sleeve_id for sleeve in sleeves}
        sleeve_hashes = {sleeve.sleeve_hash for sleeve in sleeves}
        if len(sleeve_ids) != len(sleeves) or len(sleeve_hashes) != len(sleeves):
            raise PortfolioBuildInputError("strategy sleeve ids and hashes must be unique")
        for sleeve in sleeves:
            if sleeve.as_of != request.as_of:
                raise PortfolioBuildInputError("strategy sleeve as_of does not match request")
            if sleeve.data_version != request.data_version:
                raise PortfolioBuildInputError(
                    "strategy sleeve data version does not match request"
                )

    def _scale_sleeves(
        self,
        sleeves: tuple[StrategySleeve, ...],
    ) -> tuple[dict[str, list[_ContributionInput]], dict[str, TradableInstrumentType]]:
        contributions: dict[str, list[_ContributionInput]] = {}
        instrument_types: dict[str, TradableInstrumentType] = {}
        for sleeve in sleeves:
            budget = self._config.budget_for(sleeve.kind)
            for target in sleeve.targets:
                known_type = instrument_types.setdefault(
                    target.instrument_id,
                    target.instrument_type,
                )
                if known_type is not target.instrument_type:
                    raise PortfolioBuildInputError(
                        f"instrument {target.instrument_id} has conflicting strategy types"
                    )
                contributions.setdefault(target.instrument_id, []).append(
                    (sleeve, target, target.local_target_weight * budget)
                )
        return contributions, instrument_types

    @staticmethod
    def _validate_classifications(
        *,
        request: PortfolioBuildRequest,
        classifications: tuple[PITIndustryClassification, ...],
        required_stock_ids: set[str],
    ) -> dict[str, PITIndustryClassification]:
        ids = tuple(item.instrument_id for item in classifications)
        if tuple(sorted(set(ids))) != ids:
            raise PortfolioBuildInputError("industry classifications must be unique")
        if set(ids) != required_stock_ids:
            raise PortfolioBuildInputError(
                "industry classifications must exactly cover current and active stock targets"
            )
        session_date = request.as_of.astimezone(SHANGHAI_TZ).date()
        classification_versions = {item.classification_version for item in classifications}
        if len(classification_versions) > 1:
            raise PortfolioBuildInputError("industry classification versions cannot be mixed")
        for classification in classifications:
            if classification.session_date != session_date:
                raise PortfolioBuildInputError(
                    "industry classification session does not match request"
                )
            if classification.available_at > request.as_of:
                raise PortfolioBuildInputError("future industry classifications are forbidden")
            if classification.data_version != request.data_version:
                raise PortfolioBuildInputError(
                    "industry classification data version does not match request"
                )
        return {item.instrument_id: item for item in classifications}

    @staticmethod
    def _validate_account_instrument_types(
        *,
        account: AccountSnapshot,
        instrument_types: dict[str, TradableInstrumentType],
    ) -> None:
        for position in account.positions:
            known_type = instrument_types.setdefault(
                position.instrument_id,
                position.instrument_type,
            )
            if known_type is not position.instrument_type:
                raise PortfolioBuildInputError(
                    f"instrument {position.instrument_id} conflicts with account type"
                )

    def _apply_constraints(
        self,
        *,
        proposed_weights: dict[str, Decimal],
        instrument_types: dict[str, TradableInstrumentType],
        classification_by_id: dict[str, PITIndustryClassification],
    ) -> tuple[dict[str, Decimal], tuple[PortfolioConstraintAdjustment, ...]]:
        constrained = dict(proposed_weights)
        adjustments: list[PortfolioConstraintAdjustment] = []
        for instrument_id in sorted(constrained):
            proposed = constrained[instrument_id]
            if proposed > self._config.maximum_instrument_weight:
                constrained[instrument_id] = self._config.maximum_instrument_weight
                adjustments.append(
                    PortfolioConstraintAdjustment(
                        code=PortfolioAdjustmentCode.INSTRUMENT_CAP,
                        scope_id=instrument_id,
                        before_weight=proposed,
                        after_weight=self._config.maximum_instrument_weight,
                        rationale=(
                            f"{instrument_id} reduced to the versioned single-instrument cap"
                        ),
                    )
                )

        industry_members: dict[str, list[str]] = {}
        for instrument_id, instrument_type in instrument_types.items():
            if (
                instrument_type is TradableInstrumentType.STOCK
                and constrained.get(
                    instrument_id,
                    Decimal(0),
                )
                > 0
            ):
                industry_id = classification_by_id[instrument_id].industry_id
                industry_members.setdefault(industry_id, []).append(instrument_id)
        for industry_id in sorted(industry_members):
            members = sorted(industry_members[industry_id])
            before = sum((constrained[item] for item in members), Decimal(0))
            cap = self._config.maximum_stock_industry_weight
            if before <= cap:
                continue
            with localcontext() as context:
                context.prec = 50
                scale = cap / before
                allocated = Decimal(0)
                for instrument_id in members[:-1]:
                    reduced = constrained[instrument_id] * scale
                    constrained[instrument_id] = reduced
                    allocated += reduced
                constrained[members[-1]] = cap - allocated
            adjustments.append(
                PortfolioConstraintAdjustment(
                    code=PortfolioAdjustmentCode.INDUSTRY_CAP,
                    scope_id=industry_id,
                    before_weight=before,
                    after_weight=cap,
                    rationale=f"{industry_id} stocks reduced pro rata to the industry cap",
                )
            )
        return constrained, tuple(
            sorted(adjustments, key=lambda item: (item.code.value, item.scope_id))
        )

    def _build_lines(
        self,
        *,
        account: AccountSnapshot,
        contribution_inputs: dict[str, list[_ContributionInput]],
        instrument_types: dict[str, TradableInstrumentType],
        classification_by_id: dict[str, PITIndustryClassification],
        proposed_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        adjustments: tuple[PortfolioConstraintAdjustment, ...],
    ) -> tuple[PortfolioTargetLine, ...]:
        positions_by_id = {item.instrument_id: item for item in account.positions}
        line_ids = set(positions_by_id)
        line_ids.update(
            instrument_id
            for instrument_id, target_weight in proposed_weights.items()
            if target_weight > 0
        )
        current_weights = self._current_weights(account)
        adjustment_by_instrument = {
            item.scope_id: item
            for item in adjustments
            if item.code is PortfolioAdjustmentCode.INSTRUMENT_CAP
        }
        adjusted_industries = {
            item.scope_id
            for item in adjustments
            if item.code is PortfolioAdjustmentCode.INDUSTRY_CAP
        }
        lines: list[PortfolioTargetLine] = []
        for instrument_id in sorted(line_ids):
            instrument_type = instrument_types[instrument_id]
            industry_id = (
                classification_by_id[instrument_id].industry_id
                if instrument_type is TradableInstrumentType.STOCK
                else None
            )
            industry_classification_hash = (
                classification_by_id[instrument_id].classification_hash
                if instrument_type is TradableInstrumentType.STOCK
                else None
            )
            proposed = proposed_weights.get(instrument_id, Decimal(0))
            target = target_weights.get(instrument_id, Decimal(0))
            contribution_values = self._build_contributions(
                inputs=contribution_inputs.get(instrument_id, []),
                applied_weight=target,
            )
            current_market_value = (
                positions_by_id[instrument_id].market_value
                if instrument_id in positions_by_id
                else Decimal(0)
            )
            current_weight = current_weights.get(instrument_id, Decimal(0))
            weight_delta = target - current_weight
            rationale_parts = [f"{item.sleeve_id}: {item.reason}" for item in contribution_values]
            if not rationale_parts:
                rationale_parts.append(
                    "No active strategy target; reduce the existing position to cash"
                )
            if instrument_id in adjustment_by_instrument:
                rationale_parts.append(adjustment_by_instrument[instrument_id].rationale)
            if industry_id in adjusted_industries:
                rationale_parts.append(f"{industry_id} industry cap applied pro rata")
            lines.append(
                PortfolioTargetLine(
                    instrument_id=instrument_id,
                    instrument_type=instrument_type,
                    industry_id=industry_id,
                    industry_classification_hash=industry_classification_hash,
                    current_market_value=current_market_value,
                    current_weight=current_weight,
                    proposed_target_weight=proposed,
                    target_weight=target,
                    target_market_value=account.total_equity * target,
                    weight_delta=weight_delta,
                    absolute_weight_delta=abs(weight_delta),
                    contributions=contribution_values,
                    rationale="; ".join(rationale_parts),
                )
            )
        return tuple(lines)

    @staticmethod
    def _current_weights(account: AccountSnapshot) -> dict[str, Decimal]:
        if not account.positions:
            return {}
        positions = tuple(sorted(account.positions, key=lambda item: item.instrument_id))
        weights: dict[str, Decimal] = {}
        with localcontext() as context:
            context.prec = 50
            allocated = Decimal(0)
            for position in positions[:-1]:
                weight = position.market_value / account.total_equity
                weights[position.instrument_id] = weight
                allocated += weight
            weights[positions[-1].instrument_id] = account.gross_exposure - allocated
        return weights

    def _build_contributions(
        self,
        *,
        inputs: list[_ContributionInput],
        applied_weight: Decimal,
    ) -> tuple[StrategyWeightContribution, ...]:
        ordered = sorted(inputs, key=lambda item: (item[0].sleeve_hash, item[0].sleeve_id))
        proposed_total = sum((item[2] for item in ordered), Decimal(0))
        if proposed_total == 0:
            applied_values = [Decimal(0) for _ in ordered]
        else:
            applied_values = []
            with localcontext() as context:
                context.prec = 50
                allocated = Decimal(0)
                for _, _, proposed in ordered[:-1]:
                    value = applied_weight * proposed / proposed_total
                    applied_values.append(value)
                    allocated += value
                applied_values.append(applied_weight - allocated)
        return tuple(
            StrategyWeightContribution(
                sleeve_id=sleeve.sleeve_id,
                sleeve_hash=sleeve.sleeve_hash,
                sleeve_kind=sleeve.kind,
                source_result_hash=sleeve.source_result_hash,
                local_target_weight=target.local_target_weight,
                sleeve_budget=self._config.budget_for(sleeve.kind),
                proposed_portfolio_weight=proposed,
                applied_portfolio_weight=applied,
                reason=f"{sleeve.strategy_name}: {target.reason}",
            )
            for (sleeve, target, proposed), applied in zip(
                ordered,
                applied_values,
                strict=True,
            )
        )


__all__ = ["TargetPortfolioBuilder"]
