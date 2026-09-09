"""Immutable contracts for cross-strategy target portfolio aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from enum import StrEnum

from quant_agent.backtest import TradableInstrumentType
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime.contracts import stable_hash

TARGET_PORTFOLIO_BUILDER_VERSION = "target-portfolio-builder-v1"


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _exact_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be an exact Decimal")
    canonical_decimal(value, field_name=field_name)
    return value


class PortfolioBuildInputError(ValueError):
    """Raised when strategy, account, or classification inputs cannot align safely."""


class StrategySleeveKind(StrEnum):
    """Capital-isolated strategy sleeves aggregated by the first portfolio builder."""

    ETF_CORE = "ETF_CORE"
    STOCK_ENHANCEMENT = "STOCK_ENHANCEMENT"


class PortfolioAdjustmentCode(StrEnum):
    """Deterministic constraints that may only reduce proposed target risk."""

    INSTRUMENT_CAP = "INSTRUMENT_CAP"
    INDUSTRY_CAP = "INDUSTRY_CAP"


@dataclass(frozen=True, slots=True)
class PortfolioBuilderConfig:
    """Versioned sleeve budgets, cash reserve, and concentration constraints."""

    version: str = "portfolio-builder-v1"
    etf_core_budget: Decimal = Decimal("0.70")
    stock_enhancement_budget: Decimal = Decimal("0.25")
    minimum_cash_weight: Decimal = Decimal("0.05")
    maximum_instrument_weight: Decimal = Decimal("0.25")
    maximum_stock_industry_weight: Decimal = Decimal("0.20")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "builder config version"))
        for value in (
            self.etf_core_budget,
            self.stock_enhancement_budget,
            self.minimum_cash_weight,
            self.maximum_instrument_weight,
            self.maximum_stock_industry_weight,
        ):
            _exact_decimal(value, "portfolio builder config value")
            if value < 0 or value > 1:
                raise ValueError("portfolio builder weights must be within 0..1")
        if self.maximum_instrument_weight <= 0 or self.maximum_stock_industry_weight <= 0:
            raise ValueError("portfolio concentration caps must be positive")
        if self.etf_core_budget + self.stock_enhancement_budget > (
            Decimal(1) - self.minimum_cash_weight
        ):
            raise ValueError("strategy sleeve budgets must preserve the minimum cash weight")

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    def budget_for(self, kind: StrategySleeveKind) -> Decimal:
        return {
            StrategySleeveKind.ETF_CORE: self.etf_core_budget,
            StrategySleeveKind.STOCK_ENHANCEMENT: self.stock_enhancement_budget,
        }[kind]


@dataclass(frozen=True, slots=True)
class PortfolioBuildRequest:
    """One exact target construction boundary bound to an account snapshot."""

    decision_id: str
    as_of: datetime
    data_version: str
    account_snapshot_id: str
    account_snapshot_hash: str

    def __post_init__(self) -> None:
        for field_name in ("decision_id", "data_version", "account_snapshot_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.as_of)
        _validate_hash(self.account_snapshot_hash, "account_snapshot_hash")

    def fingerprint_payload(self) -> dict[str, str]:
        return {
            "account_snapshot_hash": self.account_snapshot_hash,
            "account_snapshot_id": self.account_snapshot_id,
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "data_version": self.data_version,
            "decision_id": self.decision_id,
        }


@dataclass(frozen=True, slots=True)
class SleeveTarget:
    """One strategy-local target before sleeve budget scaling."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    local_target_weight: Decimal
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "reason", _non_empty(self.reason, "target reason"))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        _exact_decimal(self.local_target_weight, "local target weight")
        if self.local_target_weight < 0 or self.local_target_weight > 1:
            raise ValueError("local target weight must be within 0..1")


@dataclass(frozen=True, slots=True)
class StrategySleeve:
    """Complete target set from one strategy before isolated capital aggregation."""

    sleeve_id: str
    kind: StrategySleeveKind
    as_of: datetime
    data_version: str
    strategy_name: str
    strategy_version: str
    strategy_config_hash: str
    source_result_hash: str
    targets: tuple[SleeveTarget, ...]
    sleeve_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StrategySleeveKind):
            raise ValueError("sleeve kind must be a StrategySleeveKind value")
        for field_name in (
            "sleeve_id",
            "data_version",
            "strategy_name",
            "strategy_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.as_of)
        for value, field_name in (
            (self.strategy_config_hash, "strategy_config_hash"),
            (self.source_result_hash, "source_result_hash"),
            (self.sleeve_hash, "sleeve_hash"),
        ):
            _validate_hash(value, field_name)
        ids = tuple(item.instrument_id for item in self.targets)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("sleeve targets must be unique and sorted")
        expected_type = (
            TradableInstrumentType.ETF
            if self.kind is StrategySleeveKind.ETF_CORE
            else TradableInstrumentType.STOCK
        )
        if any(item.instrument_type is not expected_type for item in self.targets):
            raise ValueError("sleeve targets do not match the sleeve instrument type")
        if sum((item.local_target_weight for item in self.targets), Decimal(0)) > 1:
            raise ValueError("strategy-local target weights cannot exceed one")
        expected_hash = stable_hash(
            {
                "as_of": self.as_of,
                "data_version": self.data_version,
                "kind": self.kind,
                "sleeve_id": self.sleeve_id,
                "source_result_hash": self.source_result_hash,
                "strategy_config_hash": self.strategy_config_hash,
                "strategy_name": self.strategy_name,
                "strategy_version": self.strategy_version,
                "targets": [asdict(item) for item in self.targets],
            }
        )
        if self.sleeve_hash != expected_hash:
            raise ValueError("sleeve_hash does not match strategy targets")

    @classmethod
    def build(
        cls,
        *,
        sleeve_id: str,
        kind: StrategySleeveKind,
        as_of: datetime,
        data_version: str,
        strategy_name: str,
        strategy_version: str,
        strategy_config_hash: str,
        source_result_hash: str,
        targets: tuple[SleeveTarget, ...],
    ) -> StrategySleeve:
        payload = {
            "as_of": as_of,
            "data_version": data_version,
            "kind": kind,
            "sleeve_id": sleeve_id,
            "source_result_hash": source_result_hash,
            "strategy_config_hash": strategy_config_hash,
            "strategy_name": strategy_name,
            "strategy_version": strategy_version,
            "targets": [asdict(item) for item in targets],
        }
        return cls(
            sleeve_id=sleeve_id,
            kind=kind,
            as_of=as_of,
            data_version=data_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            strategy_config_hash=strategy_config_hash,
            source_result_hash=source_result_hash,
            targets=targets,
            sleeve_hash=stable_hash(payload),
        )


@dataclass(frozen=True, slots=True)
class PITIndustryClassification:
    """One stock industry membership known by the target construction boundary."""

    instrument_id: str
    industry_id: str
    session_date: date
    available_at: datetime
    data_version: str
    classification_version: str
    source_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_id",
            "industry_id",
            "data_version",
            "classification_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.available_at)
        if self.available_at.astimezone(SHANGHAI_TZ).date() < self.session_date:
            raise ValueError("industry classification cannot be available before its session")
        _validate_hash(self.source_hash, "classification source_hash")

    @property
    def classification_hash(self) -> str:
        """Bind the instrument, industry, PIT boundary, and source into one identity."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class StrategyWeightContribution:
    """Trace from a strategy-local target through budget scaling and final constraints."""

    sleeve_id: str
    sleeve_hash: str
    sleeve_kind: StrategySleeveKind
    source_result_hash: str
    local_target_weight: Decimal
    sleeve_budget: Decimal
    proposed_portfolio_weight: Decimal
    applied_portfolio_weight: Decimal
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sleeve_id", _non_empty(self.sleeve_id, "sleeve_id"))
        object.__setattr__(self, "reason", _non_empty(self.reason, "contribution reason"))
        if not isinstance(self.sleeve_kind, StrategySleeveKind):
            raise ValueError("sleeve_kind must be a StrategySleeveKind value")
        _validate_hash(self.sleeve_hash, "sleeve_hash")
        _validate_hash(self.source_result_hash, "source_result_hash")
        for value in (
            self.local_target_weight,
            self.sleeve_budget,
            self.proposed_portfolio_weight,
            self.applied_portfolio_weight,
        ):
            _exact_decimal(value, "strategy contribution weight")
            if value < 0 or value > 1:
                raise ValueError("strategy contribution weights must be within 0..1")
        if self.proposed_portfolio_weight != self.local_target_weight * self.sleeve_budget:
            raise ValueError("proposed weight must equal local target times sleeve budget")
        if self.applied_portfolio_weight > self.proposed_portfolio_weight:
            raise ValueError("portfolio constraints cannot increase a strategy contribution")


@dataclass(frozen=True, slots=True)
class PortfolioConstraintAdjustment:
    """One explainable risk reduction applied after isolated sleeve scaling."""

    code: PortfolioAdjustmentCode
    scope_id: str
    before_weight: Decimal
    after_weight: Decimal
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, PortfolioAdjustmentCode):
            raise ValueError("adjustment code must be a PortfolioAdjustmentCode value")
        object.__setattr__(self, "scope_id", _non_empty(self.scope_id, "adjustment scope"))
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "adjustment rationale"))
        for value in (self.before_weight, self.after_weight):
            _exact_decimal(value, "constraint adjustment weight")
            if value < 0 or value > 1:
                raise ValueError("constraint adjustment weights must be within 0..1")
        if self.after_weight >= self.before_weight:
            raise ValueError("portfolio constraint adjustment must reduce target risk")


@dataclass(frozen=True, slots=True)
class PortfolioTargetLine:
    """Current, proposed, constrained, and delta weights for one instrument."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    industry_id: str | None
    industry_classification_hash: str | None
    current_market_value: Decimal
    current_weight: Decimal
    proposed_target_weight: Decimal
    target_weight: Decimal
    target_market_value: Decimal
    weight_delta: Decimal
    absolute_weight_delta: Decimal
    contributions: tuple[StrategyWeightContribution, ...]
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        if self.industry_id is not None:
            object.__setattr__(self, "industry_id", _non_empty(self.industry_id, "industry_id"))
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "target rationale"))
        for value in (
            self.current_market_value,
            self.current_weight,
            self.proposed_target_weight,
            self.target_weight,
            self.target_market_value,
            self.weight_delta,
            self.absolute_weight_delta,
        ):
            _exact_decimal(value, "portfolio target line value")
        if (
            min(
                self.current_market_value,
                self.current_weight,
                self.proposed_target_weight,
                self.target_weight,
                self.target_market_value,
                self.absolute_weight_delta,
            )
            < 0
        ):
            raise ValueError("portfolio target values other than delta cannot be negative")
        if self.weight_delta != self.target_weight - self.current_weight:
            raise ValueError("weight_delta must equal target minus current weight")
        if self.absolute_weight_delta != abs(self.weight_delta):
            raise ValueError("absolute_weight_delta must equal absolute weight_delta")
        if self.target_weight > self.proposed_target_weight:
            raise ValueError("constrained target cannot exceed proposed target")
        if max(self.current_weight, self.proposed_target_weight, self.target_weight) > 1:
            raise ValueError("portfolio target weights must be within 0..1")
        if self.instrument_type is TradableInstrumentType.STOCK:
            if self.industry_id is None or self.industry_classification_hash is None:
                raise ValueError("stock target lines require a point-in-time industry identity")
            _validate_hash(
                self.industry_classification_hash,
                "industry_classification_hash",
            )
        elif self.industry_id is not None or self.industry_classification_hash is not None:
            raise ValueError("ETF target lines cannot carry a stock industry identity")
        if sum((item.proposed_portfolio_weight for item in self.contributions), Decimal(0)) != (
            self.proposed_target_weight
        ):
            raise ValueError("target proposed weight must equal strategy contributions")
        if sum((item.applied_portfolio_weight for item in self.contributions), Decimal(0)) != (
            self.target_weight
        ):
            raise ValueError("target applied weight must equal strategy contributions")
        contribution_keys = tuple((item.sleeve_hash, item.sleeve_id) for item in self.contributions)
        if tuple(sorted(set(contribution_keys))) != contribution_keys:
            raise ValueError("strategy contributions must be unique and deterministically sorted")
        expected_kind = (
            StrategySleeveKind.ETF_CORE
            if self.instrument_type is TradableInstrumentType.ETF
            else StrategySleeveKind.STOCK_ENHANCEMENT
        )
        if any(item.sleeve_kind is not expected_kind for item in self.contributions):
            raise ValueError("strategy contribution kind does not match the instrument type")


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Complete target weights and account differences without executable orders."""

    decision_id: str
    as_of: datetime
    data_version: str
    builder_version: str
    config_version: str
    config_hash: str
    account_snapshot_id: str
    account_snapshot_hash: str
    etf_core_budget: Decimal
    stock_enhancement_budget: Decimal
    minimum_cash_weight: Decimal
    maximum_instrument_weight: Decimal
    maximum_stock_industry_weight: Decimal
    total_equity: Decimal
    current_gross_weight: Decimal
    current_cash_weight: Decimal
    target_gross_weight: Decimal
    target_cash_weight: Decimal
    etf_target_weight: Decimal
    stock_target_weight: Decimal
    gross_instrument_turnover: Decimal
    one_way_turnover: Decimal
    lines: tuple[PortfolioTargetLine, ...]
    adjustments: tuple[PortfolioConstraintAdjustment, ...]
    sleeve_hashes: tuple[str, ...]
    classification_hashes: tuple[str, ...]
    classification_input_hash: str
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "data_version",
            "builder_version",
            "config_version",
            "account_snapshot_id",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.as_of)
        for value, field_name in (
            (self.config_hash, "config_hash"),
            (self.account_snapshot_hash, "account_snapshot_hash"),
            (self.classification_input_hash, "classification_input_hash"),
            (self.input_hash, "input_hash"),
            (self.result_hash, "result_hash"),
        ):
            _validate_hash(value, field_name)
        if (
            len(self.sleeve_hashes) != len(StrategySleeveKind)
            or tuple(sorted(set(self.sleeve_hashes))) != self.sleeve_hashes
        ):
            raise ValueError("target portfolio sleeve hashes must be non-empty, unique, and sorted")
        for value in self.sleeve_hashes:
            _validate_hash(value, "sleeve_hash")
        if tuple(sorted(set(self.classification_hashes))) != self.classification_hashes:
            raise ValueError("classification hashes must be unique and deterministically sorted")
        for value in self.classification_hashes:
            _validate_hash(value, "classification_hash")
        line_classification_hashes = tuple(
            sorted(
                item.industry_classification_hash
                for item in self.lines
                if item.industry_classification_hash is not None
            )
        )
        if self.classification_hashes != line_classification_hashes:
            raise ValueError("classification hashes must exactly cover stock target lines")
        if self.classification_input_hash != stable_hash(
            {"classification_hashes": self.classification_hashes}
        ):
            raise ValueError("classification_input_hash does not match classification identities")
        adjustment_keys = tuple((item.code.value, item.scope_id) for item in self.adjustments)
        if tuple(sorted(set(adjustment_keys))) != adjustment_keys:
            raise ValueError("portfolio adjustments must be unique and deterministically sorted")
        metrics = (
            self.etf_core_budget,
            self.stock_enhancement_budget,
            self.minimum_cash_weight,
            self.maximum_instrument_weight,
            self.maximum_stock_industry_weight,
            self.total_equity,
            self.current_gross_weight,
            self.current_cash_weight,
            self.target_gross_weight,
            self.target_cash_weight,
            self.etf_target_weight,
            self.stock_target_weight,
            self.gross_instrument_turnover,
            self.one_way_turnover,
        )
        for metric in metrics:
            _exact_decimal(metric, "target portfolio metric")
            if metric < 0:
                raise ValueError("target portfolio metrics cannot be negative")
        if self.total_equity <= 0:
            raise ValueError("target portfolio total_equity must be positive")
        config_weights = (
            self.etf_core_budget,
            self.stock_enhancement_budget,
            self.minimum_cash_weight,
            self.maximum_instrument_weight,
            self.maximum_stock_industry_weight,
        )
        if any(value > 1 for value in config_weights):
            raise ValueError("portfolio budget and constraint weights must be within 0..1")
        if self.maximum_instrument_weight <= 0 or self.maximum_stock_industry_weight <= 0:
            raise ValueError("portfolio concentration caps must be positive")
        if self.etf_core_budget + self.stock_enhancement_budget > (
            Decimal(1) - self.minimum_cash_weight
        ):
            raise ValueError("strategy sleeve budgets must preserve the minimum cash weight")
        expected_config_hash = stable_hash(
            {
                "version": self.config_version,
                "etf_core_budget": self.etf_core_budget,
                "stock_enhancement_budget": self.stock_enhancement_budget,
                "minimum_cash_weight": self.minimum_cash_weight,
                "maximum_instrument_weight": self.maximum_instrument_weight,
                "maximum_stock_industry_weight": self.maximum_stock_industry_weight,
            }
        )
        if self.config_hash != expected_config_hash:
            raise ValueError("config_hash does not match portfolio builder constraints")
        if self.current_gross_weight + self.current_cash_weight != 1:
            raise ValueError("current gross and cash weights must sum to one")
        if self.target_gross_weight + self.target_cash_weight != 1:
            raise ValueError("target gross and cash weights must sum to one")
        if self.target_gross_weight != self.etf_target_weight + self.stock_target_weight:
            raise ValueError("target gross weight must equal ETF plus stock targets")
        if self.target_cash_weight < self.minimum_cash_weight:
            raise ValueError("target cash weight violates the minimum cash reserve")
        if self.etf_target_weight > self.etf_core_budget:
            raise ValueError("ETF target weight exceeds its isolated sleeve budget")
        if self.stock_target_weight > self.stock_enhancement_budget:
            raise ValueError("stock target weight exceeds its isolated sleeve budget")
        ids = tuple(item.instrument_id for item in self.lines)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("target portfolio lines must be unique and sorted")
        with localcontext() as context:
            context.prec = 50
            current_line_weight = sum((item.current_weight for item in self.lines), Decimal(0))
            target_line_weight = sum((item.target_weight for item in self.lines), Decimal(0))
            etf_line_weight = sum(
                (
                    item.target_weight
                    for item in self.lines
                    if item.instrument_type is TradableInstrumentType.ETF
                ),
                Decimal(0),
            )
            stock_line_weight = sum(
                (
                    item.target_weight
                    for item in self.lines
                    if item.instrument_type is TradableInstrumentType.STOCK
                ),
                Decimal(0),
            )
        if self.current_gross_weight != current_line_weight:
            raise ValueError("current gross weight must equal target-line current weights")
        if self.target_gross_weight != target_line_weight:
            raise ValueError("target gross weight must equal target-line weights")
        if self.etf_target_weight != etf_line_weight:
            raise ValueError("ETF target weight must equal ETF target lines")
        if self.stock_target_weight != stock_line_weight:
            raise ValueError("stock target weight must equal stock target lines")
        if any(
            item.target_market_value != self.total_equity * item.target_weight
            for item in self.lines
        ):
            raise ValueError("target market values must equal equity times target weight")
        all_contributions = tuple(
            contribution for line in self.lines for contribution in line.contributions
        )
        if any(item.sleeve_hash not in self.sleeve_hashes for item in all_contributions):
            raise ValueError("strategy contributions must bind a declared sleeve hash")
        expected_budget_by_kind = {
            StrategySleeveKind.ETF_CORE: self.etf_core_budget,
            StrategySleeveKind.STOCK_ENHANCEMENT: self.stock_enhancement_budget,
        }
        if any(
            item.sleeve_budget != expected_budget_by_kind[item.sleeve_kind]
            for item in all_contributions
        ):
            raise ValueError("strategy contribution budget does not match builder configuration")
        budget_by_sleeve: dict[str, Decimal] = {}
        proposed_by_sleeve: dict[str, Decimal] = {}
        for contribution in all_contributions:
            prior_budget = budget_by_sleeve.setdefault(
                contribution.sleeve_hash,
                contribution.sleeve_budget,
            )
            if prior_budget != contribution.sleeve_budget:
                raise ValueError("one strategy sleeve cannot carry conflicting budgets")
            proposed_by_sleeve[contribution.sleeve_hash] = (
                proposed_by_sleeve.get(contribution.sleeve_hash, Decimal(0))
                + contribution.proposed_portfolio_weight
            )
        if any(
            proposed > budget_by_sleeve[sleeve_hash]
            for sleeve_hash, proposed in proposed_by_sleeve.items()
        ):
            raise ValueError("strategy contributions exceed an isolated sleeve budget")
        self._validate_constraint_projection()
        if self.gross_instrument_turnover != sum(
            (item.absolute_weight_delta for item in self.lines), Decimal(0)
        ):
            raise ValueError("gross instrument turnover must equal target-line deltas")
        expected_one_way_turnover = (
            self.gross_instrument_turnover + abs(self.target_cash_weight - self.current_cash_weight)
        ) / Decimal(2)
        if self.one_way_turnover != expected_one_way_turnover:
            raise ValueError("one-way turnover must include both instruments and cash")
        if self.input_hash != stable_hash(self._input_payload()):
            raise ValueError("target portfolio input_hash does not match its bound inputs")
        if self.result_hash != stable_hash(self._result_payload()):
            raise ValueError("target portfolio result_hash does not match its content")

    def _validate_constraint_projection(self) -> None:
        expected_targets = {
            line.instrument_id: min(
                line.proposed_target_weight,
                self.maximum_instrument_weight,
            )
            for line in self.lines
        }
        expected_adjustments: dict[
            tuple[PortfolioAdjustmentCode, str], tuple[Decimal, Decimal]
        ] = {}
        for line in self.lines:
            if line.proposed_target_weight > self.maximum_instrument_weight:
                expected_adjustments[
                    (PortfolioAdjustmentCode.INSTRUMENT_CAP, line.instrument_id)
                ] = (
                    line.proposed_target_weight,
                    self.maximum_instrument_weight,
                )

        members_by_industry: dict[str, list[str]] = {}
        for line in self.lines:
            if (
                line.instrument_type is TradableInstrumentType.STOCK
                and expected_targets[line.instrument_id] > 0
            ):
                assert line.industry_id is not None
                members_by_industry.setdefault(line.industry_id, []).append(line.instrument_id)
        for industry_id, unsorted_members in members_by_industry.items():
            members = sorted(unsorted_members)
            before = sum((expected_targets[item] for item in members), Decimal(0))
            if before <= self.maximum_stock_industry_weight:
                continue
            cap = self.maximum_stock_industry_weight
            with localcontext() as context:
                context.prec = 50
                scale = cap / before
                allocated = Decimal(0)
                for instrument_id in members[:-1]:
                    reduced = expected_targets[instrument_id] * scale
                    expected_targets[instrument_id] = reduced
                    allocated += reduced
                expected_targets[members[-1]] = cap - allocated
            expected_adjustments[(PortfolioAdjustmentCode.INDUSTRY_CAP, industry_id)] = (
                before,
                cap,
            )

        if any(line.target_weight != expected_targets[line.instrument_id] for line in self.lines):
            raise ValueError("target weights do not match the configured constraint projection")
        actual_adjustments = {
            (item.code, item.scope_id): (item.before_weight, item.after_weight)
            for item in self.adjustments
        }
        if len(actual_adjustments) != len(self.adjustments):
            raise ValueError("portfolio adjustments must have unique rule scopes")
        if actual_adjustments != expected_adjustments:
            raise ValueError("portfolio adjustments do not match applied constraints")

    def _input_payload(self) -> dict[str, object]:
        return {
            "account_snapshot_hash": self.account_snapshot_hash,
            "account_snapshot_id": self.account_snapshot_id,
            "as_of": self.as_of,
            "builder_version": self.builder_version,
            "config_version": self.config_version,
            "classification_input_hash": self.classification_input_hash,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "decision_id": self.decision_id,
            "sleeve_hashes": self.sleeve_hashes,
        }

    def _result_payload(self) -> dict[str, object]:
        return {
            **self._input_payload(),
            "adjustments": [asdict(item) for item in self.adjustments],
            "classification_hashes": self.classification_hashes,
            "current_cash_weight": self.current_cash_weight,
            "current_gross_weight": self.current_gross_weight,
            "etf_core_budget": self.etf_core_budget,
            "etf_target_weight": self.etf_target_weight,
            "gross_instrument_turnover": self.gross_instrument_turnover,
            "input_hash": self.input_hash,
            "lines": [asdict(item) for item in self.lines],
            "maximum_instrument_weight": self.maximum_instrument_weight,
            "maximum_stock_industry_weight": self.maximum_stock_industry_weight,
            "minimum_cash_weight": self.minimum_cash_weight,
            "one_way_turnover": self.one_way_turnover,
            "stock_enhancement_budget": self.stock_enhancement_budget,
            "stock_target_weight": self.stock_target_weight,
            "target_cash_weight": self.target_cash_weight,
            "target_gross_weight": self.target_gross_weight,
            "total_equity": self.total_equity,
        }

    @classmethod
    def build(
        cls,
        *,
        request: PortfolioBuildRequest,
        config: PortfolioBuilderConfig,
        total_equity: Decimal,
        current_gross_weight: Decimal,
        current_cash_weight: Decimal,
        target_gross_weight: Decimal,
        target_cash_weight: Decimal,
        etf_target_weight: Decimal,
        stock_target_weight: Decimal,
        gross_instrument_turnover: Decimal,
        one_way_turnover: Decimal,
        lines: tuple[PortfolioTargetLine, ...],
        adjustments: tuple[PortfolioConstraintAdjustment, ...],
        sleeve_hashes: tuple[str, ...],
        classification_hashes: tuple[str, ...],
        classification_input_hash: str,
    ) -> TargetPortfolio:
        input_payload = {
            "account_snapshot_hash": request.account_snapshot_hash,
            "account_snapshot_id": request.account_snapshot_id,
            "as_of": request.as_of,
            "builder_version": TARGET_PORTFOLIO_BUILDER_VERSION,
            "config_version": config.version,
            "classification_input_hash": classification_input_hash,
            "config_hash": config.config_hash,
            "data_version": request.data_version,
            "decision_id": request.decision_id,
            "sleeve_hashes": sleeve_hashes,
        }
        input_hash = stable_hash(input_payload)
        result_payload = {
            **input_payload,
            "adjustments": [asdict(item) for item in adjustments],
            "classification_hashes": classification_hashes,
            "current_cash_weight": current_cash_weight,
            "current_gross_weight": current_gross_weight,
            "etf_core_budget": config.etf_core_budget,
            "etf_target_weight": etf_target_weight,
            "gross_instrument_turnover": gross_instrument_turnover,
            "input_hash": input_hash,
            "lines": [asdict(item) for item in lines],
            "maximum_instrument_weight": config.maximum_instrument_weight,
            "maximum_stock_industry_weight": config.maximum_stock_industry_weight,
            "minimum_cash_weight": config.minimum_cash_weight,
            "one_way_turnover": one_way_turnover,
            "stock_enhancement_budget": config.stock_enhancement_budget,
            "stock_target_weight": stock_target_weight,
            "target_cash_weight": target_cash_weight,
            "target_gross_weight": target_gross_weight,
            "total_equity": total_equity,
        }
        return cls(
            decision_id=request.decision_id,
            as_of=request.as_of,
            data_version=request.data_version,
            builder_version=TARGET_PORTFOLIO_BUILDER_VERSION,
            config_version=config.version,
            config_hash=config.config_hash,
            account_snapshot_id=request.account_snapshot_id,
            account_snapshot_hash=request.account_snapshot_hash,
            etf_core_budget=config.etf_core_budget,
            stock_enhancement_budget=config.stock_enhancement_budget,
            minimum_cash_weight=config.minimum_cash_weight,
            maximum_instrument_weight=config.maximum_instrument_weight,
            maximum_stock_industry_weight=config.maximum_stock_industry_weight,
            total_equity=total_equity,
            current_gross_weight=current_gross_weight,
            current_cash_weight=current_cash_weight,
            target_gross_weight=target_gross_weight,
            target_cash_weight=target_cash_weight,
            etf_target_weight=etf_target_weight,
            stock_target_weight=stock_target_weight,
            gross_instrument_turnover=gross_instrument_turnover,
            one_way_turnover=one_way_turnover,
            lines=lines,
            adjustments=adjustments,
            sleeve_hashes=sleeve_hashes,
            classification_hashes=classification_hashes,
            classification_input_hash=classification_input_hash,
            input_hash=input_hash,
            result_hash=stable_hash(result_payload),
        )

    def identity_payload(self) -> dict[str, str]:
        return {
            "account_snapshot_hash": self.account_snapshot_hash,
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self.config_hash,
            "decision_id": self.decision_id,
            "input_hash": self.input_hash,
            "result_hash": self.result_hash,
        }


__all__ = [
    "PITIndustryClassification",
    "PortfolioAdjustmentCode",
    "PortfolioBuildInputError",
    "PortfolioBuildRequest",
    "PortfolioBuilderConfig",
    "PortfolioConstraintAdjustment",
    "PortfolioTargetLine",
    "SleeveTarget",
    "StrategySleeve",
    "StrategySleeveKind",
    "StrategyWeightContribution",
    "TargetPortfolio",
]
