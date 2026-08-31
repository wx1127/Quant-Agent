"""Contracts for chronological walk-forward validation and frozen parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, localcontext

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime.contracts import stable_hash


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


class WalkForwardValidationError(ValueError):
    """Raised when validation inputs violate chronology or frozen identities."""


@dataclass(frozen=True, slots=True)
class WalkForwardSplitConfig:
    """Versioned chronological train/validation/out-of-sample window policy."""

    version: str = "walk-forward-split-v1"
    train_sessions: int = 504
    validation_sessions: int = 126
    out_of_sample_sessions: int = 63
    step_sessions: int = 63
    expanding_train: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "split config version"))
        for count in (
            self.train_sessions,
            self.validation_sessions,
            self.out_of_sample_sessions,
            self.step_sessions,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("walk-forward window sizes must be positive integers")
        if self.step_sessions < self.out_of_sample_sessions:
            raise ValueError("walk-forward step cannot overlap out-of-sample windows")
        if not isinstance(self.expanding_train, bool):
            raise ValueError("expanding_train must be boolean")

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One strictly ordered train, validation, and frozen OOS split."""

    fold_id: str
    train_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    out_of_sample_sessions: tuple[date, ...]
    split_config_hash: str
    fold_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _non_empty(self.fold_id, "fold_id"))
        _validate_hash(self.split_config_hash, "split_config_hash")
        _validate_hash(self.fold_hash, "fold_hash")
        for name, values in (
            ("train", self.train_sessions),
            ("validation", self.validation_sessions),
            ("out-of-sample", self.out_of_sample_sessions),
        ):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} sessions must be non-empty, unique, and increasing")
        if not (
            self.train_sessions[-1]
            < self.validation_sessions[0]
            <= self.validation_sessions[-1]
            < self.out_of_sample_sessions[0]
        ):
            raise ValueError("walk-forward phases must be chronological and non-overlapping")
        expected = stable_hash(
            {
                "fold_id": self.fold_id,
                "out_of_sample_sessions": self.out_of_sample_sessions,
                "split_config_hash": self.split_config_hash,
                "train_sessions": self.train_sessions,
                "validation_sessions": self.validation_sessions,
            }
        )
        if self.fold_hash != expected:
            raise ValueError("fold_hash does not match split sessions")

    @property
    def train_end(self) -> date:
        return self.train_sessions[-1]

    @property
    def validation_end(self) -> date:
        return self.validation_sessions[-1]

    @property
    def out_of_sample_start(self) -> date:
        return self.out_of_sample_sessions[0]

    @property
    def out_of_sample_end(self) -> date:
        return self.out_of_sample_sessions[-1]


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    """Complete deterministic split plan over one exact trading calendar."""

    sessions: tuple[date, ...]
    split_config_version: str
    split_config_hash: str
    folds: tuple[WalkForwardFold, ...]
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.sessions))) != self.sessions:
            raise ValueError("walk-forward plan sessions must be unique and increasing")
        object.__setattr__(
            self,
            "split_config_version",
            _non_empty(self.split_config_version, "split config version"),
        )
        for hash_value, field_name in (
            (self.split_config_hash, "split_config_hash"),
            (self.input_hash, "plan input_hash"),
            (self.result_hash, "plan result_hash"),
        ):
            _validate_hash(hash_value, field_name)
        if not self.folds:
            raise ValueError("walk-forward plan requires at least one complete fold")
        fold_ids = tuple(item.fold_id for item in self.folds)
        if fold_ids != tuple(f"WF-{index:03d}" for index in range(1, len(self.folds) + 1)):
            raise ValueError("walk-forward fold IDs must be contiguous and ordered")
        if any(item.split_config_hash != self.split_config_hash for item in self.folds):
            raise ValueError("walk-forward folds must share the plan config hash")
        for earlier, later in zip(self.folds, self.folds[1:], strict=False):
            if earlier.out_of_sample_end >= later.out_of_sample_start:
                raise ValueError("out-of-sample fold windows cannot overlap")
        expected_input = stable_hash(
            {"config_hash": self.split_config_hash, "sessions": self.sessions}
        )
        if self.input_hash != expected_input:
            raise ValueError("walk-forward plan input_hash does not match inputs")
        expected_result = stable_hash(
            {
                "folds": [item.fold_hash for item in self.folds],
                "input_hash": self.input_hash,
            }
        )
        if self.result_hash != expected_result:
            raise ValueError("walk-forward plan result_hash does not match folds")


ParameterScalar = str | int | bool | Decimal


@dataclass(frozen=True, slots=True)
class ParameterValue:
    """One exact registered strategy parameter without binary floats."""

    name: str
    value: ParameterScalar

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "parameter name"))
        if isinstance(self.value, float) or not isinstance(
            self.value,
            (str, int, bool, Decimal),
        ):
            raise ValueError("parameter values must be string, integer, boolean, or Decimal")
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("string parameter values must be non-empty")
        if isinstance(self.value, Decimal):
            canonical_decimal(self.value, field_name="registered parameter value")


@dataclass(frozen=True, slots=True)
class RegisteredParameterSet:
    """One immutable named parameter set in a strategy registry."""

    strategy_name: str
    strategy_version: str
    parameter_version: str
    parameters: tuple[ParameterValue, ...]
    registered_at: datetime
    parameter_hash: str

    def __post_init__(self) -> None:
        for field_name in ("strategy_name", "strategy_version", "parameter_version"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.registered_at)
        names = tuple(item.name for item in self.parameters)
        if not names or tuple(sorted(set(names))) != names:
            raise ValueError("registered parameters must be non-empty, unique, and sorted")
        _validate_hash(self.parameter_hash, "parameter_hash")
        expected = stable_hash(
            {
                "parameter_version": self.parameter_version,
                "parameters": [asdict(item) for item in self.parameters],
                "strategy_name": self.strategy_name,
                "strategy_version": self.strategy_version,
            }
        )
        if self.parameter_hash != expected:
            raise ValueError("parameter_hash does not match registered parameters")


@dataclass(frozen=True, slots=True)
class ParameterRegistrySnapshot:
    """Sorted append-only parameter registrations and their registry identity."""

    entries: tuple[RegisteredParameterSet, ...]
    registry_hash: str

    def __post_init__(self) -> None:
        keys = tuple(
            (item.strategy_name, item.strategy_version, item.parameter_version)
            for item in self.entries
        )
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("parameter registry entries must be unique and sorted")
        _validate_hash(self.registry_hash, "registry_hash")
        if self.registry_hash != stable_hash(
            {"entries": [item.parameter_hash for item in self.entries]}
        ):
            raise ValueError("registry_hash does not match registered entries")


@dataclass(frozen=True, slots=True)
class FrozenParameterSet:
    """One registered parameter set locked before an exact OOS fold begins."""

    fold_id: str
    fold_hash: str
    strategy_name: str
    strategy_version: str
    parameter_version: str
    parameter_hash: str
    registry_hash: str
    frozen_at: datetime
    validation_end: date
    out_of_sample_start: date
    out_of_sample_end: date
    freeze_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "fold_id",
            "strategy_name",
            "strategy_version",
            "parameter_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for value, field_name in (
            (self.fold_hash, "fold_hash"),
            (self.parameter_hash, "parameter_hash"),
            (self.registry_hash, "registry_hash"),
            (self.freeze_hash, "freeze_hash"),
        ):
            _validate_hash(value, field_name)
        ensure_aware(self.frozen_at)
        frozen_date = self.frozen_at.astimezone(SHANGHAI_TZ).date()
        if frozen_date > self.validation_end:
            raise ValueError("parameters must be frozen no later than validation_end")
        if not self.validation_end < self.out_of_sample_start <= self.out_of_sample_end:
            raise ValueError("frozen parameter OOS dates must follow validation")
        expected = stable_hash(
            {
                "fold_hash": self.fold_hash,
                "fold_id": self.fold_id,
                "frozen_at": self.frozen_at,
                "out_of_sample_end": self.out_of_sample_end,
                "out_of_sample_start": self.out_of_sample_start,
                "parameter_hash": self.parameter_hash,
                "parameter_version": self.parameter_version,
                "registry_hash": self.registry_hash,
                "strategy_name": self.strategy_name,
                "strategy_version": self.strategy_version,
                "validation_end": self.validation_end,
            }
        )
        if self.freeze_hash != expected:
            raise ValueError("freeze_hash does not match frozen parameter assignment")


@dataclass(frozen=True, slots=True)
class OutOfSampleFoldResult:
    """Execution-aware metrics produced only after one frozen OOS window ends."""

    fold_id: str
    fold_hash: str
    parameter_freeze_hash: str
    observation_count: int
    gross_return: Decimal
    benchmark_return: Decimal
    net_return: Decimal
    maximum_drawdown: Decimal
    turnover: Decimal
    transaction_cost: Decimal
    data_version: str
    code_version: str
    available_at: datetime
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        for field_name in ("fold_id", "data_version", "code_version"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for value, field_name in (
            (self.fold_hash, "fold_hash"),
            (self.parameter_freeze_hash, "parameter_freeze_hash"),
            (self.input_hash, "OOS input_hash"),
            (self.result_hash, "OOS result_hash"),
        ):
            _validate_hash(value, field_name)
        if (
            not isinstance(self.observation_count, int)
            or isinstance(self.observation_count, bool)
            or self.observation_count < 1
        ):
            raise ValueError("OOS observation_count must be a positive integer")
        for metric in (
            self.gross_return,
            self.benchmark_return,
            self.net_return,
            self.maximum_drawdown,
            self.turnover,
            self.transaction_cost,
        ):
            canonical_decimal(metric, field_name="OOS metric")
        if self.net_return > self.gross_return:
            raise ValueError("net OOS return cannot exceed gross return")
        if self.maximum_drawdown > 0:
            raise ValueError("maximum_drawdown must be non-positive")
        if self.turnover < 0 or self.transaction_cost < 0:
            raise ValueError("OOS turnover and transaction cost cannot be negative")
        ensure_aware(self.available_at)
        expected = stable_hash(
            {
                "available_at": self.available_at,
                "benchmark_return": self.benchmark_return,
                "code_version": self.code_version,
                "data_version": self.data_version,
                "fold_hash": self.fold_hash,
                "fold_id": self.fold_id,
                "gross_return": self.gross_return,
                "input_hash": self.input_hash,
                "maximum_drawdown": self.maximum_drawdown,
                "net_return": self.net_return,
                "observation_count": self.observation_count,
                "parameter_freeze_hash": self.parameter_freeze_hash,
                "transaction_cost": self.transaction_cost,
                "turnover": self.turnover,
            }
        )
        if self.result_hash != expected:
            raise ValueError("OOS result_hash does not match fold metrics")


@dataclass(frozen=True, slots=True)
class ParameterPerturbationResult:
    """One symmetric/local parameter perturbation result for stability analysis."""

    parameter_freeze_hash: str
    parameter_name: str
    relative_change: Decimal
    net_metric: Decimal
    result_hash: str

    def __post_init__(self) -> None:
        _validate_hash(self.parameter_freeze_hash, "parameter_freeze_hash")
        object.__setattr__(
            self, "parameter_name", _non_empty(self.parameter_name, "parameter_name")
        )
        for value in (self.relative_change, self.net_metric):
            canonical_decimal(value, field_name="parameter perturbation value")
        if self.relative_change == 0:
            raise ValueError("parameter perturbation relative_change cannot be zero")
        _validate_hash(self.result_hash, "perturbation result_hash")
        expected = stable_hash(
            {
                "net_metric": self.net_metric,
                "parameter_freeze_hash": self.parameter_freeze_hash,
                "parameter_name": self.parameter_name,
                "relative_change": self.relative_change,
            }
        )
        if self.result_hash != expected:
            raise ValueError("perturbation result_hash does not match result")


@dataclass(frozen=True, slots=True)
class ParameterStabilitySummary:
    """Descriptive sensitivity of one frozen parameter around its base metric."""

    parameter_freeze_hash: str
    parameter_name: str
    baseline_net_metric: Decimal
    scale_floor: Decimal
    maximum_allowed_degradation: Decimal
    perturbations: tuple[ParameterPerturbationResult, ...]
    relative_degradations: tuple[Decimal, ...]
    worst_relative_degradation: Decimal
    stable: bool
    result_hash: str

    def __post_init__(self) -> None:
        _validate_hash(self.parameter_freeze_hash, "parameter_freeze_hash")
        object.__setattr__(
            self, "parameter_name", _non_empty(self.parameter_name, "parameter_name")
        )
        for metric in (
            self.baseline_net_metric,
            self.scale_floor,
            self.maximum_allowed_degradation,
            *self.relative_degradations,
            self.worst_relative_degradation,
        ):
            canonical_decimal(metric, field_name="parameter stability value")
        if self.scale_floor <= 0 or self.maximum_allowed_degradation < 0:
            raise ValueError("stability scale must be positive and threshold non-negative")
        if len(self.perturbations) < 2:
            raise ValueError("parameter stability requires at least two perturbations")
        changes = tuple(item.relative_change for item in self.perturbations)
        if tuple(sorted(set(changes))) != changes:
            raise ValueError("parameter stability changes must be unique and sorted")
        if len(self.relative_degradations) != len(self.perturbations):
            raise ValueError("each perturbation requires one degradation value")
        if any(
            item.parameter_freeze_hash != self.parameter_freeze_hash
            or item.parameter_name != self.parameter_name
            for item in self.perturbations
        ):
            raise ValueError("stability perturbations must share parameter identity")
        expected_worst = max(self.relative_degradations)
        if self.worst_relative_degradation != expected_worst:
            raise ValueError("worst degradation must match perturbation results")
        if self.stable is not (self.worst_relative_degradation <= self.maximum_allowed_degradation):
            raise ValueError("parameter stability flag does not match degradation threshold")
        _validate_hash(self.result_hash, "stability result_hash")
        expected = stable_hash(
            {
                "baseline_net_metric": self.baseline_net_metric,
                "maximum_allowed_degradation": self.maximum_allowed_degradation,
                "parameter_freeze_hash": self.parameter_freeze_hash,
                "parameter_name": self.parameter_name,
                "perturbations": [item.result_hash for item in self.perturbations],
                "relative_degradations": self.relative_degradations,
                "scale_floor": self.scale_floor,
                "stable": self.stable,
                "worst_relative_degradation": self.worst_relative_degradation,
            }
        )
        if self.result_hash != expected:
            raise ValueError("stability result_hash does not match summary")


@dataclass(frozen=True, slots=True)
class WalkForwardValidationReport:
    """Reproducible aggregate of frozen-parameter OOS folds and sensitivity."""

    plan_result_hash: str
    freezes: tuple[FrozenParameterSet, ...]
    results: tuple[OutOfSampleFoldResult, ...]
    stability: tuple[ParameterStabilitySummary, ...]
    parameter_versions: tuple[str, ...]
    data_versions: tuple[str, ...]
    code_versions: tuple[str, ...]
    mean_gross_return: Decimal
    mean_net_return: Decimal
    worst_maximum_drawdown: Decimal
    total_turnover: Decimal
    total_transaction_cost: Decimal
    input_hash: str
    result_hash: str
    interpretation: str

    def __post_init__(self) -> None:
        for hash_value, field_name in (
            (self.plan_result_hash, "plan_result_hash"),
            (self.input_hash, "report input_hash"),
            (self.result_hash, "report result_hash"),
        ):
            _validate_hash(hash_value, field_name)
        if not self.freezes or len(self.freezes) != len(self.results):
            raise ValueError("validation report requires one freeze and result per fold")
        if tuple(item.fold_id for item in self.freezes) != tuple(
            item.fold_id for item in self.results
        ):
            raise ValueError("validation report freezes and results must share fold order")
        if tuple(sorted(set(self.parameter_versions))) != self.parameter_versions:
            raise ValueError("report parameter_versions must be unique and sorted")
        if tuple(sorted(set(self.data_versions))) != self.data_versions:
            raise ValueError("report data_versions must be unique and sorted")
        if tuple(sorted(set(self.code_versions))) != self.code_versions:
            raise ValueError("report code_versions must be unique and sorted")
        for metric in (
            self.mean_gross_return,
            self.mean_net_return,
            self.worst_maximum_drawdown,
            self.total_turnover,
            self.total_transaction_cost,
        ):
            canonical_decimal(metric, field_name="walk-forward report metric")
        if self.worst_maximum_drawdown > 0:
            raise ValueError("report worst drawdown must be non-positive")
        if self.total_turnover < 0 or self.total_transaction_cost < 0:
            raise ValueError("report turnover and cost cannot be negative")
        object.__setattr__(
            self, "interpretation", _non_empty(self.interpretation, "interpretation")
        )
        expected_parameter_versions = tuple(
            sorted({item.parameter_version for item in self.freezes})
        )
        expected_data_versions = tuple(sorted({item.data_version for item in self.results}))
        expected_code_versions = tuple(sorted({item.code_version for item in self.results}))
        if (
            self.parameter_versions != expected_parameter_versions
            or self.data_versions != expected_data_versions
            or self.code_versions != expected_code_versions
        ):
            raise ValueError("report version sets must match fold inputs")
        gross = tuple(item.gross_return for item in self.results)
        net = tuple(item.net_return for item in self.results)
        with localcontext() as context:
            context.prec = 50
            expected_mean_gross = sum(gross, Decimal(0)) / Decimal(len(gross))
            expected_mean_net = sum(net, Decimal(0)) / Decimal(len(net))
        if (
            self.mean_gross_return != expected_mean_gross
            or self.mean_net_return != expected_mean_net
            or self.worst_maximum_drawdown != min(item.maximum_drawdown for item in self.results)
            or self.total_turnover != sum((item.turnover for item in self.results), Decimal(0))
            or self.total_transaction_cost
            != sum((item.transaction_cost for item in self.results), Decimal(0))
        ):
            raise ValueError("report aggregates must match OOS fold results")
        if (
            tuple(
                sorted(
                    self.stability,
                    key=lambda item: (item.parameter_freeze_hash, item.parameter_name),
                )
            )
            != self.stability
        ):
            raise ValueError("report stability summaries must use canonical order")
        expected_input = stable_hash(
            {
                "freezes": [item.freeze_hash for item in self.freezes],
                "plan_result_hash": self.plan_result_hash,
                "results": [item.result_hash for item in self.results],
                "stability": [item.result_hash for item in self.stability],
            }
        )
        if self.input_hash != expected_input:
            raise ValueError("report input_hash does not match frozen inputs")
        expected_result = stable_hash(
            {
                "code_versions": self.code_versions,
                "data_versions": self.data_versions,
                "input_hash": self.input_hash,
                "interpretation": self.interpretation,
                "mean_gross_return": self.mean_gross_return,
                "mean_net_return": self.mean_net_return,
                "parameter_versions": self.parameter_versions,
                "total_transaction_cost": self.total_transaction_cost,
                "total_turnover": self.total_turnover,
                "worst_maximum_drawdown": self.worst_maximum_drawdown,
            }
        )
        if self.result_hash != expected_result:
            raise ValueError("report result_hash does not match aggregates")

    def identity_payload(self) -> dict[str, str]:
        return {
            "input_hash": self.input_hash,
            "plan_result_hash": self.plan_result_hash,
            "result_hash": self.result_hash,
        }


__all__ = [
    "FrozenParameterSet",
    "OutOfSampleFoldResult",
    "ParameterPerturbationResult",
    "ParameterRegistrySnapshot",
    "ParameterScalar",
    "ParameterStabilitySummary",
    "ParameterValue",
    "RegisteredParameterSet",
    "WalkForwardFold",
    "WalkForwardPlan",
    "WalkForwardSplitConfig",
    "WalkForwardValidationError",
    "WalkForwardValidationReport",
]
