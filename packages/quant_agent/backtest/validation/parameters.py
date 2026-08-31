"""Append-only parameter registration and fold-specific freezing."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from quant_agent.backtest.validation.contracts import (
    FrozenParameterSet,
    ParameterRegistrySnapshot,
    ParameterScalar,
    ParameterValue,
    RegisteredParameterSet,
    WalkForwardFold,
    WalkForwardValidationError,
)
from quant_agent.core.time import ensure_aware
from quant_agent.regime.contracts import stable_hash


class ParameterRegistry:
    """In-memory append-only registry with deterministic immutable snapshots."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], RegisteredParameterSet] = {}

    def register(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        parameter_version: str,
        parameters: Mapping[str, ParameterScalar],
        registered_at: datetime,
    ) -> RegisteredParameterSet:
        """Register a new version; identical re-registration is idempotent."""

        ensure_aware(registered_at)
        values = tuple(
            ParameterValue(name=name, value=value) for name, value in sorted(parameters.items())
        )
        parameter_hash = stable_hash(
            {
                "parameter_version": parameter_version,
                "parameters": [{"name": item.name, "value": item.value} for item in values],
                "strategy_name": strategy_name,
                "strategy_version": strategy_version,
            }
        )
        registration = RegisteredParameterSet(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            parameter_version=parameter_version,
            parameters=values,
            registered_at=registered_at,
            parameter_hash=parameter_hash,
        )
        key = (strategy_name.strip(), strategy_version.strip(), parameter_version.strip())
        existing = self._entries.get(key)
        if existing is not None:
            if existing.parameter_hash == registration.parameter_hash:
                return existing
            raise WalkForwardValidationError(
                "parameter version is already registered with different values"
            )
        self._entries[key] = registration
        return registration

    def snapshot(self) -> ParameterRegistrySnapshot:
        """Freeze the registry ordering and identity without exposing mutable state."""

        entries = tuple(self._entries[key] for key in sorted(self._entries))
        return ParameterRegistrySnapshot(
            entries=entries,
            registry_hash=stable_hash({"entries": [item.parameter_hash for item in entries]}),
        )

    def freeze_for_fold(
        self,
        *,
        fold: WalkForwardFold,
        strategy_name: str,
        strategy_version: str,
        parameter_version: str,
        frozen_at: datetime,
    ) -> FrozenParameterSet:
        """Bind one pre-registered parameter version to an entire future OOS fold."""

        ensure_aware(frozen_at)
        key = (strategy_name.strip(), strategy_version.strip(), parameter_version.strip())
        registration = self._entries.get(key)
        if registration is None:
            raise WalkForwardValidationError("parameter version is not registered")
        if registration.registered_at > frozen_at:
            raise WalkForwardValidationError(
                "parameter version cannot be frozen before it is registered"
            )
        registry = self.snapshot()
        payload = {
            "fold_hash": fold.fold_hash,
            "fold_id": fold.fold_id,
            "frozen_at": frozen_at,
            "out_of_sample_end": fold.out_of_sample_end,
            "out_of_sample_start": fold.out_of_sample_start,
            "parameter_hash": registration.parameter_hash,
            "parameter_version": registration.parameter_version,
            "registry_hash": registry.registry_hash,
            "strategy_name": registration.strategy_name,
            "strategy_version": registration.strategy_version,
            "validation_end": fold.validation_end,
        }
        return FrozenParameterSet(
            fold_id=fold.fold_id,
            fold_hash=fold.fold_hash,
            strategy_name=registration.strategy_name,
            strategy_version=registration.strategy_version,
            parameter_version=registration.parameter_version,
            parameter_hash=registration.parameter_hash,
            registry_hash=registry.registry_hash,
            frozen_at=frozen_at,
            validation_end=fold.validation_end,
            out_of_sample_start=fold.out_of_sample_start,
            out_of_sample_end=fold.out_of_sample_end,
            freeze_hash=stable_hash(payload),
        )


__all__ = ["ParameterRegistry"]
