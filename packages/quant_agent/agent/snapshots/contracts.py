"""Immutable decision identities and exact references to their input versions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import ConfigDict, TypeAdapter

from quant_agent.backtest.validation.contracts import RegisteredParameterSet
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware
from quant_agent.data.snapshots import validate_data_version
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.etf_rotation import ETFRotationConfig
from quant_agent.strategies.mainline_leader import MainlineLeaderConfig

StrategyConfig = ETFRotationConfig | MainlineLeaderConfig


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _hash(value: str, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _utc(value: datetime) -> datetime:
    return ensure_aware(value).astimezone(UTC)


def _registered_value_matches(actual: object, registered: object) -> bool:
    """Compare a registry scalar without Python's cross-type numeric equality."""

    if isinstance(actual, StrEnum):
        return (type(registered) is str and registered == actual.value) or (
            type(registered) is type(actual) and registered == actual
        )
    return type(registered) is type(actual) and registered == actual


@dataclass(frozen=True, slots=True)
class StrategySnapshotRef:
    """One strategy configuration and its registered tunable parameters."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    strategy_name: str
    strategy_version: str
    config_hash: str
    parameter_version: str
    parameter_hash: str
    registered_at: datetime

    def __post_init__(self) -> None:
        for name in ("strategy_name", "strategy_version", "parameter_version"):
            _text(getattr(self, name), name)
        if self.strategy_name not in {"etf-rotation", "mainline-leader"}:
            raise ValueError("unsupported decision strategy")
        _hash(self.config_hash, "config_hash")
        _hash(self.parameter_hash, "parameter_hash")
        object.__setattr__(self, "registered_at", _utc(self.registered_at))

    @classmethod
    def from_config(
        cls, *, config: StrategyConfig, parameters: RegisteredParameterSet
    ) -> StrategySnapshotRef:
        """Bind registered scalar parameters to the exact executable configuration."""

        if not isinstance(config, ETFRotationConfig | MainlineLeaderConfig):
            raise TypeError("unsupported strategy configuration")
        if not isinstance(parameters, RegisteredParameterSet):
            raise TypeError("parameters must be a RegisteredParameterSet")
        config = replace(config)
        parameters = replace(
            parameters, parameters=tuple(replace(item) for item in parameters.parameters)
        )
        expected_name = (
            "etf-rotation" if isinstance(config, ETFRotationConfig) else "mainline-leader"
        )
        if (
            parameters.strategy_name != expected_name
            or parameters.strategy_version != config.version
        ):
            raise ValueError("parameter registration does not match strategy identity")
        config_values = asdict(config)
        for item in parameters.parameters:
            if item.name == "version" or item.name not in config_values:
                raise ValueError("registered parameter is not a configurable strategy field")
            actual = config_values[item.name]
            if not _registered_value_matches(actual, item.value):
                raise ValueError("registered parameter value does not match strategy configuration")
        return cls(
            strategy_name=expected_name,
            strategy_version=config.version,
            config_hash=config.config_hash,
            parameter_version=parameters.parameter_version,
            parameter_hash=parameters.parameter_hash,
            registered_at=parameters.registered_at,
        )


def _strategy_identity(refs: tuple[StrategySnapshotRef, ...]) -> dict[str, str]:
    if len(refs) == 1:
        item = refs[0]
        return {
            "strategy_version": item.strategy_version,
            "strategy_config_hash": item.config_hash,
            "parameter_version": item.parameter_version,
            "parameter_hash": item.parameter_hash,
        }
    strategy_hash = stable_hash(
        {
            "strategies": [
                {
                    "name": item.strategy_name,
                    "version": item.strategy_version,
                    "hash": item.config_hash,
                }
                for item in refs
            ]
        }
    )
    parameter_hash = stable_hash(
        {
            "parameters": [
                {
                    "name": item.strategy_name,
                    "version": item.parameter_version,
                    "hash": item.parameter_hash,
                }
                for item in refs
            ]
        }
    )
    return {
        "strategy_version": f"bundle:{strategy_hash}",
        "strategy_config_hash": strategy_hash,
        "parameter_version": f"bundle:{parameter_hash}",
        "parameter_hash": parameter_hash,
    }


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Write-once Harness context; future outputs reference it without changing it.

    This is an identity contract, not permission to execute. Use the creation
    service to resolve and validate authoritative inputs before storing it.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    schema_version: str
    decision_id: str
    mode: RuntimeMode
    market: str
    as_of: datetime
    data_version: str
    data_content_hash: str
    strategy_version: str
    strategy_config_hash: str
    parameter_version: str
    parameter_hash: str
    strategy_refs: tuple[StrategySnapshotRef, ...]
    risk_policy_version: str
    risk_policy_hash: str
    account_id: str
    account_snapshot_id: str
    account_snapshot_hash: str
    code_commit: str
    code_artifact_hash: str
    agent_version: str
    model_version: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("decision snapshot schema_version must be '1'")
        for name in (
            "decision_id",
            "data_version",
            "strategy_version",
            "parameter_version",
            "risk_policy_version",
            "account_id",
            "account_snapshot_id",
            "agent_version",
            "model_version",
        ):
            _text(getattr(self, name), name)
        if not isinstance(self.mode, RuntimeMode) or self.mode is RuntimeMode.LIVE_AUTO:
            raise ValueError("decision mode must be supported and cannot be LIVE_AUTO")
        if self.market != "CN_A":
            raise ValueError("decision market must be CN_A")
        validate_data_version(self.data_version)
        if (
            not isinstance(self.code_commit, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.code_commit) is None
        ):
            raise ValueError("code_commit must be a full immutable Git commit ID")
        object.__setattr__(self, "as_of", _utc(self.as_of))
        for name in (
            "data_content_hash",
            "strategy_config_hash",
            "parameter_hash",
            "risk_policy_hash",
            "account_snapshot_hash",
            "code_artifact_hash",
            "content_hash",
        ):
            _hash(getattr(self, name), name)
        if (
            not isinstance(self.strategy_refs, tuple)
            or not self.strategy_refs
            or any(not isinstance(item, StrategySnapshotRef) for item in self.strategy_refs)
        ):
            raise ValueError("strategy_refs must be a non-empty immutable tuple")
        names = tuple(item.strategy_name for item in self.strategy_refs)
        if names != tuple(sorted(set(names))):
            raise ValueError("strategy_refs must be unique and sorted by strategy_name")
        for item in self.strategy_refs:
            replace(item)
            if item.registered_at > self.as_of:
                raise ValueError("strategy parameters were not registered at the decision boundary")
        for name, expected in _strategy_identity(self.strategy_refs).items():
            if getattr(self, name) != expected:
                raise ValueError("top-level strategy identity does not bind its references")
        if self.content_hash != stable_hash(self._content_payload()):
            raise ValueError("decision content_hash does not match the complete snapshot")

    @classmethod
    def build(
        cls,
        *,
        decision_id: str,
        mode: RuntimeMode,
        market: str,
        as_of: datetime,
        data_version: str,
        data_content_hash: str,
        strategy_refs: tuple[StrategySnapshotRef, ...],
        risk_policy_version: str,
        risk_policy_hash: str,
        account_id: str,
        account_snapshot_id: str,
        account_snapshot_hash: str,
        code_commit: str,
        code_artifact_hash: str,
        agent_version: str,
        model_version: str,
    ) -> DecisionSnapshot:
        refs = tuple(sorted(strategy_refs, key=lambda item: item.strategy_name))
        values: dict[str, Any] = {
            "schema_version": "1",
            "decision_id": decision_id,
            "mode": mode,
            "market": market,
            "as_of": _utc(as_of),
            "data_version": data_version,
            "data_content_hash": data_content_hash,
            **_strategy_identity(refs),
            "strategy_refs": refs,
            "risk_policy_version": risk_policy_version,
            "risk_policy_hash": risk_policy_hash,
            "account_id": account_id,
            "account_snapshot_id": account_snapshot_id,
            "account_snapshot_hash": account_snapshot_hash,
            "code_commit": code_commit,
            "code_artifact_hash": code_artifact_hash,
            "agent_version": agent_version,
            "model_version": model_version,
        }
        payload = {**values, "strategy_refs": [asdict(item) for item in refs]}
        return cls(**values, content_hash=stable_hash(payload))

    def _content_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "content_hash"}

    def to_json(self) -> str:
        """Return a detached JSON value with deterministic ordering."""

        replace(self)
        return json.dumps(
            json.loads(_SNAPSHOT_ADAPTER.dump_json(self)), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, value: str) -> DecisionSnapshot:
        """Reject missing/extra fields, coercions, invalid references and changed hashes."""

        json.loads(value, object_pairs_hook=_unique_json_object)
        return _SNAPSHOT_ADAPTER.validate_json(value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("decision JSON contains a duplicate field")
        result[key] = value
    return result


_SNAPSHOT_ADAPTER = TypeAdapter(DecisionSnapshot)

__all__ = ["DecisionSnapshot", "StrategyConfig", "StrategySnapshotRef"]
