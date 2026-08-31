"""Immutable metadata and point-in-time contracts for rolling features."""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from quant_agent.core.time import ensure_aware
from quant_agent.features.core.identity import canonical_decimal


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _canonical_json_object(value: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("feature parameters must be a finite JSON object") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guaranteed by the annotation
        raise ValueError("feature parameters must be a JSON object")
    return encoded


def _utc_iso(value: datetime) -> str:
    return ensure_aware(value).astimezone(UTC).isoformat(timespec="microseconds")


class MissingValuePolicy(StrEnum):
    """How a selected chronological window handles explicit missing observations."""

    SKIP = "SKIP"
    REQUIRE_COMPLETE = "REQUIRE_COMPLETE"


class FeatureStatus(StrEnum):
    """Whether a feature value was produced for the requested decision time."""

    READY = "READY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Versioned rolling-window definition included in every cache identity."""

    name: str
    version: str
    calculator_id: str
    window: int
    min_observations: int
    parameters_json: str = "{}"
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.SKIP

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "name"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        object.__setattr__(
            self,
            "calculator_id",
            _non_empty(self.calculator_id, "calculator_id"),
        )
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.min_observations <= 0 or self.min_observations > self.window:
            raise ValueError("min_observations must be between 1 and window")
        try:
            parameters = json.loads(self.parameters_json)
        except json.JSONDecodeError as error:
            raise ValueError("parameters_json must contain valid JSON") from error
        if not isinstance(parameters, dict):
            raise ValueError("parameters_json must contain a JSON object")
        object.__setattr__(self, "parameters_json", _canonical_json_object(parameters))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        calculator_id: str,
        window: int,
        min_observations: int,
        parameters: dict[str, Any] | None = None,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.SKIP,
    ) -> "FeatureDefinition":
        """Create a definition while canonicalizing arbitrary JSON parameters."""

        return cls(
            name=name,
            version=version,
            calculator_id=calculator_id,
            window=window,
            min_observations=min_observations,
            parameters_json=_canonical_json_object(parameters or {}),
            missing_value_policy=missing_value_policy,
        )

    @property
    def definition_hash(self) -> str:
        """Return a stable identity for code version, window semantics, and parameters."""

        payload = {
            "calculator_id": self.calculator_id,
            "min_observations": self.min_observations,
            "missing_value_policy": self.missing_value_policy.value,
            "name": self.name,
            "parameters": json.loads(self.parameters_json),
            "version": self.version,
            "window": self.window,
        }
        return hashlib.sha256(_canonical_json_object(payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One revisable numeric observation with an explicit information-availability time."""

    observation_key: str
    observed_at: datetime
    available_at: datetime
    value: Decimal | None
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_key",
            _non_empty(self.observation_key, "observation_key"),
        )
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        ensure_aware(self.observed_at)
        ensure_aware(self.available_at)
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        if self.value is not None and not self.value.is_finite():
            raise ValueError("observation value must be finite")

    def fingerprint_payload(self) -> dict[str, str | None]:
        """Return canonical, timezone-normalized fields used to bind results to inputs."""

        return {
            "available_at": _utc_iso(self.available_at),
            "observation_key": self.observation_key,
            "observed_at": _utc_iso(self.observed_at),
            "revision": self.revision,
            "value": canonical_decimal(self.value) if self.value is not None else None,
        }


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """Decision-time boundary and immutable data identity for one feature computation."""

    entity_id: str
    as_of: datetime
    data_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _non_empty(self.entity_id, "entity_id"))
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        ensure_aware(self.as_of)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Reproducible feature result, including the exact selected-input fingerprint."""

    definition: FeatureDefinition
    request: FeatureRequest
    status: FeatureStatus
    value: Decimal | None
    observation_count: int
    window_start: datetime | None
    window_end: datetime | None
    input_hash: str
    cache_key: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.observation_count < 0:
            raise ValueError("observation_count cannot be negative")
        if len(self.input_hash) != 64 or len(self.cache_key) != 64:
            raise ValueError("feature hashes must be SHA-256 hexadecimal digests")
        if self.status is FeatureStatus.READY:
            if self.value is None or not self.value.is_finite():
                raise ValueError("ready feature result requires a finite value")
            if self.window_start is None or self.window_end is None:
                raise ValueError("ready feature result requires window boundaries")
            if self.reason is not None:
                raise ValueError("ready feature result cannot include a failure reason")
        elif self.value is not None:
            raise ValueError("unavailable feature result cannot contain a value")


def stable_feature_hash(payload: dict[str, Any]) -> str:
    """Hash a finite JSON payload using one canonical representation."""

    return hashlib.sha256(_canonical_json_object(payload).encode()).hexdigest()


def finite_decimal(value: Decimal | int | float | str) -> Decimal:
    """Normalize calculator output and reject floating-point NaN or infinity."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("feature calculator returned a non-finite value")
    normalized = value if isinstance(value, Decimal) else Decimal(str(value))
    if not normalized.is_finite():
        raise ValueError("feature calculator returned a non-finite value")
    return normalized
