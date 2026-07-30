"""Shared strategy output contracts that never submit orders directly."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from quant_agent.core.time import ensure_aware


@dataclass(frozen=True, slots=True)
class TargetWeight:
    instrument_id: str
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if not 0 <= self.weight <= 1:
            raise ValueError("target weight must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    as_of: datetime
    strategy_id: str
    targets: tuple[TargetWeight, ...]
    cash_weight: float
    data_version: str
    strategy_version: str
    parameter_version: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        total = self.cash_weight + sum(item.weight for item in self.targets)
        if not 0 <= self.cash_weight <= 1 or abs(total - 1.0) > 1e-8:
            raise ValueError("target portfolio weights must sum to one")

    @property
    def content_hash(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()
