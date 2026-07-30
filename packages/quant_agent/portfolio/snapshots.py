"""Immutable, hash-addressed account and holding snapshots."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from quant_agent.backtest.contracts import AssetType
from quant_agent.core.time import ensure_aware


@dataclass(frozen=True, slots=True)
class HoldingSnapshot:
    instrument_id: str
    asset_type: AssetType
    industry_id: str | None
    quantity: int
    available_quantity: int
    frozen_quantity: int
    average_cost: float
    last_price: float

    def __post_init__(self) -> None:
        if self.quantity < 0 or self.available_quantity < 0 or self.frozen_quantity < 0:
            raise ValueError("holding quantities cannot be negative")
        if self.available_quantity + self.frozen_quantity != self.quantity:
            raise ValueError("available and frozen quantities must equal total quantity")
        if self.average_cost < 0 or self.last_price <= 0:
            raise ValueError("holding prices are invalid")

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    snapshot_id: str
    account_id: str
    as_of: datetime
    available_cash: float
    frozen_cash: float
    holdings: tuple[HoldingSnapshot, ...]
    source: str
    version: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.available_cash < 0 or self.frozen_cash < 0:
            raise ValueError("account cash cannot be negative")
        identifiers = [item.instrument_id for item in self.holdings]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("holdings must have unique instrument ids")

    @property
    def total_cash(self) -> float:
        return self.available_cash + self.frozen_cash

    @property
    def total_market_value(self) -> float:
        return sum(item.market_value for item in self.holdings)

    @property
    def total_equity(self) -> float:
        return self.total_cash + self.total_market_value

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "AccountSnapshot":
        values = json.loads(payload)
        values["as_of"] = datetime.fromisoformat(values["as_of"])
        values["holdings"] = tuple(
            HoldingSnapshot(
                **{
                    **item,
                    "asset_type": AssetType(item["asset_type"]),
                }
            )
            for item in values["holdings"]
        )
        return cls(**values)
