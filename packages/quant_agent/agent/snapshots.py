"""Immutable, hash-addressed decision snapshots."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol

from quant_agent.config.models import RuntimeMode
from quant_agent.core.identifiers import new_decision_id
from quant_agent.core.time import ensure_aware


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """All authoritative versions used by one Agent decision."""

    decision_id: str
    mode: RuntimeMode
    market: str
    as_of: datetime
    data_version: str
    strategy_version: str
    parameter_version: str
    risk_policy_version: str
    account_snapshot_id: str | None
    code_commit: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        required = (
            self.decision_id,
            self.market,
            self.data_version,
            self.strategy_version,
            self.parameter_version,
            self.risk_policy_version,
            self.code_commit,
        )
        if any(not item.strip() for item in required):
            raise ValueError("decision snapshot version fields cannot be empty")
        if not self.decision_id.startswith("dec_"):
            raise ValueError("invalid decision id")
        if (
            self.mode in {RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}
            and not self.account_snapshot_id
        ):
            raise ValueError("account snapshot is required in trading modes")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True, separators=(",", ":"))

    @classmethod
    def create(
        cls,
        *,
        mode: RuntimeMode,
        market: str,
        as_of: datetime,
        data_version: str,
        strategy_version: str,
        parameter_version: str,
        risk_policy_version: str,
        account_snapshot_id: str | None,
        code_commit: str,
    ) -> "DecisionSnapshot":
        return cls(
            decision_id=str(new_decision_id(as_of)),
            mode=mode,
            market=market,
            as_of=as_of,
            data_version=data_version,
            strategy_version=strategy_version,
            parameter_version=parameter_version,
            risk_policy_version=risk_policy_version,
            account_snapshot_id=account_snapshot_id,
            code_commit=code_commit,
        )

    @classmethod
    def from_json(cls, payload: str) -> "DecisionSnapshot":
        values = json.loads(payload)
        values["as_of"] = datetime.fromisoformat(values["as_of"])
        values["mode"] = RuntimeMode(values["mode"])
        return cls(**values)


class DecisionSnapshotStore(Protocol):
    def save(self, snapshot: DecisionSnapshot) -> None: ...

    def get(self, decision_id: str) -> DecisionSnapshot: ...


class InMemoryDecisionSnapshotStore:
    """Append-only store that rejects silent replacement."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, str]] = {}

    def save(self, snapshot: DecisionSnapshot) -> None:
        existing = self._items.get(snapshot.decision_id)
        item = (snapshot.content_hash, snapshot.to_json())
        if existing is not None and existing != item:
            raise ValueError("decision snapshot is immutable")
        self._items[snapshot.decision_id] = item

    def get(self, decision_id: str) -> DecisionSnapshot:
        try:
            expected_hash, payload = self._items[decision_id]
        except KeyError as exc:
            raise KeyError(f"decision snapshot not found: {decision_id}") from exc
        snapshot = DecisionSnapshot.from_json(payload)
        if snapshot.content_hash != expected_hash:
            raise ValueError("decision snapshot content hash mismatch")
        return snapshot
