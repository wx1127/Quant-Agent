"""Create and resume decisions only against their verified input versions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from quant_agent.backtest.validation.contracts import RegisteredParameterSet
from quant_agent.config import RuntimeMode
from quant_agent.core.identifiers import new_decision_id
from quant_agent.data.snapshots import SnapshotManifest
from quant_agent.portfolio import AccountSnapshot
from quant_agent.risk.portfolio import PortfolioRiskPolicy

from .contracts import DecisionSnapshot, StrategyConfig, StrategySnapshotRef
from .repository import DecisionSnapshotRepository


class DecisionDataStore(Protocol):
    """Trusted immutable dataset store, implemented by data.snapshots.SnapshotStore."""

    def load_verified_manifest(self, data_version: str) -> SnapshotManifest:
        """Return the exact manifest whose referenced content was verified."""

        ...


@dataclass(frozen=True, slots=True)
class DecisionStrategyBinding:
    config: StrategyConfig
    parameters: RegisteredParameterSet

    def reference(self) -> StrategySnapshotRef:
        return StrategySnapshotRef.from_config(config=self.config, parameters=self.parameters)


class DecisionSnapshotService:
    """Resolve input identities at creation and reject replacement on resume.

    Dataset integrity does not itself establish point-in-time correctness of
    every row. Downstream deterministic tools must still filter availability
    against the frozen decision.as_of. No method grants execution authority.
    """

    def __init__(
        self, *, repository: DecisionSnapshotRepository, data_store: DecisionDataStore
    ) -> None:
        self._repository = repository
        self._data_store = data_store

    def create(
        self,
        *,
        mode: RuntimeMode,
        market: str,
        as_of: datetime,
        data_version: str,
        account: AccountSnapshot,
        strategies: tuple[DecisionStrategyBinding, ...],
        risk_policy: PortfolioRiskPolicy,
        code_commit: str,
        code_artifact_hash: str,
        agent_version: str,
        model_version: str,
        decision_id: str | None = None,
    ) -> DecisionSnapshot:
        snapshot = self._resolve(
            decision_id=decision_id if decision_id is not None else new_decision_id(at=as_of),
            mode=mode,
            market=market,
            as_of=as_of,
            data_version=data_version,
            account=account,
            strategies=strategies,
            risk_policy=risk_policy,
            code_commit=code_commit,
            code_artifact_hash=code_artifact_hash,
            agent_version=agent_version,
            model_version=model_version,
        )
        return self._repository.create(snapshot)

    def resume(
        self,
        decision_id: str,
        *,
        account: AccountSnapshot,
        strategies: tuple[DecisionStrategyBinding, ...],
        risk_policy: PortfolioRiskPolicy,
        code_commit: str,
        code_artifact_hash: str,
        agent_version: str,
        model_version: str,
    ) -> DecisionSnapshot:
        """Revalidate exact inputs; changed inputs require a newly created decision."""

        stored = self._repository.get(decision_id)
        current = self._resolve(
            decision_id=stored.decision_id,
            mode=stored.mode,
            market=stored.market,
            as_of=stored.as_of,
            data_version=stored.data_version,
            account=account,
            strategies=strategies,
            risk_policy=risk_policy,
            code_commit=code_commit,
            code_artifact_hash=code_artifact_hash,
            agent_version=agent_version,
            model_version=model_version,
        )
        if current != stored:
            raise ValueError("decision inputs changed; create a new decision_id")
        return stored

    def _resolve(
        self,
        *,
        decision_id: str,
        mode: RuntimeMode,
        market: str,
        as_of: datetime,
        data_version: str,
        account: AccountSnapshot,
        strategies: tuple[DecisionStrategyBinding, ...],
        risk_policy: PortfolioRiskPolicy,
        code_commit: str,
        code_artifact_hash: str,
        agent_version: str,
        model_version: str,
    ) -> DecisionSnapshot:
        if not isinstance(account, AccountSnapshot):
            raise TypeError("account must be an AccountSnapshot")
        account = AccountSnapshot.from_json(account.to_json())
        if (account.runtime_mode, account.as_of, account.data_version) != (
            mode,
            as_of,
            data_version,
        ):
            raise ValueError("account mode, time and data version must match the decision")
        if not isinstance(risk_policy, PortfolioRiskPolicy):
            raise TypeError("risk_policy must be a PortfolioRiskPolicy")
        risk_policy = replace(risk_policy)
        if (
            not isinstance(strategies, tuple)
            or not strategies
            or any(not isinstance(item, DecisionStrategyBinding) for item in strategies)
        ):
            raise ValueError("strategies must be a non-empty immutable tuple of bindings")
        refs = tuple(item.reference() for item in strategies)

        manifest = self._load_detached_manifest(data_version)

        return DecisionSnapshot.build(
            decision_id=decision_id,
            mode=mode,
            market=market,
            as_of=as_of,
            data_version=data_version,
            data_content_hash=manifest.content_hash,
            strategy_refs=refs,
            risk_policy_version=risk_policy.version,
            risk_policy_hash=risk_policy.policy_hash,
            account_id=account.account_id,
            account_snapshot_id=account.snapshot_id,
            account_snapshot_hash=account.content_hash,
            code_commit=code_commit,
            code_artifact_hash=code_artifact_hash,
            agent_version=agent_version,
            model_version=model_version,
        )

    def _load_detached_manifest(self, data_version: str) -> SnapshotManifest:
        manifest = self._data_store.load_verified_manifest(data_version)
        if not isinstance(manifest, SnapshotManifest):
            raise TypeError("data_store must return a SnapshotManifest")
        detached = SnapshotManifest.model_validate_json(manifest.model_dump_json())
        if detached.data_version != data_version:
            raise ValueError("decision dataset manifest does not match data_version")
        return detached


__all__ = ["DecisionDataStore", "DecisionSnapshotService", "DecisionStrategyBinding"]
