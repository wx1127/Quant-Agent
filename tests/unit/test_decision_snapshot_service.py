"""Authoritative-source binding tests for the decision-snapshot service."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest
from tests.unit import test_portfolio_snapshots as account_fixtures

import quant_agent.agent as agent_api
from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.agent.snapshots.repository import (
    DecisionSnapshotConflict,
    InMemoryDecisionSnapshotRepository,
)
from quant_agent.agent.snapshots.service import (
    DecisionSnapshotService,
    DecisionStrategyBinding,
)
from quant_agent.backtest.validation import ParameterRegistry, RegisteredParameterSet
from quant_agent.config import RuntimeMode
from quant_agent.data.quality import QualityReport
from quant_agent.data.snapshots import SnapshotManifest, SnapshotStore
from quant_agent.portfolio import AccountSnapshot
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.portfolio import PortfolioRiskPolicy
from quant_agent.strategies.etf_rotation import ETFRotationConfig

AS_OF = account_fixtures.AS_OF
DATA_VERSION = "snapshot-v1"
SECOND_DATA_VERSION = "snapshot-v2"
DECISION_ID = "dec_20260907_0123456789abcdef0123456789abcdef"
CODE_COMMIT = "a" * 40
CODE_ARTIFACT_HASH = stable_hash({"artifact": "quant-agent-source-tree-v1"})
AGENT_VERSION = "quant-agent-harness-v1"
MODEL_VERSION = "model-release-v1"


def test_snapshot_api_is_exported_from_the_agent_package() -> None:
    assert agent_api.DecisionSnapshot is DecisionSnapshot
    assert agent_api.DecisionSnapshotService is DecisionSnapshotService
    assert agent_api.InMemoryDecisionSnapshotRepository is InMemoryDecisionSnapshotRepository


def _publish(store: SnapshotStore, data_version: str, value: int) -> SnapshotManifest:
    return store.create(
        data_version,
        {"daily_bars": pa.table({"instrument_id": ["CN.SSE.510300"], "close": [value]})},
        quality_report=QualityReport(
            data_version=data_version,
            observed_at=AS_OF,
            issues=(),
        ),
        metadata={"as_of": AS_OF.isoformat(), "source": "service-test"},
    )


@pytest.fixture
def store_and_root(tmp_path: Path) -> tuple[SnapshotStore, Path]:
    root = tmp_path / "market-snapshots"
    store = SnapshotStore(root)
    _publish(store, DATA_VERSION, 1)
    return store, root


def _account(
    *,
    data_version: str = DATA_VERSION,
    mode: RuntimeMode = RuntimeMode.PAPER,
    snapshot_id: str = "account-snapshot-20260828",
) -> AccountSnapshot:
    source = account_fixtures._account()
    return AccountSnapshot.build(
        snapshot_id=snapshot_id,
        account_id=source.account_id,
        runtime_mode=mode,
        as_of=source.as_of,
        valuation_at=source.valuation_at,
        data_version=data_version,
        currency=source.currency,
        cash=source.cash,
        positions=source.positions,
        previous_snapshot_hash=source.previous_snapshot_hash,
        source_event_log_hash=source.source_event_log_hash,
    )


def _binding(
    *,
    config: ETFRotationConfig | None = None,
    parameter_version: str = "etf-balanced-v1",
    registered_at: datetime | None = None,
) -> DecisionStrategyBinding:
    active = config or ETFRotationConfig()
    parameters = ParameterRegistry().register(
        strategy_name="etf-rotation",
        strategy_version=active.version,
        parameter_version=parameter_version,
        parameters={
            "force_downtrend_cash": active.force_downtrend_cash,
            "maximum_instrument_weight": active.maximum_instrument_weight,
            "top_n": active.top_n,
        },
        registered_at=registered_at or AS_OF - timedelta(days=1),
    )
    return DecisionStrategyBinding(config=active, parameters=parameters)


def _create(
    service: DecisionSnapshotService,
    *,
    decision_id: str | None = DECISION_ID,
    mode: RuntimeMode = RuntimeMode.PAPER,
    as_of: datetime = AS_OF,
    data_version: str = DATA_VERSION,
    account: AccountSnapshot | None = None,
    strategies: tuple[DecisionStrategyBinding, ...] | None = None,
    risk_policy: PortfolioRiskPolicy | None = None,
    code_commit: str = CODE_COMMIT,
    code_artifact_hash: str = CODE_ARTIFACT_HASH,
    agent_version: str = AGENT_VERSION,
    model_version: str = MODEL_VERSION,
) -> DecisionSnapshot:
    return service.create(
        decision_id=decision_id,
        mode=mode,
        market="CN_A",
        as_of=as_of,
        data_version=data_version,
        account=account or _account(data_version=data_version, mode=mode),
        strategies=strategies if strategies is not None else (_binding(),),
        risk_policy=risk_policy or PortfolioRiskPolicy(),
        code_commit=code_commit,
        code_artifact_hash=code_artifact_hash,
        agent_version=agent_version,
        model_version=model_version,
    )


def test_create_binds_verified_real_sources_and_replays_exactly(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    repository = InMemoryDecisionSnapshotRepository()
    service = DecisionSnapshotService(repository=repository, data_store=store)
    account = _account()
    binding = _binding()
    risk_policy = PortfolioRiskPolicy()
    manifest = store.load_manifest(DATA_VERSION)

    created = _create(
        service,
        account=account,
        strategies=(binding,),
        risk_policy=risk_policy,
    )
    replayed = _create(
        service,
        account=account,
        strategies=(binding,),
        risk_policy=risk_policy,
    )

    assert replayed == created == repository.get(DECISION_ID)
    assert created.as_of == AS_OF.astimezone(UTC)
    assert created.mode is RuntimeMode.PAPER
    assert created.market == "CN_A"
    assert (created.data_version, created.data_content_hash) == (
        manifest.data_version,
        manifest.content_hash,
    )
    assert (created.account_id, created.account_snapshot_id, created.account_snapshot_hash) == (
        account.account_id,
        account.snapshot_id,
        account.content_hash,
    )
    assert (created.risk_policy_version, created.risk_policy_hash) == (
        risk_policy.version,
        risk_policy.policy_hash,
    )
    assert created.strategy_refs == (binding.reference(),)
    assert created.code_commit == CODE_COMMIT
    assert created.code_artifact_hash == CODE_ARTIFACT_HASH
    assert created.agent_version == AGENT_VERSION
    assert created.model_version == MODEL_VERSION


def test_create_generates_a_decision_id_and_accepts_same_instant_in_utc(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    created = _create(service, decision_id=None, as_of=AS_OF.astimezone(UTC))

    assert created.decision_id.startswith(f"dec_{AS_OF:%Y%m%d}_")
    assert created.as_of == AS_OF.astimezone(UTC)


def test_updated_data_requires_a_new_decision_identity(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    _publish(store, SECOND_DATA_VERSION, 2)
    repository = InMemoryDecisionSnapshotRepository()
    service = DecisionSnapshotService(repository=repository, data_store=store)
    original = _create(service)
    updated_account = _account(
        data_version=SECOND_DATA_VERSION,
        snapshot_id="account-snapshot-20260828-v2",
    )

    with pytest.raises(DecisionSnapshotConflict):
        _create(
            service,
            data_version=SECOND_DATA_VERSION,
            account=updated_account,
        )

    updated = _create(
        service,
        decision_id="dec_20260907_ffffffffffffffffffffffffffffffff",
        data_version=SECOND_DATA_VERSION,
        account=updated_account,
    )

    assert repository.get(original.decision_id) == original
    assert updated.decision_id != original.decision_id
    assert updated.data_version == SECOND_DATA_VERSION
    assert updated.data_content_hash != original.data_content_hash


def test_resume_revalidates_every_current_source_and_rejects_changed_inputs(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )
    account = _account()
    strategies = (_binding(),)
    policy = PortfolioRiskPolicy()
    created = _create(service, account=account, strategies=strategies, risk_policy=policy)

    resumed = service.resume(
        created.decision_id,
        account=account,
        strategies=strategies,
        risk_policy=policy,
        code_commit=CODE_COMMIT,
        code_artifact_hash=CODE_ARTIFACT_HASH,
        agent_version=AGENT_VERSION,
        model_version=MODEL_VERSION,
    )
    assert resumed == created

    with pytest.raises(ValueError, match=r"inputs changed.*new decision_id"):
        service.resume(
            created.decision_id,
            account=account,
            strategies=strategies,
            risk_policy=policy,
            code_commit=CODE_COMMIT,
            code_artifact_hash=stable_hash({"artifact": "changed"}),
            agent_version=AGENT_VERSION,
            model_version=MODEL_VERSION,
        )
    with pytest.raises(ValueError, match=r"inputs changed.*new decision_id"):
        service.resume(
            created.decision_id,
            account=account,
            strategies=strategies,
            risk_policy=replace(policy, version="portfolio-risk-policy-v2"),
            code_commit=CODE_COMMIT,
            code_artifact_hash=CODE_ARTIFACT_HASH,
            agent_version=AGENT_VERSION,
            model_version=MODEL_VERSION,
        )


@pytest.mark.parametrize(
    ("mode", "as_of", "data_version"),
    (
        (RuntimeMode.RESEARCH, AS_OF, DATA_VERSION),
        (RuntimeMode.PAPER, AS_OF + timedelta(microseconds=1), DATA_VERSION),
        (RuntimeMode.PAPER, AS_OF, SECOND_DATA_VERSION),
    ),
    ids=("mode", "as-of", "data-version"),
)
def test_account_boundary_must_exactly_match_the_decision(
    store_and_root: tuple[SnapshotStore, Path],
    mode: RuntimeMode,
    as_of: datetime,
    data_version: str,
) -> None:
    store, _ = store_and_root
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    with pytest.raises(ValueError, match="account mode, time and data version"):
        _create(
            service,
            mode=mode,
            as_of=as_of,
            data_version=data_version,
            account=_account(),
        )


def test_service_rejects_live_auto_empty_bindings_and_future_registration(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    with pytest.raises(ValueError, match="LIVE_AUTO"):
        _create(service, mode=RuntimeMode.LIVE_AUTO)
    with pytest.raises(ValueError, match="non-empty immutable tuple"):
        _create(service, strategies=())
    with pytest.raises(ValueError, match="registered at the decision boundary"):
        _create(service, strategies=(_binding(registered_at=AS_OF + timedelta(seconds=1)),))


def test_service_rejects_wrong_runtime_source_types(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    with pytest.raises(TypeError, match="AccountSnapshot"):
        _create(service, account=cast(AccountSnapshot, object()))
    with pytest.raises(TypeError, match="PortfolioRiskPolicy"):
        _create(service, risk_policy=cast(PortfolioRiskPolicy, object()))
    with pytest.raises(ValueError, match="non-empty immutable tuple"):
        _create(
            service,
            strategies=cast(tuple[DecisionStrategyBinding, ...], [_binding()]),
        )


def test_real_snapshot_file_tampering_fails_closed(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, root = store_and_root
    manifest = store.load_manifest(DATA_VERSION)
    data_file = root / DATA_VERSION / manifest.files[0].relative_path
    with data_file.open("ab") as target:
        target.write(b"tampered")
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    with pytest.raises(ValueError, match="snapshot integrity verification failed"):
        _create(service)


def test_snapshot_store_returns_the_same_manifest_instance_it_verified(
    store_and_root: tuple[SnapshotStore, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = store_and_root
    loaded = store.load_manifest(DATA_VERSION)
    load_count = 0

    def load_once(data_version: str) -> SnapshotManifest:
        nonlocal load_count
        assert data_version == DATA_VERSION
        load_count += 1
        if load_count > 1:
            raise AssertionError("verified-manifest loading must not reload by version")
        return loaded

    monkeypatch.setattr(store, "load_manifest", load_once)

    verified = store.load_verified_manifest(DATA_VERSION)

    assert verified == loaded
    assert verified is not loaded
    assert load_count == 1


def _manifest(label: str, *, data_version: str = DATA_VERSION) -> SnapshotManifest:
    return SnapshotManifest(
        data_version=data_version,
        created_at=AS_OF.isoformat(),
        content_hash=stable_hash({"scripted-manifest": label}),
        files=(),
        metadata={"label": label},
    )


class _ScriptedDataStore:
    def __init__(self, manifest: object) -> None:
        self._manifest = manifest

    def load_verified_manifest(self, data_version: str) -> SnapshotManifest:
        del data_version
        return cast(SnapshotManifest, self._manifest)


class _FailingDataStore:
    def load_verified_manifest(self, data_version: str) -> SnapshotManifest:
        del data_version
        raise ValueError("snapshot integrity verification failed")


def test_verified_manifest_loader_failure_is_not_downgraded() -> None:
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=_FailingDataStore(),
    )

    with pytest.raises(ValueError, match="integrity verification failed"):
        _create(service)


class _ABATrapDataStore:
    """Expose the old A-B-A failure mode while returning the truly verified B."""

    def __init__(self) -> None:
        self.a = _manifest("A")
        self.b = _manifest("B")
        self.legacy_load_calls = 0
        self.legacy_verify_calls = 0
        self.verified_load_calls = 0

    def load_verified_manifest(self, data_version: str) -> SnapshotManifest:
        del data_version
        self.verified_load_calls += 1
        return self.b

    def load_manifest(self, data_version: str) -> SnapshotManifest:
        del data_version
        sequence = (self.a, self.b, self.a)
        item = sequence[min(self.legacy_load_calls, len(sequence) - 1)]
        self.legacy_load_calls += 1
        return item

    def verify(self, data_version: str) -> bool:
        del data_version
        self.legacy_verify_calls += 1
        return True


def test_service_binds_the_exact_verified_manifest_instead_of_an_aba_reload() -> None:
    store = _ABATrapDataStore()
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    created = _create(service)

    assert created.data_content_hash == store.b.content_hash
    assert created.data_content_hash != store.a.content_hash
    assert store.verified_load_calls == 1
    assert store.legacy_load_calls == 0
    assert store.legacy_verify_calls == 0


@pytest.mark.parametrize(
    ("manifest", "exception", "message"),
    (
        (_manifest("wrong-version", data_version=SECOND_DATA_VERSION), ValueError, "version"),
        (object(), TypeError, "SnapshotManifest"),
    ),
    ids=("wrong-version", "wrong-type"),
)
def test_manifest_must_have_the_requested_identity_and_type(
    manifest: object,
    exception: type[Exception],
    message: str,
) -> None:
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=_ScriptedDataStore(manifest),
    )

    with pytest.raises(exception, match=message):
        _create(service)


def test_strategy_registration_is_resolved_against_the_executable_config(
    store_and_root: tuple[SnapshotStore, Path],
) -> None:
    store, _ = store_and_root
    config = ETFRotationConfig(top_n=2)
    parameters: RegisteredParameterSet = ParameterRegistry().register(
        strategy_name="etf-rotation",
        strategy_version=config.version,
        parameter_version="mismatched-v1",
        parameters={"top_n": 3},
        registered_at=AS_OF - timedelta(days=1),
    )
    service = DecisionSnapshotService(
        repository=InMemoryDecisionSnapshotRepository(),
        data_store=store,
    )

    with pytest.raises(ValueError, match="value does not match strategy configuration"):
        _create(
            service,
            strategies=(DecisionStrategyBinding(config=config, parameters=parameters),),
        )
