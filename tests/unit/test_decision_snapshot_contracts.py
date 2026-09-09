"""Decision snapshot identity, immutability, and strict serialization contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.backtest.validation import ParameterRegistry, RegisteredParameterSet
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.portfolio import PortfolioRiskPolicy
from quant_agent.strategies.etf_rotation import ETFRotationConfig
from quant_agent.strategies.mainline_leader import (
    MainlineLeaderConfig,
    MainlineLeaderWeightingMode,
)

AS_OF = datetime(2026, 9, 7, 15, 10, tzinfo=SHANGHAI_TZ)
REGISTERED_AT = AS_OF - timedelta(days=1)
CODE_COMMIT = "1" * 40
CODE_ARTIFACT_HASH = stable_hash({"artifact": "quant-agent-source-tree"})
DATA_CONTENT_HASH = stable_hash({"dataset": "market_20260907_eod_v1"})
ACCOUNT_SNAPSHOT_HASH = stable_hash({"account": "paper-account-1", "as_of": AS_OF})
RISK_POLICY = PortfolioRiskPolicy()


def _registration(
    *,
    strategy_name: str,
    strategy_version: str,
    parameter_version: str,
    parameters: dict[str, str | int | bool | Decimal],
    registered_at: datetime = REGISTERED_AT,
) -> RegisteredParameterSet:
    return ParameterRegistry().register(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        parameter_version=parameter_version,
        parameters=parameters,
        registered_at=registered_at,
    )


def _etf_ref(
    *,
    config: ETFRotationConfig | None = None,
    registered_at: datetime = REGISTERED_AT,
) -> StrategySnapshotRef:
    active = config or ETFRotationConfig()
    registration = _registration(
        strategy_name="etf-rotation",
        strategy_version=active.version,
        parameter_version="etf-balanced-v1",
        parameters={
            "force_downtrend_cash": active.force_downtrend_cash,
            "maximum_instrument_weight": active.maximum_instrument_weight,
            "top_n": active.top_n,
        },
        registered_at=registered_at,
    )
    return StrategySnapshotRef.from_config(config=active, parameters=registration)


def _mainline_ref(
    *,
    config: MainlineLeaderConfig | None = None,
    registered_at: datetime = REGISTERED_AT,
) -> StrategySnapshotRef:
    active = config or MainlineLeaderConfig()
    registration = _registration(
        strategy_name="mainline-leader",
        strategy_version=active.version,
        parameter_version="mainline-balanced-v1",
        parameters={
            "force_downtrend_cash": active.force_downtrend_cash,
            "maximum_positions": active.maximum_positions,
            "minimum_candidate_score": active.minimum_candidate_score,
        },
        registered_at=registered_at,
    )
    return StrategySnapshotRef.from_config(config=active, parameters=registration)


def _snapshot(
    *,
    strategy_refs: tuple[StrategySnapshotRef, ...] | None = None,
    **changes: Any,
) -> DecisionSnapshot:
    values: dict[str, Any] = {
        "decision_id": "dec_20260907_0123456789abcdef0123456789abcdef",
        "mode": RuntimeMode.PAPER,
        "market": "CN_A",
        "as_of": AS_OF,
        "data_version": "market_20260907_eod_v1",
        "data_content_hash": DATA_CONTENT_HASH,
        "strategy_refs": strategy_refs if strategy_refs is not None else (_etf_ref(),),
        "risk_policy_version": RISK_POLICY.version,
        "risk_policy_hash": RISK_POLICY.policy_hash,
        "account_id": "paper-account-1",
        "account_snapshot_id": "paper-account-1:20260907T151000",
        "account_snapshot_hash": ACCOUNT_SNAPSHOT_HASH,
        "code_commit": CODE_COMMIT,
        "code_artifact_hash": CODE_ARTIFACT_HASH,
        "agent_version": "quant-agent-harness-v1",
        "model_version": "model-release-v1",
    }
    values.update(changes)
    return DecisionSnapshot.build(**values)


def _json_payload(snapshot: DecisionSnapshot) -> dict[str, Any]:
    payload = json.loads(snapshot.to_json())
    assert isinstance(payload, dict)
    return payload


def test_strategy_reference_binds_real_config_and_registered_parameters() -> None:
    config = ETFRotationConfig(top_n=2, maximum_instrument_weight=Decimal("0.45"))
    registration = _registration(
        strategy_name="etf-rotation",
        strategy_version=config.version,
        parameter_version="etf-conservative-v1",
        parameters={
            "maximum_instrument_weight": config.maximum_instrument_weight,
            "top_n": config.top_n,
        },
    )

    reference = StrategySnapshotRef.from_config(config=config, parameters=registration)

    assert reference.strategy_name == "etf-rotation"
    assert reference.strategy_version == config.version
    assert reference.config_hash == config.config_hash
    assert reference.parameter_version == registration.parameter_version
    assert reference.parameter_hash == registration.parameter_hash
    assert reference.registered_at == REGISTERED_AT.astimezone(UTC)


@pytest.mark.parametrize(
    ("strategy_name", "strategy_version", "parameters"),
    [
        ("mainline-leader", "etf-rotation-v1", {"top_n": 3}),
        ("etf-rotation", "different-version", {"top_n": 3}),
        ("etf-rotation", "etf-rotation-v1", {"unknown_parameter": 3}),
        ("etf-rotation", "etf-rotation-v1", {"top_n": 2}),
    ],
)
def test_strategy_reference_rejects_registration_identity_or_value_mismatch(
    strategy_name: str,
    strategy_version: str,
    parameters: dict[str, str | int | bool | Decimal],
) -> None:
    config = ETFRotationConfig()
    registration = _registration(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        parameter_version="params-v1",
        parameters=parameters,
    )

    with pytest.raises(ValueError):
        StrategySnapshotRef.from_config(config=config, parameters=registration)


def test_strategy_reference_rejects_bool_integer_aliasing() -> None:
    config = ETFRotationConfig(force_downtrend_cash=True)
    registration = _registration(
        strategy_name="etf-rotation",
        strategy_version=config.version,
        parameter_version="wrong-type-v1",
        parameters={"force_downtrend_cash": 1},
    )

    with pytest.raises(ValueError, match="value does not match"):
        StrategySnapshotRef.from_config(config=config, parameters=registration)


@pytest.mark.parametrize(
    ("parameter_name", "registered_value"),
    [
        ("minimum_weighted_momentum", 0),
        ("top_n", Decimal("3")),
    ],
)
def test_strategy_reference_rejects_integer_decimal_aliasing(
    parameter_name: str, registered_value: int | Decimal
) -> None:
    config = ETFRotationConfig()
    registration = _registration(
        strategy_name="etf-rotation",
        strategy_version=config.version,
        parameter_version="wrong-numeric-type-v1",
        parameters={parameter_name: registered_value},
    )

    with pytest.raises(ValueError, match="value does not match"):
        StrategySnapshotRef.from_config(config=config, parameters=registration)


@pytest.mark.parametrize(
    "registered_value",
    (MainlineLeaderWeightingMode.EQUAL.value, MainlineLeaderWeightingMode.EQUAL),
    ids=("canonical-string", "same-enum"),
)
def test_strategy_reference_compares_strenum_parameters_explicitly(
    registered_value: str,
) -> None:
    config = MainlineLeaderConfig(weighting_mode=MainlineLeaderWeightingMode.EQUAL)
    registration = _registration(
        strategy_name="mainline-leader",
        strategy_version=config.version,
        parameter_version="weighting-v1",
        parameters={"weighting_mode": registered_value},
    )

    reference = StrategySnapshotRef.from_config(config=config, parameters=registration)

    assert reference.config_hash == config.config_hash


def test_strategy_reference_rejects_a_different_strenum_parameter() -> None:
    config = MainlineLeaderConfig(weighting_mode=MainlineLeaderWeightingMode.EQUAL)
    registration = _registration(
        strategy_name="mainline-leader",
        strategy_version=config.version,
        parameter_version="wrong-weighting-v1",
        parameters={"weighting_mode": MainlineLeaderWeightingMode.RISK_ADJUSTED.value},
    )

    with pytest.raises(ValueError, match="value does not match"):
        StrategySnapshotRef.from_config(config=config, parameters=registration)


def test_strategy_reference_rejects_untyped_or_unsupported_sources() -> None:
    config = ETFRotationConfig()
    registration = _registration(
        strategy_name="etf-rotation",
        strategy_version=config.version,
        parameter_version="params-v1",
        parameters={"top_n": config.top_n},
    )

    with pytest.raises(TypeError, match="unsupported strategy"):
        StrategySnapshotRef.from_config(  # type: ignore[arg-type]
            config=object(),
            parameters=registration,
        )
    with pytest.raises(TypeError, match="RegisteredParameterSet"):
        StrategySnapshotRef.from_config(  # type: ignore[arg-type]
            config=config,
            parameters=object(),
        )
    with pytest.raises(ValueError, match="unsupported decision strategy"):
        StrategySnapshotRef(
            strategy_name="unsupported",
            strategy_version="v1",
            config_hash="a" * 64,
            parameter_version="params-v1",
            parameter_hash="b" * 64,
            registered_at=REGISTERED_AT,
        )


def test_single_strategy_identity_projects_reference_fields_exactly() -> None:
    reference = _etf_ref()
    snapshot = _snapshot(strategy_refs=(reference,))

    assert snapshot.strategy_refs == (reference,)
    assert snapshot.strategy_version == reference.strategy_version
    assert snapshot.strategy_config_hash == reference.config_hash
    assert snapshot.parameter_version == reference.parameter_version
    assert snapshot.parameter_hash == reference.parameter_hash


def test_two_strategy_bundle_is_sorted_deterministic_and_hash_typed() -> None:
    etf = _etf_ref()
    mainline = _mainline_ref()

    reversed_input = _snapshot(strategy_refs=(mainline, etf))
    canonical_input = _snapshot(strategy_refs=(etf, mainline))

    assert reversed_input == canonical_input
    assert tuple(item.strategy_name for item in reversed_input.strategy_refs) == (
        "etf-rotation",
        "mainline-leader",
    )
    assert reversed_input.strategy_version.startswith("bundle:")
    assert len(reversed_input.strategy_version.removeprefix("bundle:")) == 64
    assert reversed_input.parameter_version.startswith("bundle:")
    assert len(reversed_input.parameter_version.removeprefix("bundle:")) == 64
    assert len(reversed_input.strategy_config_hash) == 64
    assert len(reversed_input.parameter_hash) == 64

    changed_config = ETFRotationConfig(top_n=2)
    changed = _snapshot(strategy_refs=(_etf_ref(config=changed_config), mainline))
    assert changed.strategy_version != reversed_input.strategy_version
    assert changed.strategy_config_hash != reversed_input.strategy_config_hash
    assert changed.content_hash != reversed_input.content_hash


def test_snapshot_and_nested_references_are_immutable_tuples() -> None:
    snapshot = _snapshot(strategy_refs=(_etf_ref(), _mainline_ref()))

    assert isinstance(snapshot.strategy_refs, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.market = "CN_B"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.strategy_refs[0].parameter_version = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(snapshot, strategy_refs=list(snapshot.strategy_refs))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique and sorted"):
        _snapshot(strategy_refs=(_etf_ref(), _etf_ref()))


def test_canonical_json_round_trip_preserves_every_field_and_hash() -> None:
    snapshot = _snapshot(strategy_refs=(_etf_ref(), _mainline_ref()))

    encoded = snapshot.to_json()
    decoded = DecisionSnapshot.from_json(encoded)

    assert decoded == snapshot
    assert decoded.to_json() == encoded
    assert set(_json_payload(snapshot)) == {
        "account_id",
        "account_snapshot_hash",
        "account_snapshot_id",
        "agent_version",
        "as_of",
        "code_artifact_hash",
        "code_commit",
        "content_hash",
        "data_content_hash",
        "data_version",
        "decision_id",
        "market",
        "mode",
        "model_version",
        "parameter_hash",
        "parameter_version",
        "risk_policy_hash",
        "risk_policy_version",
        "schema_version",
        "strategy_config_hash",
        "strategy_refs",
        "strategy_version",
    }


@pytest.mark.parametrize("missing", ["decision_id", "model_version", "content_hash"])
def test_json_rejects_missing_root_fields(missing: str) -> None:
    payload = _json_payload(_snapshot())
    payload.pop(missing)

    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(payload))


def test_json_rejects_extra_root_and_nested_fields() -> None:
    root_extra = _json_payload(_snapshot())
    root_extra["unexpected"] = "not-hashed"
    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(root_extra))

    nested_extra = _json_payload(_snapshot())
    nested_extra["strategy_refs"][0]["unexpected"] = "not-hashed"
    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(nested_extra))

    nested_missing = _json_payload(_snapshot())
    nested_missing["strategy_refs"][0].pop("parameter_hash")
    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(nested_missing))


def test_json_rejects_duplicate_keys_and_type_coercion() -> None:
    encoded = _snapshot().to_json()
    duplicate = encoded[:-1] + ',"decision_id":"dec_duplicate"}'
    with pytest.raises(ValueError, match="duplicate"):
        DecisionSnapshot.from_json(duplicate)

    coerced = _json_payload(_snapshot())
    coerced["agent_version"] = 1
    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(coerced))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "2"),
        ("decision_id", "dec_20260907_ffffffffffffffffffffffffffffffff"),
        ("mode", "BACKTEST"),
        ("market", "CN_B"),
        ("as_of", "2026-09-07T07:11:00+00:00"),
        ("data_version", "market_20260907_eod_v2"),
        ("data_content_hash", "2" * 64),
        ("strategy_version", "different-strategy-v1"),
        ("strategy_config_hash", "3" * 64),
        ("parameter_version", "different-parameters-v1"),
        ("parameter_hash", "4" * 64),
        ("risk_policy_version", "different-risk-v1"),
        ("risk_policy_hash", "5" * 64),
        ("account_id", "different-account"),
        ("account_snapshot_id", "different-account-snapshot"),
        ("account_snapshot_hash", "6" * 64),
        ("code_commit", "7" * 40),
        ("code_artifact_hash", "8" * 64),
        ("agent_version", "different-agent-v1"),
        ("model_version", "different-model-v1"),
        ("content_hash", "9" * 64),
    ],
)
def test_content_hash_or_identity_validation_rejects_each_changed_root_field(
    field: str, replacement: str
) -> None:
    payload = _json_payload(_snapshot())
    payload[field] = replacement

    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("strategy_name", "mainline-leader"),
        ("strategy_version", "different-strategy-v1"),
        ("config_hash", "a" * 64),
        ("parameter_version", "different-parameters-v1"),
        ("parameter_hash", "b" * 64),
        ("registered_at", "2026-09-05T07:10:00+00:00"),
    ],
)
def test_content_hash_or_identity_validation_rejects_changed_strategy_reference(
    field: str, replacement: str
) -> None:
    payload = _json_payload(_snapshot())
    payload["strategy_refs"][0][field] = replacement

    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(payload))


def test_naive_decision_and_registration_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _snapshot(as_of=AS_OF.replace(tzinfo=None))

    reference = _etf_ref()
    with pytest.raises(ValueError, match="timezone"):
        replace(reference, registered_at=REGISTERED_AT.replace(tzinfo=None))

    payload = _json_payload(_snapshot())
    payload["as_of"] = "2026-09-07T15:10:00"
    with pytest.raises(ValueError):
        DecisionSnapshot.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field",
    [
        "decision_id",
        "data_version",
        "strategy_version",
        "parameter_version",
        "risk_policy_version",
        "account_id",
        "account_snapshot_id",
        "agent_version",
        "model_version",
    ],
)
def test_empty_or_whitespace_snapshot_identity_fields_are_rejected(field: str) -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="non-empty"):
        replace(snapshot, **{field: " "})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["strategy_name", "strategy_version", "parameter_version"])
def test_empty_strategy_identity_fields_are_rejected(field: str) -> None:
    reference = _etf_ref()

    with pytest.raises(ValueError, match="non-empty"):
        replace(reference, **{field: ""})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "data_content_hash",
        "strategy_config_hash",
        "parameter_hash",
        "risk_policy_hash",
        "account_snapshot_hash",
        "code_artifact_hash",
        "content_hash",
    ],
)
def test_invalid_snapshot_hash_fields_are_rejected(field: str) -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="SHA-256"):
        replace(snapshot, **{field: "A" * 64})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["config_hash", "parameter_hash"])
def test_invalid_strategy_reference_hash_fields_are_rejected(field: str) -> None:
    reference = _etf_ref()

    with pytest.raises(ValueError, match="SHA-256"):
        replace(reference, **{field: "a" * 63})  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", [RuntimeMode.LIVE_AUTO, "PAPER", None])
def test_live_auto_and_untyped_modes_are_rejected(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        _snapshot(mode=mode)


@pytest.mark.parametrize("market", ["CN_B", "cn_a", "CN_A ", ""])
def test_only_exact_cn_a_market_is_accepted(market: str) -> None:
    with pytest.raises(ValueError, match="CN_A"):
        _snapshot(market=market)


def test_parameters_registered_after_decision_boundary_are_rejected() -> None:
    future = _etf_ref(registered_at=AS_OF + timedelta(microseconds=1))

    with pytest.raises(ValueError, match="not registered"):
        _snapshot(strategy_refs=(future,))


@pytest.mark.parametrize(
    "commit",
    [
        "1" * 39,
        "1" * 41,
        "A" * 40,
        "git-sha",
        "",
    ],
)
def test_code_commit_requires_full_lowercase_sha1_or_sha256(commit: str) -> None:
    with pytest.raises(ValueError, match="Git commit"):
        _snapshot(code_commit=commit)


def test_code_commit_and_artifact_hash_are_independent_content_identities() -> None:
    baseline = _snapshot()
    sha256_commit = _snapshot(code_commit="2" * 64)
    changed_artifact = _snapshot(code_artifact_hash="3" * 64)

    assert sha256_commit.code_commit == "2" * 64
    assert sha256_commit.content_hash != baseline.content_hash
    assert changed_artifact.code_commit == baseline.code_commit
    assert changed_artifact.code_artifact_hash != baseline.code_artifact_hash
    assert changed_artifact.content_hash != baseline.content_hash


def test_same_instants_in_different_timezones_have_identical_identity_and_json() -> None:
    shanghai = _snapshot(
        as_of=AS_OF,
        strategy_refs=(_etf_ref(registered_at=REGISTERED_AT),),
    )
    utc = _snapshot(
        as_of=AS_OF.astimezone(UTC),
        strategy_refs=(_etf_ref(registered_at=REGISTERED_AT.astimezone(UTC)),),
    )

    assert shanghai == utc
    assert shanghai.as_of.tzinfo is UTC
    assert shanghai.strategy_refs[0].registered_at.tzinfo is UTC
    assert shanghai.content_hash == utc.content_hash
    assert shanghai.to_json() == utc.to_json()
