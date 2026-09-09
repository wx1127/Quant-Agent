"""Exact v1 Agent-tool allow-list and fail-closed runtime-mode policy tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from quant_agent.agent.tools.contracts import ToolCapability, ToolCategory, ToolEffect
from quant_agent.agent.tools.policy import (
    HUMAN_ONLY_TOOL_NAMES,
    MODE_TOOL_PERMISSIONS,
    TOOL_POLICY_VERSION,
    V1_TOOL_POLICIES,
    ToolPolicy,
    allowed_v1_tool_names,
    get_v1_tool_policy,
    is_v1_tool_allowed,
    validate_v1_tool_policy,
)
from quant_agent.config import RuntimeMode

NON_AUTO_MODES = frozenset(
    {
        RuntimeMode.RESEARCH,
        RuntimeMode.BACKTEST,
        RuntimeMode.PAPER,
        RuntimeMode.LIVE_ASSISTED,
    }
)
PAPER_AND_LIVE_MODES = frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED})

RESEARCH_NAMES = frozenset(
    {
        "detect_market_regime",
        "explain_candidate",
        "generate_decision_report",
        "get_market_snapshot",
        "rank_market_themes",
        "rank_stock_candidates",
        "rank_theme_leaders",
        "validate_market_data",
    }
)
ACCOUNT_COMMON_NAMES = frozenset(
    {
        "build_target_portfolio",
        "check_portfolio_risk",
        "create_order_draft",
        "get_order_draft",
        "get_portfolio_snapshot",
        "reconcile_account",
    }
)
EXPECTED_MODE_PERMISSIONS = {
    RuntimeMode.RESEARCH: RESEARCH_NAMES,
    RuntimeMode.BACKTEST: RESEARCH_NAMES | {"run_backtest"},
    RuntimeMode.PAPER: RESEARCH_NAMES | ACCOUNT_COMMON_NAMES | {"submit_paper_orders"},
    RuntimeMode.LIVE_ASSISTED: RESEARCH_NAMES | ACCOUNT_COMMON_NAMES | {"submit_approved_orders"},
    RuntimeMode.LIVE_AUTO: frozenset(),
}
EXPECTED_HUMAN_ONLY_NAMES = frozenset(
    {
        "approve_order_batch",
        "change_risk_limits",
        "change_runtime_mode",
        "deactivate_kill_switch",
        "disable_kill_switch",
        "recover_kill_switch",
    }
)

PolicyRow = tuple[
    ToolCategory,
    ToolCapability,
    ToolEffect,
    frozenset[RuntimeMode],
    bool,
    bool,
    bool,
]

EXPECTED_POLICIES: dict[str, PolicyRow] = {
    "get_market_snapshot": (
        ToolCategory.DATA,
        ToolCapability.MARKET_DATA_READ,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        False,
        False,
    ),
    "validate_market_data": (
        ToolCategory.VALIDATION,
        ToolCapability.MARKET_DATA_VALIDATE,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        False,
        False,
    ),
    "detect_market_regime": (
        ToolCategory.ANALYSIS,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        False,
        False,
    ),
    "rank_market_themes": (
        ToolCategory.MAINLINE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        False,
        False,
    ),
    "rank_theme_leaders": (
        ToolCategory.LEADER,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        True,
        False,
    ),
    "rank_stock_candidates": (
        ToolCategory.CANDIDATE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        True,
        False,
    ),
    "explain_candidate": (
        ToolCategory.CANDIDATE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        True,
        False,
    ),
    "run_backtest": (
        ToolCategory.BACKTEST,
        ToolCapability.BACKTEST_RUN,
        ToolEffect.ISOLATED_COMPUTE,
        frozenset({RuntimeMode.BACKTEST}),
        False,
        False,
        False,
    ),
    "get_portfolio_snapshot": (
        ToolCategory.PORTFOLIO,
        ToolCapability.ACCOUNT_READ,
        ToolEffect.READ_ONLY,
        PAPER_AND_LIVE_MODES,
        False,
        True,
        False,
    ),
    "build_target_portfolio": (
        ToolCategory.PORTFOLIO,
        ToolCapability.PORTFOLIO_BUILD,
        ToolEffect.TARGET_ONLY,
        PAPER_AND_LIVE_MODES,
        False,
        True,
        False,
    ),
    "check_portfolio_risk": (
        ToolCategory.RISK,
        ToolCapability.RISK_CHECK,
        ToolEffect.READ_ONLY,
        PAPER_AND_LIVE_MODES,
        False,
        True,
        False,
    ),
    "create_order_draft": (
        ToolCategory.ORDER_DRAFT,
        ToolCapability.ORDER_DRAFT_CREATE,
        ToolEffect.ARTIFACT_WRITE,
        PAPER_AND_LIVE_MODES,
        True,
        True,
        False,
    ),
    "get_order_draft": (
        ToolCategory.ORDER_DRAFT,
        ToolCapability.ORDER_DRAFT_READ,
        ToolEffect.READ_ONLY,
        PAPER_AND_LIVE_MODES,
        False,
        True,
        False,
    ),
    "submit_paper_orders": (
        ToolCategory.EXECUTION,
        ToolCapability.PAPER_ORDER_SUBMIT,
        ToolEffect.PAPER_EXECUTION_WRITE,
        frozenset({RuntimeMode.PAPER}),
        True,
        True,
        False,
    ),
    "submit_approved_orders": (
        ToolCategory.EXECUTION,
        ToolCapability.APPROVED_LIVE_ORDER_SUBMIT,
        ToolEffect.LIVE_EXTERNAL_WRITE,
        frozenset({RuntimeMode.LIVE_ASSISTED}),
        True,
        True,
        True,
    ),
    "reconcile_account": (
        ToolCategory.RECONCILIATION,
        ToolCapability.ACCOUNT_RECONCILE,
        ToolEffect.READ_ONLY,
        PAPER_AND_LIVE_MODES,
        False,
        True,
        False,
    ),
    "generate_decision_report": (
        ToolCategory.REPORT,
        ToolCapability.REPORT_READ,
        ToolEffect.READ_ONLY,
        NON_AUTO_MODES,
        False,
        False,
        False,
    ),
}


def _policy(**changes: Any) -> ToolPolicy:
    values: dict[str, Any] = {
        "name": "example_tool",
        "category": ToolCategory.DATA,
        "capability": ToolCapability.MARKET_DATA_READ,
        "effect": ToolEffect.READ_ONLY,
        "allowed_modes": frozenset({RuntimeMode.RESEARCH}),
        "requires_idempotency_key": False,
        "account_scoped": False,
        "requires_external_authorization": False,
    }
    values.update(changes)
    return ToolPolicy(**values)


def test_v1_catalog_matches_every_reviewed_policy_field_exactly() -> None:
    assert TOOL_POLICY_VERSION == "agent-tool-policy-v1"
    assert frozenset(V1_TOOL_POLICIES) == frozenset(EXPECTED_POLICIES)
    assert len(V1_TOOL_POLICIES) == 17

    for name, expected in EXPECTED_POLICIES.items():
        policy = V1_TOOL_POLICIES[name]
        actual: PolicyRow = (
            policy.category,
            policy.capability,
            policy.effect,
            policy.allowed_modes,
            policy.requires_idempotency_key,
            policy.account_scoped,
            policy.requires_external_authorization,
        )
        assert policy.name == name
        assert actual == expected

    validate_v1_tool_policy()


def test_complete_five_mode_matrix_is_exact_and_bidirectionally_bound() -> None:
    assert set(MODE_TOOL_PERMISSIONS) == set(RuntimeMode)
    assert set(EXPECTED_MODE_PERMISSIONS) == set(RuntimeMode)
    assert dict(MODE_TOOL_PERMISSIONS) == EXPECTED_MODE_PERMISSIONS

    for mode, expected_names in EXPECTED_MODE_PERMISSIONS.items():
        assert allowed_v1_tool_names(mode) == expected_names
        assert isinstance(allowed_v1_tool_names(mode), frozenset)
        for name, policy in V1_TOOL_POLICIES.items():
            assert is_v1_tool_allowed(name, mode) is (name in expected_names)
            assert (mode in policy.allowed_modes) is (name in expected_names)

    assert frozenset().union(*MODE_TOOL_PERMISSIONS.values()) == frozenset(V1_TOOL_POLICIES)


def test_research_boundary_is_read_only_and_excludes_operational_account_tools() -> None:
    allowed = allowed_v1_tool_names(RuntimeMode.RESEARCH)

    assert allowed == RESEARCH_NAMES
    assert all(V1_TOOL_POLICIES[name].effect is ToolEffect.READ_ONLY for name in allowed)
    assert not (
        {
            "build_target_portfolio",
            "check_portfolio_risk",
            "create_order_draft",
            "get_order_draft",
            "get_portfolio_snapshot",
            "reconcile_account",
            "run_backtest",
            "submit_approved_orders",
            "submit_paper_orders",
        }
        & allowed
    )


def test_account_sized_research_tools_require_the_frozen_decision_account() -> None:
    account_sized = {
        "rank_theme_leaders",
        "rank_stock_candidates",
        "explain_candidate",
    }

    assert all(V1_TOOL_POLICIES[name].account_scoped for name in account_sized)
    assert all(V1_TOOL_POLICIES[name].effect is ToolEffect.READ_ONLY for name in account_sized)
    assert all(
        V1_TOOL_POLICIES[name].capability is ToolCapability.RESEARCH_ANALYSIS
        for name in account_sized
    )


def test_backtest_boundary_adds_only_isolated_backtest_compute() -> None:
    allowed = allowed_v1_tool_names(RuntimeMode.BACKTEST)

    assert allowed - RESEARCH_NAMES == {"run_backtest"}
    assert V1_TOOL_POLICIES["run_backtest"].effect is ToolEffect.ISOLATED_COMPUTE
    assert not {
        "create_order_draft",
        "submit_approved_orders",
        "submit_paper_orders",
    }.intersection(allowed)


def test_paper_and_live_assisted_have_disjoint_submission_authority() -> None:
    paper = allowed_v1_tool_names(RuntimeMode.PAPER)
    live = allowed_v1_tool_names(RuntimeMode.LIVE_ASSISTED)

    assert "submit_paper_orders" in paper
    assert "submit_approved_orders" not in paper
    assert "submit_approved_orders" in live
    assert "submit_paper_orders" not in live
    assert "run_backtest" not in paper | live
    assert paper ^ live == {"submit_paper_orders", "submit_approved_orders"}

    paper_submission = V1_TOOL_POLICIES["submit_paper_orders"]
    live_submission = V1_TOOL_POLICIES["submit_approved_orders"]
    assert paper_submission.requires_idempotency_key is True
    assert paper_submission.requires_external_authorization is False
    assert live_submission.requires_idempotency_key is True
    assert live_submission.requires_external_authorization is True


def test_live_auto_is_a_valid_mode_with_no_agent_authority() -> None:
    assert allowed_v1_tool_names(RuntimeMode.LIVE_AUTO) == frozenset()
    assert MODE_TOOL_PERMISSIONS[RuntimeMode.LIVE_AUTO] == frozenset()
    assert all(not is_v1_tool_allowed(name, RuntimeMode.LIVE_AUTO) for name in V1_TOOL_POLICIES)


def test_every_human_only_control_is_excluded_in_every_mode_and_lookup() -> None:
    assert HUMAN_ONLY_TOOL_NAMES == EXPECTED_HUMAN_ONLY_NAMES
    assert HUMAN_ONLY_TOOL_NAMES.isdisjoint(V1_TOOL_POLICIES)

    for name in HUMAN_ONLY_TOOL_NAMES:
        assert get_v1_tool_policy(name) is None
        for mode in RuntimeMode:
            assert name not in allowed_v1_tool_names(mode)
            assert is_v1_tool_allowed(name, mode) is False


class StringSubclass(str):
    """An untrusted string subtype must not pass exact-name checks."""


@pytest.mark.parametrize(
    "name",
    [
        None,
        1,
        b"get_market_snapshot",
        RuntimeMode.RESEARCH,
        StringSubclass("get_market_snapshot"),
        "GET_MARKET_SNAPSHOT",
        " get_market_snapshot",
        "get_market_snapshot ",
        "unknown_tool",
    ],
)
def test_policy_lookup_fails_closed_for_non_exact_or_unknown_names(name: object) -> None:
    assert get_v1_tool_policy(name) is None
    for mode in RuntimeMode:
        assert is_v1_tool_allowed(name, mode) is False


@pytest.mark.parametrize(
    "mode",
    [None, 1, True, "RESEARCH", b"RESEARCH", object()],
)
def test_mode_lookup_fails_closed_for_non_enum_values(mode: object) -> None:
    assert allowed_v1_tool_names(mode) == frozenset()
    assert is_v1_tool_allowed("get_market_snapshot", mode) is False


def test_exact_registered_policy_lookup_returns_the_immutable_catalog_entry() -> None:
    for name, policy in V1_TOOL_POLICIES.items():
        assert get_v1_tool_policy(name) is policy


def test_policy_catalog_matrix_rows_and_policy_objects_are_immutable() -> None:
    policy = V1_TOOL_POLICIES["get_market_snapshot"]
    names = allowed_v1_tool_names(RuntimeMode.RESEARCH)

    with pytest.raises(FrozenInstanceError):
        policy.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.allowed_modes = frozenset({RuntimeMode.PAPER})  # type: ignore[misc]
    with pytest.raises(TypeError):
        V1_TOOL_POLICIES["changed"] = policy  # type: ignore[index]
    with pytest.raises(TypeError):
        MODE_TOOL_PERMISSIONS[RuntimeMode.RESEARCH] = frozenset()  # type: ignore[index]
    with pytest.raises(AttributeError):
        names.add("changed")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        HUMAN_ONLY_TOOL_NAMES.add("changed")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("name", "Bad Name", ValueError),
        ("name", "a" * 65, ValueError),
        ("category", "DATA", TypeError),
        ("capability", "MARKET_DATA_READ", TypeError),
        ("effect", "READ_ONLY", TypeError),
        ("allowed_modes", {RuntimeMode.RESEARCH}, TypeError),
        ("allowed_modes", frozenset(), TypeError),
        ("allowed_modes", frozenset({"RESEARCH"}), TypeError),
        ("allowed_modes", frozenset({RuntimeMode.LIVE_AUTO}), ValueError),
        ("requires_idempotency_key", 0, TypeError),
        ("requires_idempotency_key", None, TypeError),
        ("account_scoped", 1, TypeError),
        ("account_scoped", None, TypeError),
        ("requires_external_authorization", 1, TypeError),
        ("requires_external_authorization", None, TypeError),
    ],
)
def test_tool_policy_constructor_rejects_untyped_mutable_or_unsafe_values(
    field: str,
    value: object,
    exception: type[BaseException],
) -> None:
    with pytest.raises(exception):
        _policy(**{field: value})


def test_tool_policy_requires_a_nonempty_frozen_mode_set_and_strict_booleans() -> None:
    policy = _policy(
        allowed_modes=frozenset({RuntimeMode.RESEARCH, RuntimeMode.BACKTEST}),
        requires_idempotency_key=True,
        account_scoped=True,
        requires_external_authorization=True,
    )

    assert policy.allowed_modes == frozenset({RuntimeMode.RESEARCH, RuntimeMode.BACKTEST})
    assert policy.requires_idempotency_key is True
    assert policy.account_scoped is True
    assert policy.requires_external_authorization is True
