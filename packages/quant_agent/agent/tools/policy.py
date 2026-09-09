"""Fixed v1 allow-list and fail-closed mode policy for Agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quant_agent.agent.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolEffect,
    validate_tool_name,
)
from quant_agent.config import RuntimeMode

TOOL_POLICY_VERSION = "agent-tool-policy-v1"

_V1_RUNTIME_MODES = frozenset(
    {
        RuntimeMode.RESEARCH,
        RuntimeMode.BACKTEST,
        RuntimeMode.PAPER,
        RuntimeMode.LIVE_ASSISTED,
        RuntimeMode.LIVE_AUTO,
    }
)
_NON_AUTO_MODES = frozenset(
    {
        RuntimeMode.RESEARCH,
        RuntimeMode.BACKTEST,
        RuntimeMode.PAPER,
        RuntimeMode.LIVE_ASSISTED,
    }
)
_PAPER_AND_LIVE_MODES = frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED})
_WRITE_EFFECTS = frozenset(
    {
        ToolEffect.ARTIFACT_WRITE,
        ToolEffect.PAPER_EXECUTION_WRITE,
        ToolEffect.LIVE_EXTERNAL_WRITE,
    }
)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Immutable maximum authority for one explicitly approved v1 tool name."""

    name: str
    category: ToolCategory
    capability: ToolCapability
    effect: ToolEffect
    allowed_modes: frozenset[RuntimeMode]
    requires_idempotency_key: bool
    account_scoped: bool
    requires_external_authorization: bool

    def __post_init__(self) -> None:
        validate_tool_name(self.name)
        if not isinstance(self.category, ToolCategory):
            raise TypeError("tool policy category must be a ToolCategory")
        if not isinstance(self.capability, ToolCapability):
            raise TypeError("tool policy capability must be a ToolCapability")
        if not isinstance(self.effect, ToolEffect):
            raise TypeError("tool policy effect must be a ToolEffect")
        if not isinstance(self.allowed_modes, frozenset) or not self.allowed_modes:
            raise TypeError("tool policy allowed_modes must be a non-empty frozenset")
        if any(not isinstance(mode, RuntimeMode) for mode in self.allowed_modes):
            raise TypeError("tool policy allowed_modes must contain only RuntimeMode values")
        if RuntimeMode.LIVE_AUTO in self.allowed_modes:
            raise ValueError("LIVE_AUTO cannot be allowed by an Agent tool policy")
        for field_name in (
            "requires_idempotency_key",
            "account_scoped",
            "requires_external_authorization",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"tool policy {field_name} must be a bool")


HUMAN_ONLY_TOOL_NAMES = frozenset(
    {
        "approve_order_batch",
        "change_risk_limits",
        "change_runtime_mode",
        "deactivate_kill_switch",
        "disable_kill_switch",
        "recover_kill_switch",
    }
)
_EXPECTED_V1_TOOL_NAMES = frozenset(
    {
        "build_target_portfolio",
        "check_portfolio_risk",
        "create_order_draft",
        "detect_market_regime",
        "explain_candidate",
        "generate_decision_report",
        "get_market_snapshot",
        "get_order_draft",
        "get_portfolio_snapshot",
        "rank_market_themes",
        "rank_stock_candidates",
        "rank_theme_leaders",
        "reconcile_account",
        "run_backtest",
        "submit_approved_orders",
        "submit_paper_orders",
        "validate_market_data",
    }
)

_CATEGORY_CAPABILITIES: Mapping[ToolCategory, frozenset[ToolCapability]] = MappingProxyType(
    {
        ToolCategory.DATA: frozenset({ToolCapability.MARKET_DATA_READ}),
        ToolCategory.VALIDATION: frozenset({ToolCapability.MARKET_DATA_VALIDATE}),
        ToolCategory.ANALYSIS: frozenset({ToolCapability.RESEARCH_ANALYSIS}),
        ToolCategory.MAINLINE: frozenset({ToolCapability.RESEARCH_ANALYSIS}),
        ToolCategory.LEADER: frozenset({ToolCapability.RESEARCH_ANALYSIS}),
        ToolCategory.CANDIDATE: frozenset({ToolCapability.RESEARCH_ANALYSIS}),
        ToolCategory.BACKTEST: frozenset({ToolCapability.BACKTEST_RUN}),
        ToolCategory.PORTFOLIO: frozenset(
            {ToolCapability.ACCOUNT_READ, ToolCapability.PORTFOLIO_BUILD}
        ),
        ToolCategory.RISK: frozenset({ToolCapability.RISK_CHECK}),
        ToolCategory.ORDER_DRAFT: frozenset(
            {ToolCapability.ORDER_DRAFT_READ, ToolCapability.ORDER_DRAFT_CREATE}
        ),
        ToolCategory.EXECUTION: frozenset(
            {
                ToolCapability.PAPER_ORDER_SUBMIT,
                ToolCapability.APPROVED_LIVE_ORDER_SUBMIT,
            }
        ),
        ToolCategory.RECONCILIATION: frozenset({ToolCapability.ACCOUNT_RECONCILE}),
        ToolCategory.REPORT: frozenset({ToolCapability.REPORT_READ}),
    }
)

_CAPABILITY_EFFECTS: Mapping[ToolCapability, ToolEffect] = MappingProxyType(
    {
        ToolCapability.MARKET_DATA_READ: ToolEffect.READ_ONLY,
        ToolCapability.MARKET_DATA_VALIDATE: ToolEffect.READ_ONLY,
        ToolCapability.RESEARCH_ANALYSIS: ToolEffect.READ_ONLY,
        ToolCapability.BACKTEST_RUN: ToolEffect.ISOLATED_COMPUTE,
        ToolCapability.ACCOUNT_READ: ToolEffect.READ_ONLY,
        ToolCapability.PORTFOLIO_BUILD: ToolEffect.TARGET_ONLY,
        ToolCapability.RISK_CHECK: ToolEffect.READ_ONLY,
        ToolCapability.ORDER_DRAFT_READ: ToolEffect.READ_ONLY,
        ToolCapability.ORDER_DRAFT_CREATE: ToolEffect.ARTIFACT_WRITE,
        ToolCapability.PAPER_ORDER_SUBMIT: ToolEffect.PAPER_EXECUTION_WRITE,
        ToolCapability.APPROVED_LIVE_ORDER_SUBMIT: ToolEffect.LIVE_EXTERNAL_WRITE,
        ToolCapability.ACCOUNT_RECONCILE: ToolEffect.READ_ONLY,
        ToolCapability.REPORT_READ: ToolEffect.READ_ONLY,
    }
)

_CAPABILITY_MODE_CEILINGS: Mapping[ToolCapability, frozenset[RuntimeMode]] = MappingProxyType(
    {
        ToolCapability.MARKET_DATA_READ: _NON_AUTO_MODES,
        ToolCapability.MARKET_DATA_VALIDATE: _NON_AUTO_MODES,
        ToolCapability.RESEARCH_ANALYSIS: _NON_AUTO_MODES,
        ToolCapability.BACKTEST_RUN: frozenset({RuntimeMode.BACKTEST}),
        ToolCapability.ACCOUNT_READ: _PAPER_AND_LIVE_MODES,
        ToolCapability.PORTFOLIO_BUILD: _PAPER_AND_LIVE_MODES,
        ToolCapability.RISK_CHECK: _PAPER_AND_LIVE_MODES,
        ToolCapability.ORDER_DRAFT_READ: _PAPER_AND_LIVE_MODES,
        ToolCapability.ORDER_DRAFT_CREATE: _PAPER_AND_LIVE_MODES,
        ToolCapability.PAPER_ORDER_SUBMIT: frozenset({RuntimeMode.PAPER}),
        ToolCapability.APPROVED_LIVE_ORDER_SUBMIT: frozenset({RuntimeMode.LIVE_ASSISTED}),
        ToolCapability.ACCOUNT_RECONCILE: _PAPER_AND_LIVE_MODES,
        ToolCapability.REPORT_READ: _NON_AUTO_MODES,
    }
)

_ACCOUNT_SCOPED_CAPABILITIES = frozenset(
    {
        ToolCapability.ACCOUNT_READ,
        ToolCapability.PORTFOLIO_BUILD,
        ToolCapability.RISK_CHECK,
        ToolCapability.ORDER_DRAFT_READ,
        ToolCapability.ORDER_DRAFT_CREATE,
        ToolCapability.PAPER_ORDER_SUBMIT,
        ToolCapability.APPROVED_LIVE_ORDER_SUBMIT,
        ToolCapability.ACCOUNT_RECONCILE,
    }
)
_ACCOUNT_SCOPED_RESEARCH_TOOLS = frozenset(
    {
        "rank_theme_leaders",
        "rank_stock_candidates",
        "explain_candidate",
    }
)


def _policy(
    name: str,
    category: ToolCategory,
    capability: ToolCapability,
    effect: ToolEffect,
    allowed_modes: frozenset[RuntimeMode],
    *,
    requires_idempotency_key: bool = False,
    account_scoped: bool = False,
    requires_external_authorization: bool = False,
) -> ToolPolicy:
    return ToolPolicy(
        name=name,
        category=category,
        capability=capability,
        effect=effect,
        allowed_modes=allowed_modes,
        requires_idempotency_key=requires_idempotency_key,
        account_scoped=account_scoped,
        requires_external_authorization=requires_external_authorization,
    )


_POLICIES = (
    _policy(
        "get_market_snapshot",
        ToolCategory.DATA,
        ToolCapability.MARKET_DATA_READ,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
    ),
    _policy(
        "validate_market_data",
        ToolCategory.VALIDATION,
        ToolCapability.MARKET_DATA_VALIDATE,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
    ),
    _policy(
        "detect_market_regime",
        ToolCategory.ANALYSIS,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
    ),
    _policy(
        "rank_market_themes",
        ToolCategory.MAINLINE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
    ),
    _policy(
        "rank_theme_leaders",
        ToolCategory.LEADER,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
        account_scoped=True,
    ),
    _policy(
        "rank_stock_candidates",
        ToolCategory.CANDIDATE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
        account_scoped=True,
    ),
    _policy(
        "explain_candidate",
        ToolCategory.CANDIDATE,
        ToolCapability.RESEARCH_ANALYSIS,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
        account_scoped=True,
    ),
    _policy(
        "run_backtest",
        ToolCategory.BACKTEST,
        ToolCapability.BACKTEST_RUN,
        ToolEffect.ISOLATED_COMPUTE,
        frozenset({RuntimeMode.BACKTEST}),
    ),
    _policy(
        "get_portfolio_snapshot",
        ToolCategory.PORTFOLIO,
        ToolCapability.ACCOUNT_READ,
        ToolEffect.READ_ONLY,
        _PAPER_AND_LIVE_MODES,
        account_scoped=True,
    ),
    _policy(
        "build_target_portfolio",
        ToolCategory.PORTFOLIO,
        ToolCapability.PORTFOLIO_BUILD,
        ToolEffect.TARGET_ONLY,
        _PAPER_AND_LIVE_MODES,
        account_scoped=True,
    ),
    _policy(
        "check_portfolio_risk",
        ToolCategory.RISK,
        ToolCapability.RISK_CHECK,
        ToolEffect.READ_ONLY,
        _PAPER_AND_LIVE_MODES,
        account_scoped=True,
    ),
    _policy(
        "create_order_draft",
        ToolCategory.ORDER_DRAFT,
        ToolCapability.ORDER_DRAFT_CREATE,
        ToolEffect.ARTIFACT_WRITE,
        _PAPER_AND_LIVE_MODES,
        requires_idempotency_key=True,
        account_scoped=True,
    ),
    _policy(
        "get_order_draft",
        ToolCategory.ORDER_DRAFT,
        ToolCapability.ORDER_DRAFT_READ,
        ToolEffect.READ_ONLY,
        _PAPER_AND_LIVE_MODES,
        account_scoped=True,
    ),
    _policy(
        "submit_paper_orders",
        ToolCategory.EXECUTION,
        ToolCapability.PAPER_ORDER_SUBMIT,
        ToolEffect.PAPER_EXECUTION_WRITE,
        frozenset({RuntimeMode.PAPER}),
        requires_idempotency_key=True,
        account_scoped=True,
    ),
    _policy(
        "submit_approved_orders",
        ToolCategory.EXECUTION,
        ToolCapability.APPROVED_LIVE_ORDER_SUBMIT,
        ToolEffect.LIVE_EXTERNAL_WRITE,
        frozenset({RuntimeMode.LIVE_ASSISTED}),
        requires_idempotency_key=True,
        account_scoped=True,
        requires_external_authorization=True,
    ),
    _policy(
        "reconcile_account",
        ToolCategory.RECONCILIATION,
        ToolCapability.ACCOUNT_RECONCILE,
        ToolEffect.READ_ONLY,
        _PAPER_AND_LIVE_MODES,
        account_scoped=True,
    ),
    _policy(
        "generate_decision_report",
        ToolCategory.REPORT,
        ToolCapability.REPORT_READ,
        ToolEffect.READ_ONLY,
        _NON_AUTO_MODES,
    ),
)

V1_TOOL_POLICIES: Mapping[str, ToolPolicy] = MappingProxyType(
    {policy.name: policy for policy in _POLICIES}
)

MODE_TOOL_PERMISSIONS: Mapping[RuntimeMode, frozenset[str]] = MappingProxyType(
    {
        mode: frozenset(policy.name for policy in _POLICIES if mode in policy.allowed_modes)
        for mode in RuntimeMode
    }
)


def get_v1_tool_policy(name: object) -> ToolPolicy | None:
    """Return policy only for one exact plain-string v1 name; otherwise deny."""

    if type(name) is not str or name in HUMAN_ONLY_TOOL_NAMES:
        return None
    return V1_TOOL_POLICIES.get(name)


def allowed_v1_tool_names(mode: object) -> frozenset[str]:
    """Return a frozen allow-list, defaulting to empty for any untrusted mode value."""

    if not isinstance(mode, RuntimeMode):
        return frozenset()
    return MODE_TOOL_PERMISSIONS.get(mode, frozenset())


def is_v1_tool_allowed(name: object, mode: object) -> bool:
    """Apply exact-name and exact-enum checks without normalizing untrusted input."""

    if type(name) is not str or not isinstance(mode, RuntimeMode):
        return False
    return name in allowed_v1_tool_names(mode) and name in V1_TOOL_POLICIES


def validate_v1_tool_policy() -> None:
    """Fail import/startup when the fixed catalog or its complete matrix drifts."""

    if not TOOL_POLICY_VERSION or TOOL_POLICY_VERSION.strip() != TOOL_POLICY_VERSION:
        raise RuntimeError("tool policy version must be one exact non-empty identifier")
    if frozenset(RuntimeMode) != _V1_RUNTIME_MODES:
        raise RuntimeError("RuntimeMode changed without a reviewed v1 tool policy update")
    if set(MODE_TOOL_PERMISSIONS) != set(RuntimeMode):
        raise RuntimeError("mode tool permissions must cover every RuntimeMode exactly")
    if MODE_TOOL_PERMISSIONS[RuntimeMode.LIVE_AUTO]:
        raise RuntimeError("LIVE_AUTO must have an empty Agent tool allow-list")
    if len(V1_TOOL_POLICIES) != len(_POLICIES):
        raise RuntimeError("v1 Agent tool policy contains duplicate names")
    if frozenset(V1_TOOL_POLICIES) != _EXPECTED_V1_TOOL_NAMES:
        raise RuntimeError("v1 Agent tool allow-list changed without a policy review")
    if not HUMAN_ONLY_TOOL_NAMES:
        raise RuntimeError("human-only Agent exclusion list cannot be empty")
    for name in HUMAN_ONLY_TOOL_NAMES:
        validate_tool_name(name)
    if HUMAN_ONLY_TOOL_NAMES.intersection(V1_TOOL_POLICIES):
        raise RuntimeError("human-only tools cannot appear in the Agent allow-list")

    matrix_names = frozenset().union(*MODE_TOOL_PERMISSIONS.values())
    if matrix_names != frozenset(V1_TOOL_POLICIES):
        raise RuntimeError("mode matrix and v1 Agent tool catalog do not bind exactly")
    if set(_CATEGORY_CAPABILITIES) != set(ToolCategory):
        raise RuntimeError("category capability policy must cover every ToolCategory")
    if set(_CAPABILITY_EFFECTS) != set(ToolCapability):
        raise RuntimeError("capability effect policy must cover every ToolCapability")
    if set(_CAPABILITY_MODE_CEILINGS) != set(ToolCapability):
        raise RuntimeError("capability mode policy must cover every ToolCapability")

    for name, policy in V1_TOOL_POLICIES.items():
        if name != policy.name:
            raise RuntimeError("tool policy key must equal its exact name")
        if policy.capability not in _CATEGORY_CAPABILITIES[policy.category]:
            raise RuntimeError(f"tool category cannot grant capability: {name}")
        if policy.effect is not _CAPABILITY_EFFECTS[policy.capability]:
            raise RuntimeError(f"tool capability cannot use this effect: {name}")
        if policy.allowed_modes != _CAPABILITY_MODE_CEILINGS[policy.capability]:
            raise RuntimeError(f"tool modes exceed or drift from capability ceiling: {name}")
        if policy.requires_idempotency_key is not (policy.effect in _WRITE_EFFECTS):
            raise RuntimeError(f"tool idempotency policy does not match its effect: {name}")
        expected_account_scope = (
            policy.capability in _ACCOUNT_SCOPED_CAPABILITIES
            or name in _ACCOUNT_SCOPED_RESEARCH_TOOLS
        )
        if policy.account_scoped is not expected_account_scope:
            raise RuntimeError(f"tool account scope does not match its capability: {name}")
        if policy.requires_external_authorization is not (
            policy.capability is ToolCapability.APPROVED_LIVE_ORDER_SUBMIT
        ):
            raise RuntimeError(f"tool external authorization policy is unsafe: {name}")
        matrix_modes = frozenset(
            mode for mode, names in MODE_TOOL_PERMISSIONS.items() if name in names
        )
        if matrix_modes != policy.allowed_modes:
            raise RuntimeError(f"tool policy modes do not match the mode matrix: {name}")

    forbidden_in_research = {
        "create_order_draft",
        "submit_paper_orders",
        "submit_approved_orders",
    }
    if forbidden_in_research.intersection(MODE_TOOL_PERMISSIONS[RuntimeMode.RESEARCH]):
        raise RuntimeError("RESEARCH cannot create or submit orders")
    if {
        "submit_paper_orders",
        "submit_approved_orders",
    }.intersection(MODE_TOOL_PERMISSIONS[RuntimeMode.BACKTEST]):
        raise RuntimeError("BACKTEST cannot submit operational orders")


validate_v1_tool_policy()

__all__ = [
    "HUMAN_ONLY_TOOL_NAMES",
    "MODE_TOOL_PERMISSIONS",
    "TOOL_POLICY_VERSION",
    "V1_TOOL_POLICIES",
    "ToolPolicy",
    "allowed_v1_tool_names",
    "get_v1_tool_policy",
    "is_v1_tool_allowed",
    "validate_v1_tool_policy",
]
