"""Focused service tests for the state-gated Agent runtime facade."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier, Lock
from time import sleep
from typing import cast

import pytest

from quant_agent.agent.runtime.contracts import (
    AgentArtifactKind,
    AgentArtifactRef,
    AgentExternalApproval,
    AgentPendingToolCall,
    AgentRunEventKind,
    AgentRunGoal,
    AgentRunLimits,
    AgentRunSnapshot,
    AgentRunState,
    AgentRunTrigger,
)
from quant_agent.agent.runtime.replay import replay_agent_run
from quant_agent.agent.runtime.repository import InMemoryAgentRunRepository
from quant_agent.agent.runtime.rules import AgentGoalModeMismatch
from quant_agent.agent.runtime.service import (
    AgentApprovalInvalid,
    AgentRequestConflict,
    AgentRunTerminal,
    AgentRuntime,
    AgentRuntimeClockError,
    AgentRuntimeOutputInvalid,
    AgentToolCallLimitExceeded,
    AgentToolStateDenied,
)
from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools.contracts import (
    ToolArgumentsRejected,
    ToolAuditUnavailable,
    ToolCallContext,
    ToolDenialReason,
    ToolDescriptor,
    ToolEffect,
    ToolExecutionFailed,
    ToolOutputRejected,
    ToolPermissionDenied,
)
from quant_agent.agent.tools.policy import V1_TOOL_POLICIES
from quant_agent.agent.tools.portfolio_execution.contracts import (
    CashReconciliationOutput,
    OrderDraftOutput,
    PaperAccountStateOutput,
    PaperExecutionOutput,
    ReconciliationOutput,
    ReconciliationStopSignalOutput,
)
from quant_agent.agent.tools.research.contracts import MarketSnapshotOutput
from quant_agent.config import RuntimeMode
from quant_agent.core import Provenance, ToolResponse
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import ToolIssue
from quant_agent.execution.order_drafts import DraftFundingPolicy
from quant_agent.reconciliation import (
    ReconciliationCheckStatus,
    ReconciliationSeverity,
    ReconciliationStatus,
    ReconciliationStopAction,
    ReconciliationStopScope,
)
from quant_agent.risk import RiskCheckStatus

AS_OF = datetime(2026, 9, 8, 7, 0, tzinfo=UTC)
NOW = AS_OF + timedelta(minutes=1)
DECISION_ID = "dec_runtime_0123456789abcdef0123456789abcdef"
ACCOUNT_ID = "paper-account-1"
BATCH_HASH = "b" * 64


@dataclass
class _Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


@dataclass
class _SequenceClock:
    values: tuple[datetime, ...]
    call_count: int = 0

    def __call__(self) -> datetime:
        value = self.values[min(self.call_count, len(self.values) - 1)]
        self.call_count += 1
        return value


class _RecordingRegistry:
    def __init__(
        self,
        *,
        mode: RuntimeMode,
        decision: DecisionSnapshot,
        response_delay: float = 0,
    ) -> None:
        self._mode = mode
        self._decision = decision
        self._response_delay = response_delay
        self._lock = Lock()
        self.catalog_contexts: list[ToolCallContext] = []
        self.invoke_calls: list[tuple[str, object, ToolCallContext]] = []
        self.invoke_error: Exception | None = None
        self.response_error_code: ErrorCode | None = None
        self.response_error_with_data = False
        self.on_invoke: Callable[[], None] | None = None
        self.response_factory: Callable[[str, ToolCallContext], object] | None = None
        self._descriptors = tuple(
            ToolDescriptor(
                name=policy.name,
                version="agent-runtime-test-v1",
                description=f"Test descriptor for {policy.name}",
                category=policy.category,
                capability=policy.capability,
                effect=policy.effect,
                argument_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            for policy in V1_TOOL_POLICIES.values()
        )

    @property
    def runtime_mode(self) -> RuntimeMode:
        return self._mode

    def catalog(self, context: ToolCallContext) -> tuple[ToolDescriptor, ...]:
        self.catalog_contexts.append(context)
        return self._descriptors

    def invoke(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, object],
        context: ToolCallContext,
    ) -> object:
        with self._lock:
            self.invoke_calls.append((tool_name, raw_arguments, context))
        if self._response_delay:
            sleep(self._response_delay)
        if self.on_invoke is not None:
            self.on_invoke()
        if self.invoke_error is not None:
            raise self.invoke_error
        if self.response_factory is not None:
            return self.response_factory(tool_name, context)
        if self.response_error_code is not None:
            error_data = (
                MarketSnapshotOutput(
                    decision_id=self._decision.decision_id,
                    as_of=self._decision.as_of,
                    data_version=self._decision.data_version,
                    market="CN_A",
                    data_content_hash=self._decision.data_content_hash,
                    created_at=self._decision.as_of.isoformat(),
                    total_file_count=1,
                    returned_file_count=0,
                    total_row_count=0,
                    files_truncated=True,
                    files=(),
                )
                if self.response_error_with_data
                else None
            )
            return ToolResponse[object](
                ok=False,
                request_id=context.request_id,
                decision_id=context.decision_id,
                as_of=self._decision.as_of,
                data=error_data,
                errors=(
                    ToolIssue(
                        code=self.response_error_code,
                        message=f"test failure: {self.response_error_code.value}",
                    ),
                ),
                provenance=Provenance(
                    service="agent-runtime-test",
                    version="1.0.0",
                    data_version=self._decision.data_version,
                ),
            )
        if tool_name != "get_market_snapshot":
            raise AssertionError(f"unexpected test registry invocation: {tool_name}")
        data = MarketSnapshotOutput(
            decision_id=self._decision.decision_id,
            as_of=self._decision.as_of,
            data_version=self._decision.data_version,
            market="CN_A",
            data_content_hash=self._decision.data_content_hash,
            created_at=self._decision.as_of.isoformat(),
            total_file_count=1,
            returned_file_count=0,
            total_row_count=0,
            files_truncated=True,
            files=(),
        )
        return ToolResponse[MarketSnapshotOutput](
            ok=True,
            request_id=context.request_id,
            decision_id=context.decision_id,
            as_of=self._decision.as_of,
            data=data,
            provenance=Provenance(
                service="agent-runtime-test",
                version="1.0.0",
                data_version=self._decision.data_version,
            ),
        )


class _IncidentHandler:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, str, datetime]] = []

    def handle_verified_reconciliation(
        self,
        *,
        decision_id: str,
        reconciliation_id: str,
        result_hash: str,
        requested_at: datetime,
    ) -> str:
        self.calls.append((decision_id, reconciliation_id, result_hash, requested_at))
        if self.fail:
            raise RuntimeError("test incident adapter unavailable")
        return "kill-switch-activation-1"


@dataclass(frozen=True)
class _IncidentResult:
    reconciliation_id: str
    result_hash: str


def _decision(mode: RuntimeMode) -> DecisionSnapshot:
    reference = StrategySnapshotRef(
        strategy_name="etf-rotation",
        strategy_version="etf-rotation-v1",
        config_hash="1" * 64,
        parameter_version="params-v1",
        parameter_hash="2" * 64,
        registered_at=AS_OF - timedelta(days=1),
    )
    return DecisionSnapshot.build(
        decision_id=DECISION_ID,
        mode=mode,
        market="CN_A",
        as_of=AS_OF,
        data_version="market_20260908_eod_v1",
        data_content_hash="3" * 64,
        strategy_refs=(reference,),
        risk_policy_version="portfolio-risk-v1",
        risk_policy_hash="4" * 64,
        account_id=ACCOUNT_ID,
        account_snapshot_id=f"{ACCOUNT_ID}:20260908T070000",
        account_snapshot_hash="5" * 64,
        code_commit="6" * 40,
        code_artifact_hash="7" * 64,
        agent_version="quant-agent-harness-v1",
        model_version="model-release-v1",
    )


def _order_draft_response(
    decision: DecisionSnapshot,
    context: ToolCallContext,
) -> ToolResponse[OrderDraftOutput]:
    data = OrderDraftOutput(
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data_version=decision.data_version,
        account_id=decision.account_id,
        account_snapshot_id=decision.account_snapshot_id,
        account_snapshot_hash=decision.account_snapshot_hash,
        draft_as_of=NOW,
        trading_day=NOW.date(),
        expires_at=NOW + timedelta(minutes=30),
        validity_seconds=1_800,
        runtime_mode=decision.mode,
        currency="CNY",
        portfolio_proposal_hash="7" * 64,
        risk_request_hash="a" * 64,
        risk_result_hash="8" * 64,
        risk_status=RiskCheckStatus.PASS,
        risk_checked_at=NOW,
        risk_engine_version="portfolio-risk-v1",
        risk_policy_version=decision.risk_policy_version,
        risk_policy_hash=decision.risk_policy_hash,
        generator_version="order-draft-generator-v1",
        generator_config_version="order-draft-config-v1",
        generator_config_hash="b" * 64,
        funding_policy=DraftFundingPolicy.SNAPSHOT_AVAILABLE_CASH_ONLY,
        available_cash=Decimal("100000"),
        reserved_cash_required=Decimal(0),
        estimated_buy_cash_required=Decimal(0),
        estimated_sell_cash_proceeds=Decimal(0),
        estimated_total_fees=Decimal(0),
        estimated_total_slippage=Decimal(0),
        state_hashes=(),
        market_rule_hashes=(),
        fee_rule_hashes=(),
        slippage_model_hashes=(),
        input_hash="c" * 64,
        batch_hash=BATCH_HASH,
        batch_id=f"draft:{BATCH_HASH}",
        is_executable=False,
        requires_human_approval=True,
        total_line_count=0,
        returned_line_count=0,
        lines_truncated=False,
        lines=(),
    )
    return ToolResponse[OrderDraftOutput](
        ok=True,
        request_id=context.request_id,
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data=data,
        provenance=Provenance(
            service="agent-runtime-test",
            version="1.0.0",
            data_version=decision.data_version,
        ),
    )


def _paper_execution_response(
    decision: DecisionSnapshot,
    context: ToolCallContext,
) -> ToolResponse[PaperExecutionOutput]:
    account_before = PaperAccountStateOutput(
        account_id=decision.account_id,
        runtime_mode=RuntimeMode.PAPER,
        data_version=decision.data_version,
        currency="CNY",
        source_snapshot_id=decision.account_snapshot_id,
        source_snapshot_hash=decision.account_snapshot_hash,
        source_snapshot_as_of=decision.as_of,
        as_of=NOW,
        total_cash=Decimal("100000"),
        available_cash=Decimal("100000"),
        frozen_cash=Decimal(0),
        external_frozen_cash=Decimal(0),
        state_hash="d" * 64,
        event_log_hash="e" * 64,
        total_position_count=0,
        returned_position_count=0,
        positions_truncated=False,
        positions=(),
    )
    account_after = account_before.model_copy(
        update={
            "as_of": NOW + timedelta(seconds=1),
            "state_hash": "f" * 64,
            "event_log_hash": "0" * 64,
        }
    )
    data = PaperExecutionOutput(
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data_version=decision.data_version,
        account_id=decision.account_id,
        account_snapshot_id=decision.account_snapshot_id,
        account_snapshot_hash=decision.account_snapshot_hash,
        receipt_id="receipt-runtime-lineage-1",
        request_hash="1" * 64,
        batch_hash=BATCH_HASH,
        request_submitted_at=NOW,
        processed_at=NOW + timedelta(seconds=1),
        account_before_hash=account_before.state_hash,
        account_after_hash=account_after.state_hash,
        event_log_before_hash=account_before.event_log_hash,
        event_log_hash=account_after.event_log_hash,
        receipt_hash="2" * 64,
        account_before=account_before,
        account_after=account_after,
        total_order_count=0,
        returned_order_count=0,
        orders_truncated=False,
        orders=(),
        total_attempt_count=0,
        returned_attempt_count=0,
        attempts_truncated=False,
        attempts=(),
        total_fill_count=0,
        returned_fill_count=0,
        fills_truncated=False,
        fills=(),
    )
    return ToolResponse[PaperExecutionOutput](
        ok=True,
        request_id=context.request_id,
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data=data,
        provenance=Provenance(
            service="agent-runtime-test",
            version="1.0.0",
            data_version=decision.data_version,
        ),
    )


def _failed_reconciliation_response(
    decision: DecisionSnapshot,
    context: ToolCallContext,
) -> ToolResponse[ReconciliationOutput]:
    data = ReconciliationOutput(
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data_version=decision.data_version,
        account_id=decision.account_id,
        account_snapshot_id=decision.account_snapshot_id,
        account_snapshot_hash=decision.account_snapshot_hash,
        reconciliation_id="reconciliation-runtime-failed-typed",
        engine_version="reconciliation-test-v1",
        input_hash="1" * 64,
        batch_hash=BATCH_HASH,
        draft_hash=BATCH_HASH,
        receipt_id="receipt-runtime-1",
        receipt_hash="9" * 64,
        expected_account_before_hash="2" * 64,
        expected_account_after_hash="3" * 64,
        observed_snapshot_id="observed-account-state-1",
        observed_snapshot_hash="4" * 64,
        observed_as_of=NOW,
        reconciled_at=NOW,
        policy_version="reconciliation-policy-v1",
        policy_hash="5" * 64,
        status=ReconciliationStatus.UNRECONCILABLE,
        max_severity=ReconciliationSeverity.CRITICAL,
        result_hash="6" * 64,
        total_finding_count=0,
        returned_finding_count=0,
        findings_truncated=False,
        findings=(),
        total_check_count=0,
        returned_check_count=0,
        checks_truncated=False,
        checks=(),
        duplicate_group_count=0,
        cash=CashReconciliationOutput(
            status=ReconciliationCheckStatus.CONFLICT,
            severity=ReconciliationSeverity.CRITICAL,
            expected_total_cash=Decimal(0),
            observed_total_cash=Decimal(0),
            total_delta=Decimal(0),
            expected_available_cash=Decimal(0),
            observed_available_cash=Decimal(0),
            available_delta=Decimal(0),
            expected_frozen_cash=Decimal(0),
            observed_frozen_cash=Decimal(0),
            frozen_delta=Decimal(0),
            mismatched_fields=(),
            check_hash="7" * 64,
        ),
        stop_signal=ReconciliationStopSignalOutput(
            required=True,
            action=ReconciliationStopAction.STOP_NEW_ORDERS,
            scope=ReconciliationStopScope.ACCOUNT,
            severity=ReconciliationSeverity.CRITICAL,
            reason_codes=(),
            trigger_hashes=(),
            generated_at=NOW,
            signal_hash="8" * 64,
        ),
    )
    return ToolResponse[ReconciliationOutput](
        ok=False,
        request_id=context.request_id,
        decision_id=decision.decision_id,
        as_of=decision.as_of,
        data=data,
        errors=(
            ToolIssue(
                code=ErrorCode.INTERNAL_ERROR,
                message="reconciliation failed despite producing typed diagnostics",
            ),
        ),
        provenance=Provenance(
            service="agent-runtime-test",
            version="1.0.0",
            data_version=decision.data_version,
        ),
    )


def _runtime(
    mode: RuntimeMode,
    *,
    clock: _Clock | None = None,
    response_delay: float = 0,
    incident_handler: _IncidentHandler | None = None,
) -> tuple[AgentRuntime, InMemoryAgentRunRepository, _RecordingRegistry, _Clock]:
    selected_clock = clock or _Clock(NOW)
    decision = _decision(mode)
    repository = InMemoryAgentRunRepository()
    registry = _RecordingRegistry(
        mode=mode,
        decision=decision,
        response_delay=response_delay,
    )
    runtime = AgentRuntime(
        registry=registry,
        repository=repository,
        clock=selected_clock,
        incident_handler=incident_handler,
    )
    return runtime, repository, registry, selected_clock


def _start(
    runtime: AgentRuntime,
    mode: RuntimeMode,
    goal: AgentRunGoal,
    *,
    limits: AgentRunLimits | None = None,
    run_id: str = "run_agent_runtime_test",
) -> AgentRunSnapshot:
    return runtime.start_run(
        decision_snapshot=_decision(mode),
        goal=goal,
        limits=limits,
        run_id=run_id,
    )


def _append_state_path(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
    snapshot: AgentRunSnapshot,
    states: tuple[AgentRunState, ...],
    *,
    approval_deadline: datetime | None = None,
) -> AgentRunSnapshot:
    before = snapshot
    current = snapshot
    for state in states:
        current = runtime._append_event(
            current,
            kind=AgentRunEventKind.STATE_TRANSITION,
            trigger=AgentRunTrigger.SYSTEM,
            state_after=state,
            reason=f"TEST_ADVANCE_TO_{state.value}",
            occurred_at=current.updated_at,
            approval_deadline=(
                approval_deadline if state is AgentRunState.PENDING_APPROVAL else None
            ),
        )
    return repository.save(current, expected_revision=before.revision)


def _append_replayable_artifact_call(
    runtime: AgentRuntime,
    snapshot: AgentRunSnapshot,
    *,
    tool_name: str,
    state_after: AgentRunState,
    artifact_kind: AgentArtifactKind,
    artifact_id: str,
    content_hash: str,
    expires_at: datetime | None = None,
    batch_hash: str | None = None,
) -> AgentRunSnapshot:
    policy = V1_TOOL_POLICIES[tool_name]
    request_id = f"req_replayable_{tool_name}"
    argument_hash = "1" * 64
    response_hash = "2" * 64
    idempotency_key_hash = "3" * 64 if policy.requires_idempotency_key else None
    pending = AgentPendingToolCall(
        request_id=request_id,
        tool_name=tool_name,
        argument_hash=argument_hash,
        idempotency_key_hash=idempotency_key_hash,
        batch_hash=batch_hash,
        started_at=snapshot.updated_at,
        effect=policy.effect,
    )
    reserved = runtime._append_event(
        snapshot,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_after=snapshot.state,
        reason="TOOL_CALL_RESERVED",
        occurred_at=snapshot.updated_at,
        request_id=request_id,
        tool_name=tool_name,
        tool_effect=policy.effect,
        argument_hash=argument_hash,
        idempotency_key_hash=idempotency_key_hash,
        batch_hash=batch_hash,
        pending_call=pending,
        tool_call_count=snapshot.tool_call_count + 1,
        reconciliation_attempt_count=(
            snapshot.reconciliation_attempt_count + (1 if tool_name == "reconcile_account" else 0)
        ),
    )
    artifact = AgentArtifactRef(
        kind=artifact_kind,
        artifact_id=artifact_id,
        content_hash=content_hash,
        tool_name=tool_name,
        response_hash=response_hash,
        expires_at=expires_at,
    )
    return runtime._append_event(
        reserved,
        kind=AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        trigger=AgentRunTrigger.TOOL,
        state_after=state_after,
        reason=f"TEST_{tool_name.upper()}_SUCCEEDED",
        occurred_at=reserved.updated_at,
        request_id=request_id,
        tool_name=tool_name,
        tool_effect=policy.effect,
        argument_hash=argument_hash,
        response_hash=response_hash,
        idempotency_key_hash=idempotency_key_hash,
        batch_hash=batch_hash,
        artifact=artifact,
    )


def _replayable_portfolio_ready_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
    *,
    mode: RuntimeMode,
    goal: AgentRunGoal,
) -> AgentRunSnapshot:
    started = _start(
        runtime,
        mode,
        goal,
        limits=AgentRunLimits(timeout_seconds=3_600),
        run_id=f"run_replayable_{mode.value.lower()}",
    )
    current = _append_replayable_artifact_call(
        runtime,
        started,
        tool_name="get_market_snapshot",
        state_after=AgentRunState.SNAPSHOT_READY,
        artifact_kind=AgentArtifactKind.MARKET_SNAPSHOT,
        artifact_id="market-runtime-1",
        content_hash="3" * 64,
    )
    current = _append_replayable_artifact_call(
        runtime,
        current,
        tool_name="validate_market_data",
        state_after=AgentRunState.DATA_VALIDATED,
        artifact_kind=AgentArtifactKind.DATA_QUALITY,
        artifact_id="quality-runtime-1",
        content_hash="4" * 64,
    )
    current = _append_replayable_artifact_call(
        runtime,
        current,
        tool_name="rank_stock_candidates",
        state_after=AgentRunState.ANALYZED,
        artifact_kind=AgentArtifactKind.STOCK_CANDIDATES,
        artifact_id="candidates-runtime-1",
        content_hash="5" * 64,
    )
    current = _append_replayable_artifact_call(
        runtime,
        current,
        tool_name="get_portfolio_snapshot",
        state_after=AgentRunState.ANALYZED,
        artifact_kind=AgentArtifactKind.PORTFOLIO_SNAPSHOT,
        artifact_id="portfolio-runtime-1",
        content_hash="6" * 64,
    )
    current = _append_replayable_artifact_call(
        runtime,
        current,
        tool_name="build_target_portfolio",
        state_after=AgentRunState.PORTFOLIO_READY,
        artifact_kind=AgentArtifactKind.TARGET_PORTFOLIO,
        artifact_id="target-runtime-1",
        content_hash="7" * 64,
    )
    return repository.save(current, expected_revision=started.revision)


def _replayable_draft_ready_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
    *,
    mode: RuntimeMode,
    goal: AgentRunGoal,
    draft_expires_at: datetime,
) -> AgentRunSnapshot:
    risk_checked = _replayable_risk_checked_run(
        runtime,
        repository,
        mode=mode,
        goal=goal,
    )
    current = _append_replayable_artifact_call(
        runtime,
        risk_checked,
        tool_name="create_order_draft",
        state_after=AgentRunState.DRAFT_READY,
        artifact_kind=AgentArtifactKind.ORDER_DRAFT,
        artifact_id=f"draft:{BATCH_HASH}",
        content_hash=BATCH_HASH,
        expires_at=draft_expires_at,
    )
    return repository.save(current, expected_revision=risk_checked.revision)


def _replayable_risk_checked_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
    *,
    mode: RuntimeMode,
    goal: AgentRunGoal,
) -> AgentRunSnapshot:
    portfolio_ready = _replayable_portfolio_ready_run(
        runtime,
        repository,
        mode=mode,
        goal=goal,
    )
    current = _append_replayable_artifact_call(
        runtime,
        portfolio_ready,
        tool_name="check_portfolio_risk",
        state_after=AgentRunState.RISK_CHECKED,
        artifact_kind=AgentArtifactKind.RISK_CHECK,
        artifact_id="risk-runtime-1",
        content_hash="8" * 64,
    )
    return repository.save(current, expected_revision=portfolio_ready.revision)


def _replayable_paper_executing_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
    *,
    draft_expires_at: datetime,
) -> AgentRunSnapshot:
    draft_ready = _replayable_draft_ready_run(
        runtime,
        repository,
        mode=RuntimeMode.PAPER,
        goal=AgentRunGoal.PAPER_EXECUTION,
        draft_expires_at=draft_expires_at,
    )
    return _append_state_path(
        runtime,
        repository,
        draft_ready,
        (AgentRunState.EXECUTING,),
    )


def _replayable_paper_reconciling_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
) -> AgentRunSnapshot:
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    reconciliating = _append_replayable_artifact_call(
        runtime,
        executing,
        tool_name="submit_paper_orders",
        state_after=AgentRunState.RECONCILING,
        artifact_kind=AgentArtifactKind.EXECUTION_RECEIPT,
        artifact_id="receipt-runtime-1",
        content_hash="9" * 64,
        batch_hash=BATCH_HASH,
    )
    return repository.save(reconciliating, expected_revision=executing.revision)


def _replayable_live_pending_run(
    runtime: AgentRuntime,
    repository: InMemoryAgentRunRepository,
) -> AgentRunSnapshot:
    draft_ready = _replayable_draft_ready_run(
        runtime,
        repository,
        mode=RuntimeMode.LIVE_ASSISTED,
        goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    return _append_state_path(
        runtime,
        repository,
        draft_ready,
        (AgentRunState.PENDING_APPROVAL,),
        approval_deadline=NOW + timedelta(minutes=15),
    )


def test_start_run_binds_exact_decision_mode_goal_and_rejects_mismatches() -> None:
    runtime, _, _, _ = _runtime(RuntimeMode.PAPER)

    result = _start(runtime, RuntimeMode.PAPER, AgentRunGoal.PAPER_EXECUTION)

    assert result.state is AgentRunState.SNAPSHOT_READY
    assert result.runtime_mode is RuntimeMode.PAPER
    assert result.goal is AgentRunGoal.PAPER_EXECUTION
    assert result.decision_id == DECISION_ID
    assert result.decision_snapshot_hash == _decision(RuntimeMode.PAPER).content_hash
    assert result.events[-1].reason == "DECISION_SNAPSHOT_BOUND"

    with pytest.raises(AgentGoalModeMismatch):
        _start(
            runtime,
            RuntimeMode.PAPER,
            AgentRunGoal.BACKTEST_REPORT,
            run_id="run_wrong_goal",
        )
    with pytest.raises(ValueError, match="fixed registry mode"):
        _start(
            runtime,
            RuntimeMode.RESEARCH,
            AgentRunGoal.MARKET_RESEARCH,
            run_id="run_wrong_registry",
        )


def test_catalog_and_invoke_fail_closed_before_registry_dispatch() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.PAPER)
    run = _start(runtime, RuntimeMode.PAPER, AgentRunGoal.PAPER_EXECUTION)

    assert {descriptor.name for descriptor in runtime.catalog(run.run_id)} == {
        "get_market_snapshot"
    }
    assert len(registry.catalog_contexts) == 1

    with pytest.raises(AgentToolStateDenied, match="required artifact is missing"):
        runtime.invoke_tool(
            run.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "paper-submit-1"},
            request_id="req_skip_risk_and_draft",
        )

    current = runtime.get_run(run.run_id)
    assert registry.invoke_calls == []
    assert current.tool_call_count == 1
    assert current.invalid_tool_call_count == 1
    assert current.state is AgentRunState.SNAPSHOT_READY


def test_invalid_attempts_consume_budget_and_enter_explicit_limit_state() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(max_tool_calls=2, max_invalid_tool_calls=2),
    )

    with pytest.raises(AgentToolStateDenied):
        runtime.invoke_tool(
            run.run_id,
            "detect_market_regime",
            {},
            request_id="req_invalid_1",
        )
    with pytest.raises(AgentToolCallLimitExceeded):
        runtime.invoke_tool(
            run.run_id,
            "rank_market_themes",
            {},
            request_id="req_invalid_2",
        )

    limited = runtime.get_run(run.run_id)
    assert limited.state is AgentRunState.LIMIT_EXCEEDED
    assert limited.tool_call_count == 2
    assert limited.invalid_tool_call_count == 2
    assert limited.terminal_reason == "INVALID_TOOL_CALL_LIMIT_REACHED"
    assert registry.invoke_calls == []
    assert runtime.catalog(run.run_id) == ()
    with pytest.raises(AgentRunTerminal):
        runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {},
            request_id="req_after_limit",
        )


def test_deadline_is_inclusive_and_transitions_to_timed_out() -> None:
    clock = _Clock(NOW)
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH, clock=clock)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(timeout_seconds=60),
    )

    clock.value = run.deadline
    timed_out = runtime.get_run(run.run_id)

    assert timed_out.state is AgentRunState.TIMED_OUT
    assert timed_out.completed_at == run.deadline
    assert timed_out.terminal_reason == "RUN_DEADLINE_EXCEEDED"
    assert runtime.catalog(run.run_id) == ()
    assert registry.invoke_calls == []


def test_invoke_reuses_gate_time_for_reservation_at_microsecond_boundary() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(timeout_seconds=60),
    )
    sequence = _SequenceClock((run.deadline - timedelta(microseconds=1), run.deadline))
    runtime._clock = sequence

    response = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_deadline_microsecond_boundary",
    )

    timed_out = repository.get(run.run_id)
    assert response.ok is True
    assert timed_out.state is AgentRunState.TIMED_OUT
    assert timed_out.events[-1].kind is AgentRunEventKind.TIMEOUT_RECORDED
    assert timed_out.events[-1].occurred_at == run.deadline
    assert len(registry.invoke_calls) == 1
    assert sequence.call_count == 2
    assert replay_agent_run(timed_out.events) == timed_out


def test_cancel_before_execution_is_terminal() -> None:
    runtime, _, _, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)

    cancelled = runtime.request_cancel(run.run_id)

    assert cancelled.state is AgentRunState.CANCELLED
    assert cancelled.cancel_requested is True
    assert cancelled.terminal_reason == "CANCELLED_BEFORE_EXECUTION"


def test_cancel_reuses_one_clock_snapshot_at_microsecond_boundary() -> None:
    runtime, _, _, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(timeout_seconds=60),
    )
    sequence = _SequenceClock((run.deadline - timedelta(microseconds=1), run.deadline))
    runtime._clock = sequence

    cancelled = runtime.request_cancel(run.run_id)

    assert cancelled.state is AgentRunState.CANCELLED
    assert cancelled.events[-1].occurred_at == run.deadline - timedelta(microseconds=1)
    assert sequence.call_count == 1
    assert replay_agent_run(cancelled.events) == cancelled


@pytest.mark.parametrize(
    ("mode", "goal", "submit_tool"),
    [
        (RuntimeMode.PAPER, AgentRunGoal.PAPER_EXECUTION, "submit_paper_orders"),
        (
            RuntimeMode.LIVE_ASSISTED,
            AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            "submit_approved_orders",
        ),
    ],
)
def test_cancel_unsubmitted_execution_is_safe_and_blocks_submit(
    mode: RuntimeMode,
    goal: AgentRunGoal,
    submit_tool: str,
) -> None:
    runtime, repository, registry, clock = _runtime(mode)
    if goal is AgentRunGoal.PAPER_EXECUTION:
        executing = _replayable_paper_executing_run(
            runtime,
            repository,
            draft_expires_at=NOW + timedelta(minutes=30),
        )
    else:
        pending = _replayable_live_pending_run(runtime, repository)
        executing = runtime.record_external_approval(
            pending.run_id,
            AgentExternalApproval(
                approval_id="approval-cancel-test",
                approval_hash="a" * 64,
                decision_id=DECISION_ID,
                batch_hash=BATCH_HASH,
                approved_at=clock.value,
                expires_at=clock.value + timedelta(minutes=5),
            ),
        )

    result = runtime.request_cancel(executing.run_id)

    assert result.state is AgentRunState.CANCELLED
    assert result.cancel_requested is True
    assert result.terminal_reason == "CANCELLED_BEFORE_EXECUTION"
    assert runtime.catalog(result.run_id) == ()
    with pytest.raises(AgentRunTerminal):
        runtime.invoke_tool(
            result.run_id,
            submit_tool,
            {"batch_hash": BATCH_HASH, "idempotency_key": "cancelled-submit"},
            request_id="req_cancelled_submit",
        )
    assert registry.invoke_calls == []
    assert replay_agent_run(result.events) == result


def test_cancel_pending_execution_write_only_records_flag() -> None:
    runtime, repository, _, _ = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    pending = AgentPendingToolCall(
        request_id="req_pending_submit",
        tool_name="submit_paper_orders",
        argument_hash="a" * 64,
        idempotency_key_hash="b" * 64,
        batch_hash=BATCH_HASH,
        started_at=executing.updated_at,
        effect=ToolEffect.PAPER_EXECUTION_WRITE,
    )
    reserved = runtime._append_event(
        executing,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_after=AgentRunState.EXECUTING,
        reason="TOOL_CALL_RESERVED",
        occurred_at=executing.updated_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        idempotency_key_hash=pending.idempotency_key_hash,
        batch_hash=pending.batch_hash,
        pending_call=pending,
        tool_call_count=executing.tool_call_count + 1,
    )
    staged = repository.save(reserved, expected_revision=executing.revision)

    result = runtime.request_cancel(staged.run_id)

    assert result.state is AgentRunState.EXECUTING
    assert result.cancel_requested is True
    assert result.pending_call == pending
    assert result.completed_at is None
    assert result.events[-1].reason == "CANCEL_REQUESTED_RECONCILIATION_REQUIRED"
    assert replay_agent_run(result.events) == result


def test_cancelled_pending_execution_rejection_fails_closed_to_incident() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    pending = AgentPendingToolCall(
        request_id="req_cancelled_pending_submit",
        tool_name="submit_paper_orders",
        argument_hash="a" * 64,
        idempotency_key_hash="b" * 64,
        batch_hash=BATCH_HASH,
        started_at=executing.updated_at,
        effect=ToolEffect.PAPER_EXECUTION_WRITE,
    )
    reserved = runtime._append_event(
        executing,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_after=AgentRunState.EXECUTING,
        reason="TOOL_CALL_RESERVED",
        occurred_at=executing.updated_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        idempotency_key_hash=pending.idempotency_key_hash,
        batch_hash=pending.batch_hash,
        pending_call=pending,
        tool_call_count=executing.tool_call_count + 1,
    )
    staged = repository.save(reserved, expected_revision=executing.revision)
    cancelled = runtime.request_cancel(staged.run_id)

    rejected = runtime._append_event(
        cancelled,
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        trigger=AgentRunTrigger.TOOL,
        state_after=AgentRunState.EXECUTING,
        reason="TOOL_ARGUMENTS_REJECTED",
        occurred_at=cancelled.updated_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        idempotency_key_hash=pending.idempotency_key_hash,
        batch_hash=pending.batch_hash,
        error_codes=("REGISTRY_INVALID_ARGUMENTS",),
        invalid_tool_call_count=cancelled.invalid_tool_call_count + 1,
    )
    incident = runtime._append_event(
        rejected,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        trigger=AgentRunTrigger.SYSTEM,
        state_after=AgentRunState.INCIDENT,
        reason="CANCELLED_EXECUTION_OUTCOME_REJECTED",
        occurred_at=cancelled.updated_at,
    )
    closed = repository.save(incident, expected_revision=cancelled.revision)

    assert closed.state is AgentRunState.INCIDENT
    assert closed.cancel_requested is True
    assert closed.pending_call is None
    assert runtime.catalog(closed.run_id) == ()
    with pytest.raises(
        AgentRunTerminal,
        match="run is terminal",
    ):
        runtime.invoke_tool(
            closed.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "second-submit"},
            request_id="req_second_submit_after_cancel",
        )

    current = runtime.get_run(closed.run_id)
    assert current.state is AgentRunState.INCIDENT
    assert current.cancel_requested is True
    assert current.pending_call is None
    assert registry.invoke_calls == []
    assert replay_agent_run(current.events) == current


def test_cancel_reconciling_run_only_records_flag() -> None:
    runtime, repository, _, _ = _runtime(RuntimeMode.PAPER)
    reconciling = _replayable_paper_reconciling_run(runtime, repository)

    result = runtime.request_cancel(reconciling.run_id)

    assert result.state is AgentRunState.RECONCILING
    assert result.cancel_requested is True
    assert result.completed_at is None
    assert result.events[-1].reason == "CANCEL_REQUESTED_RECONCILIATION_REQUIRED"
    assert replay_agent_run(result.events) == result


@pytest.mark.parametrize(
    ("decision_id", "batch_hash", "approved_delta", "expires_delta"),
    [
        ("another-decision", BATCH_HASH, timedelta(0), timedelta(minutes=1)),
        (DECISION_ID, "f" * 64, timedelta(0), timedelta(minutes=1)),
        (DECISION_ID, BATCH_HASH, timedelta(seconds=1), timedelta(minutes=1)),
        (DECISION_ID, BATCH_HASH, -timedelta(seconds=1), timedelta(0)),
    ],
)
def test_external_approval_rejects_unbound_or_expired_evidence(
    decision_id: str,
    batch_hash: str,
    approved_delta: timedelta,
    expires_delta: timedelta,
) -> None:
    runtime, repository, _, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    approval = AgentExternalApproval(
        approval_id="approval-invalid-test",
        approval_hash="a" * 64,
        decision_id=decision_id,
        batch_hash=batch_hash,
        approved_at=clock.value + approved_delta,
        expires_at=clock.value + expires_delta,
    )

    with pytest.raises(AgentApprovalInvalid, match="does not bind"):
        runtime.record_external_approval(pending.run_id, approval)

    assert runtime.get_run(pending.run_id).state is AgentRunState.PENDING_APPROVAL


def test_external_approval_cannot_predate_pending_approval_stage() -> None:
    runtime, repository, _, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    pending_since = next(
        event.occurred_at
        for event in reversed(pending.events)
        if event.state_after is AgentRunState.PENDING_APPROVAL
    )
    approval = AgentExternalApproval(
        approval_id="approval-before-pending-stage",
        approval_hash="a" * 64,
        decision_id=DECISION_ID,
        batch_hash=BATCH_HASH,
        approved_at=pending_since - timedelta(microseconds=1),
        expires_at=clock.value + timedelta(minutes=1),
    )

    with pytest.raises(AgentApprovalInvalid, match="does not bind"):
        runtime.record_external_approval(pending.run_id, approval)

    unchanged = repository.get(pending.run_id)
    assert unchanged == pending
    assert replay_agent_run(unchanged.events) == unchanged


def test_approval_signed_after_pending_entry_survives_draft_read_self_loop() -> None:
    runtime, repository, registry, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    pending_since = next(
        event.occurred_at
        for event in reversed(pending.events)
        if (
            event.state_before is not AgentRunState.PENDING_APPROVAL
            and event.state_after is AgentRunState.PENDING_APPROVAL
        )
    )
    signed_at = pending_since + timedelta(seconds=1)
    approval = AgentExternalApproval(
        approval_id="approval-before-draft-read-self-loop",
        approval_hash="a" * 64,
        decision_id=DECISION_ID,
        batch_hash=BATCH_HASH,
        approved_at=signed_at,
        expires_at=signed_at + timedelta(minutes=5),
    )
    decision = _decision(RuntimeMode.LIVE_ASSISTED)
    registry.response_factory = lambda tool_name, context: (
        _order_draft_response(decision, context)
        if tool_name == "get_order_draft"
        else pytest.fail(f"unexpected registry call: {tool_name}")
    )
    clock.value = signed_at + timedelta(seconds=1)

    draft_response = runtime.invoke_tool(
        pending.run_id,
        "get_order_draft",
        {"batch_hash": BATCH_HASH},
        request_id="req_pending_approval_draft_read",
    )
    after_read = repository.get(pending.run_id)
    executing = runtime.record_external_approval(pending.run_id, approval)

    assert draft_response.ok is True
    assert after_read.state is AgentRunState.PENDING_APPROVAL
    assert after_read.events[-1].kind is AgentRunEventKind.TOOL_CALL_SUCCEEDED
    assert after_read.events[-1].state_before is AgentRunState.PENDING_APPROVAL
    assert after_read.events[-1].state_after is AgentRunState.PENDING_APPROVAL
    assert executing.state is AgentRunState.EXECUTING
    assert executing.events[-2].external_approval == approval
    assert [call[0] for call in registry.invoke_calls] == ["get_order_draft"]
    assert replay_agent_run(executing.events) == executing


def test_record_approval_reuses_one_clock_snapshot_at_microsecond_boundary() -> None:
    runtime, repository, _, _ = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    assert pending.approval_deadline is not None
    operation_time = pending.approval_deadline - timedelta(microseconds=1)
    sequence = _SequenceClock((operation_time, pending.approval_deadline))
    runtime._clock = sequence
    approval = AgentExternalApproval(
        approval_id="approval-deadline-microsecond",
        approval_hash="a" * 64,
        decision_id=DECISION_ID,
        batch_hash=BATCH_HASH,
        approved_at=operation_time,
        expires_at=pending.approval_deadline + timedelta(minutes=1),
    )

    executing = runtime.record_external_approval(pending.run_id, approval)

    assert executing.state is AgentRunState.EXECUTING
    assert executing.approval_deadline == pending.approval_deadline
    assert executing.events[-2].occurred_at == operation_time
    assert executing.events[-1].occurred_at == operation_time
    assert sequence.call_count == 1
    assert replay_agent_run(executing.events) == executing


def test_reject_approval_reuses_one_clock_snapshot_at_microsecond_boundary() -> None:
    runtime, repository, _, _ = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    assert pending.approval_deadline is not None
    operation_time = pending.approval_deadline - timedelta(microseconds=1)
    sequence = _SequenceClock((operation_time, pending.approval_deadline))
    runtime._clock = sequence

    rejected = runtime.reject_external_approval(
        pending.run_id,
        rejection_hash="f" * 64,
    )

    assert rejected.state is AgentRunState.REJECTED
    assert rejected.events[-1].occurred_at == operation_time
    assert sequence.call_count == 1
    assert replay_agent_run(rejected.events) == rejected


def test_external_approval_unlocks_live_execution_only_for_exact_batch() -> None:
    runtime, repository, _, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    approval = AgentExternalApproval(
        approval_id="approval-runtime-1",
        approval_hash="a" * 64,
        decision_id=DECISION_ID,
        batch_hash=BATCH_HASH,
        approved_at=clock.value,
        expires_at=clock.value + timedelta(minutes=1),
    )

    approved = runtime.record_external_approval(pending.run_id, approval)

    assert approved.state is AgentRunState.EXECUTING
    assert approved.approval_deadline == approval.expires_at
    assert approved.events[-2].kind is AgentRunEventKind.APPROVAL_RECORDED
    assert approved.events[-2].batch_hash == BATCH_HASH
    assert approved.events[-2].control_hash == approval.approval_hash
    assert approved.events[-2].external_approval == approval
    assert approved.events[-1].reason == "LIVE_EXECUTION_READY"
    assert {descriptor.name for descriptor in runtime.catalog(pending.run_id)} == {
        "submit_approved_orders"
    }
    assert replay_agent_run(approved.events) == approved


def test_request_id_replays_exact_result_and_conflicts_on_changed_arguments() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)

    first = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_exact_replay",
    )
    replay = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_exact_replay",
    )

    assert replay == first
    assert len(registry.invoke_calls) == 1
    assert runtime.get_run(run.run_id).tool_call_count == 1
    with pytest.raises(AgentRequestConflict, match="different tool call"):
        runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {"max_files": 1},
            request_id="req_exact_replay",
        )


def test_concurrent_same_request_dispatches_registry_handler_once() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH, response_delay=0.05)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    ready = Barrier(2)

    def invoke() -> ToolResponse[object]:
        ready.wait()
        return runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {},
            request_id="req_concurrent_replay",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: invoke(), range(2)))

    assert results[0] == results[1]
    assert len(registry.invoke_calls) == 1
    assert runtime.get_run(run.run_id).tool_call_count == 1


def test_invalid_json_is_counted_and_omitted_request_id_is_generated() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)

    with pytest.raises(AgentToolStateDenied, match="valid JSON object"):
        runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            '{"duplicate":1,"duplicate":2}',
            request_id="req_duplicate_json_key",
        )
    response = runtime.invoke_tool(run.run_id, "get_market_snapshot", {})

    current = runtime.get_run(run.run_id)
    reserved = tuple(
        event for event in current.events if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED
    )
    assert current.tool_call_count == 2
    assert current.invalid_tool_call_count == 1
    assert response.request_id.startswith("tool_")
    assert reserved[-1].request_id == response.request_id
    assert len(registry.invoke_calls) == 1


@pytest.mark.parametrize(
    ("error", "expected_state", "expected_invalid", "expected_reason"),
    [
        (
            ToolArgumentsRejected(
                ToolDenialReason.INVALID_ARGUMENTS,
                mode=RuntimeMode.RESEARCH,
                tool_name="get_market_snapshot",
            ),
            AgentRunState.SNAPSHOT_READY,
            1,
            "TOOL_ARGUMENTS_REJECTED",
        ),
        (
            ToolPermissionDenied(
                ToolDenialReason.CAPABILITY_NOT_GRANTED,
                mode=RuntimeMode.RESEARCH,
                tool_name="get_market_snapshot",
            ),
            AgentRunState.REJECTED,
            1,
            "REGISTRY_AUTHORIZATION_REJECTED",
        ),
        (
            ToolAuditUnavailable(None),
            AgentRunState.FAILED,
            0,
            "TOOL_AUDIT_UNAVAILABLE",
        ),
        (
            ToolExecutionFailed("safe handler failure"),
            AgentRunState.FAILED,
            0,
            "TOOL_EXECUTION_FAILED",
        ),
        (
            ToolOutputRejected("safe output failure"),
            AgentRunState.FAILED,
            0,
            "TOOL_EXECUTION_FAILED",
        ),
    ],
)
def test_registry_exceptions_are_recorded_and_mapped_to_explicit_states(
    error: Exception,
    expected_state: AgentRunState,
    expected_invalid: int,
    expected_reason: str,
) -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    registry.invoke_error = error

    with pytest.raises(type(error)):
        runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {},
            request_id=f"req_{type(error).__name__}",
        )

    current = runtime.get_run(run.run_id)
    assert current.state is expected_state
    assert current.invalid_tool_call_count == expected_invalid
    assert current.pending_call is None
    assert any(event.reason == expected_reason for event in current.events)
    assert len(registry.invoke_calls) == 1


def test_reconciliation_argument_rejection_is_same_state_and_retryable() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    reconciling = _replayable_paper_reconciling_run(runtime, repository)
    registry.invoke_error = ToolArgumentsRejected(
        ToolDenialReason.INVALID_ARGUMENTS,
        mode=RuntimeMode.PAPER,
        tool_name="reconcile_account",
    )

    with pytest.raises(ToolArgumentsRejected):
        runtime.invoke_tool(
            reconciling.run_id,
            "reconcile_account",
            {"batch_hash": BATCH_HASH},
            request_id="req_reconciliation_arguments_rejected",
        )

    retryable = repository.get(reconciling.run_id)
    assert retryable.state is AgentRunState.RECONCILING
    assert retryable.pending_call is None
    assert retryable.invalid_tool_call_count == reconciling.invalid_tool_call_count + 1
    assert retryable.tool_call_count == reconciling.tool_call_count + 1
    assert retryable.reconciliation_attempt_count == reconciling.reconciliation_attempt_count + 1
    assert retryable.events[-1].kind is AgentRunEventKind.TOOL_CALL_REJECTED
    assert replay_agent_run(retryable.events) == retryable

    decision = _decision(RuntimeMode.PAPER)
    registry.invoke_error = None
    registry.response_factory = lambda tool_name, context: (
        _failed_reconciliation_response(decision, context)
        if tool_name == "reconcile_account"
        else pytest.fail(f"unexpected registry call: {tool_name}")
    )
    response = runtime.invoke_tool(
        retryable.run_id,
        "reconcile_account",
        {"batch_hash": BATCH_HASH},
        request_id="req_reconciliation_retry",
    )

    incident = repository.get(retryable.run_id)
    assert response.ok is False
    assert incident.state is AgentRunState.INCIDENT
    assert incident.pending_call is None
    assert incident.reconciliation_attempt_count == reconciling.reconciliation_attempt_count + 2
    assert [call[0] for call in registry.invoke_calls] == [
        "reconcile_account",
        "reconcile_account",
    ]
    assert replay_agent_run(incident.events) == incident


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"batch_hash": BATCH_HASH}, "idempotency key"),
        (
            {"batch_hash": "f" * 64, "idempotency_key": "paper-submit-1"},
            "batch_hash does not match",
        ),
    ],
)
def test_invalid_execution_write_is_blocked_before_registry_and_enters_incident(
    arguments: dict[str, object],
    message: str,
) -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )

    with pytest.raises(AgentToolStateDenied, match=message):
        runtime.invoke_tool(
            executing.run_id,
            "submit_paper_orders",
            arguments,
            request_id="req_invalid_execution_write",
        )

    current = runtime.get_run(executing.run_id)
    assert current.state is AgentRunState.INCIDENT
    assert current.invalid_tool_call_count == 1
    assert registry.invoke_calls == []


def test_rejected_execution_write_does_not_lock_idempotency_for_retry() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    registry.invoke_error = ToolArgumentsRejected(
        ToolDenialReason.INVALID_ARGUMENTS,
        mode=RuntimeMode.PAPER,
        tool_name="submit_paper_orders",
    )

    with pytest.raises(ToolArgumentsRejected):
        runtime.invoke_tool(
            executing.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "paper-submit-1"},
            request_id="req_first_execution_reservation",
        )
    registry.invoke_error = ToolExecutionFailed("controlled retry outcome")
    with pytest.raises(ToolExecutionFailed):
        runtime.invoke_tool(
            executing.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "paper-submit-2"},
            request_id="req_changed_execution_identity",
        )

    current = runtime.get_run(executing.run_id)
    assert current.state is AgentRunState.INCIDENT
    assert current.tool_call_count == executing.tool_call_count + 2
    assert len(registry.invoke_calls) == 2
    assert replay_agent_run(current.events) == current


def test_rejected_draft_without_idempotency_can_retry_and_advance() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    risk_checked = _replayable_risk_checked_run(
        runtime,
        repository,
        mode=RuntimeMode.PAPER,
        goal=AgentRunGoal.PAPER_EXECUTION,
    )
    decision = _decision(RuntimeMode.PAPER)
    registry.response_factory = lambda tool_name, context: (
        _order_draft_response(decision, context)
        if tool_name == "create_order_draft"
        else pytest.fail(f"unexpected registry call: {tool_name}")
    )

    with pytest.raises(AgentToolStateDenied, match="idempotency key"):
        runtime.invoke_tool(
            risk_checked.run_id,
            "create_order_draft",
            {},
            request_id="req_draft_missing_idempotency",
        )
    response = runtime.invoke_tool(
        risk_checked.run_id,
        "create_order_draft",
        {"idempotency_key": "draft-valid-retry"},
        request_id="req_draft_valid_retry",
    )

    executing = runtime.get_run(risk_checked.run_id)
    assert response.ok is True
    assert executing.state is AgentRunState.EXECUTING
    assert executing.invalid_tool_call_count == 1
    assert [call[0] for call in registry.invoke_calls] == ["create_order_draft"]
    assert any(artifact.kind is AgentArtifactKind.ORDER_DRAFT for artifact in executing.artifacts)
    assert replay_agent_run(executing.events) == executing


def test_get_order_draft_lineage_mismatch_fails_without_pending_and_replays() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    draft_ready = _replayable_draft_ready_run(
        runtime,
        repository,
        mode=RuntimeMode.PAPER,
        goal=AgentRunGoal.ORDER_DRAFT,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    decision = _decision(RuntimeMode.PAPER)

    def mismatched_draft(
        tool_name: str,
        context: ToolCallContext,
    ) -> ToolResponse[OrderDraftOutput]:
        assert tool_name == "get_order_draft"
        valid = _order_draft_response(decision, context)
        assert valid.data is not None
        wrong_hash = "f" * 64
        wrong = valid.data.model_copy(
            update={
                "batch_hash": wrong_hash,
                "batch_id": f"draft:{wrong_hash}",
            }
        )
        return valid.model_copy(update={"data": wrong})

    registry.response_factory = mismatched_draft

    with pytest.raises(AgentRuntimeOutputInvalid):
        runtime.invoke_tool(
            draft_ready.run_id,
            "get_order_draft",
            {"batch_hash": BATCH_HASH},
            request_id="req_mismatched_loaded_draft",
        )

    failed = runtime.get_run(draft_ready.run_id)
    assert failed.state is AgentRunState.FAILED
    assert failed.pending_call is None
    assert len(registry.invoke_calls) == 1
    assert replay_agent_run(failed.events) == failed


def test_execution_receipt_source_snapshot_hash_mismatch_fails_closed() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    decision = _decision(RuntimeMode.PAPER)

    def mismatched_receipt(
        tool_name: str,
        context: ToolCallContext,
    ) -> ToolResponse[PaperExecutionOutput]:
        assert tool_name == "submit_paper_orders"
        valid = _paper_execution_response(decision, context)
        assert valid.data is not None
        wrong_before = valid.data.account_before.model_copy(
            update={"source_snapshot_hash": "f" * 64}
        )
        wrong = valid.data.model_copy(update={"account_before": wrong_before})
        return valid.model_copy(update={"data": wrong})

    registry.response_factory = mismatched_receipt

    with pytest.raises(AgentRuntimeOutputInvalid):
        runtime.invoke_tool(
            executing.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "paper-lineage-test"},
            request_id="req_mismatched_receipt_lineage",
        )

    incident = runtime.get_run(executing.run_id)
    assert incident.state is AgentRunState.INCIDENT
    assert incident.pending_call is None
    assert not any(
        artifact.kind is AgentArtifactKind.EXECUTION_RECEIPT for artifact in incident.artifacts
    )
    assert len(registry.invoke_calls) == 1
    assert replay_agent_run(incident.events) == incident


def test_failed_data_response_enters_data_invalid() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    registry.response_error_code = ErrorCode.DATA_INVALID

    response = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_data_invalid",
    )

    assert response.ok is False
    assert runtime.get_run(run.run_id).state is AgentRunState.DATA_INVALID


def test_failed_risk_response_enters_rejected() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    ready = _replayable_portfolio_ready_run(
        runtime,
        repository,
        mode=RuntimeMode.PAPER,
        goal=AgentRunGoal.PORTFOLIO_REPORT,
    )
    registry.response_error_code = ErrorCode.RISK_REJECTED

    response = runtime.invoke_tool(
        ready.run_id,
        "check_portfolio_risk",
        {},
        request_id="req_risk_rejected",
    )

    assert response.ok is False
    rejected = runtime.get_run(ready.run_id)
    assert rejected.state is AgentRunState.REJECTED
    assert rejected.terminal_reason == "PORTFOLIO_RISK_REJECTED"


def test_unexpected_approval_required_response_is_rejected_fail_closed() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.PAPER)
    run = _start(runtime, RuntimeMode.PAPER, AgentRunGoal.MARKET_RESEARCH)
    registry.response_error_code = ErrorCode.APPROVAL_REQUIRED

    runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_unexpected_approval",
    )

    result = runtime.get_run(run.run_id)
    assert result.state is AgentRunState.REJECTED
    assert result.terminal_reason == "UNEXPECTED_APPROVAL_REQUIREMENT"


def test_live_ordinary_tool_approval_required_rejects_without_pending_and_replays() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.LIVE_ASSISTED)
    run = _start(
        runtime,
        RuntimeMode.LIVE_ASSISTED,
        AgentRunGoal.MARKET_RESEARCH,
    )
    registry.response_error_code = ErrorCode.APPROVAL_REQUIRED

    response = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_live_unexpected_approval",
    )

    rejected = runtime.get_run(run.run_id)
    assert response.ok is False
    assert rejected.state is AgentRunState.REJECTED
    assert rejected.terminal_reason == "UNEXPECTED_APPROVAL_REQUIREMENT"
    assert rejected.pending_call is None
    assert len(registry.invoke_calls) == 1
    assert replay_agent_run(rejected.events) == rejected


def test_early_approval_expired_tool_response_fails_closed() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    draft_ready = _replayable_draft_ready_run(
        runtime,
        repository,
        mode=RuntimeMode.PAPER,
        goal=AgentRunGoal.ORDER_DRAFT,
        draft_expires_at=NOW + timedelta(minutes=30),
    )
    registry.response_error_code = ErrorCode.APPROVAL_EXPIRED

    with pytest.raises(AgentRuntimeOutputInvalid):
        runtime.invoke_tool(
            draft_ready.run_id,
            "get_order_draft",
            {"batch_hash": BATCH_HASH},
            request_id="req_approval_expired",
        )

    failed = runtime.get_run(draft_ready.run_id)
    assert failed.state is AgentRunState.FAILED
    assert failed.pending_call is None
    assert replay_agent_run(failed.events) == failed


def test_external_approval_rejection_is_terminal_and_hash_bound() -> None:
    runtime, repository, _, _ = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)

    rejected = runtime.reject_external_approval(
        pending.run_id,
        rejection_hash="7" * 64,
    )

    assert rejected.state is AgentRunState.REJECTED
    assert rejected.terminal_reason == "EXTERNAL_APPROVAL_REJECTED"
    assert rejected.events[-1].control_hash == "7" * 64
    with pytest.raises(AgentRunTerminal):
        runtime.reject_external_approval(
            pending.run_id,
            rejection_hash="8" * 64,
        )


def test_pending_external_approval_expires_at_inclusive_deadline() -> None:
    runtime, repository, _, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    assert pending.approval_deadline is not None

    clock.value = pending.approval_deadline
    expired = runtime.get_run(pending.run_id)

    assert expired.state is AgentRunState.EXPIRED
    assert expired.terminal_reason == "APPROVAL_WINDOW_EXPIRED"
    with pytest.raises(AgentRunTerminal):
        runtime.record_external_approval(
            pending.run_id,
            AgentExternalApproval(
                approval_id="approval-after-expiry",
                approval_hash="a" * 64,
                decision_id=DECISION_ID,
                batch_hash=BATCH_HASH,
                approved_at=clock.value - timedelta(seconds=1),
                expires_at=clock.value + timedelta(minutes=1),
            ),
        )


def test_clock_rollback_is_rejected() -> None:
    runtime, _, _, clock = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    clock.value = run.updated_at - timedelta(microseconds=1)

    with pytest.raises(AgentRuntimeClockError, match="moved backwards"):
        runtime.get_run(run.run_id)


def test_cancel_is_idempotent_without_appending_duplicate_events() -> None:
    runtime, _, _, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)

    first = runtime.request_cancel(run.run_id)
    second = runtime.request_cancel(run.run_id)

    assert second == first
    assert (
        sum(event.kind is AgentRunEventKind.CANCELLATION_REQUESTED for event in second.events) == 1
    )


def test_cancel_request_after_run_deadline_preserves_timed_out_terminal_state() -> None:
    clock = _Clock(NOW)
    runtime, _, _, _ = _runtime(RuntimeMode.RESEARCH, clock=clock)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(timeout_seconds=1),
    )
    clock.value = run.deadline

    timed_out = runtime.request_cancel(run.run_id)

    assert timed_out.state is AgentRunState.TIMED_OUT
    assert timed_out.terminal_reason == "RUN_DEADLINE_EXCEEDED"
    assert timed_out.cancel_requested is False
    assert all(
        event.kind is not AgentRunEventKind.CANCELLATION_REQUESTED for event in timed_out.events
    )
    assert replay_agent_run(timed_out.events) == timed_out


@pytest.mark.parametrize("expiry_source", ["draft", "approval"])
def test_cancel_request_after_earlier_authorization_deadline_preserves_expired_state(
    expiry_source: str,
) -> None:
    if expiry_source == "draft":
        runtime, repository, _, clock = _runtime(RuntimeMode.PAPER)
        executing = _replayable_paper_executing_run(
            runtime,
            repository,
            draft_expires_at=NOW + timedelta(seconds=1),
        )
        authorization_deadline = NOW + timedelta(seconds=1)
    else:
        runtime, repository, _, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
        pending = _replayable_live_pending_run(runtime, repository)
        approval = AgentExternalApproval(
            approval_id="approval-cancel-expiry",
            approval_hash="a" * 64,
            decision_id=DECISION_ID,
            batch_hash=BATCH_HASH,
            approved_at=NOW,
            expires_at=NOW + timedelta(seconds=1),
        )
        executing = runtime.record_external_approval(pending.run_id, approval)
        authorization_deadline = approval.expires_at
    clock.value = authorization_deadline

    expired = runtime.request_cancel(executing.run_id)

    assert expired.state is AgentRunState.EXPIRED
    assert expired.cancel_requested is False
    assert all(
        event.kind is not AgentRunEventKind.CANCELLATION_REQUESTED for event in expired.events
    )
    assert replay_agent_run(expired.events) == expired


@pytest.mark.parametrize("fail", [False, True])
def test_reconciliation_incident_handler_is_deduplicated_and_exception_isolated(
    fail: bool,
) -> None:
    handler = _IncidentHandler(fail=fail)
    runtime, _, _, _ = _runtime(RuntimeMode.PAPER, incident_handler=handler)
    run = _start(runtime, RuntimeMode.PAPER, AgentRunGoal.PAPER_EXECUTION)
    result = cast(
        ReconciliationOutput,
        _IncidentResult(
            reconciliation_id="reconciliation-runtime-1",
            result_hash="6" * 64,
        ),
    )

    runtime._notify_reconciliation_incident(run, result, NOW)
    runtime._notify_reconciliation_incident(run, result, NOW)

    assert handler.calls == [(DECISION_ID, "reconciliation-runtime-1", "6" * 64, NOW)]


def test_failed_reconciliation_typed_data_does_not_notify_incident_handler() -> None:
    handler = _IncidentHandler()
    runtime, repository, registry, _ = _runtime(
        RuntimeMode.PAPER,
        incident_handler=handler,
    )
    reconciling = _replayable_paper_reconciling_run(runtime, repository)
    decision = _decision(RuntimeMode.PAPER)
    registry.response_factory = lambda tool_name, context: (
        _failed_reconciliation_response(decision, context)
        if tool_name == "reconcile_account"
        else pytest.fail(f"unexpected tool: {tool_name}")
    )

    response = runtime.invoke_tool(
        reconciling.run_id,
        "reconcile_account",
        {"batch_hash": BATCH_HASH},
        request_id="req_failed_reconciliation_typed_data",
    )

    incident = runtime.get_run(reconciling.run_id)
    assert response.ok is False
    assert isinstance(response.data, ReconciliationOutput)
    assert response.data.stop_signal.required is True
    assert incident.state is AgentRunState.INCIDENT
    assert incident.pending_call is None
    assert all(
        artifact.kind is not AgentArtifactKind.RECONCILIATION for artifact in incident.artifacts
    )
    assert handler.calls == []
    assert replay_agent_run(incident.events) == incident


def test_expired_draft_closes_unsubmitted_execution_before_catalog() -> None:
    runtime, repository, registry, clock = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(seconds=1),
    )

    clock.value = NOW + timedelta(seconds=1)
    assert runtime.catalog(executing.run_id) == ()

    expired = runtime.get_run(executing.run_id)
    assert expired.state is AgentRunState.EXPIRED
    assert expired.terminal_reason == "ORDER_DRAFT_EXPIRED_BEFORE_EXECUTION"
    assert registry.catalog_contexts == []
    assert registry.invoke_calls == []
    assert replay_agent_run(expired.events) == expired


def test_expired_draft_with_pending_execution_write_enters_incident_before_invoke() -> None:
    runtime, repository, registry, clock = _runtime(RuntimeMode.PAPER)
    executing = _replayable_paper_executing_run(
        runtime,
        repository,
        draft_expires_at=NOW + timedelta(seconds=1),
    )
    pending = AgentPendingToolCall(
        request_id="req_expiring_pending_submit",
        tool_name="submit_paper_orders",
        argument_hash="a" * 64,
        idempotency_key_hash="b" * 64,
        batch_hash=BATCH_HASH,
        started_at=executing.updated_at,
        effect=ToolEffect.PAPER_EXECUTION_WRITE,
    )
    reserved = runtime._append_event(
        executing,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_after=AgentRunState.EXECUTING,
        reason="TOOL_CALL_RESERVED",
        occurred_at=executing.updated_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        idempotency_key_hash=pending.idempotency_key_hash,
        batch_hash=pending.batch_hash,
        pending_call=pending,
        tool_call_count=executing.tool_call_count + 1,
    )
    staged = repository.save(reserved, expected_revision=executing.revision)

    clock.value = NOW + timedelta(seconds=1)
    with pytest.raises(AgentRunTerminal):
        runtime.invoke_tool(
            staged.run_id,
            "submit_paper_orders",
            {"batch_hash": BATCH_HASH, "idempotency_key": "pending-submit"},
            request_id="req_after_pending_expiry",
        )

    incident = runtime.get_run(staged.run_id)
    assert incident.state is AgentRunState.INCIDENT
    assert incident.pending_call is None
    assert registry.invoke_calls == []
    assert replay_agent_run(incident.events) == incident


def test_external_approval_expiry_narrows_deadline_and_closes_before_catalog() -> None:
    runtime, repository, registry, clock = _runtime(RuntimeMode.LIVE_ASSISTED)
    pending = _replayable_live_pending_run(runtime, repository)
    approval = AgentExternalApproval(
        approval_id="approval-short-window",
        approval_hash="a" * 64,
        decision_id=DECISION_ID,
        batch_hash=BATCH_HASH,
        approved_at=NOW,
        expires_at=NOW + timedelta(seconds=1),
    )
    executing = runtime.record_external_approval(pending.run_id, approval)
    assert executing.approval_deadline == approval.expires_at

    clock.value = approval.expires_at
    assert runtime.catalog(executing.run_id) == ()

    expired = runtime.get_run(executing.run_id)
    assert expired.state is AgentRunState.EXPIRED
    assert registry.catalog_contexts == []
    assert registry.invoke_calls == []
    assert replay_agent_run(expired.events) == expired


def test_wrong_reconciliation_batch_records_replayable_incident_and_attempt() -> None:
    runtime, repository, registry, _ = _runtime(RuntimeMode.PAPER)
    reconciling = _replayable_paper_reconciling_run(runtime, repository)

    with pytest.raises(AgentToolStateDenied, match="batch_hash does not match"):
        runtime.invoke_tool(
            reconciling.run_id,
            "reconcile_account",
            {"batch_hash": "f" * 64},
            request_id="req_wrong_reconciliation_batch",
        )

    incident = runtime.get_run(reconciling.run_id)
    assert incident.state is AgentRunState.INCIDENT
    assert incident.reconciliation_attempt_count == 1
    assert incident.tool_call_count == reconciling.tool_call_count + 1
    assert registry.invoke_calls == []
    assert replay_agent_run(incident.events) == incident


@pytest.mark.parametrize("handler_raises", [False, True])
def test_normal_handler_crossing_deadline_closes_with_replayable_timeout(
    handler_raises: bool,
) -> None:
    clock = _Clock(NOW)
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH, clock=clock)
    run = _start(
        runtime,
        RuntimeMode.RESEARCH,
        AgentRunGoal.MARKET_RESEARCH,
        limits=AgentRunLimits(timeout_seconds=1),
    )
    registry.on_invoke = lambda: setattr(clock, "value", run.deadline)
    if handler_raises:
        registry.invoke_error = ToolExecutionFailed("handler crossed deadline")
        with pytest.raises(ToolExecutionFailed):
            runtime.invoke_tool(
                run.run_id,
                "get_market_snapshot",
                {},
                request_id="req_cross_deadline_error",
            )
    else:
        response = runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {},
            request_id="req_cross_deadline_success",
        )
        assert response.ok is True

    timed_out = runtime.get_run(run.run_id)
    assert timed_out.state is AgentRunState.TIMED_OUT
    assert timed_out.events[-1].kind is AgentRunEventKind.TIMEOUT_RECORDED
    assert timed_out.events[-1].trigger is AgentRunTrigger.TIMEOUT
    assert replay_agent_run(timed_out.events) == timed_out


def test_failed_response_with_typed_data_never_persists_artifact_and_replays() -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    registry.response_error_code = ErrorCode.DATA_INVALID
    registry.response_error_with_data = True

    response = runtime.invoke_tool(
        run.run_id,
        "get_market_snapshot",
        {},
        request_id="req_failed_typed_data",
    )

    result = runtime.get_run(run.run_id)
    assert response.ok is False
    assert response.data is not None
    assert result.state is AgentRunState.DATA_INVALID
    assert result.artifacts == ()
    assert all(event.artifact is None for event in result.events)
    assert replay_agent_run(result.events) == result


@pytest.mark.parametrize("drift", ["as_of", "provenance_data_version"])
def test_response_envelope_lineage_drift_fails_and_clears_pending(drift: str) -> None:
    runtime, _, registry, _ = _runtime(RuntimeMode.RESEARCH)
    run = _start(runtime, RuntimeMode.RESEARCH, AgentRunGoal.MARKET_RESEARCH)
    decision = _decision(RuntimeMode.RESEARCH)

    def drifted_response(
        tool_name: str,
        context: ToolCallContext,
    ) -> ToolResponse[MarketSnapshotOutput]:
        assert tool_name == "get_market_snapshot"
        data = MarketSnapshotOutput(
            decision_id=decision.decision_id,
            as_of=decision.as_of,
            data_version=decision.data_version,
            market="CN_A",
            data_content_hash=decision.data_content_hash,
            created_at=decision.as_of.isoformat(),
            total_file_count=1,
            returned_file_count=0,
            total_row_count=0,
            files_truncated=True,
            files=(),
        )
        response = ToolResponse[MarketSnapshotOutput](
            ok=True,
            request_id=context.request_id,
            decision_id=decision.decision_id,
            as_of=decision.as_of,
            data=data,
            provenance=Provenance(
                service="agent-runtime-test",
                version="1.0.0",
                data_version=decision.data_version,
            ),
        )
        if drift == "as_of":
            return response.model_copy(update={"as_of": decision.as_of + timedelta(seconds=1)})
        return response.model_copy(
            update={
                "provenance": response.provenance.model_copy(
                    update={"data_version": "drifted_data_version"}
                )
            }
        )

    registry.response_factory = drifted_response

    with pytest.raises(AgentRuntimeOutputInvalid):
        runtime.invoke_tool(
            run.run_id,
            "get_market_snapshot",
            {},
            request_id=f"req_envelope_drift_{drift}",
        )

    failed = runtime.get_run(run.run_id)
    assert failed.state is AgentRunState.FAILED
    assert failed.pending_call is None
    assert failed.artifacts == ()
    assert len(registry.invoke_calls) == 1
    assert replay_agent_run(failed.events) == failed
