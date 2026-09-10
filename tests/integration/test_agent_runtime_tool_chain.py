"""Real PAPER-chain acceptance test for the explicit Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from tests.unit import test_agent_portfolio_execution_tools as portfolio_support
from tests.unit import test_agent_research_service as research_service_support
from tests.unit import test_agent_research_tools as research_tool_support
from tests.unit import test_order_draft_generator as draft_support
from tests.unit import test_reconciliation_engine as reconciliation_support

from quant_agent.agent.runtime import (
    AgentArtifactKind,
    AgentRunGoal,
    AgentRunLimits,
    AgentRunState,
    AgentRuntime,
    InMemoryAgentRunRepository,
    replay_agent_run,
)
from quant_agent.agent.tools.contracts import ToolCallContext, ToolDescriptor
from quant_agent.agent.tools.portfolio_execution.inputs import ReconciliationInputs
from quant_agent.agent.tools.registry import AgentToolRegistry
from quant_agent.agent.tools.research.contracts import (
    CandidateRankingOutput,
    MarketDataQualityOutput,
    MarketSnapshotOutput,
)
from quant_agent.config import RuntimeMode
from quant_agent.core.responses import Provenance, ToolResponse
from quant_agent.execution.paper import PaperExecutionReceipt
from quant_agent.reconciliation import (
    ReconciliationPolicy,
    observed_evidence_from_paper_receipt,
)


@dataclass
class _Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


class _BootstrappedPaperRegistry:
    """Add deterministic bootstrap responses to the real P6-T04 registry."""

    runtime_mode = RuntimeMode.PAPER

    def __init__(
        self,
        registry: AgentToolRegistry,
        candidate: CandidateRankingOutput,
        fixture: portfolio_support._Fixture,
    ) -> None:
        self._registry = registry
        self._candidate = candidate
        self._fixture = fixture

    def catalog(self, context: ToolCallContext) -> tuple[ToolDescriptor, ...]:
        return self._registry.catalog(context)

    def invoke(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, object],
        context: ToolCallContext,
    ) -> object:
        decision = self._fixture.decision
        if tool_name == "get_market_snapshot":
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
        elif tool_name == "validate_market_data":
            data = MarketDataQualityOutput(
                decision_id=decision.decision_id,
                as_of=decision.as_of,
                data_version=decision.data_version,
                data_content_hash=decision.data_content_hash,
                observed_at=decision.as_of,
                qualified=True,
                total_issue_count=0,
                blocking_issue_count=0,
                returned_issue_count=0,
                issues_truncated=False,
                issues=(),
            )
        elif tool_name == "rank_stock_candidates":
            data = self._candidate
        else:
            return self._registry.invoke(tool_name, raw_arguments, context)
        return ToolResponse[object](
            ok=True,
            request_id=context.request_id,
            decision_id=decision.decision_id,
            as_of=decision.as_of,
            data=data,
            provenance=Provenance(
                service="runtime-bootstrap-test",
                version="1",
                data_version=decision.data_version,
            ),
        ).validate_consistency()


def _candidate_for_paper(
    fixture: portfolio_support._Fixture,
) -> CandidateRankingOutput:
    research_registry, _ = research_tool_support._registry(research_service_support._source())
    response = research_registry.invoke(
        "rank_stock_candidates",
        {"limit": 20, "include_excluded": True},
        research_tool_support._call_context("runtime-candidate-seed"),
    )
    assert type(response.data) is CandidateRankingOutput
    decision = fixture.decision
    return response.data.model_copy(
        update={
            "decision_id": decision.decision_id,
            "as_of": decision.as_of,
            "data_version": decision.data_version,
            "account_id": decision.account_id,
            "account_snapshot_id": decision.account_snapshot_id,
            "account_snapshot_hash": decision.account_snapshot_hash,
        }
    )


def test_runtime_completes_real_paper_execution_and_mandatory_reconciliation() -> None:
    fixture = portfolio_support._fixture()
    portfolio_registry, _ = portfolio_support._registry(fixture)
    registry = _BootstrappedPaperRegistry(
        portfolio_registry,
        _candidate_for_paper(fixture),
        fixture,
    )
    clock = _Clock(draft_support.DRAFT_AS_OF)
    runtime = AgentRuntime(
        registry=registry,
        repository=InMemoryAgentRunRepository(),
        clock=clock,
    )
    run = runtime.start_run(
        decision_snapshot=fixture.decision,
        goal=AgentRunGoal.PAPER_EXECUTION,
        limits=AgentRunLimits(timeout_seconds=300),
        run_id="runtime-real-paper-chain",
    )

    calls: tuple[tuple[str, dict[str, object]], ...] = (
        ("get_market_snapshot", {}),
        ("validate_market_data", {}),
        ("rank_stock_candidates", {"limit": 20, "include_excluded": True}),
        ("get_portfolio_snapshot", {"max_positions": 1}),
        ("build_target_portfolio", {"max_lines": 2, "max_adjustments": 1}),
        ("check_portfolio_risk", {"max_findings": 50}),
        ("create_order_draft", {"idempotency_key": "runtime-draft", "max_lines": 2}),
    )
    for index, (tool_name, arguments) in enumerate(calls):
        response = runtime.invoke_tool(
            run.run_id,
            tool_name,
            arguments,
            request_id=f"runtime-step-{index}",
        )
        assert response.ok is True

    run = runtime.get_run(run.run_id)
    assert run.state is AgentRunState.EXECUTING
    draft_artifact = next(
        item for item in run.artifacts if item.kind is AgentArtifactKind.ORDER_DRAFT
    )

    clock.value = draft_support.DRAFT_AS_OF + timedelta(minutes=1)
    submitted = runtime.invoke_tool(
        run.run_id,
        "submit_paper_orders",
        {
            "idempotency_key": "runtime-paper",
            "batch_hash": draft_artifact.content_hash,
            "max_orders": 500,
            "max_fills": 1_000,
        },
        request_id="runtime-submit",
    )
    assert submitted.ok is True
    assert runtime.get_run(run.run_id).state is AgentRunState.RECONCILING

    # Exact transport replay must not submit the same side effect again.
    assert (
        runtime.invoke_tool(
            run.run_id,
            "submit_paper_orders",
            {
                "idempotency_key": "runtime-paper",
                "batch_hash": draft_artifact.content_hash,
                "max_orders": 500,
                "max_fills": 1_000,
            },
            request_id="runtime-submit",
        )
        is submitted
    )

    assert fixture.repository is not None
    receipt = fixture.repository.receipt_by_idempotency_key(
        fixture.account.account_id,
        "runtime-paper",
    )
    assert type(receipt) is PaperExecutionReceipt
    observed_snapshot = reconciliation_support._snapshot(receipt)
    evidence_available_at = receipt.processed_at + timedelta(seconds=1)
    observed = observed_evidence_from_paper_receipt(
        receipt=receipt,
        account_snapshot=observed_snapshot,
        available_at=evidence_available_at,
        source_cursor="runtime-integration-test",
    )
    fixture.source.reconciliation_inputs = ReconciliationInputs(
        receipt=receipt,
        observed=observed,
        policy=ReconciliationPolicy(),
    )

    # This is after the five-minute model deadline. A submitted order may not
    # become TIMED_OUT; the independent reconciliation budget remains usable.
    clock.value = max(
        evidence_available_at + timedelta(seconds=1),
        run.deadline + timedelta(seconds=1),
    )
    assert clock.value >= run.deadline
    reconciled = runtime.invoke_tool(
        run.run_id,
        "reconcile_account",
        {"batch_hash": draft_artifact.content_hash},
        request_id="runtime-reconcile",
    )
    assert reconciled.ok is True

    completed = runtime.get_run(run.run_id)
    assert completed.state is AgentRunState.COMPLETED
    assert completed.reconciliation_attempt_count == 1
    assert completed.tool_call_count == 9
    assert completed.pending_call is None
    assert completed.terminal_reason == "RECONCILIATION_COMPLETED"
    assert completed.events[-1].state_after is AgentRunState.COMPLETED
    assert any(item.kind is AgentArtifactKind.EXECUTION_RECEIPT for item in completed.artifacts)
    assert any(item.kind is AgentArtifactKind.RECONCILIATION for item in completed.artifacts)
    assert replay_agent_run(completed.events) == completed
