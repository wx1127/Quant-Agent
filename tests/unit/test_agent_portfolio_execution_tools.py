"""Acceptance tests for the decision-bound portfolio/execution Agent tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from threading import Barrier

import pytest
from tests.unit import test_order_draft_generator as draft_support
from tests.unit import test_paper_repository_service as paper_support
from tests.unit import test_reconciliation_engine as reconciliation_support

from quant_agent.agent.snapshots import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools.contracts import (
    ToolCallContext,
    ToolCapability,
    ToolDenialReason,
    ToolPermissionDenied,
)
from quant_agent.agent.tools.portfolio_execution.contracts import (
    OrderDraftOutput,
    PaperExecutionOutput,
    PortfolioRiskOutput,
    PortfolioSnapshotOutput,
    ReconciliationOutput,
    TargetPortfolioOutput,
)
from quant_agent.agent.tools.portfolio_execution.gateway import (
    ServiceBackedPaperOrderGateway,
)
from quant_agent.agent.tools.portfolio_execution.inputs import (
    PortfolioExecutionInputInvalid,
    PortfolioExecutionInputUnavailable,
    ReconciliationInputs,
)
from quant_agent.agent.tools.portfolio_execution.repository import (
    DraftArtifactNotFound,
    InMemoryDraftArtifactStore,
)
from quant_agent.agent.tools.portfolio_execution.service import PortfolioExecutionPipeline
from quant_agent.agent.tools.portfolio_execution.toolset import (
    PORTFOLIO_EXECUTION_TOOL_VERSION,
    build_portfolio_execution_tools,
)
from quant_agent.agent.tools.registry import AgentToolRegistry, AgentToolRegistryBuilder
from quant_agent.backtest import TradableInstrumentType
from quant_agent.backtest.cn_market import CNSlippageModelBook, MarketSessionState
from quant_agent.config import RuntimeMode
from quant_agent.execution.order_drafts import OrderDraftBatch, OrderDraftGenerator
from quant_agent.execution.paper import (
    InMemoryPaperRepository,
    PaperExecutionEngine,
    PaperExecutionReceipt,
    PaperExecutionService,
)
from quant_agent.observability.audit import AuditEvent
from quant_agent.portfolio import (
    AccountSnapshot,
    PITIndustryClassification,
    PortfolioBuildRequest,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    TargetPortfolio,
    TargetPortfolioBuilder,
)
from quant_agent.reconciliation import (
    ReconciliationPolicy,
    ReconciliationStatus,
    observed_evidence_from_paper_receipt,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import (
    PortfolioRiskContext,
    PortfolioRiskEvaluator,
    PortfolioRiskPolicy,
    RiskCheckStatus,
)

EXPECTED_NAMES = {
    "build_target_portfolio",
    "check_portfolio_risk",
    "create_order_draft",
    "get_order_draft",
    "get_portfolio_snapshot",
    "reconcile_account",
    "submit_paper_orders",
}
ALL_CAPABILITIES = frozenset(
    {
        ToolCapability.ACCOUNT_READ,
        ToolCapability.PORTFOLIO_BUILD,
        ToolCapability.RISK_CHECK,
        ToolCapability.ORDER_DRAFT_CREATE,
        ToolCapability.ORDER_DRAFT_READ,
        ToolCapability.PAPER_ORDER_SUBMIT,
        ToolCapability.ACCOUNT_RECONCILE,
    }
)
CAPABILITY_BY_NAME = {
    "get_portfolio_snapshot": ToolCapability.ACCOUNT_READ,
    "build_target_portfolio": ToolCapability.PORTFOLIO_BUILD,
    "check_portfolio_risk": ToolCapability.RISK_CHECK,
    "create_order_draft": ToolCapability.ORDER_DRAFT_CREATE,
    "get_order_draft": ToolCapability.ORDER_DRAFT_READ,
    "submit_paper_orders": ToolCapability.PAPER_ORDER_SUBMIT,
    "reconcile_account": ToolCapability.ACCOUNT_RECONCILE,
}


class _Resolver:
    def __init__(self, decision: DecisionSnapshot) -> None:
        self.decision = decision

    def resolve(self, decision_id: str) -> DecisionSnapshot:
        assert decision_id == self.decision.decision_id
        return self.decision


class _AuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


class _Source:
    def __init__(
        self,
        *,
        account: AccountSnapshot,
        sleeves: tuple[StrategySleeve, ...],
        context: PortfolioRiskContext,
        states: tuple[MarketSessionState, ...],
    ) -> None:
        self.account = account
        self.sleeves = sleeves
        self.context = context
        self.states = states
        self.fail_reads = False
        self.reconciliation_inputs: ReconciliationInputs | None = None

    def _available(self) -> None:
        if self.fail_reads:
            raise PortfolioExecutionInputUnavailable("test source is unavailable")

    def load_account_snapshot(self, decision: DecisionSnapshot) -> AccountSnapshot:
        self._available()
        assert decision.account_id == self.account.account_id
        return self.account

    def load_strategy_sleeves(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[StrategySleeve, ...]:
        self._available()
        assert decision.data_version == self.account.data_version
        return self.sleeves

    def load_industry_classifications(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[PITIndustryClassification, ...]:
        self._available()
        assert decision.account_id == self.account.account_id
        return ()

    def load_risk_context(self, decision: DecisionSnapshot) -> PortfolioRiskContext:
        self._available()
        assert decision.account_snapshot_hash == self.context.account_snapshot_hash
        return self.context

    def load_market_states(
        self,
        decision: DecisionSnapshot,
        target: TargetPortfolio,
    ) -> tuple[MarketSessionState, ...]:
        self._available()
        assert target.decision_id == decision.decision_id
        return self.states

    def load_reconciliation_inputs(
        self,
        decision: DecisionSnapshot,
        draft: OrderDraftBatch,
    ) -> ReconciliationInputs:
        self._available()
        assert draft.decision_id == decision.decision_id
        if self.reconciliation_inputs is None:
            raise PortfolioExecutionInputUnavailable(
                "independent reconciliation evidence is not ready"
            )
        return self.reconciliation_inputs


class _BarrierSource(_Source):
    """Force two draft deliveries past their initial empty-key lookup."""

    def __init__(
        self,
        *,
        account: AccountSnapshot,
        sleeves: tuple[StrategySleeve, ...],
        context: PortfolioRiskContext,
        states: tuple[MarketSessionState, ...],
    ) -> None:
        super().__init__(
            account=account,
            sleeves=sleeves,
            context=context,
            states=states,
        )
        self._account_barrier = Barrier(2)

    def load_account_snapshot(self, decision: DecisionSnapshot) -> AccountSnapshot:
        self._account_barrier.wait(timeout=10)
        return super().load_account_snapshot(decision)


@dataclass(frozen=True, slots=True)
class _Fixture:
    account: AccountSnapshot
    sleeves: tuple[StrategySleeve, ...]
    decision: DecisionSnapshot
    policy: PortfolioRiskPolicy
    target: TargetPortfolio
    source: _Source
    builder: TargetPortfolioBuilder
    risk_evaluator: PortfolioRiskEvaluator
    generator: OrderDraftGenerator
    store: InMemoryDraftArtifactStore
    repository: InMemoryPaperRepository | None
    gateway: ServiceBackedPaperOrderGateway | None


def _digest(label: str) -> str:
    return stable_hash({"portfolio-execution-agent-test": label})


def _sleeves(account: AccountSnapshot) -> tuple[StrategySleeve, ...]:
    etf_targets = tuple(
        SleeveTarget(
            instrument_id=instrument_id,
            instrument_type=TradableInstrumentType.ETF,
            local_target_weight=weight,
            reason=f"target {instrument_id} for Agent portfolio tool tests",
        )
        for instrument_id, weight in sorted(draft_support.DEFAULT_LOCAL_TARGETS.items())
    )
    values = (
        (
            "etf-core",
            StrategySleeveKind.ETF_CORE,
            "etf-rotation",
            etf_targets,
        ),
        (
            "stock-enhancement",
            StrategySleeveKind.STOCK_ENHANCEMENT,
            "mainline-leader",
            (),
        ),
    )
    return tuple(
        StrategySleeve.build(
            sleeve_id=sleeve_id,
            kind=kind,
            as_of=account.as_of,
            data_version=account.data_version,
            strategy_name=strategy_name,
            strategy_version="strategy-v1",
            strategy_config_hash=_digest(f"{strategy_name}-config"),
            source_result_hash=_digest(f"{strategy_name}-result"),
            targets=targets,
        )
        for sleeve_id, kind, strategy_name, targets in values
    )


def _decision(
    account: AccountSnapshot,
    sleeves: tuple[StrategySleeve, ...],
    policy: PortfolioRiskPolicy,
    *,
    decision_id: str = "portfolio-execution-decision-1",
    account_id: str | None = None,
) -> DecisionSnapshot:
    refs = tuple(
        StrategySnapshotRef(
            strategy_name=item.strategy_name,
            strategy_version=item.strategy_version,
            config_hash=item.strategy_config_hash,
            parameter_version="registered-parameters-v1",
            parameter_hash=_digest(f"{item.strategy_name}-parameters"),
            registered_at=account.as_of - timedelta(days=1),
        )
        for item in sleeves
    )
    return DecisionSnapshot.build(
        decision_id=decision_id,
        mode=account.runtime_mode,
        market="CN_A",
        as_of=account.as_of,
        data_version=account.data_version,
        data_content_hash=_digest("market-data"),
        strategy_refs=refs,
        risk_policy_version=policy.version,
        risk_policy_hash=policy.policy_hash,
        account_id=account_id or account.account_id,
        account_snapshot_id=account.snapshot_id,
        account_snapshot_hash=account.content_hash,
        code_commit="1" * 40,
        code_artifact_hash=_digest("code-artifact"),
        agent_version="agent-v1",
        model_version="model-v1",
    )


def _risk_context(account: AccountSnapshot) -> PortfolioRiskContext:
    return PortfolioRiskContext.build(
        account_snapshot_hash=account.content_hash,
        as_of=account.as_of,
        valuation_version="account-mark-v1",
        current_equity=account.total_equity,
        equity_high_watermark=account.total_equity,
        high_watermark_at=account.as_of - timedelta(days=1),
        drawdown_source_hash=_digest("drawdown"),
        regime_risk_budget=Decimal("0.90"),
        regime_result_hash=_digest("regime"),
    )


def _fixture(
    *,
    policy: PortfolioRiskPolicy | None = None,
    mode: RuntimeMode = RuntimeMode.PAPER,
    with_gateway: bool = True,
    paper_repository: InMemoryPaperRepository | None = None,
) -> _Fixture:
    account = draft_support._account(runtime_mode=mode)
    sleeves = _sleeves(account)
    selected_policy = policy or draft_support._risk_policy()
    decision = _decision(account, sleeves, selected_policy)
    builder = TargetPortfolioBuilder(draft_support._builder_config())
    target = builder.build(
        request=PortfolioBuildRequest(
            decision_id=decision.decision_id,
            as_of=decision.as_of,
            data_version=decision.data_version,
            account_snapshot_id=decision.account_snapshot_id,
            account_snapshot_hash=decision.account_snapshot_hash,
        ),
        account=account,
        sleeves=sleeves,
        classifications=(),
    )
    states = draft_support._market_states(target)
    next_trading_day = draft_support.TRADING_DAY + timedelta(days=3)
    rule = replace(
        draft_support._market_rule(states),
        trading_calendar=(draft_support.TRADING_DAY, next_trading_day),
    )
    generator = draft_support._generator(market_rules=(rule,))
    source = _Source(
        account=account,
        sleeves=sleeves,
        context=_risk_context(account),
        states=states,
    )
    store = InMemoryDraftArtifactStore()
    repository: InMemoryPaperRepository | None = None
    gateway: ServiceBackedPaperOrderGateway | None = None
    if with_gateway:
        assert mode is RuntimeMode.PAPER
        engine = PaperExecutionEngine(
            market_rules=(rule,),
            fee_rule_book=draft_support._fee_rule_book(),
            slippage_model_book=CNSlippageModelBook((draft_support._slippage_model(),)),
        )
        repository = paper_repository or InMemoryPaperRepository()
        opener = PaperExecutionService(
            engine=engine,
            repository=repository,
            new_order_gate=paper_support._TEST_ALLOW_GATE,
        )
        opened = opener.open_account(snapshot=account)
        assert opened.source_snapshot_hash == decision.account_snapshot_hash
        gateway = ServiceBackedPaperOrderGateway(
            engine=engine,
            repository=repository,
            new_order_gate=paper_support._TEST_ALLOW_GATE,
            clock=lambda: draft_support.DRAFT_AS_OF + timedelta(minutes=10),
        )
    return _Fixture(
        account=account,
        sleeves=sleeves,
        decision=decision,
        policy=selected_policy,
        target=target,
        source=source,
        builder=builder,
        risk_evaluator=PortfolioRiskEvaluator(selected_policy),
        generator=generator,
        store=store,
        repository=repository,
        gateway=gateway,
    )


def _tools(fixture: _Fixture) -> tuple[object, ...]:
    return build_portfolio_execution_tools(
        source=fixture.source,
        artifact_store=fixture.store,
        draft_generator=fixture.generator,
        builder=fixture.builder,
        risk_evaluator=fixture.risk_evaluator,
        paper_gateway=fixture.gateway,
    )


def _registry(
    fixture: _Fixture,
    *,
    capabilities: frozenset[ToolCapability] = ALL_CAPABILITIES,
) -> tuple[AgentToolRegistry, _AuditSink]:
    audit = _AuditSink()
    builder = AgentToolRegistryBuilder(
        runtime_mode=fixture.decision.mode,
        principal_id="portfolio-execution-agent",
        granted_capabilities=capabilities,
        granted_account_ids=frozenset({fixture.decision.account_id}),
        decision_resolver=_Resolver(fixture.decision),
        audit_sink=audit,
    )
    for tool in _tools(fixture):
        builder.register(tool)  # type: ignore[arg-type]
    return builder.build(), audit


def _call_context(
    fixture: _Fixture,
    request_id: str,
    *,
    requested_at: datetime | None = None,
) -> ToolCallContext:
    return ToolCallContext(
        request_id=request_id,
        decision_id=fixture.decision.decision_id,
        requested_at=requested_at or draft_support.DRAFT_AS_OF,
    )


def _pipeline(fixture: _Fixture) -> PortfolioExecutionPipeline:
    return PortfolioExecutionPipeline(
        source=fixture.source,
        artifact_store=fixture.store,
        draft_generator=fixture.generator,
        builder=fixture.builder,
        risk_evaluator=fixture.risk_evaluator,
        paper_gateway=fixture.gateway,
    )


def test_all_seven_tools_register_and_complete_the_real_paper_chain() -> None:
    fixture = _fixture()
    registry, audit = _registry(fixture)
    catalog_context = _call_context(fixture, "portfolio-catalog-1")
    descriptors = registry.catalog(catalog_context)

    assert {item.name for item in descriptors} == EXPECTED_NAMES
    assert all(item.version == PORTFOLIO_EXECUTION_TOOL_VERSION for item in descriptors)
    assert {item.name: item.capability for item in descriptors} == CAPABILITY_BY_NAME

    snapshot = registry.invoke(
        "get_portfolio_snapshot",
        {"max_positions": 1},
        _call_context(fixture, "portfolio-snapshot-1"),
    )
    assert snapshot.ok is True
    assert type(snapshot.data) is PortfolioSnapshotOutput
    assert snapshot.data.content_hash == fixture.account.content_hash
    assert snapshot.data.total_position_count == len(fixture.account.positions)
    assert snapshot.data.returned_position_count == 1
    assert snapshot.data.positions_truncated is True

    target = registry.invoke(
        "build_target_portfolio",
        {"max_lines": 2, "max_adjustments": 1},
        _call_context(fixture, "portfolio-target-1"),
    )
    assert target.ok is True
    assert type(target.data) is TargetPortfolioOutput
    assert target.data.result_hash == fixture.target.result_hash
    assert target.data.input_hash == fixture.target.input_hash
    assert target.data.returned_line_count == 2
    assert target.data.lines_truncated is True

    risk = registry.invoke(
        "check_portfolio_risk",
        {"max_findings": 50},
        _call_context(fixture, "portfolio-risk-1"),
    )
    assert risk.ok is True
    assert type(risk.data) is PortfolioRiskOutput
    assert risk.data.request.portfolio_proposal_hash == fixture.target.result_hash
    assert risk.data.request.policy_hash == fixture.policy.policy_hash
    assert risk.data.status in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}

    created = registry.invoke(
        "create_order_draft",
        {"idempotency_key": "draft-delivery-1", "max_lines": 2},
        _call_context(fixture, "portfolio-draft-create-1"),
    )
    assert created.ok is True
    assert type(created.data) is OrderDraftOutput
    assert created.data.portfolio_proposal_hash == fixture.target.result_hash
    assert created.data.is_executable is False
    assert created.data.requires_human_approval is True
    stored = fixture.store.get(fixture.account.account_id, created.data.batch_hash)
    assert stored.idempotency_key == "draft-delivery-1"
    assert stored.draft.batch_hash == created.data.batch_hash

    loaded = registry.invoke(
        "get_order_draft",
        {"batch_hash": created.data.batch_hash, "max_lines": 500},
        _call_context(fixture, "portfolio-draft-read-1"),
    )
    assert loaded.ok is True
    assert type(loaded.data) is OrderDraftOutput
    assert loaded.data.batch_hash == created.data.batch_hash
    assert tuple(item.line_hash for item in loaded.data.lines) == tuple(
        item.line_hash for item in stored.draft.lines
    )
    assert loaded.data.total_line_count == len(stored.draft.lines)

    submitted_at = draft_support.DRAFT_AS_OF + timedelta(minutes=1)
    submitted = registry.invoke(
        "submit_paper_orders",
        {
            "idempotency_key": "paper-delivery-1",
            "batch_hash": created.data.batch_hash,
            "max_orders": 500,
            "max_fills": 1000,
        },
        _call_context(
            fixture,
            "portfolio-paper-submit-1",
            requested_at=submitted_at,
        ),
    )
    assert submitted.ok is True
    assert type(submitted.data) is PaperExecutionOutput
    assert submitted.data.batch_hash == created.data.batch_hash
    assert submitted.data.total_order_count == len(stored.draft.lines)
    assert submitted.data.account_before_hash == submitted.data.account_before.state_hash
    assert submitted.data.account_after_hash == submitted.data.account_after.state_hash
    assert fixture.repository is not None
    receipt = fixture.repository.receipt_by_idempotency_key(
        fixture.account.account_id,
        "paper-delivery-1",
    )
    assert type(receipt) is PaperExecutionReceipt
    assert receipt.receipt_hash == submitted.data.receipt_hash

    account_after_first_submit = fixture.repository.get_account(fixture.account.account_id)
    retried = registry.invoke(
        "submit_paper_orders",
        {
            "idempotency_key": "paper-delivery-1",
            "batch_hash": created.data.batch_hash,
        },
        _call_context(
            fixture,
            "portfolio-paper-submit-retry",
            requested_at=submitted_at + timedelta(minutes=1),
        ),
    )
    assert retried.ok is True
    assert type(retried.data) is PaperExecutionOutput
    assert retried.data.receipt_hash == submitted.data.receipt_hash
    assert (
        fixture.repository.get_account(fixture.account.account_id).state_hash
        == account_after_first_submit.state_hash
    )

    observed_snapshot = reconciliation_support._snapshot(receipt)
    evidence_available_at = receipt.processed_at + timedelta(seconds=1)
    observed = observed_evidence_from_paper_receipt(
        receipt=receipt,
        account_snapshot=observed_snapshot,
        available_at=evidence_available_at,
        source_cursor="portfolio-agent-test-cursor",
    )
    fixture.source.reconciliation_inputs = ReconciliationInputs(
        receipt=receipt,
        observed=observed,
        policy=ReconciliationPolicy(),
    )
    reconciled = registry.invoke(
        "reconcile_account",
        {"batch_hash": created.data.batch_hash},
        _call_context(
            fixture,
            "portfolio-reconcile-1",
            requested_at=evidence_available_at + timedelta(seconds=1),
        ),
    )
    assert reconciled.ok is True
    assert type(reconciled.data) is ReconciliationOutput
    assert reconciled.as_of == fixture.decision.as_of
    assert reconciled.data.as_of == fixture.decision.as_of
    assert reconciled.data.batch_hash == created.data.batch_hash
    assert reconciled.data.receipt_hash == receipt.receipt_hash
    assert reconciled.data.observed_snapshot_hash == observed_snapshot.content_hash
    assert reconciled.data.status is ReconciliationStatus.MATCHED
    assert reconciled.data.stop_signal.required is False

    allowed = [item for item in audit.events if item.result == "ALLOWED"]
    assert len(allowed) == 8


def test_draft_replay_survives_source_outage_and_stays_account_and_decision_scoped() -> None:
    fixture = _fixture()
    registry, _ = _registry(fixture)
    first = registry.invoke(
        "create_order_draft",
        {"idempotency_key": "durable-draft-replay"},
        _call_context(fixture, "portfolio-draft-replay-1"),
    )
    assert type(first.data) is OrderDraftOutput

    fixture.source.fail_reads = True
    repeated = registry.invoke(
        "create_order_draft",
        {"idempotency_key": "durable-draft-replay"},
        _call_context(
            fixture,
            "portfolio-draft-replay-2",
            requested_at=draft_support.DRAFT_AS_OF + timedelta(minutes=5),
        ),
    )
    assert repeated.ok is True
    assert type(repeated.data) is OrderDraftOutput
    assert repeated.data.batch_hash == first.data.batch_hash
    assert repeated.data.draft_as_of == first.data.draft_as_of
    fixture.source.fail_reads = False

    other_decision = _decision(
        fixture.account,
        fixture.sleeves,
        fixture.policy,
        decision_id="portfolio-execution-decision-other",
    )
    with pytest.raises(PortfolioExecutionInputInvalid, match="frozen decision"):
        _pipeline(fixture).get_order_draft(
            other_decision,
            batch_hash=first.data.batch_hash,
        )

    other_account_decision = _decision(
        fixture.account,
        fixture.sleeves,
        fixture.policy,
        decision_id="portfolio-execution-decision-other-account",
        account_id="paper-account-other",
    )
    with pytest.raises(DraftArtifactNotFound, match="unknown account-scoped"):
        _pipeline(fixture).get_order_draft(
            other_account_decision,
            batch_hash=first.data.batch_hash,
        )


def test_risk_rejection_returns_evidence_and_cannot_persist_an_order_draft() -> None:
    rejecting_policy = draft_support._risk_policy(
        maximum_total_weight=Decimal("0.50"),
        maximum_etf_weight=Decimal("0.50"),
    )
    fixture = _fixture(policy=rejecting_policy)
    registry, _ = _registry(fixture)

    risk = registry.invoke(
        "check_portfolio_risk",
        {},
        _call_context(fixture, "portfolio-risk-reject-1"),
    )
    assert risk.ok is False
    assert type(risk.data) is PortfolioRiskOutput
    assert risk.data.status is RiskCheckStatus.REJECT
    assert risk.data.allows_execution is False
    assert risk.errors[0].code.value == "RISK_REJECTED"

    draft = registry.invoke(
        "create_order_draft",
        {"idempotency_key": "risk-rejected-draft"},
        _call_context(fixture, "portfolio-draft-reject-1"),
    )
    assert draft.ok is False
    assert draft.data is None
    assert draft.errors[0].code.value == "RISK_REJECTED"
    assert (
        fixture.store.by_idempotency_key(
            fixture.account.account_id,
            fixture.decision.decision_id,
            "risk-rejected-draft",
        )
        is None
    )


def test_paper_submit_is_hidden_without_its_capability_and_absent_from_live_composition() -> None:
    fixture = _fixture()
    capabilities = ALL_CAPABILITIES - {ToolCapability.PAPER_ORDER_SUBMIT}
    registry, _ = _registry(fixture, capabilities=capabilities)
    names = {item.name for item in registry.catalog(_call_context(fixture, "catalog-no-paper"))}
    assert names == EXPECTED_NAMES - {"submit_paper_orders"}

    with pytest.raises(ToolPermissionDenied) as denied:
        registry.invoke(
            "submit_paper_orders",
            {"idempotency_key": "not-authorized", "batch_hash": "f" * 64},
            _call_context(fixture, "paper-not-authorized"),
        )
    assert denied.value.reason is ToolDenialReason.CAPABILITY_NOT_GRANTED

    live = _fixture(mode=RuntimeMode.LIVE_ASSISTED, with_gateway=False)
    live_tools = _tools(live)
    assert {tool.name for tool in live_tools} == EXPECTED_NAMES - {"submit_paper_orders"}  # type: ignore[attr-defined]


def test_concurrent_duplicate_deliveries_replay_the_first_draft_and_receipt() -> None:
    fixture = _fixture(with_gateway=False)
    source = _BarrierSource(
        account=fixture.account,
        sleeves=fixture.sleeves,
        context=fixture.source.context,
        states=fixture.source.states,
    )
    store = InMemoryDraftArtifactStore()
    pipeline = PortfolioExecutionPipeline(
        source=source,
        artifact_store=store,
        draft_generator=fixture.generator,
        builder=fixture.builder,
        risk_evaluator=fixture.risk_evaluator,
    )

    def create(offset: int) -> OrderDraftBatch:
        return pipeline.create_order_draft(
            fixture.decision,
            idempotency_key="concurrent-draft-delivery",
            draft_as_of=draft_support.DRAFT_AS_OF + timedelta(minutes=offset),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        drafts = tuple(executor.map(create, (0, 1)))
    assert drafts[0].batch_hash == drafts[1].batch_hash
    assert drafts[0].draft_as_of == drafts[1].draft_as_of

    barrier_repository = paper_support._BarrierPaperRepository()
    paper_fixture = _fixture(paper_repository=barrier_repository)
    paper_pipeline = _pipeline(paper_fixture)
    paper_draft = paper_pipeline.create_order_draft(
        paper_fixture.decision,
        idempotency_key="paper-race-draft",
        draft_as_of=draft_support.DRAFT_AS_OF,
    )

    def submit(offset: int) -> PaperExecutionReceipt:
        return paper_pipeline.submit_paper_orders(
            paper_fixture.decision,
            request_id=f"paper-race-request-{offset}",
            idempotency_key="concurrent-paper-delivery",
            batch_hash=paper_draft.batch_hash,
            submitted_at=draft_support.DRAFT_AS_OF + timedelta(minutes=offset + 1),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(executor.map(submit, (0, 1)))
    assert receipts[0].receipt_hash == receipts[1].receipt_hash
    assert barrier_repository.get_account(
        paper_fixture.account.account_id
    ).processed_batch_hashes == (paper_draft.batch_hash,)
