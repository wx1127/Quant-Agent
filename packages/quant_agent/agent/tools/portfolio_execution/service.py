"""Deterministic orchestration for decision-bound portfolio/execution tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from pydantic import TypeAdapter, ValidationError

from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.backtest.cn_market import MarketSessionState
from quant_agent.execution.order_drafts import (
    OrderDraftBatch,
    OrderDraftGenerator,
    OrderDraftInputError,
)
from quant_agent.execution.paper import PaperExecutionReceipt
from quant_agent.portfolio import (
    AccountSnapshot,
    PITIndustryClassification,
    PortfolioBuildInputError,
    PortfolioBuildRequest,
    StrategySleeve,
    TargetPortfolio,
    TargetPortfolioBuilder,
)
from quant_agent.reconciliation import (
    ReconciliationEngine,
    ReconciliationRequest,
    ReconciliationResult,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import (
    PortfolioRiskContext,
    PortfolioRiskEvaluator,
    PortfolioRiskInputError,
    RiskCheckResult,
    build_portfolio_risk_request,
)

from .gateway import PaperOrderGateway
from .inputs import (
    PortfolioExecutionInputInvalid,
    PortfolioExecutionInputSource,
    PortfolioExecutionInputUnavailable,
    PortfolioRiskBlocked,
    ReconciliationInputs,
)
from .repository import DraftArtifact, DraftArtifactConflict, DraftArtifactStore

_MAX_SLEEVES = 32
_MAX_CLASSIFICATIONS = 20_000
_MAX_MARKET_STATES = 20_000


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _detach[T](value: object, expected: type[T], label: str) -> T:
    if type(value) is not expected:
        raise PortfolioExecutionInputInvalid(f"{label} has an unsupported type")
    try:
        adapter = TypeAdapter(expected)
        detached = adapter.validate_json(adapter.dump_json(value, warnings="error"), strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise PortfolioExecutionInputInvalid(f"{label} failed strict revalidation") from error
    if type(detached) is not expected:
        raise PortfolioExecutionInputInvalid(f"{label} failed exact-type revalidation")
    return detached


def _decision(value: object) -> DecisionSnapshot:
    if type(value) is not DecisionSnapshot:
        raise PortfolioExecutionInputInvalid(
            "portfolio pipeline requires an exact DecisionSnapshot"
        )
    try:
        return DecisionSnapshot.from_json(value.to_json())
    except (TypeError, ValueError) as error:
        raise PortfolioExecutionInputInvalid(
            "decision snapshot failed strict revalidation"
        ) from error


def _tuple(
    value: object,
    expected: type[object],
    label: str,
    *,
    maximum: int,
    allow_empty: bool,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise PortfolioExecutionInputInvalid(f"{label} must be an exact tuple")
    result = cast(tuple[object, ...], value)
    if (not allow_empty and not result) or len(result) > maximum:
        raise PortfolioExecutionInputInvalid(f"{label} has an invalid collection size")
    if any(type(item) is not expected for item in result):
        raise PortfolioExecutionInputInvalid(f"{label} contains an unsupported item type")
    return result


def _require_source(value: object) -> PortfolioExecutionInputSource:
    methods = (
        "load_account_snapshot",
        "load_strategy_sleeves",
        "load_industry_classifications",
        "load_risk_context",
        "load_market_states",
        "load_reconciliation_inputs",
    )
    if any(not callable(getattr(value, name, None)) for name in methods):
        raise TypeError("portfolio input source does not implement the complete protocol")
    return cast(PortfolioExecutionInputSource, value)


def _require_store(value: object) -> DraftArtifactStore:
    methods = ("by_idempotency_key", "put", "get")
    if any(not callable(getattr(value, name, None)) for name in methods):
        raise TypeError("draft artifact store does not implement the complete atomic protocol")
    return cast(DraftArtifactStore, value)


def _require_gateway(value: object | None) -> PaperOrderGateway | None:
    if value is None:
        return None
    if not callable(getattr(value, "submit", None)) or not isinstance(
        getattr(value, "version", None), str
    ):
        raise TypeError("paper gateway does not implement the complete trusted protocol")
    return cast(PaperOrderGateway, value)


class PortfolioExecutionPipeline:
    """Reuse P5 engines while enforcing P6 decision, ordering, and storage bounds."""

    __slots__ = (
        "_builder",
        "_builder_config_hash",
        "_draft_config_hash",
        "_draft_generator",
        "_gateway",
        "_reconciliation_engine",
        "_risk_evaluator",
        "_risk_policy_hash",
        "_source",
        "_store",
    )

    def __init__(
        self,
        *,
        source: PortfolioExecutionInputSource,
        artifact_store: DraftArtifactStore,
        draft_generator: OrderDraftGenerator,
        builder: TargetPortfolioBuilder | None = None,
        risk_evaluator: PortfolioRiskEvaluator | None = None,
        reconciliation_engine: ReconciliationEngine | None = None,
        paper_gateway: PaperOrderGateway | None = None,
    ) -> None:
        if type(draft_generator) is not OrderDraftGenerator:
            raise TypeError("portfolio pipeline requires an exact OrderDraftGenerator")
        selected_builder = builder or TargetPortfolioBuilder()
        selected_risk = risk_evaluator or PortfolioRiskEvaluator()
        selected_reconciliation = reconciliation_engine or ReconciliationEngine()
        if type(selected_builder) is not TargetPortfolioBuilder:
            raise TypeError("builder must be an exact TargetPortfolioBuilder")
        if type(selected_risk) is not PortfolioRiskEvaluator:
            raise TypeError("risk evaluator must be an exact PortfolioRiskEvaluator")
        if type(selected_reconciliation) is not ReconciliationEngine:
            raise TypeError("reconciliation engine must be an exact ReconciliationEngine")
        self._source = _require_source(source)
        self._store = _require_store(artifact_store)
        self._draft_generator = draft_generator
        self._builder = selected_builder
        self._risk_evaluator = selected_risk
        self._reconciliation_engine = selected_reconciliation
        self._gateway = _require_gateway(paper_gateway)
        self._builder_config_hash = selected_builder.config.config_hash
        self._risk_policy_hash = selected_risk.policy.policy_hash
        self._draft_config_hash = draft_generator.config.config_hash

    @property
    def builder_version(self) -> str:
        return self._builder.config.version

    @property
    def risk_version(self) -> str:
        return self._risk_evaluator.policy.version

    @property
    def draft_version(self) -> str:
        return self._draft_generator.config.version

    @property
    def paper_version(self) -> str | None:
        return self._gateway.version if self._gateway is not None else None

    def get_account_snapshot(self, decision: DecisionSnapshot) -> AccountSnapshot:
        frozen = _decision(decision)
        try:
            raw = self._source.load_account_snapshot(frozen)
        except PortfolioExecutionInputUnavailable:
            raise
        account = _detach(raw, AccountSnapshot, "account snapshot")
        if (
            account.account_id != frozen.account_id
            or account.snapshot_id != frozen.account_snapshot_id
            or account.content_hash != frozen.account_snapshot_hash
            or account.runtime_mode is not frozen.mode
            or _utc(account.as_of) != _utc(frozen.as_of)
            or account.data_version != frozen.data_version
        ):
            raise PortfolioExecutionInputInvalid(
                "account snapshot does not match the frozen decision"
            )
        return account

    def build_target_portfolio(self, decision: DecisionSnapshot) -> TargetPortfolio:
        frozen = _decision(decision)
        account = self.get_account_snapshot(frozen)
        return self._build_target(frozen, account)

    def check_portfolio_risk(
        self,
        decision: DecisionSnapshot,
        *,
        evaluated_at: datetime | None = None,
    ) -> RiskCheckResult:
        frozen = _decision(decision)
        account = self.get_account_snapshot(frozen)
        target = self._build_target(frozen, account)
        return self._check_risk(
            frozen,
            account,
            target,
            evaluated_at=evaluated_at or frozen.as_of,
        )

    def create_order_draft(
        self,
        decision: DecisionSnapshot,
        *,
        idempotency_key: str,
        draft_as_of: datetime,
    ) -> OrderDraftBatch:
        frozen = _decision(decision)
        existing = self._store.by_idempotency_key(
            frozen.account_id,
            frozen.decision_id,
            idempotency_key,
        )
        if existing is not None:
            return self._validate_artifact(existing, frozen).draft

        account = self.get_account_snapshot(frozen)
        target = self._build_target(frozen, account)
        risk = self._check_risk(
            frozen,
            account,
            target,
            # Risk is a decision-bound input, not a transport-time observation.
            # Re-evaluate at the frozen decision boundary so the draft binds the
            # exact result exposed by ``check_portfolio_risk`` instead of silently
            # changing its hash merely because draft delivery happened later.
            evaluated_at=frozen.as_of,
        )
        if not risk.allows_execution:
            raise PortfolioRiskBlocked(risk)
        self._assert_configs_unchanged()
        raw_states = self._source.load_market_states(frozen, target)
        states = tuple(
            _detach(item, MarketSessionState, "market state")
            for item in _tuple(
                raw_states,
                MarketSessionState,
                "market states",
                maximum=_MAX_MARKET_STATES,
                allow_empty=False,
            )
        )
        try:
            draft = self._draft_generator.generate(
                account_snapshot=account,
                target_portfolio=target,
                risk_result=risk,
                market_states=states,
                draft_as_of=draft_as_of,
            )
        except OrderDraftInputError as error:
            raise PortfolioExecutionInputInvalid(
                "order-draft inputs failed domain validation"
            ) from error
        try:
            stored = self._store.put(
                DraftArtifact(
                    account_id=frozen.account_id,
                    decision_id=frozen.decision_id,
                    idempotency_key=idempotency_key,
                    draft=draft,
                )
            )
        except DraftArtifactConflict:
            # Concurrent deliveries can carry different transport times and
            # therefore independently derive different draft hashes. The first
            # atomic write defines this key's result; replay it exactly.
            raced = self._store.by_idempotency_key(
                frozen.account_id,
                frozen.decision_id,
                idempotency_key,
            )
            if raced is None:
                raise
            stored = raced
        return self._validate_artifact(stored, frozen).draft

    def get_order_draft(
        self,
        decision: DecisionSnapshot,
        *,
        batch_hash: str,
    ) -> OrderDraftBatch:
        frozen = _decision(decision)
        artifact = self._store.get(frozen.account_id, batch_hash)
        return self._validate_artifact(artifact, frozen).draft

    def submit_paper_orders(
        self,
        decision: DecisionSnapshot,
        *,
        request_id: str,
        idempotency_key: str,
        batch_hash: str,
        submitted_at: datetime,
    ) -> PaperExecutionReceipt:
        frozen = _decision(decision)
        gateway = self._gateway
        if gateway is None:
            raise PortfolioExecutionInputUnavailable("paper execution gateway is unavailable")
        draft = self.get_order_draft(frozen, batch_hash=batch_hash)
        receipt = gateway.submit(
            decision=frozen,
            request_id=request_id,
            idempotency_key=idempotency_key,
            submitted_at=submitted_at,
            draft=draft,
        )
        return _detach(receipt, PaperExecutionReceipt, "paper receipt")

    def reconcile_account(
        self,
        decision: DecisionSnapshot,
        *,
        batch_hash: str,
        requested_at: datetime,
    ) -> ReconciliationResult:
        frozen = _decision(decision)
        draft = self.get_order_draft(frozen, batch_hash=batch_hash)
        inputs = _detach(
            self._source.load_reconciliation_inputs(frozen, draft),
            ReconciliationInputs,
            "reconciliation inputs",
        )
        receipt = inputs.receipt
        observed = inputs.observed
        if (
            receipt.account_before.account_id != frozen.account_id
            or receipt.account_after.account_id != frozen.account_id
            or observed.account_id != frozen.account_id
            or observed.account_snapshot.account_id != frozen.account_id
        ):
            raise PortfolioExecutionInputInvalid(
                "reconciliation inputs cross the authorized account boundary"
            )
        if _utc(receipt.processed_at) > _utc(requested_at) or _utc(observed.available_at) > _utc(
            requested_at
        ):
            raise PortfolioExecutionInputInvalid(
                "reconciliation inputs were unavailable at the request boundary"
            )
        identity = stable_hash(
            {
                "decision_id": frozen.decision_id,
                "batch_hash": draft.batch_hash,
                "receipt_hash": receipt.receipt_hash,
                "evidence_hash": observed.evidence_hash,
                "policy_hash": inputs.policy.policy_hash,
                "previous_result_hash": inputs.previous_result_hash,
            }
        )
        reconciled_at = max(
            frozen.as_of,
            receipt.processed_at,
            observed.available_at,
            key=_utc,
        )
        try:
            request = ReconciliationRequest.build(
                request_id=f"agent-reconcile:{identity}",
                idempotency_key=f"agent-reconcile:{identity}",
                reconciled_at=reconciled_at,
                draft=draft,
                receipt=receipt,
                observed=observed,
                policy=inputs.policy,
                previous_result_hash=inputs.previous_result_hash,
            )
            result = self._reconciliation_engine.reconcile(request)
        except ValueError as error:
            raise PortfolioExecutionInputInvalid(
                "reconciliation inputs failed domain validation"
            ) from error
        detached = _detach(result, ReconciliationResult, "reconciliation result")
        if detached.account_id != frozen.account_id:
            raise PortfolioExecutionInputInvalid(
                "reconciliation result crossed the authorized account boundary"
            )
        return detached

    def _build_target(
        self,
        decision: DecisionSnapshot,
        account: AccountSnapshot,
    ) -> TargetPortfolio:
        self._assert_configs_unchanged()
        raw_sleeves = self._source.load_strategy_sleeves(decision)
        sleeves = tuple(
            _detach(item, StrategySleeve, "strategy sleeve")
            for item in _tuple(
                raw_sleeves,
                StrategySleeve,
                "strategy sleeves",
                maximum=_MAX_SLEEVES,
                allow_empty=False,
            )
        )
        expected_refs = {
            (item.strategy_name, item.strategy_version, item.config_hash)
            for item in decision.strategy_refs
        }
        actual_refs = {
            (item.strategy_name, item.strategy_version, item.strategy_config_hash)
            for item in sleeves
        }
        if actual_refs != expected_refs or len(actual_refs) != len(sleeves):
            raise PortfolioExecutionInputInvalid(
                "strategy sleeves do not exactly match decision strategy references"
            )
        raw_classifications = self._source.load_industry_classifications(decision)
        classifications = tuple(
            _detach(item, PITIndustryClassification, "industry classification")
            for item in _tuple(
                raw_classifications,
                PITIndustryClassification,
                "industry classifications",
                maximum=_MAX_CLASSIFICATIONS,
                allow_empty=True,
            )
        )
        request = PortfolioBuildRequest(
            decision_id=decision.decision_id,
            as_of=decision.as_of,
            data_version=decision.data_version,
            account_snapshot_id=decision.account_snapshot_id,
            account_snapshot_hash=decision.account_snapshot_hash,
        )
        try:
            result = self._builder.build(
                request=request,
                account=account,
                sleeves=sleeves,
                classifications=classifications,
            )
        except PortfolioBuildInputError as error:
            raise PortfolioExecutionInputInvalid(
                "target-portfolio inputs failed domain validation"
            ) from error
        detached = _detach(result, TargetPortfolio, "target portfolio")
        if (
            detached.decision_id != decision.decision_id
            or detached.account_snapshot_id != decision.account_snapshot_id
            or detached.account_snapshot_hash != decision.account_snapshot_hash
            or detached.data_version != decision.data_version
            or _utc(detached.as_of) != _utc(decision.as_of)
        ):
            raise PortfolioExecutionInputInvalid(
                "target portfolio drifted from the decision boundary"
            )
        return detached

    def _check_risk(
        self,
        decision: DecisionSnapshot,
        account: AccountSnapshot,
        target: TargetPortfolio,
        *,
        evaluated_at: datetime,
    ) -> RiskCheckResult:
        self._assert_configs_unchanged()
        policy = self._risk_evaluator.policy
        if (
            policy.version != decision.risk_policy_version
            or policy.policy_hash != decision.risk_policy_hash
        ):
            raise PortfolioExecutionInputInvalid("risk policy does not match the frozen decision")
        context = _detach(
            self._source.load_risk_context(decision),
            PortfolioRiskContext,
            "portfolio risk context",
        )
        try:
            request = build_portfolio_risk_request(
                account=account,
                proposal=target,
                context=context,
                policy=policy,
            )
        except PortfolioRiskInputError as error:
            raise PortfolioExecutionInputInvalid(
                "risk inputs failed request construction"
            ) from error
        result = self._risk_evaluator.check(
            request=request,
            account=account,
            proposal=target,
            context=context,
            evaluated_at=evaluated_at,
        )
        detached = _detach(result, RiskCheckResult, "portfolio risk result")
        if (
            detached.request.decision_id != decision.decision_id
            or detached.request.account_snapshot_id != decision.account_snapshot_id
            or detached.request.account_snapshot_hash != decision.account_snapshot_hash
            or detached.request.portfolio_proposal_hash != target.result_hash
            or detached.request.data_version != decision.data_version
            or detached.request.policy_hash != decision.risk_policy_hash
        ):
            raise PortfolioExecutionInputInvalid(
                "portfolio risk result drifted from the decision boundary"
            )
        return detached

    def _validate_artifact(
        self,
        artifact: DraftArtifact,
        decision: DecisionSnapshot,
    ) -> DraftArtifact:
        detached = _detach(artifact, DraftArtifact, "draft artifact")
        draft = detached.draft
        if (
            detached.account_id != decision.account_id
            or detached.decision_id != decision.decision_id
            or draft.decision_id != decision.decision_id
            or draft.account_snapshot_id != decision.account_snapshot_id
            or draft.account_snapshot_hash != decision.account_snapshot_hash
            or draft.account_snapshot_as_of != decision.as_of
            or draft.runtime_mode is not decision.mode
            or draft.data_version != decision.data_version
        ):
            raise PortfolioExecutionInputInvalid(
                "stored draft artifact does not match the frozen decision"
            )
        return detached

    def _assert_configs_unchanged(self) -> None:
        if (
            self._builder.config.config_hash != self._builder_config_hash
            or self._risk_evaluator.policy.policy_hash != self._risk_policy_hash
            or self._draft_generator.config.config_hash != self._draft_config_hash
        ):
            raise PortfolioExecutionInputInvalid(
                "portfolio tool configuration changed after toolset construction"
            )


__all__ = ["PortfolioExecutionPipeline"]
