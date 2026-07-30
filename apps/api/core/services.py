"""Thread-safe service boundaries used by the versioned API routes."""

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Any

from quant_agent.agent.responses import AgentAnswer
from quant_agent.core.errors import ErrorCode, QuantAgentError
from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.observability.audit import AuditEvent, AuditSink
from quant_agent.observability.monitoring import MonitoringRegistry
from quant_agent.risk.kill_switch import KillSwitch


@dataclass(frozen=True, slots=True)
class ResearchRecord:
    record_id: str
    as_of: datetime
    data_version: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)


class ResearchCatalog:
    """Immutable read-model snapshots for API pagination and filtering."""

    _KINDS = frozenset({"themes", "leaders", "candidates", "evidence"})

    def __init__(self) -> None:
        self._regime: ResearchRecord | None = None
        self._records: dict[str, tuple[ResearchRecord, ...]] = {kind: () for kind in self._KINDS}

    def publish_regime(self, record: ResearchRecord) -> None:
        self._regime = record

    def publish(self, kind: str, records: list[ResearchRecord]) -> None:
        if kind not in self._KINDS:
            raise ValueError("unsupported research record kind")
        identifiers = [item.record_id for item in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research record ids must be unique")
        self._records[kind] = tuple(sorted(records, key=lambda item: item.record_id))

    def regime(self) -> ResearchRecord:
        if self._regime is None:
            raise QuantAgentError(ErrorCode.DATA_UNAVAILABLE, "market regime is not available")
        return self._regime

    def list(
        self,
        kind: str,
        *,
        offset: int,
        limit: int,
        filters: dict[str, str],
    ) -> tuple[tuple[ResearchRecord, ...], int]:
        if kind not in self._KINDS:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "research collection not found")
        rows = self._records[kind]
        for key, expected in sorted(filters.items()):
            rows = tuple(row for row in rows if str(row.payload.get(key, "")) == expected)
        return rows[offset : offset + limit], len(rows)

    def get(self, kind: str, record_id: str) -> ResearchRecord:
        rows, _ = self.list(kind, offset=0, limit=10_000, filters={})
        for row in rows:
            if row.record_id == record_id:
                return row
        raise QuantAgentError(ErrorCode.NOT_FOUND, "research record not found")


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class BacktestTask:
    run_id: str
    strategy_id: str
    parameter_version: str
    data_version: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = None
    error: str | None = None


BacktestRunner = Callable[[str, str, str], dict[str, Any]]


class BacktestTaskStore:
    """Runs published strategy jobs outside the API request thread."""

    def __init__(
        self,
        published: set[tuple[str, str]],
        runner: BacktestRunner,
        *,
        max_workers: int = 2,
    ) -> None:
        self._published = frozenset(published)
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, BacktestTask] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = RLock()

    def submit(self, strategy_id: str, parameter_version: str, data_version: str) -> BacktestTask:
        if (strategy_id, parameter_version) not in self._published:
            raise QuantAgentError(
                ErrorCode.INVALID_ARGUMENT,
                "strategy and parameter version are not published",
            )
        now = shanghai_now()
        run_id = f"bt_{secrets.token_hex(12)}"
        task = BacktestTask(
            run_id,
            strategy_id,
            parameter_version,
            data_version,
            TaskStatus.QUEUED,
            now,
            now,
        )
        with self._lock:
            self._tasks[run_id] = task
            self._futures[run_id] = self._executor.submit(self._execute, run_id)
        return task

    def get(self, run_id: str) -> BacktestTask:
        with self._lock:
            task = self._tasks.get(run_id)
        if task is None:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "backtest run not found")
        return task

    def _execute(self, run_id: str) -> None:
        with self._lock:
            current = self._tasks[run_id]
            self._tasks[run_id] = _replace_task(current, status=TaskStatus.RUNNING)
        try:
            result = self._runner(
                current.strategy_id,
                current.parameter_version,
                current.data_version,
            )
        except Exception as exc:
            with self._lock:
                self._tasks[run_id] = _replace_task(
                    self._tasks[run_id],
                    status=TaskStatus.FAILED,
                    error=str(exc),
                )
        else:
            with self._lock:
                self._tasks[run_id] = _replace_task(
                    self._tasks[run_id],
                    status=TaskStatus.SUCCEEDED,
                    result=result,
                )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def _replace_task(
    task: BacktestTask,
    *,
    status: TaskStatus,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> BacktestTask:
    return BacktestTask(
        task.run_id,
        task.strategy_id,
        task.parameter_version,
        task.data_version,
        status,
        task.created_at,
        shanghai_now(),
        result,
        error,
    )


@dataclass(frozen=True, slots=True)
class PortfolioProposal:
    proposal_id: str
    account_id: str
    decision_id: str
    as_of: datetime
    data_version: str
    executable: bool
    target: dict[str, Any]
    risk_result: dict[str, Any]


class PortfolioService:
    def __init__(self) -> None:
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._proposals: dict[str, PortfolioProposal] = {}

    def publish_snapshot(self, account_id: str, snapshot: dict[str, Any]) -> None:
        self._snapshots[account_id] = snapshot

    def get_snapshot(self, account_id: str) -> dict[str, Any]:
        try:
            return self._snapshots[account_id]
        except KeyError as exc:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "portfolio snapshot not found") from exc

    def create_proposal(
        self,
        *,
        account_id: str,
        decision_id: str,
        as_of: datetime,
        data_version: str,
        target: dict[str, Any],
        risk_result: dict[str, Any],
    ) -> PortfolioProposal:
        if risk_result.get("passed") is not True:
            raise QuantAgentError(ErrorCode.RISK_REJECTED, "portfolio risk check rejected")
        proposal = PortfolioProposal(
            f"proposal_{secrets.token_hex(10)}",
            account_id,
            decision_id,
            ensure_aware(as_of),
            data_version,
            False,
            target,
            risk_result,
        )
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def get_proposal(self, proposal_id: str) -> PortfolioProposal:
        try:
            return self._proposals[proposal_id]
        except KeyError as exc:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "portfolio proposal not found") from exc


@dataclass(frozen=True, slots=True)
class OrderDraftRecord:
    draft_id: str
    account_id: str
    decision_id: str
    batch_hash: str
    expires_at: datetime
    orders: tuple[dict[str, Any], ...]
    estimated_fees: float
    risk_warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.expires_at)


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    token_hash: str
    nonce: str
    approver_id: str
    draft_id: str
    account_id: str
    decision_id: str
    batch_hash: str
    expires_at: datetime
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class IssuedApproval:
    token: str
    expires_at: datetime


class OrderApprovalService:
    """Human-only, hash-bound, expiring and one-time approval tokens."""

    def __init__(
        self,
        signing_secret: bytes,
        kill_switch: KillSwitch,
        submitter: Callable[[OrderDraftRecord], dict[str, Any]],
        audit_sink: AuditSink | None = None,
        approval_ttl: timedelta = timedelta(minutes=5),
        price_provider: Callable[[OrderDraftRecord], dict[str, float]] | None = None,
        maximum_price_deviation_bps: float = 100.0,
        monitoring: MonitoringRegistry | None = None,
    ) -> None:
        if len(signing_secret) < 32:
            raise ValueError("approval signing secret must be at least 32 bytes")
        if not timedelta(seconds=1) <= approval_ttl <= timedelta(minutes=15):
            raise ValueError("approval TTL must be between one second and fifteen minutes")
        if maximum_price_deviation_bps <= 0:
            raise ValueError("maximum price deviation must be positive")
        self._secret = signing_secret
        self._kill_switch = kill_switch
        self._submitter = submitter
        self._audit_sink = audit_sink
        self._approval_ttl = approval_ttl
        self._price_provider = price_provider or _draft_reference_prices
        self._maximum_price_deviation_bps = maximum_price_deviation_bps
        self._monitoring = monitoring
        self._drafts: dict[str, OrderDraftRecord] = {}
        self._grants: dict[str, ApprovalGrant] = {}
        self._submissions: dict[str, tuple[str, dict[str, Any]]] = {}
        self._lock = RLock()

    def put_draft(self, draft: OrderDraftRecord) -> None:
        self._drafts[draft.draft_id] = draft

    @property
    def kill_switch(self) -> KillSwitch:
        """Expose the shared deployment gate for identity validation."""

        return self._kill_switch

    def get_draft(self, draft_id: str) -> OrderDraftRecord:
        try:
            return self._drafts[draft_id]
        except KeyError as exc:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "order draft not found") from exc

    def approve(
        self,
        draft_id: str,
        *,
        approver_id: str,
        approved_at: datetime,
        request_id: str,
    ) -> IssuedApproval:
        draft = self.get_draft(draft_id)
        moment = ensure_aware(approved_at)
        if moment >= draft.expires_at:
            raise QuantAgentError(ErrorCode.APPROVAL_EXPIRED, "order draft has expired")
        grant_expires_at = min(draft.expires_at, moment + self._approval_ttl)
        payload = {
            "nonce": secrets.token_urlsafe(18),
            "approver_id": approver_id,
            "draft_id": draft.draft_id,
            "account_id": draft.account_id,
            "decision_id": draft.decision_id,
            "batch_hash": draft.batch_hash,
            "expires_at": grant_expires_at.isoformat(),
        }
        encoded = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        signature = _b64(hmac.digest(self._secret, encoded.encode(), "sha256"))
        token = f"{encoded}.{signature}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._grants[token_hash] = ApprovalGrant(
            token_hash=token_hash,
            nonce=payload["nonce"],
            approver_id=approver_id,
            draft_id=draft.draft_id,
            account_id=draft.account_id,
            decision_id=draft.decision_id,
            batch_hash=draft.batch_hash,
            expires_at=grant_expires_at,
        )
        self._audit(
            actor_id=approver_id,
            action="APPROVE_ORDER_DRAFT",
            result="approved",
            request_id=request_id,
            decision_id=draft.decision_id,
            occurred_at=moment,
            metadata={
                "draft_id": draft.draft_id,
                "account_id": draft.account_id,
                "batch_hash": draft.batch_hash,
                "approval_token_hash": token_hash,
            },
        )
        return IssuedApproval(token, grant_expires_at)

    def submit(
        self,
        draft_id: str,
        *,
        approval_token: str,
        idempotency_key: str,
        submitted_at: datetime,
        submitted_by: str,
        request_id: str,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise QuantAgentError(ErrorCode.INVALID_ARGUMENT, "idempotency key is required")
        with self._lock:
            existing = self._submissions.get(idempotency_key)
            if existing is not None:
                existing_draft_id, existing_result = existing
                if existing_draft_id != draft_id:
                    if self._monitoring is not None:
                        self._monitoring.record_order_submission(duplicate_risk=True)
                    raise QuantAgentError(
                        ErrorCode.CONFLICT,
                        "idempotency key is already bound to another order draft",
                    )
                return existing_result
            draft = self.get_draft(draft_id)
            moment = ensure_aware(submitted_at)
            grant_key = hashlib.sha256(approval_token.encode()).hexdigest()
            grant = self._grants.get(grant_key)
            self._validate_token(approval_token, grant, draft, moment)
            if not self._kill_switch.order_allowed(draft.account_id):
                raise QuantAgentError(
                    ErrorCode.KILL_SWITCH_ACTIVE,
                    "Kill Switch is active for this account",
                )
            self._validate_latest_prices(draft)
            assert grant is not None
            self._grants[grant_key] = ApprovalGrant(**{**asdict(grant), "consumed": True})
            result = self._submitter(draft)
            self._submissions[idempotency_key] = (draft_id, result)
            if self._monitoring is not None:
                self._monitoring.record_order_submission()
            self._audit(
                actor_id=submitted_by,
                action="SUBMIT_APPROVED_PAPER_ORDERS",
                result="submitted",
                request_id=request_id,
                decision_id=draft.decision_id,
                occurred_at=moment,
                metadata={
                    "draft_id": draft.draft_id,
                    "account_id": draft.account_id,
                    "batch_hash": draft.batch_hash,
                    "idempotency_key_hash": hashlib.sha256(idempotency_key.encode()).hexdigest(),
                },
            )
            return result

    def _validate_latest_prices(self, draft: OrderDraftRecord) -> None:
        latest = self._price_provider(draft)
        for order in draft.orders:
            instrument_id = str(order.get("instrument_id", ""))
            reference = order.get("reference_price")
            if reference is None:
                continue
            current = latest.get(instrument_id)
            if current is None or current <= 0:
                raise QuantAgentError(
                    ErrorCode.DATA_UNAVAILABLE,
                    "latest price is unavailable before order submission",
                    details={"instrument_id": instrument_id},
                )
            deviation_bps = abs(float(current) / float(reference) - 1) * 10_000
            if deviation_bps > self._maximum_price_deviation_bps:
                raise QuantAgentError(
                    ErrorCode.APPROVAL_REQUIRED,
                    "price moved beyond the approved order boundary",
                    details={
                        "instrument_id": instrument_id,
                        "deviation_bps": round(deviation_bps, 2),
                    },
                )

    def _validate_token(
        self,
        token: str,
        grant: ApprovalGrant | None,
        draft: OrderDraftRecord,
        now: datetime,
    ) -> None:
        try:
            encoded, signature = token.split(".", 1)
        except ValueError as exc:
            raise QuantAgentError(ErrorCode.APPROVAL_REQUIRED, "invalid approval token") from exc
        expected = _b64(hmac.digest(self._secret, encoded.encode(), "sha256"))
        if not hmac.compare_digest(signature, expected) or grant is None:
            raise QuantAgentError(ErrorCode.APPROVAL_REQUIRED, "invalid approval token")
        if grant.consumed:
            raise QuantAgentError(ErrorCode.CONFLICT, "approval token was already consumed")
        if now >= grant.expires_at:
            raise QuantAgentError(ErrorCode.APPROVAL_EXPIRED, "approval token has expired")
        if (
            grant.draft_id != draft.draft_id
            or grant.account_id != draft.account_id
            or grant.decision_id != draft.decision_id
            or grant.batch_hash != draft.batch_hash
        ):
            raise QuantAgentError(
                ErrorCode.APPROVAL_REQUIRED,
                "approval token does not match the current order draft",
            )

    def _audit(
        self,
        *,
        actor_id: str,
        action: str,
        result: str,
        request_id: str,
        decision_id: str,
        occurred_at: datetime,
        metadata: dict[str, Any],
    ) -> None:
        if self._audit_sink is None:
            return
        self._audit_sink.append(
            AuditEvent(
                event_type="order_approval",
                actor_id=actor_id,
                action=action,
                result=result,
                request_id=request_id,
                decision_id=decision_id,
                occurred_at=occurred_at,
                metadata=metadata,
            )
        )


@dataclass(slots=True)
class DecisionStore:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, decision_id: str) -> dict[str, Any]:
        try:
            return self.records[decision_id]
        except KeyError as exc:
            raise QuantAgentError(ErrorCode.NOT_FOUND, "decision not found") from exc


AgentHandler = Callable[[str, str], AgentAnswer]
ReconcileHandler = Callable[[str], dict[str, Any]]
RiskHandler = Callable[[str, str, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class ApiServices:
    research: ResearchCatalog
    backtests: BacktestTaskStore
    portfolios: PortfolioService
    orders: OrderApprovalService
    decisions: DecisionStore
    agent_handler: AgentHandler
    reconcile_handler: ReconcileHandler
    risk_handler: RiskHandler


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _draft_reference_prices(draft: OrderDraftRecord) -> dict[str, float]:
    return {
        str(order["instrument_id"]): float(order["reference_price"])
        for order in draft.orders
        if order.get("instrument_id") and order.get("reference_price") is not None
    }
