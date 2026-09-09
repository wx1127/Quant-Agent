"""Trusted adapter that preserves the complete P5 paper-execution boundary."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, cast

from pydantic import TypeAdapter, ValidationError

from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.execution.paper import (
    PaperExecutionEngine,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperExecutionService,
    PaperIdempotencyConflict,
    PaperNewOrderGate,
    PaperRepository,
)
from quant_agent.risk import KillSwitchActor


class PaperOrderGateway(Protocol):
    """Submit a stored draft through a guarded, repository-backed paper service."""

    @property
    def version(self) -> str:
        """Return the immutable underlying paper engine/config version."""

        ...

    def submit(
        self,
        *,
        decision: DecisionSnapshot,
        request_id: str,
        idempotency_key: str,
        submitted_at: datetime,
        draft: OrderDraftBatch,
    ) -> PaperExecutionReceipt:
        """Submit once or replay the exact already-committed receipt."""

        ...


def _detach[T](value: object, expected: type[T], label: str) -> T:
    if type(value) is not expected:
        raise PaperExecutionInputError(f"{label} has an unsupported type")
    try:
        adapter = TypeAdapter(expected)
        detached = adapter.validate_json(adapter.dump_json(value, warnings="error"), strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise PaperExecutionInputError(f"{label} failed strict revalidation") from error
    if type(detached) is not expected:
        raise PaperExecutionInputError(f"{label} failed exact-type revalidation")
    return detached


def _require_repository(value: object) -> PaperRepository:
    methods = (
        "open_account",
        "get_account",
        "refresh_account",
        "account_by_refresh_snapshot_hash",
        "receipt_by_idempotency_key",
        "receipt_by_batch_hash",
        "commit_execution",
    )
    if any(not callable(getattr(value, name, None)) for name in methods):
        raise TypeError("paper repository does not implement the complete atomic protocol")
    return cast(PaperRepository, value)


def _validate_receipt_binding(
    receipt: PaperExecutionReceipt,
    *,
    decision: DecisionSnapshot,
    draft: OrderDraftBatch,
    idempotency_key: str,
) -> PaperExecutionReceipt:
    """Fail closed if a repository/service returns another account's receipt."""

    before = receipt.account_before
    after = receipt.account_after
    if receipt.idempotency_key != idempotency_key or receipt.batch_hash != draft.batch_hash:
        raise PaperIdempotencyConflict(
            "paper receipt identity does not match the requested idempotent operation"
        )
    if (
        before.account_id != decision.account_id
        or after.account_id != decision.account_id
        or before.runtime_mode is not RuntimeMode.PAPER
        or after.runtime_mode is not RuntimeMode.PAPER
        or before.source_snapshot_id != decision.account_snapshot_id
        or after.source_snapshot_id != decision.account_snapshot_id
        or before.source_snapshot_hash != decision.account_snapshot_hash
        or after.source_snapshot_hash != decision.account_snapshot_hash
        or before.source_snapshot_as_of != decision.as_of
        or after.source_snapshot_as_of != decision.as_of
        or before.data_version != decision.data_version
        or after.data_version != decision.data_version
        or any(order.decision_id != decision.decision_id for order in receipt.orders)
    ):
        raise PaperExecutionInputError(
            "paper receipt does not match the frozen decision account boundary"
        )
    return receipt


class ServiceBackedPaperOrderGateway:
    """Build requests and invoke ``PaperExecutionService``, never the raw engine.

    Exact committed receipts are looked up before constructing a new request. This
    lets retries with a new transport timestamp replay safely without changing the
    P5 request hash or colliding with its idempotency contract.
    """

    __slots__ = ("_engine", "_repository", "_service", "_version")

    def __init__(
        self,
        *,
        engine: PaperExecutionEngine,
        repository: PaperRepository,
        new_order_gate: PaperNewOrderGate,
        gate_actor: KillSwitchActor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(engine) is not PaperExecutionEngine:
            raise TypeError("paper gateway requires an exact PaperExecutionEngine")
        selected_repository = _require_repository(repository)
        if not callable(getattr(new_order_gate, "guard_new_order", None)):
            raise TypeError("paper gateway requires a complete new-order gate")
        service_kwargs: dict[str, object] = {
            "engine": engine,
            "repository": selected_repository,
            "new_order_gate": new_order_gate,
            "gate_actor": gate_actor,
        }
        if clock is not None:
            if not callable(clock):
                raise TypeError("paper gateway clock must be callable")
            service_kwargs["clock"] = clock
        self._engine = engine
        self._repository = selected_repository
        self._service = PaperExecutionService(**service_kwargs)  # type: ignore[arg-type]
        self._version = f"{engine.config.version}:{engine.config.config_hash}"

    @property
    def version(self) -> str:
        return self._version

    def submit(
        self,
        *,
        decision: DecisionSnapshot,
        request_id: str,
        idempotency_key: str,
        submitted_at: datetime,
        draft: OrderDraftBatch,
    ) -> PaperExecutionReceipt:
        frozen = DecisionSnapshot.from_json(decision.to_json())
        batch = _detach(draft, OrderDraftBatch, "order draft")
        ensure_aware(submitted_at)
        if frozen.mode is not RuntimeMode.PAPER or batch.runtime_mode is not RuntimeMode.PAPER:
            raise PaperExecutionInputError("paper gateway requires PAPER decision and draft modes")
        if (
            batch.decision_id != frozen.decision_id
            or batch.account_snapshot_id != frozen.account_snapshot_id
            or batch.account_snapshot_hash != frozen.account_snapshot_hash
            or batch.data_version != frozen.data_version
        ):
            raise PaperExecutionInputError("paper draft does not match the frozen decision")

        existing = self._repository.receipt_by_idempotency_key(
            frozen.account_id,
            idempotency_key,
        )
        if existing is not None:
            receipt = _detach(existing, PaperExecutionReceipt, "paper receipt")
            return _validate_receipt_binding(
                receipt,
                decision=frozen,
                draft=batch,
                idempotency_key=idempotency_key,
            )
        consumed = self._repository.receipt_by_batch_hash(
            frozen.account_id,
            batch.batch_hash,
        )
        if consumed is not None:
            receipt = _detach(consumed, PaperExecutionReceipt, "paper receipt")
            return _validate_receipt_binding(
                receipt,
                decision=frozen,
                draft=batch,
                idempotency_key=idempotency_key,
            )

        account = self._repository.get_account(frozen.account_id)
        if (
            account.account_id != frozen.account_id
            or account.runtime_mode is not RuntimeMode.PAPER
            or account.source_snapshot_id != frozen.account_snapshot_id
            or account.source_snapshot_hash != frozen.account_snapshot_hash
            or account.source_snapshot_as_of != frozen.as_of
            or account.data_version != frozen.data_version
        ):
            raise PaperExecutionInputError(
                "paper account state does not match the decision account"
            )
        request = PaperExecutionRequest.build(
            request_id=request_id,
            idempotency_key=idempotency_key,
            account_id=frozen.account_id,
            expected_account_state_hash=account.state_hash,
            expected_account_snapshot_hash=frozen.account_snapshot_hash,
            batch_hash=batch.batch_hash,
            submitted_at=submitted_at,
            config=self._engine.config,
        )
        try:
            receipt = self._service.submit_and_match(request=request, draft=batch)
        except PaperIdempotencyConflict:
            # A competing delivery can commit the same logical Agent operation
            # after our preflight lookup but before the service acquires its
            # gate. P5 intentionally compares the lower-level request hash,
            # whose transport timestamp differs; recover only the exact
            # account/key/batch receipt at this higher-level boundary.
            raced = self._repository.receipt_by_idempotency_key(
                frozen.account_id,
                idempotency_key,
            )
            if raced is None:
                raise
            return _validate_receipt_binding(
                _detach(raced, PaperExecutionReceipt, "paper receipt"),
                decision=frozen,
                draft=batch,
                idempotency_key=idempotency_key,
            )
        detached = _detach(receipt, PaperExecutionReceipt, "paper receipt")
        if detached.request_hash != request.request_hash:
            raise PaperExecutionInputError("paper receipt drifted from its authorized request")
        return _validate_receipt_binding(
            detached,
            decision=frozen,
            draft=batch,
            idempotency_key=idempotency_key,
        )


__all__ = ["PaperOrderGateway", "ServiceBackedPaperOrderGateway"]
