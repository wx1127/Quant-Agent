"""Transactional orchestration for credential-free paper execution."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.portfolio import AccountSnapshot
from quant_agent.risk.kill_switch.contracts import (
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchGateDecision,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchGateStatus,
)
from quant_agent.risk.kill_switch.service import KillSwitchOrderBlocked

from .contracts import (
    PaperAccountState,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperIdempotencyConflict,
    PaperPositionLot,
)
from .engine import PaperExecutionEngine
from .repository import PaperRepository


class PaperNewOrderGate(Protocol):
    """Lock-spanning safety gate required around each new paper commit."""

    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> AbstractContextManager[KillSwitchGateDecision]: ...


class PaperExecutionService:
    """Coordinate deterministic planning with an atomic paper-only repository."""

    def __init__(
        self,
        *,
        engine: PaperExecutionEngine,
        repository: PaperRepository,
        new_order_gate: PaperNewOrderGate,
        gate_actor: KillSwitchActor | None = None,
        clock: Callable[[], datetime] = shanghai_now,
    ) -> None:
        if not isinstance(engine, PaperExecutionEngine):
            raise TypeError("engine must be a PaperExecutionEngine")
        self._engine = engine
        self._repository = repository
        self._new_order_gate = new_order_gate
        self._gate_actor = gate_actor or KillSwitchActor.build(
            actor_id="paper-execution-service",
            kind=KillSwitchActorKind.SYSTEM,
            role=KillSwitchActorRole.SYSTEM,
        )
        self._clock = clock

    @property
    def engine(self) -> PaperExecutionEngine:
        return self._engine

    def open_account(
        self,
        *,
        snapshot: AccountSnapshot,
        opening_lots: tuple[PaperPositionLot, ...] = (),
    ) -> PaperAccountState:
        """Open an account idempotently from one exact PAPER snapshot."""

        state = self._engine.open_account(snapshot=snapshot, opening_lots=opening_lots)
        return self._repository.open_account(state)

    def refresh_account(self, *, snapshot: AccountSnapshot) -> PaperAccountState:
        """Advance an opened account to a new hash-linked snapshot boundary."""

        try:
            snapshot = AccountSnapshot.from_json(snapshot.to_json())
        except (AttributeError, TypeError, ValueError) as error:
            raise PaperExecutionInputError(str(error)) from error
        current = self._repository.get_account(snapshot.account_id)
        committed_refresh = self._repository.account_by_refresh_snapshot_hash(
            snapshot.account_id,
            snapshot.content_hash,
        )
        if committed_refresh is not None:
            return current
        state_after = self._engine.refresh_account(account=current, snapshot=snapshot)
        return self._repository.refresh_account(
            current.state_hash,
            snapshot,
            state_after,
        )

    def submit_and_match(
        self,
        *,
        request: PaperExecutionRequest,
        draft: OrderDraftBatch,
    ) -> PaperExecutionReceipt:
        """Execute once or return the exact committed receipt for a retry."""

        try:
            request = replace(request)
            draft = replace(
                draft,
                lines=tuple(replace(line) for line in draft.lines),
            )
        except PaperExecutionInputError:
            raise
        except (TypeError, ValueError) as error:
            raise PaperExecutionInputError(str(error)) from error
        if request.batch_hash != draft.batch_hash:
            raise PaperExecutionInputError("request batch hash does not match order draft")
        committed = self._committed_receipt(request)
        if committed is not None:
            return committed
        checked_at = max(ensure_aware(self._clock()), request.submitted_at)
        gate_request = KillSwitchGateRequest.build(
            request_id=request.request_id,
            account_id=request.account_id,
            idempotency_key=request.idempotency_key,
            batch_hash=request.batch_hash,
            source_request_hash=request.request_hash,
            checked_at=checked_at,
        )
        try:
            with self._new_order_gate.guard_new_order(gate_request, self._gate_actor) as decision:
                # Another submission may have committed while this call waited
                # for the gate lock. A receipt replay is not a new order.
                committed = self._committed_receipt(request)
                if committed is not None:
                    return committed
                if not isinstance(decision, KillSwitchGateDecision) or (
                    decision.allowed
                    and (
                        decision.request_hash != gate_request.request_hash
                        or decision.checked_at != gate_request.checked_at
                    )
                ):
                    raise KillSwitchOrderBlocked(
                        KillSwitchGateDecision.build(
                            request=gate_request,
                            status=KillSwitchGateStatus.BLOCKED,
                            global_state_hash=None,
                            account_state_hash=None,
                            reason_codes=(KillSwitchGateReason.REPOSITORY_FAILURE,),
                        )
                    )
                if not decision.allowed:
                    raise KillSwitchOrderBlocked(decision)
                return self._execute_new(request=request, draft=draft)
        except Exception:
            # A concurrent commit or a lost acknowledgement can precede a
            # block/error. Only an exact durable receipt may resolve that race.
            committed = self._committed_receipt(request)
            if committed is not None:
                return committed
            raise
        raise PaperExecutionInputError("new-order gate suppressed execution without a receipt")

    def _execute_new(
        self,
        *,
        request: PaperExecutionRequest,
        draft: OrderDraftBatch,
    ) -> PaperExecutionReceipt:
        """Execute and commit while any configured new-order guard remains held."""

        account = self._repository.get_account(request.account_id)
        try:
            receipt = self._engine.execute(
                request=request,
                draft=draft,
                account=account,
            )
        except PaperExecutionInputError:
            committed = self._committed_receipt(request)
            if committed is not None:
                return committed
            raise
        return self._repository.commit_execution(
            request.expected_account_state_hash,
            request,
            receipt.account_after,
            receipt,
        )

    def _committed_receipt(
        self,
        request: PaperExecutionRequest,
    ) -> PaperExecutionReceipt | None:
        existing = self._repository.receipt_by_idempotency_key(
            request.account_id,
            request.idempotency_key,
        )
        if existing is not None:
            if existing.request_hash != request.request_hash:
                raise PaperIdempotencyConflict(
                    "paper idempotency key belongs to a different execution request"
                )
            return existing
        consumed = self._repository.receipt_by_batch_hash(
            request.account_id,
            request.batch_hash,
        )
        if consumed is not None:
            if consumed.request_hash == request.request_hash:
                return consumed
            raise PaperIdempotencyConflict(
                "paper draft batch belongs to a different execution request"
            )
        return None


__all__ = ["PaperExecutionService", "PaperNewOrderGate"]
