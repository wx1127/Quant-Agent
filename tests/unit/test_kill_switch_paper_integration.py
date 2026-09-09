"""Execution-gateway tests for the lock-spanning kill-switch guard."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event

import pytest

from quant_agent.execution.paper import (
    InMemoryPaperRepository,
    PaperAccountState,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperExecutionService,
    PaperNewOrderGate,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.kill_switch import (
    InMemoryKillSwitchRepository,
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchGateDecision,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchIncident,
    KillSwitchOrderBlocked,
    KillSwitchScope,
    KillSwitchService,
    KillSwitchStatus,
    KillSwitchTransition,
    KillSwitchTriggerSource,
)

from .test_paper_repository_service import (
    _execution_fixture,
    _ExecutionFixture,
    _LookupRacePaperRepository,
)

SYSTEM = KillSwitchActor.build(
    actor_id="kill-switch-paper-integration",
    kind=KillSwitchActorKind.SYSTEM,
    role=KillSwitchActorRole.SYSTEM,
)


class _BlockingPaperRepository(InMemoryPaperRepository):
    def __init__(self) -> None:
        super().__init__()
        self.commit_entered = Event()
        self.release_commit = Event()

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        self.commit_entered.set()
        assert self.release_commit.wait(timeout=10)
        return super().commit_execution(
            expected_state_hash,
            request,
            state_after,
            receipt,
        )


class _ObservableActivationRepository(InMemoryKillSwitchRepository):
    def __init__(self) -> None:
        super().__init__()
        self.activation_called = Event()

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        self.activation_called.set()
        return super().activate(request)


def _initialized_kill_service(*, account_id: str, at: datetime) -> KillSwitchService:
    service = KillSwitchService(repository=InMemoryKillSwitchRepository())
    service.initialize(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=SYSTEM,
        changed_at=at,
    )
    service.initialize(
        scope=KillSwitchScope.ACCOUNT,
        account_id=account_id,
        actor=SYSTEM,
        changed_at=at,
    )
    return service


def _activation(
    *,
    scope: KillSwitchScope,
    account_id: str | None,
    at: datetime,
    label: str,
) -> KillSwitchActivationRequest:
    source_hash = stable_hash({"kill-switch-paper-trigger": label})
    incident = KillSwitchIncident.build(
        scope=scope,
        account_id=account_id,
        trigger_source=KillSwitchTriggerSource.MANUAL,
        reason_codes=(f"MANUAL_{label.upper()}",),
        evidence_hashes=(source_hash,),
        source_reference_hash=source_hash,
        summary=f"paper integration {label}",
        triggered_at=at,
    )
    return KillSwitchActivationRequest.build(
        request_id=f"activation-{label}",
        idempotency_key=f"activation-key-{label}",
        incident=incident,
        actor=SYSTEM,
        requested_at=at,
    )


def _gated_paper_service(
    fixture: _ExecutionFixture,
    kill_service: PaperNewOrderGate,
) -> PaperExecutionService:
    return PaperExecutionService(
        engine=fixture.engine,
        repository=fixture.repository,
        new_order_gate=kill_service,
        clock=lambda: fixture.request.submitted_at + timedelta(seconds=2),
    )


def test_inactive_global_and_account_switches_allow_one_new_execution() -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id,
        at=fixture.request.submitted_at - timedelta(seconds=1),
    )

    receipt = _gated_paper_service(fixture, kill_service).submit_and_match(
        request=fixture.request,
        draft=fixture.draft,
    )

    assert receipt.request_hash == fixture.request.request_hash


@pytest.mark.parametrize(
    ("scope", "reason"),
    (
        (KillSwitchScope.ACCOUNT, KillSwitchGateReason.ACCOUNT_ACTIVE),
        (KillSwitchScope.GLOBAL, KillSwitchGateReason.GLOBAL_ACTIVE),
    ),
)
def test_active_switch_blocks_before_paper_state_or_receipt_changes(
    scope: KillSwitchScope,
    reason: KillSwitchGateReason,
) -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id,
        at=fixture.request.submitted_at - timedelta(seconds=1),
    )
    kill_service.activate(
        _activation(
            scope=scope,
            account_id=(fixture.account.account_id if scope is KillSwitchScope.ACCOUNT else None),
            at=fixture.request.submitted_at,
            label=scope.value.lower(),
        )
    )
    state_before = fixture.repository.get_account(fixture.account.account_id)

    with pytest.raises(KillSwitchOrderBlocked) as raised:
        _gated_paper_service(fixture, kill_service).submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        )

    assert reason in raised.value.decision.reason_codes
    assert fixture.repository.get_account(fixture.account.account_id) == state_before
    assert (
        fixture.repository.receipt_by_idempotency_key(
            fixture.account.account_id,
            fixture.request.idempotency_key,
        )
        is None
    )


def test_missing_account_switch_fails_closed() -> None:
    fixture = _execution_fixture()
    kill_service = KillSwitchService(repository=InMemoryKillSwitchRepository())
    kill_service.initialize(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=SYSTEM,
        changed_at=fixture.request.submitted_at,
    )

    with pytest.raises(KillSwitchOrderBlocked) as raised:
        _gated_paper_service(fixture, kill_service).submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        )

    assert raised.value.decision.reason_codes == (KillSwitchGateReason.STATE_UNAVAILABLE,)


def test_committed_idempotent_retry_is_returned_even_after_activation() -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id,
        at=fixture.request.submitted_at - timedelta(seconds=1),
    )
    service = _gated_paper_service(fixture, kill_service)
    committed = service.submit_and_match(request=fixture.request, draft=fixture.draft)
    kill_service.activate(
        _activation(
            scope=KillSwitchScope.ACCOUNT,
            account_id=fixture.account.account_id,
            at=fixture.request.submitted_at + timedelta(seconds=3),
            label="after-commit",
        )
    )

    replay = service.submit_and_match(request=fixture.request, draft=fixture.draft)

    assert replay is committed
    assert not any(
        event.event_type.value == "ORDER_BLOCKED" for event in kill_service.repository.events()
    )


def test_activation_waits_for_an_allowed_submission_to_commit() -> None:
    paper_repository = _BlockingPaperRepository()
    fixture = _execution_fixture(repository=paper_repository)
    kill_repository = _ObservableActivationRepository()
    kill_service = KillSwitchService(repository=kill_repository)
    initialized_at = fixture.request.submitted_at - timedelta(seconds=1)
    for scope, account_id in (
        (KillSwitchScope.GLOBAL, None),
        (KillSwitchScope.ACCOUNT, fixture.account.account_id),
    ):
        kill_service.initialize(
            scope=scope,
            account_id=account_id,
            actor=SYSTEM,
            changed_at=initialized_at,
        )
    service = _gated_paper_service(fixture, kill_service)
    activation = _activation(
        scope=KillSwitchScope.ACCOUNT,
        account_id=fixture.account.account_id,
        at=fixture.request.submitted_at + timedelta(seconds=3),
        label="linearized",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        submit_future = executor.submit(
            service.submit_and_match,
            request=fixture.request,
            draft=fixture.draft,
        )
        assert paper_repository.commit_entered.wait(timeout=10)
        activation_future = executor.submit(kill_service.activate, activation)
        assert kill_repository.activation_called.wait(timeout=10)
        with pytest.raises(TimeoutError):
            activation_future.result(timeout=0.05)
        paper_repository.release_commit.set()
        receipt = submit_future.result(timeout=10)
        activated = activation_future.result(timeout=10)

    assert receipt.request_hash == fixture.request.request_hash
    assert activated.state_after.status is KillSwitchStatus.ACTIVE


def test_direct_repository_gate_cannot_bypass_an_active_switch() -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id,
        at=fixture.request.submitted_at - timedelta(seconds=1),
    )
    kill_service.activate(
        _activation(
            scope=KillSwitchScope.ACCOUNT,
            account_id=fixture.account.account_id,
            at=fixture.request.submitted_at,
            label="direct-repository",
        )
    )
    with pytest.raises(KillSwitchOrderBlocked):
        _gated_paper_service(fixture, kill_service.repository).submit_and_match(
            request=fixture.request, draft=fixture.draft
        )
    assert fixture.repository.get_account(fixture.account.account_id) == fixture.account


class _FailOncePaperRepository(InMemoryPaperRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = True

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        if self.fail_next:
            self.fail_next = False
            raise OSError("transient paper commit failure")
        return super().commit_execution(expected_state_hash, request, state_after, receipt)


def test_order_retry_after_transient_failure_uses_stable_gate_identity() -> None:
    fixture = _execution_fixture(repository=_FailOncePaperRepository())
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id, at=fixture.request.submitted_at
    )
    times = iter(fixture.request.submitted_at + timedelta(seconds=seconds) for seconds in (1, 2))
    service = PaperExecutionService(
        engine=fixture.engine,
        repository=fixture.repository,
        new_order_gate=kill_service,
        clock=lambda: next(times),
    )
    with pytest.raises(OSError, match="transient paper commit"):
        service.submit_and_match(request=fixture.request, draft=fixture.draft)
    receipt = service.submit_and_match(request=fixture.request, draft=fixture.draft)
    assert receipt.request_hash == fixture.request.request_hash


def test_blocked_retry_with_later_clock_replays_original_decision() -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id, at=fixture.request.submitted_at
    )
    kill_service.activate(
        _activation(
            scope=KillSwitchScope.ACCOUNT,
            account_id=fixture.account.account_id,
            at=fixture.request.submitted_at,
            label="retry-blocked",
        )
    )
    times = iter(fixture.request.submitted_at + timedelta(seconds=seconds) for seconds in (1, 2))
    service = PaperExecutionService(
        engine=fixture.engine,
        repository=fixture.repository,
        new_order_gate=kill_service,
        clock=lambda: next(times),
    )
    decisions = []
    for _ in range(2):
        with pytest.raises(KillSwitchOrderBlocked) as raised:
            service.submit_and_match(request=fixture.request, draft=fixture.draft)
        decisions.append(raised.value.decision)
    assert decisions[0] == decisions[1]
    assert KillSwitchGateReason.ACCOUNT_ACTIVE in decisions[1].reason_codes
    assert (
        sum(event.event_type.value == "ORDER_BLOCKED" for event in kill_service.repository.events())
        == 1
    )


def test_retry_returns_concurrent_receipt_even_when_activation_wins_gate_race() -> None:
    repository = _LookupRacePaperRepository()
    fixture = _execution_fixture(repository=repository)
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id, at=fixture.request.submitted_at
    )
    service = _gated_paper_service(fixture, kill_service)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="delayed-paper-retry") as executor:
        retry = executor.submit(
            service.submit_and_match, request=fixture.request, draft=fixture.draft
        )
        assert repository.empty_lookup_observed.wait(timeout=5)
        try:
            committed = service.submit_and_match(request=fixture.request, draft=fixture.draft)
            kill_service.activate(
                _activation(
                    scope=KillSwitchScope.ACCOUNT,
                    account_id=fixture.account.account_id,
                    at=fixture.request.submitted_at + timedelta(seconds=1),
                    label="concurrent-retry",
                )
            )
        finally:
            repository.commit_completed.set()
        assert retry.result(timeout=5) == committed


def test_allowed_decision_for_another_request_fails_closed() -> None:
    fixture = _execution_fixture()
    kill_service = _initialized_kill_service(
        account_id=fixture.account.account_id, at=fixture.request.submitted_at
    )

    class WrongRequestGate:
        @contextmanager
        def guard_new_order(
            self, request: KillSwitchGateRequest, actor: KillSwitchActor
        ) -> Iterator[KillSwitchGateDecision]:
            wrong = KillSwitchGateRequest.build(
                request_id="wrong-request",
                account_id=request.account_id,
                idempotency_key=request.idempotency_key,
                batch_hash=request.batch_hash,
                source_request_hash=request.source_request_hash,
                checked_at=request.checked_at,
            )
            with kill_service.guard_new_order(wrong, actor) as decision:
                yield decision

    with pytest.raises(KillSwitchOrderBlocked) as raised:
        _gated_paper_service(fixture, WrongRequestGate()).submit_and_match(
            request=fixture.request, draft=fixture.draft
        )
    assert raised.value.decision.reason_codes == (KillSwitchGateReason.REPOSITORY_FAILURE,)
    assert fixture.repository.get_account(fixture.account.account_id) == fixture.account
