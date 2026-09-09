"""Repository, service idempotency, and isolation tests for paper execution."""

from __future__ import annotations

import ast
import http.client
import inspect
import socket
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier, Event, current_thread
from typing import NoReturn

import pytest
from tests.unit import test_order_draft_generator as draft_support
from tests.unit import test_paper_execution_engine as engine_support

import quant_agent.execution.paper as paper_package
from quant_agent.backtest.cn_market import CNSlippageModelBook
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.execution.paper import (
    InMemoryPaperRepository,
    PaperAccountConflict,
    PaperAccountNotFound,
    PaperAccountState,
    PaperConcurrentUpdate,
    PaperExecutionEngine,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperExecutionService,
    PaperIdempotencyConflict,
    PaperRepositoryError,
    SQLitePaperRepository,
)
from quant_agent.portfolio import AccountSnapshot
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.kill_switch import (
    KillSwitchActor,
    KillSwitchGateDecision,
    KillSwitchGateRequest,
    KillSwitchGateStatus,
)


@dataclass(frozen=True, slots=True)
class _ExecutionFixture:
    snapshot: AccountSnapshot
    draft: OrderDraftBatch
    engine: PaperExecutionEngine
    repository: InMemoryPaperRepository
    service: PaperExecutionService
    account: PaperAccountState
    request: PaperExecutionRequest


class _ExplicitTestAllowGate:
    """Test-only opt-in that makes legacy paper repository tests explicit."""

    @contextmanager
    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> Iterator[KillSwitchGateDecision]:
        del actor
        state_hash = stable_hash({"test-only-allow-gate": request.request_hash})
        yield KillSwitchGateDecision.build(
            request=request,
            status=KillSwitchGateStatus.ALLOWED,
            global_state_hash=state_hash,
            account_state_hash=state_hash,
        )


_TEST_ALLOW_GATE = _ExplicitTestAllowGate()


class _BarrierPaperRepository(InMemoryPaperRepository):
    """Hold competing commits until both executions have planned from one CAS state."""

    def __init__(self) -> None:
        super().__init__()
        self._commit_barrier = Barrier(2)

    def commit_execution(
        self,
        expected_state_hash: str,
        request: PaperExecutionRequest,
        state_after: PaperAccountState,
        receipt: PaperExecutionReceipt,
    ) -> PaperExecutionReceipt:
        self._commit_barrier.wait(timeout=10)
        return super().commit_execution(
            expected_state_hash=expected_state_hash,
            request=request,
            state_after=state_after,
            receipt=receipt,
        )


class _LookupRacePaperRepository(InMemoryPaperRepository):
    """Pause one retry after its empty batch lookup to force a commit race."""

    def __init__(self) -> None:
        super().__init__()
        self.empty_lookup_observed = Event()
        self.commit_completed = Event()

    def receipt_by_batch_hash(
        self,
        account_id: str,
        batch_hash: str,
    ) -> PaperExecutionReceipt | None:
        result = super().receipt_by_batch_hash(account_id, batch_hash)
        if current_thread().name.startswith("delayed-paper-retry") and result is None:
            self.empty_lookup_observed.set()
            assert self.commit_completed.wait(timeout=10)
        return result


def _digest(label: str) -> str:
    return stable_hash({"paper-repository-service-test": label})


def _replace_receipt_account_after(
    receipt: PaperExecutionReceipt,
    account_after: PaperAccountState,
) -> PaperExecutionReceipt:
    payload = asdict(receipt)
    payload.pop("receipt_hash")
    payload["account_after_hash"] = account_after.state_hash
    payload["account_after"] = asdict(account_after)
    return replace(
        receipt,
        account_after_hash=account_after.state_hash,
        account_after=account_after,
        receipt_hash=stable_hash(payload),
    )


def _request(
    *,
    engine: PaperExecutionEngine,
    account: PaperAccountState,
    draft: OrderDraftBatch,
    request_id: str = "paper-service-request-1",
    idempotency_key: str = "paper-service-key-1",
    batch_hash: str | None = None,
    expected_snapshot_hash: str | None = None,
) -> PaperExecutionRequest:
    return PaperExecutionRequest.build(
        request_id=request_id,
        idempotency_key=idempotency_key,
        account_id=account.account_id,
        expected_account_state_hash=account.state_hash,
        expected_account_snapshot_hash=expected_snapshot_hash or account.source_snapshot_hash,
        batch_hash=batch_hash or draft.batch_hash,
        submitted_at=draft.draft_as_of + timedelta(minutes=1),
        config=engine.config,
    )


def _execution_fixture(
    repository: InMemoryPaperRepository | None = None,
) -> _ExecutionFixture:
    chain = draft_support._chain()
    states = draft_support._market_states(chain.proposal)
    next_trading_day = draft_support.TRADING_DAY + timedelta(days=3)
    rule = replace(
        draft_support._market_rule(states),
        trading_calendar=(draft_support.TRADING_DAY, next_trading_day),
    )
    draft = draft_support._generate(chain, states=states, rules=(rule,))
    engine = PaperExecutionEngine(
        market_rules=(rule,),
        fee_rule_book=draft_support._fee_rule_book(),
        slippage_model_book=CNSlippageModelBook((draft_support._slippage_model(),)),
    )
    active_repository = repository or InMemoryPaperRepository()
    service = PaperExecutionService(
        engine=engine,
        repository=active_repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    account = service.open_account(snapshot=chain.account)
    request = _request(engine=engine, account=account, draft=draft)
    return _ExecutionFixture(
        snapshot=chain.account,
        draft=draft,
        engine=engine,
        repository=active_repository,
        service=service,
        account=account,
        request=request,
    )


def _capture_outcome(
    future: Future[PaperExecutionReceipt],
) -> PaperExecutionReceipt | Exception:
    try:
        return future.result(timeout=15)
    except Exception as error:
        return error


def _race(
    *calls: Callable[[], PaperExecutionReceipt],
) -> tuple[PaperExecutionReceipt | Exception, ...]:
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = tuple(executor.submit(call) for call in calls)
        return tuple(_capture_outcome(future) for future in futures)


def _different_batch_draft() -> OrderDraftBatch:
    chain = draft_support._chain()
    states = draft_support._market_states(chain.proposal)
    next_trading_day = draft_support.TRADING_DAY + timedelta(days=3)
    rule = replace(
        draft_support._market_rule(states),
        trading_calendar=(draft_support.TRADING_DAY, next_trading_day),
    )
    return draft_support._generate(
        chain,
        states=states,
        rules=(rule,),
        draft_as_of=draft_support.DRAFT_AS_OF + timedelta(seconds=1),
    )


def test_open_account_is_exactly_idempotent_and_rejects_conflicts_and_unknown_ids() -> None:
    fixture = _execution_fixture()

    reopened = fixture.service.open_account(snapshot=fixture.snapshot)

    assert reopened is fixture.account
    assert fixture.repository.get_account(fixture.account.account_id) is fixture.account
    conflicting_snapshot = draft_support._account(available_cash=Decimal(7000))
    with pytest.raises(PaperAccountConflict, match="different state"):
        fixture.service.open_account(snapshot=conflicting_snapshot)
    with pytest.raises(PaperAccountNotFound, match="unknown paper account"):
        fixture.repository.get_account("paper-account-does-not-exist")


def test_same_idempotency_key_and_request_return_one_receipt_without_duplicate_fills() -> None:
    fixture = _execution_fixture()

    first = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    committed_after_first = fixture.repository.get_account(fixture.account.account_id)
    retried = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)

    assert first.fills
    assert retried is first
    assert retried.receipt_hash == first.receipt_hash
    assert fixture.repository.get_account(fixture.account.account_id) is committed_after_first
    assert len(retried.fills) == len(first.fills)
    assert committed_after_first.processed_batch_hashes == (fixture.draft.batch_hash,)
    assert {order.order_hash for order in committed_after_first.orders} == {
        order.order_hash for order in first.orders
    }


def test_same_idempotency_key_with_different_request_is_a_conflict() -> None:
    fixture = _execution_fixture()
    committed = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    changed_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-request-changed",
        idempotency_key=fixture.request.idempotency_key,
    )

    with pytest.raises(PaperIdempotencyConflict, match="different execution request"):
        fixture.service.submit_and_match(request=changed_request, draft=fixture.draft)

    assert fixture.repository.get_account(fixture.account.account_id) == committed.account_after


def test_same_batch_with_a_new_delivery_key_is_a_conflict() -> None:
    fixture = _execution_fixture()
    first = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    redelivery = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-redelivery-request",
        idempotency_key="paper-service-redelivery-key",
    )

    with pytest.raises(PaperIdempotencyConflict, match="different execution request"):
        fixture.service.submit_and_match(request=redelivery, draft=fixture.draft)

    assert (
        fixture.repository.receipt_by_batch_hash(
            fixture.account.account_id,
            fixture.draft.batch_hash,
        )
        is first
    )


def test_consumed_batch_does_not_bypass_draft_validation() -> None:
    fixture = _execution_fixture()
    committed = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    different_draft = _different_batch_draft()

    with pytest.raises(PaperExecutionInputError, match="request batch hash"):
        fixture.service.submit_and_match(request=fixture.request, draft=different_draft)

    assert fixture.repository.get_account(fixture.account.account_id) == committed.account_after


def test_concurrent_same_key_and_request_commit_exactly_one_transition() -> None:
    fixture = _execution_fixture(_BarrierPaperRepository())

    outcomes = _race(
        lambda: fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        ),
        lambda: fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        ),
    )

    receipts = tuple(outcome for outcome in outcomes if isinstance(outcome, PaperExecutionReceipt))
    assert len(receipts) == 2
    assert receipts[0] is receipts[1]
    assert not tuple(outcome for outcome in outcomes if isinstance(outcome, Exception))
    assert fixture.repository.get_account(fixture.account.account_id).processed_batch_hashes == (
        fixture.draft.batch_hash,
    )


def test_retry_recovers_when_the_first_commit_wins_after_its_empty_lookup() -> None:
    repository = _LookupRacePaperRepository()
    fixture = _execution_fixture(repository)
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="delayed-paper-retry",
    ) as executor:
        delayed = executor.submit(
            fixture.service.submit_and_match,
            request=fixture.request,
            draft=fixture.draft,
        )
        assert repository.empty_lookup_observed.wait(timeout=10)
        committed = fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        )
        repository.commit_completed.set()
        recovered = delayed.result(timeout=10)

    assert recovered is committed
    assert repository.get_account(fixture.account.account_id) == committed.account_after


def test_concurrent_same_key_with_different_requests_conflicts() -> None:
    fixture = _execution_fixture(_BarrierPaperRepository())
    competing_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-concurrent-same-key",
        idempotency_key=fixture.request.idempotency_key,
    )

    outcomes = _race(
        lambda: fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        ),
        lambda: fixture.service.submit_and_match(
            request=competing_request,
            draft=fixture.draft,
        ),
    )

    assert sum(isinstance(outcome, PaperExecutionReceipt) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, PaperIdempotencyConflict) for outcome in outcomes) == 1
    assert fixture.repository.get_account(fixture.account.account_id).processed_batch_hashes == (
        fixture.draft.batch_hash,
    )


def test_concurrent_different_keys_for_same_batch_conflict() -> None:
    fixture = _execution_fixture(_BarrierPaperRepository())
    competing_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-concurrent-same-batch",
        idempotency_key="paper-service-concurrent-same-batch-key",
    )

    outcomes = _race(
        lambda: fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        ),
        lambda: fixture.service.submit_and_match(
            request=competing_request,
            draft=fixture.draft,
        ),
    )

    assert sum(isinstance(outcome, PaperExecutionReceipt) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, PaperIdempotencyConflict) for outcome in outcomes) == 1
    assert fixture.repository.get_account(fixture.account.account_id).processed_batch_hashes == (
        fixture.draft.batch_hash,
    )


def test_concurrent_different_batches_from_same_cas_allow_only_one_commit() -> None:
    fixture = _execution_fixture(_BarrierPaperRepository())
    competing_draft = _different_batch_draft()
    assert competing_draft.batch_hash != fixture.draft.batch_hash
    competing_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=competing_draft,
        request_id="paper-service-concurrent-different-batch",
        idempotency_key="paper-service-concurrent-different-batch-key",
    )

    outcomes = _race(
        lambda: fixture.service.submit_and_match(
            request=fixture.request,
            draft=fixture.draft,
        ),
        lambda: fixture.service.submit_and_match(
            request=competing_request,
            draft=competing_draft,
        ),
    )

    assert sum(isinstance(outcome, PaperExecutionReceipt) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, PaperConcurrentUpdate) for outcome in outcomes) == 1
    committed = fixture.repository.get_account(fixture.account.account_id)
    assert len(committed.processed_batch_hashes) == 1
    assert committed.processed_batch_hashes[0] in {
        fixture.draft.batch_hash,
        competing_draft.batch_hash,
    }


def test_repository_detects_a_stale_compare_and_swap_commit() -> None:
    fixture = _execution_fixture()
    committed = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    stale_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-racing-request",
        idempotency_key="paper-service-racing-key",
        batch_hash=_digest("racing-batch"),
    )

    with pytest.raises(PaperConcurrentUpdate, match="changed before execution commit"):
        fixture.repository.commit_execution(
            expected_state_hash=fixture.account.state_hash,
            request=stale_request,
            state_after=committed.account_after,
            receipt=committed,
        )

    assert fixture.repository.get_account(fixture.account.account_id) == committed.account_after


def test_recreated_service_with_the_same_repository_preserves_idempotency() -> None:
    fixture = _execution_fixture()
    first = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    restarted_service = PaperExecutionService(
        engine=fixture.engine,
        repository=fixture.repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )

    recovered = restarted_service.submit_and_match(
        request=fixture.request,
        draft=fixture.draft,
    )

    assert recovered is first
    assert recovered.account_after == fixture.repository.get_account(fixture.account.account_id)


def test_sqlite_repository_preserves_account_and_idempotency_after_reopen(
    tmp_path: Path,
) -> None:
    fixture = _execution_fixture()
    database_path = tmp_path / "paper-execution.db"
    first_repository = SQLitePaperRepository(database_path)
    first_service = PaperExecutionService(
        engine=fixture.engine,
        repository=first_repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    opened = first_service.open_account(snapshot=fixture.snapshot)
    assert opened == fixture.account
    first = first_service.submit_and_match(request=fixture.request, draft=fixture.draft)

    reopened_repository = SQLitePaperRepository(database_path)
    reopened_service = PaperExecutionService(
        engine=fixture.engine,
        repository=reopened_repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    recovered = reopened_service.submit_and_match(
        request=fixture.request,
        draft=fixture.draft,
    )

    assert recovered == first
    assert reopened_repository.get_account(opened.account_id) == first.account_after
    assert (
        reopened_repository.receipt_by_idempotency_key(
            opened.account_id,
            fixture.request.idempotency_key,
        )
        == first
    )
    assert (
        reopened_repository.receipt_by_batch_hash(
            opened.account_id,
            fixture.draft.batch_hash,
        )
        == first
    )
    database_path.unlink()
    assert not database_path.exists()


def test_sqlite_repository_conflicts_refresh_and_cas_are_transactional(tmp_path: Path) -> None:
    fixture = _execution_fixture()
    repository = SQLitePaperRepository(tmp_path / "paper-boundaries.db")
    with pytest.raises(PaperAccountNotFound, match="unknown paper account"):
        repository.get_account("missing-paper-account")
    assert repository.receipt_by_idempotency_key("missing", "key") is None
    assert repository.receipt_by_batch_hash("missing", _digest("batch")) is None

    opened = repository.open_account(fixture.account)
    assert repository.open_account(fixture.account) == opened
    conflicting_open = engine_support._rebuild_account(
        opened,
        event_log_hash=_digest("different-opening-event-log"),
    )
    with pytest.raises(PaperAccountConflict, match="different state"):
        repository.open_account(conflicting_open)

    planned = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=opened,
    )
    committed = repository.commit_execution(
        opened.state_hash,
        fixture.request,
        planned.account_after,
        planned,
    )
    assert (
        repository.commit_execution(
            opened.state_hash,
            fixture.request,
            planned.account_after,
            planned,
        )
        == committed
    )
    same_key_different_request = _request(
        engine=fixture.engine,
        account=opened,
        draft=fixture.draft,
        request_id="sqlite-same-key-different-request",
        idempotency_key=fixture.request.idempotency_key,
    )
    with pytest.raises(PaperIdempotencyConflict, match="idempotency key"):
        repository.commit_execution(
            opened.state_hash,
            same_key_different_request,
            planned.account_after,
            planned,
        )
    same_batch_different_key = _request(
        engine=fixture.engine,
        account=opened,
        draft=fixture.draft,
        request_id="sqlite-same-batch-different-request",
        idempotency_key="sqlite-same-batch-different-key",
    )
    with pytest.raises(PaperIdempotencyConflict, match="draft batch"):
        repository.commit_execution(
            opened.state_hash,
            same_batch_different_key,
            planned.account_after,
            planned,
        )

    refresh_snapshot = engine_support._refresh_snapshot(
        committed.account_after,
        as_of=committed.processed_at + timedelta(days=3),
        snapshot_id="sqlite-paper-refresh",
    )
    refreshed = fixture.engine.refresh_account(
        account=committed.account_after,
        snapshot=refresh_snapshot,
    )
    assert (
        repository.refresh_account(
            committed.account_after.state_hash,
            refresh_snapshot,
            refreshed,
        )
        == refreshed
    )
    assert (
        repository.refresh_account(
            committed.account_after.state_hash,
            refresh_snapshot,
            refreshed,
        )
        == refreshed
    )
    next_snapshot = engine_support._refresh_snapshot(
        refreshed,
        as_of=refresh_snapshot.as_of + timedelta(days=1),
        snapshot_id="sqlite-paper-refresh-next",
    )
    next_state = fixture.engine.refresh_account(account=refreshed, snapshot=next_snapshot)
    assert repository.refresh_account(refreshed.state_hash, next_snapshot, next_state) == next_state
    sqlite_service = PaperExecutionService(
        engine=fixture.engine,
        repository=repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    assert sqlite_service.refresh_account(snapshot=refresh_snapshot) == next_state
    assert (
        repository.account_by_refresh_snapshot_hash(
            refreshed.account_id,
            refresh_snapshot.content_hash,
        )
        == refreshed
    )
    third_snapshot = engine_support._refresh_snapshot(
        next_state,
        as_of=next_snapshot.as_of + timedelta(days=1),
        snapshot_id="sqlite-paper-refresh-third",
    )
    third_state = fixture.engine.refresh_account(account=next_state, snapshot=third_snapshot)
    with pytest.raises(PaperConcurrentUpdate, match="refresh commit"):
        repository.refresh_account(
            committed.account_after.state_hash,
            third_snapshot,
            third_state,
        )
    reopened = SQLitePaperRepository(repository.path)
    assert reopened.get_account(opened.account_id) == next_state
    assert (
        reopened.account_by_refresh_snapshot_hash(
            refreshed.account_id,
            refresh_snapshot.content_hash,
        )
        == refreshed
    )


def test_sqlite_repository_rejects_unknown_schema_version(tmp_path: Path) -> None:
    database_path = tmp_path / "future-paper-schema.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(PaperRepositoryError, match=r"unsupported.*schema version"):
        SQLitePaperRepository(database_path)


def test_sqlite_repository_rejects_sql_row_identity_tampering(tmp_path: Path) -> None:
    fixture = _execution_fixture()
    account_path = tmp_path / "tampered-account-row.db"
    account_repository = SQLitePaperRepository(account_path)
    account_repository.open_account(fixture.account)
    with sqlite3.connect(account_path) as connection:
        connection.execute(
            "UPDATE paper_accounts SET account_id = ? WHERE account_id = ?",
            ("paper-account-alias", fixture.account.account_id),
        )
    with pytest.raises(PaperRepositoryError, match="row identity does not match"):
        account_repository.get_account("paper-account-alias")

    receipt_path = tmp_path / "tampered-receipt-row.db"
    receipt_repository = SQLitePaperRepository(receipt_path)
    service = PaperExecutionService(
        engine=fixture.engine,
        repository=receipt_repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    service.open_account(snapshot=fixture.snapshot)
    service.submit_and_match(request=fixture.request, draft=fixture.draft)
    with sqlite3.connect(receipt_path) as connection:
        connection.execute(
            "UPDATE paper_receipts SET idempotency_key = ?, request_hash = ?",
            ("paper-key-alias", _digest("request-alias")),
        )
    with pytest.raises(PaperRepositoryError, match="row identity does not match"):
        receipt_repository.receipt_by_idempotency_key(
            fixture.account.account_id,
            "paper-key-alias",
        )

    refresh_path = tmp_path / "tampered-refresh-row.db"
    refresh_repository = SQLitePaperRepository(refresh_path)
    refresh_service = PaperExecutionService(
        engine=fixture.engine,
        repository=refresh_repository,
        new_order_gate=_TEST_ALLOW_GATE,
    )
    refresh_service.open_account(snapshot=fixture.snapshot)
    first = refresh_service.submit_and_match(request=fixture.request, draft=fixture.draft)
    refresh_snapshot = engine_support._refresh_snapshot(
        first.account_after,
        as_of=first.processed_at + timedelta(days=3),
        snapshot_id="tampered-refresh-snapshot",
    )
    refresh_service.refresh_account(snapshot=refresh_snapshot)
    with sqlite3.connect(refresh_path) as connection:
        connection.execute("UPDATE paper_refreshes SET state_before_json = state_after_json")
    with pytest.raises(PaperRepositoryError, match="row identity does not match"):
        refresh_repository.account_by_refresh_snapshot_hash(
            fixture.account.account_id,
            refresh_snapshot.content_hash,
        )


def test_sqlite_backend_error_rolls_back_account_and_uses_repository_error(tmp_path: Path) -> None:
    fixture = _execution_fixture()
    database_path = tmp_path / "paper-rollback.db"
    repository = SQLitePaperRepository(database_path)
    repository.open_account(fixture.account)
    planned = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_paper_receipt BEFORE INSERT ON paper_receipts "
            "BEGIN SELECT RAISE(ABORT, 'forced receipt failure'); END"
        )

    with pytest.raises(PaperRepositoryError, match="SQLite paper repository operation failed"):
        repository.commit_execution(
            fixture.account.state_hash,
            fixture.request,
            planned.account_after,
            planned,
        )
    assert repository.get_account(fixture.account.account_id) == fixture.account
    assert (
        repository.receipt_by_idempotency_key(
            fixture.account.account_id,
            fixture.request.idempotency_key,
        )
        is None
    )


def test_service_commits_snapshot_linked_refresh_and_repository_enforces_cas() -> None:
    fixture = _execution_fixture()
    first = fixture.service.submit_and_match(request=fixture.request, draft=fixture.draft)
    with pytest.raises(PaperExecutionInputError, match="snapshot must be newer"):
        fixture.service.refresh_account(snapshot=fixture.snapshot)
    refresh_snapshot = engine_support._refresh_snapshot(
        first.account_after,
        as_of=first.processed_at + timedelta(days=3),
    )

    refreshed = fixture.service.refresh_account(snapshot=refresh_snapshot)

    assert refreshed.source_snapshot_hash == refresh_snapshot.content_hash
    assert refreshed.previous_state_hash == first.account_after.state_hash
    assert fixture.repository.get_account(refreshed.account_id) == refreshed
    assert fixture.service.refresh_account(snapshot=refresh_snapshot) == refreshed
    assert (
        fixture.repository.refresh_account(
            first.account_after.state_hash,
            refresh_snapshot,
            refreshed,
        )
        == refreshed
    )
    next_snapshot = engine_support._refresh_snapshot(
        refreshed,
        as_of=refresh_snapshot.as_of + timedelta(days=1),
        snapshot_id="paper-refresh-next",
    )
    next_state = fixture.engine.refresh_account(account=refreshed, snapshot=next_snapshot)
    assert fixture.service.refresh_account(snapshot=next_snapshot) == next_state
    assert fixture.service.refresh_account(snapshot=refresh_snapshot) == next_state
    assert (
        fixture.repository.account_by_refresh_snapshot_hash(
            refreshed.account_id,
            refresh_snapshot.content_hash,
        )
        == refreshed
    )
    third_snapshot = engine_support._refresh_snapshot(
        next_state,
        as_of=next_snapshot.as_of + timedelta(days=1),
        snapshot_id="paper-refresh-third",
    )
    third_state = fixture.engine.refresh_account(account=next_state, snapshot=third_snapshot)
    with pytest.raises(PaperConcurrentUpdate, match="refresh commit"):
        fixture.repository.refresh_account(
            first.account_after.state_hash,
            third_snapshot,
            third_state,
        )


def test_repository_rejects_tampered_request_and_receipt_commit_bindings() -> None:
    fixture = _execution_fixture()
    planned = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )
    wrong_snapshot_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-wrong-snapshot-request",
        idempotency_key="paper-service-wrong-snapshot-key",
        expected_snapshot_hash=_digest("wrong-source-snapshot"),
    )
    with pytest.raises(PaperAccountConflict, match="source snapshot"):
        fixture.repository.commit_execution(
            expected_state_hash=fixture.account.state_hash,
            request=wrong_snapshot_request,
            state_after=planned.account_after,
            receipt=planned,
        )

    wrong_receipt_request = _request(
        engine=fixture.engine,
        account=fixture.account,
        draft=fixture.draft,
        request_id="paper-service-wrong-receipt-request",
        idempotency_key="paper-service-wrong-receipt-key",
    )
    with pytest.raises(PaperAccountConflict, match="receipt does not match"):
        fixture.repository.commit_execution(
            expected_state_hash=fixture.account.state_hash,
            request=wrong_receipt_request,
            state_after=planned.account_after,
            receipt=planned,
        )

    assert fixture.repository.get_account(fixture.account.account_id) is fixture.account


def test_receipt_contract_replays_economics_before_repository_commit() -> None:
    fixture = _execution_fixture()
    planned = fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )
    expected = planned.account_after
    forged_state = PaperAccountState.build(
        account_id=expected.account_id,
        as_of=expected.as_of,
        data_version=expected.data_version,
        currency=expected.currency,
        source_snapshot_id=expected.source_snapshot_id,
        source_snapshot_hash=expected.source_snapshot_hash,
        source_snapshot_as_of=expected.source_snapshot_as_of,
        total_cash=expected.total_cash + Decimal(1_000_000),
        external_frozen_cash=expected.external_frozen_cash,
        positions=expected.positions,
        orders=expected.orders,
        processed_batch_hashes=expected.processed_batch_hashes,
        previous_state_hash=expected.previous_state_hash,
        event_log_hash=expected.event_log_hash,
    )
    with pytest.raises(ValueError, match="canonical paper execution replay"):
        _replace_receipt_account_after(planned, forged_state)
    assert fixture.repository.get_account(fixture.account.account_id) is fixture.account


def _paper_source_files() -> tuple[Path, ...]:
    package_file = paper_package.__file__
    assert package_file is not None
    return tuple(sorted(Path(package_file).parent.glob("*.py")))


def test_paper_package_has_no_network_or_broker_credential_dependency_boundary() -> None:
    forbidden_import_roots = {
        "aiohttp",
        "http",
        "httpx",
        "requests",
        "socket",
        "urllib",
        "websockets",
    }
    forbidden_identifiers = ("broker", "client", "token", "secret")

    for source_file in _paper_source_files():
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        imported_roots: set[str] = set()
        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
            elif isinstance(node, ast.Name):
                identifiers.add(node.id.casefold())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.casefold())
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg.casefold())
        assert imported_roots.isdisjoint(forbidden_import_roots)
        assert not {
            identifier
            for identifier in identifiers
            if any(marker in identifier for marker in forbidden_identifiers)
        }

    for export_name in paper_package.__all__:
        exported = getattr(paper_package, export_name)
        if not inspect.isclass(exported):
            continue
        if "__init__" not in exported.__dict__ and "__new__" not in exported.__dict__:
            continue
        constructor = str(inspect.signature(exported)).casefold()
        assert not any(marker in constructor for marker in forbidden_identifiers)


def test_paper_execution_does_not_use_network_or_leak_environment_broker_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("paper execution must not access a network transport")

    monkeypatch.setattr(socket, "socket", network_forbidden)
    monkeypatch.setattr(http.client.HTTPConnection, "request", network_forbidden)

    first_bait = "paper-broker-token-bait-alpha"
    monkeypatch.setenv("BROKER_TOKEN", first_bait)
    first_fixture = _execution_fixture()
    first_receipt = first_fixture.service.submit_and_match(
        request=first_fixture.request,
        draft=first_fixture.draft,
    )

    second_bait = "paper-broker-token-bait-beta"
    monkeypatch.setenv("BROKER_TOKEN", second_bait)
    second_fixture = _execution_fixture()
    second_receipt = second_fixture.service.submit_and_match(
        request=second_fixture.request,
        draft=second_fixture.draft,
    )

    assert second_fixture.account.state_hash == first_fixture.account.state_hash
    assert second_receipt == first_receipt
    assert second_receipt.receipt_hash == first_receipt.receipt_hash
    exposed_content = repr(
        (
            first_fixture.account,
            asdict(first_fixture.account),
            first_receipt,
            asdict(first_receipt),
            second_fixture.account,
            asdict(second_fixture.account),
            second_receipt,
            asdict(second_receipt),
        )
    )
    assert first_bait not in exposed_content
    assert second_bait not in exposed_content
