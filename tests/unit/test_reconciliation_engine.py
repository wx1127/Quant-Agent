"""Real-chain acceptance tests for independent paper execution reconciliation."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_agent.execution.paper import (
    PaperExecutionReceipt,
    PaperMatchAttempt,
    PaperNoFillReason,
    PaperOrder,
    PaperOrderStatus,
)
from quant_agent.execution.paper.transition import replay_execution_transition
from quant_agent.portfolio import AccountSnapshot, CashSnapshot, PortfolioPositionSnapshot
from quant_agent.reconciliation import (
    DuplicateClassification,
    ObservedExecutionEvidence,
    ObservedFillReport,
    ObservedOrderReport,
    ReconciliationDifferenceCode,
    ReconciliationEngine,
    ReconciliationPolicy,
    ReconciliationRequest,
    ReconciliationSeverity,
    ReconciliationStatus,
    ReconciliationStopAction,
    observed_evidence_from_paper_receipt,
)
from quant_agent.regime.contracts import stable_hash

from . import test_order_draft_generator as draft_fixtures
from .test_paper_execution_engine import (
    BUY_ONLY_TARGETS,
    _execution_fixture,
    _ExecutionFixture,
    _StateChange,
)


def _snapshot(
    receipt: PaperExecutionReceipt,
    *,
    cash: CashSnapshot | None = None,
    positions: tuple[PortfolioPositionSnapshot, ...] | None = None,
    snapshot_id: str = "reconciliation-observed-1",
    previous_snapshot_hash: str | None = None,
    source_event_log_hash: str | None = None,
    account_id: str | None = None,
    data_version: str | None = None,
) -> AccountSnapshot:
    account = receipt.account_after
    as_of = receipt.processed_at
    observed_cash = cash or CashSnapshot(
        as_of=as_of,
        currency=account.currency,
        total_cash=account.total_cash,
        available_cash=account.available_cash,
        frozen_cash=account.frozen_cash,
    )
    observed_positions = positions
    if observed_positions is None:
        observed_positions = tuple(
            PortfolioPositionSnapshot.build(
                instrument_id=item.instrument_id,
                instrument_type=item.instrument_type,
                available_quantity=item.available_quantity,
                frozen_quantity=item.frozen_quantity,
                unsettled_quantity=item.unsettled_quantity,
                average_cost=item.average_cost,
                valuation_price=Decimal(10),
                price_observed_at=as_of,
                price_available_at=as_of,
                position_as_of=as_of,
                price_data_version=account.data_version,
                price_source_hash=stable_hash(
                    {"reconciliation-price": item.instrument_id, "as_of": as_of}
                ),
                mark_policy_hash=stable_hash({"reconciliation-mark": "v1"}),
            )
            for item in account.positions
        )
    return AccountSnapshot.build(
        snapshot_id=snapshot_id,
        account_id=account_id or account.account_id,
        runtime_mode=account.runtime_mode,
        as_of=as_of,
        valuation_at=as_of,
        data_version=data_version or account.data_version,
        currency=account.currency,
        cash=observed_cash,
        positions=observed_positions,
        previous_snapshot_hash=previous_snapshot_hash or account.source_snapshot_hash,
        source_event_log_hash=source_event_log_hash or receipt.event_log_hash,
    )


def _request(
    fixture: _ExecutionFixture,
    receipt: PaperExecutionReceipt,
    *,
    snapshot: AccountSnapshot | None = None,
    order_reports: tuple[ObservedOrderReport, ...] | None = None,
    fill_reports: tuple[ObservedFillReport, ...] | None = None,
    evidence_account_id: str | None = None,
    evidence_available_at: datetime | None = None,
    reconciled_at: datetime | None = None,
    policy: ReconciliationPolicy | None = None,
) -> ReconciliationRequest:
    observed_snapshot = snapshot or _snapshot(receipt)
    available_at = evidence_available_at or receipt.processed_at + timedelta(seconds=1)
    base = observed_evidence_from_paper_receipt(
        receipt=receipt,
        account_snapshot=observed_snapshot,
        available_at=available_at,
    )
    evidence = ObservedExecutionEvidence.build(
        source_system=base.source_system,
        account_id=evidence_account_id or base.account_id,
        observed_at=base.observed_at,
        available_at=available_at,
        account_snapshot=observed_snapshot,
        order_reports=base.order_reports if order_reports is None else order_reports,
        fill_reports=base.fill_reports if fill_reports is None else fill_reports,
        source_cursor="paper-cursor-1",
    )
    return ReconciliationRequest.build(
        request_id="reconciliation-request-1",
        idempotency_key="reconciliation-key-1",
        reconciled_at=reconciled_at or available_at + timedelta(seconds=1),
        draft=fixture.draft,
        receipt=receipt,
        observed=evidence,
        policy=policy,
    )


def _execute_full() -> tuple[_ExecutionFixture, PaperExecutionReceipt]:
    fixture = _execution_fixture()
    return fixture, fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )


def _execute_partial() -> tuple[_ExecutionFixture, PaperExecutionReceipt]:
    fixture = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(available_quantity=Decimal(1000)),
        },
    )
    return fixture, fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )


def _execute_no_fill() -> tuple[_ExecutionFixture, PaperExecutionReceipt]:
    fixture = _execution_fixture(
        local_targets=BUY_ONLY_TARGETS,
        state_changes={
            draft_fixtures.BUY_A: _StateChange(available_quantity=Decimal(0)),
        },
    )
    return fixture, fixture.engine.execute(
        request=fixture.request,
        draft=fixture.draft,
        account=fixture.account,
    )


def _execute_rejected() -> tuple[_ExecutionFixture, PaperExecutionReceipt]:
    fixture = _execution_fixture(local_targets=BUY_ONLY_TARGETS)
    line = fixture.draft.lines[0]
    request = fixture.request
    attempt_id = "paper-attempt-rejected"
    order = PaperOrder.build(
        order_id="paper-order-rejected",
        batch_hash=fixture.draft.batch_hash,
        line_hash=line.line_hash,
        decision_id=fixture.draft.decision_id,
        instrument_id=line.instrument_id,
        instrument_type=line.instrument_type,
        side=line.side,
        quantity=line.quantity,
        filled_quantity=Decimal(0),
        is_full_liquidation=line.is_full_liquidation,
        status=PaperOrderStatus.REJECTED,
        created_at=request.submitted_at,
        expires_at=fixture.draft.expires_at,
        cash_reserved=Decimal(0),
        sell_reserved_quantity=Decimal(0),
        market_rule_version=line.market_rule_version,
        market_rule_hash=line.market_rule_hash,
        fee_rule_version=line.fee_rule_version,
        fee_rule_hash=line.fee_rule_hash,
        slippage_model_version=line.slippage_model_version,
        slippage_model_hash=line.slippage_model_hash,
        last_attempt_id=attempt_id,
    )
    processed_at = request.submitted_at + timedelta(seconds=1)
    attempt = PaperMatchAttempt.build(
        attempt_id=attempt_id,
        order_id=order.order_id,
        attempted_at=processed_at,
        status=PaperOrderStatus.REJECTED,
        no_fill_reason=PaperNoFillReason.INSUFFICIENT_CASH,
        requested_quantity=order.quantity,
        filled_quantity=Decimal(0),
        reference_price=line.reference_price,
        execution_price=None,
        participation_rate=Decimal(0),
        state_revision=line.state_revision,
        data_version=line.data_version,
        state_hash=line.state_hash,
        market_rule_version=line.market_rule_version,
        market_rule_hash=line.market_rule_hash,
        slippage_model_version=line.slippage_model_version,
        slippage_model_hash=line.slippage_model_hash,
    )
    event_log_hash = PaperExecutionReceipt.event_log_hash_for(
        previous_event_log_hash=fixture.account.event_log_hash,
        request_hash=request.request_hash,
        batch_hash=fixture.draft.batch_hash,
        orders=(order,),
        attempts=(attempt,),
        fills=(),
    )
    account_after = replay_execution_transition(
        account_before=fixture.account,
        batch_hash=fixture.draft.batch_hash,
        processed_at=processed_at,
        event_log_hash=event_log_hash,
        orders=(order,),
        fills=(),
    )
    receipt = PaperExecutionReceipt.build(
        receipt_id="paper-receipt-rejected",
        request=request,
        account_before=fixture.account,
        account_after=account_after,
        processed_at=processed_at,
        orders=(order,),
        attempts=(attempt,),
        fills=(),
    )
    return fixture, receipt


def _finding_codes(result: object) -> set[ReconciliationDifferenceCode]:
    assert hasattr(result, "all_findings")
    return {item.code for item in result.all_findings}  # type: ignore[attr-defined]


def test_full_fill_reconciles_every_domain_without_stop() -> None:
    fixture, receipt = _execute_full()

    result = ReconciliationEngine().reconcile(_request(fixture, receipt))

    assert result.status is ReconciliationStatus.MATCHED
    assert result.max_severity is ReconciliationSeverity.INFO
    assert not result.difference_hashes
    assert not result.stop_signal.required
    assert result.stop_signal.action is ReconciliationStopAction.NONE
    assert len(result.order_checks) == len(fixture.draft.lines)
    assert len(result.fill_checks) == len(receipt.orders)
    assert len(result.position_checks) == len(receipt.account_after.positions)
    assert not result.is_executable
    assert not result.stop_signal.is_executable


@pytest.mark.parametrize(
    ("execution", "expected_code"),
    (
        (_execute_partial, ReconciliationDifferenceCode.ORDER_PARTIAL),
        (_execute_no_fill, ReconciliationDifferenceCode.ORDER_NO_FILL),
        (_execute_rejected, ReconciliationDifferenceCode.ORDER_REJECTED),
    ),
)
def test_expected_partial_no_fill_and_rejection_are_warnings_without_stop(
    execution: object,
    expected_code: ReconciliationDifferenceCode,
) -> None:
    fixture, receipt = execution()  # type: ignore[operator]

    result = ReconciliationEngine().reconcile(_request(fixture, receipt))

    assert result.status is ReconciliationStatus.RECONCILED_WITH_VARIANCE
    assert result.max_severity is ReconciliationSeverity.WARNING
    assert expected_code in _finding_codes(result)
    assert not result.stop_signal.required


def test_identical_duplicate_reports_are_counted_once_and_retained_as_warning() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    request = _request(
        fixture,
        receipt,
        order_reports=(*base.order_reports, base.order_reports[0]),
        fill_reports=(*base.fill_reports, base.fill_reports[0]),
    )

    result = ReconciliationEngine().reconcile(request)

    assert result.status is ReconciliationStatus.RECONCILED_WITH_VARIANCE
    assert len(result.duplicate_groups) == 2
    assert all(
        item.classification is DuplicateClassification.IDENTICAL for item in result.duplicate_groups
    )
    assert all(item.occurrence_count == 2 for item in result.duplicate_groups)
    assert result.max_severity is ReconciliationSeverity.WARNING
    assert not result.stop_signal.required
    assert all(item.quantity_delta == 0 for item in result.fill_checks)


def test_conflicting_duplicate_fill_is_unreconcilable_and_requests_stop() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    conflict = ObservedFillReport.build(
        source_system=source.source_system,
        account_id=source.account_id,
        report_id=f"{source.report_id}:conflict",
        source_fill_id=source.source_fill_id,
        source_order_id=source.source_order_id,
        client_order_id=source.client_order_id,
        client_fill_id=source.client_fill_id,
        filled_at=source.filled_at,
        available_at=source.available_at,
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        side=source.side,
        currency=source.currency,
        quantity=source.quantity,
        price=source.price + Decimal(1),
        commission=source.commission,
        stamp_duty=source.stamp_duty,
        transfer_fee=source.transfer_fee,
        other_fee=source.other_fee,
    )
    request = _request(
        fixture,
        receipt,
        fill_reports=(*base.fill_reports, conflict),
    )

    result = ReconciliationEngine().reconcile(request)

    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert any(
        item.classification is DuplicateClassification.CONFLICTING
        for item in result.duplicate_groups
    )
    assert result.stop_signal.required
    assert (
        ReconciliationDifferenceCode.CONFLICTING_DUPLICATE_REPORT in result.stop_signal.reason_codes
    )
    assert result.stop_signal.action is ReconciliationStopAction.STOP_NEW_ORDERS


def test_missing_order_and_fill_reports_are_not_silently_ignored() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    omitted_order_id = receipt.orders[0].order_id
    request = _request(
        fixture,
        receipt,
        order_reports=tuple(
            item for item in base.order_reports if item.client_order_id != omitted_order_id
        ),
        fill_reports=tuple(
            item for item in base.fill_reports if item.client_order_id != omitted_order_id
        ),
    )

    result = ReconciliationEngine().reconcile(request)

    assert result.status is ReconciliationStatus.MISMATCH
    assert {
        ReconciliationDifferenceCode.ORDER_MISSING,
        ReconciliationDifferenceCode.FILL_MISSING,
    }.issubset(_finding_codes(result))
    assert result.stop_signal.required


def test_unexpected_explicitly_linked_fill_requests_stop() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    extra = ObservedFillReport.build(
        source_system=source.source_system,
        account_id=source.account_id,
        report_id="unexpected-fill-report",
        source_fill_id="unexpected-source-fill",
        source_order_id="unexpected-source-order",
        client_order_id="unexpected-client-order",
        client_fill_id="unexpected-client-fill",
        filled_at=source.filled_at,
        available_at=source.available_at,
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        side=source.side,
        currency=source.currency,
        quantity=source.quantity,
        price=source.price,
        commission=source.commission,
        stamp_duty=source.stamp_duty,
        transfer_fee=source.transfer_fee,
        other_fee=source.other_fee,
    )

    result = ReconciliationEngine().reconcile(
        _request(fixture, receipt, fill_reports=(*base.fill_reports, extra))
    )

    assert ReconciliationDifferenceCode.FILL_UNEXPECTED in _finding_codes(result)
    assert result.stop_signal.required


def test_cash_difference_is_critical_and_preserves_each_mismatched_bucket() -> None:
    fixture, receipt = _execute_full()
    account = receipt.account_after
    cash = CashSnapshot(
        as_of=receipt.processed_at,
        currency=account.currency,
        total_cash=account.total_cash + Decimal(1),
        available_cash=account.available_cash + Decimal(1),
        frozen_cash=account.frozen_cash,
    )

    result = ReconciliationEngine().reconcile(
        _request(fixture, receipt, snapshot=_snapshot(receipt, cash=cash))
    )

    assert result.cash_check.mismatched_fields == ("available_cash", "total_cash")
    assert result.cash_check.total_delta == 1
    assert result.cash_check.available_delta == 1
    assert result.stop_signal.required
    assert {
        ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
        ReconciliationDifferenceCode.CASH_AVAILABLE_MISMATCH,
    }.issubset(_finding_codes(result))


def test_nonzero_cash_difference_within_tolerance_remains_visible_warning() -> None:
    fixture, receipt = _execute_full()
    account = receipt.account_after
    cash = CashSnapshot(
        as_of=receipt.processed_at,
        currency=account.currency,
        total_cash=account.total_cash + Decimal("0.01"),
        available_cash=account.available_cash + Decimal("0.01"),
        frozen_cash=account.frozen_cash,
    )
    policy = ReconciliationPolicy(cash_tolerance=Decimal("0.02"))

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(receipt, cash=cash),
            policy=policy,
        )
    )

    assert result.status is ReconciliationStatus.RECONCILED_WITH_VARIANCE
    assert result.cash_check.mismatched_fields
    assert result.max_severity is ReconciliationSeverity.WARNING
    assert not result.stop_signal.required


def test_position_bucket_difference_is_critical_but_valuation_is_out_of_scope() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(receipt)
    source = snapshot.positions[0]
    assert source.unsettled_quantity > 0
    changed = PortfolioPositionSnapshot.build(
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        available_quantity=source.available_quantity + Decimal(1),
        frozen_quantity=source.frozen_quantity,
        unsettled_quantity=source.unsettled_quantity - Decimal(1),
        average_cost=source.average_cost,
        valuation_price=source.valuation_price + Decimal(5),
        price_observed_at=source.price_observed_at,
        price_available_at=source.price_available_at,
        position_as_of=source.position_as_of,
        price_data_version=source.price_data_version,
        price_source_hash=stable_hash({"changed-price": source.instrument_id}),
        mark_policy_hash=source.mark_policy_hash,
    )
    positions = tuple(
        changed if item.instrument_id == changed.instrument_id else item
        for item in snapshot.positions
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(receipt, positions=positions, snapshot_id="bucket-mismatch"),
        )
    )

    assert result.stop_signal.required
    assert ReconciliationDifferenceCode.POSITION_BUCKET_MISMATCH in _finding_codes(result)
    position = next(
        item for item in result.position_checks if item.instrument_id == source.instrument_id
    )
    assert position.total_delta == 0
    assert position.available_delta == 1
    assert position.unsettled_delta == -1


def test_valuation_only_change_does_not_create_a_false_execution_difference() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(receipt)
    changed_positions = tuple(
        PortfolioPositionSnapshot.build(
            instrument_id=item.instrument_id,
            instrument_type=item.instrument_type,
            available_quantity=item.available_quantity,
            frozen_quantity=item.frozen_quantity,
            unsettled_quantity=item.unsettled_quantity,
            average_cost=item.average_cost,
            valuation_price=item.valuation_price + Decimal(5),
            price_observed_at=item.price_observed_at,
            price_available_at=item.price_available_at,
            position_as_of=item.position_as_of,
            price_data_version=item.price_data_version,
            price_source_hash=stable_hash({"valuation-only": item.instrument_id}),
            mark_policy_hash=item.mark_policy_hash,
        )
        for item in snapshot.positions
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(
                receipt,
                positions=changed_positions,
                snapshot_id="valuation-only-change",
            ),
        )
    )

    assert result.status is ReconciliationStatus.MATCHED
    assert not result.difference_hashes


def test_snapshot_lineage_and_stale_evidence_are_unreconcilable() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(
        receipt,
        snapshot_id="wrong-lineage",
        source_event_log_hash=stable_hash({"wrong": "event-log"}),
    )
    available_at = receipt.processed_at + timedelta(seconds=1)

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=snapshot,
            evidence_available_at=available_at,
            reconciled_at=available_at + timedelta(seconds=301),
        )
    )

    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert {
        ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
        ReconciliationDifferenceCode.STALE_EVIDENCE,
    }.issubset(_finding_codes(result))
    assert result.stop_signal.required


def test_reports_predating_order_and_attempt_cannot_hide_in_a_fresh_envelope() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    order_source = base.order_reports[0]
    stale_order = ObservedOrderReport.build(
        source_system=order_source.source_system,
        account_id=order_source.account_id,
        report_id=f"{order_source.report_id}:pre-order",
        source_order_id=order_source.source_order_id,
        client_order_id=order_source.client_order_id,
        revision=order_source.revision,
        event_time=receipt.request_submitted_at - timedelta(seconds=1),
        available_at=order_source.available_at,
        instrument_id=order_source.instrument_id,
        instrument_type=order_source.instrument_type,
        side=order_source.side,
        status=order_source.status,
        requested_quantity=order_source.requested_quantity,
        cumulative_filled_quantity=order_source.cumulative_filled_quantity,
        remaining_quantity=order_source.remaining_quantity,
        outcome_code=order_source.outcome_code,
    )
    fill_source = base.fill_reports[0]
    stale_fill = ObservedFillReport.build(
        source_system=fill_source.source_system,
        account_id=fill_source.account_id,
        report_id=f"{fill_source.report_id}:pre-attempt",
        source_fill_id=fill_source.source_fill_id,
        source_order_id=fill_source.source_order_id,
        client_order_id=fill_source.client_order_id,
        client_fill_id=fill_source.client_fill_id,
        filled_at=receipt.request_submitted_at - timedelta(seconds=1),
        available_at=fill_source.available_at,
        instrument_id=fill_source.instrument_id,
        instrument_type=fill_source.instrument_type,
        side=fill_source.side,
        currency=fill_source.currency,
        quantity=fill_source.quantity,
        price=fill_source.price,
        commission=fill_source.commission,
        stamp_duty=fill_source.stamp_duty,
        transfer_fee=fill_source.transfer_fee,
        other_fee=fill_source.other_fee,
    )
    request = _request(
        fixture,
        receipt,
        order_reports=(stale_order, *base.order_reports[1:]),
        fill_reports=(stale_fill, *base.fill_reports[1:]),
    )

    result = ReconciliationEngine().reconcile(request)

    assert result.status is ReconciliationStatus.UNRECONCILABLE
    boundary_findings = tuple(
        item
        for item in result.all_findings
        if item.code is ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH
    )
    assert len(boundary_findings) >= 2
    assert result.stop_signal.required


def test_business_identity_mismatch_builds_a_result_instead_of_being_rejected_early() -> None:
    fixture, receipt = _execute_full()

    request = _request(fixture, receipt, evidence_account_id="other-paper-account")
    result = ReconciliationEngine().reconcile(request)

    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert ReconciliationDifferenceCode.ACCOUNT_IDENTITY_MISMATCH in _finding_codes(result)
    assert result.stop_signal.required


def test_order_revision_regression_is_retained_and_requests_stop() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    regressed = ObservedOrderReport.build(
        source_system=source.source_system,
        account_id=source.account_id,
        report_id=f"{source.report_id}:regressed",
        source_order_id=source.source_order_id,
        client_order_id=source.client_order_id,
        revision=2,
        event_time=source.event_time + timedelta(microseconds=1),
        available_at=source.available_at,
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        side=source.side,
        status=PaperOrderStatus.ACCEPTED,
        requested_quantity=source.requested_quantity,
        cumulative_filled_quantity=Decimal(0),
        remaining_quantity=source.requested_quantity,
        outcome_code=PaperNoFillReason.ZERO_CAPACITY.value,
    )

    result = ReconciliationEngine().reconcile(
        _request(fixture, receipt, order_reports=(*base.order_reports, regressed))
    )

    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert ReconciliationDifferenceCode.ORDER_REPORT_REGRESSION in _finding_codes(result)
    assert result.stop_signal.required


def test_report_hash_tampering_and_evidence_order_do_not_bypass_contracts() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    report = base.order_reports[0]
    with pytest.raises(ValueError, match="report_hash"):
        replace(report, report_hash=stable_hash({"forged": "order-report"}))

    first = ObservedExecutionEvidence.build(
        source_system=base.source_system,
        account_id=base.account_id,
        observed_at=base.observed_at,
        available_at=base.available_at,
        account_snapshot=base.account_snapshot,
        order_reports=base.order_reports,
        fill_reports=base.fill_reports,
    )
    second = ObservedExecutionEvidence.build(
        source_system=base.source_system,
        account_id=base.account_id,
        observed_at=base.observed_at,
        available_at=base.available_at,
        account_snapshot=base.account_snapshot,
        order_reports=tuple(reversed(base.order_reports)),
        fill_reports=tuple(reversed(base.fill_reports)),
    )
    assert first == second
    assert first.evidence_hash == second.evidence_hash


def test_reconciliation_is_deterministic_and_has_no_execution_authority() -> None:
    fixture, receipt = _execute_full()
    request = _request(fixture, receipt)
    engine = ReconciliationEngine()

    first = engine.reconcile(request)
    second = engine.reconcile(request)

    assert first == second
    assert first.result_hash == second.result_hash
    package = Path(__file__).parents[2] / "packages" / "quant_agent" / "reconciliation"
    forbidden_modules = {
        "quant_agent.execution.paper.service",
        "quant_agent.execution.paper.repository",
        "quant_agent.execution.paper.sqlite_repository",
        "quant_agent.risk.kill_switch",
    }
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert imported.isdisjoint(forbidden_modules)
    assert not hasattr(engine, "execute")
    assert not hasattr(engine, "submit")
    assert not hasattr(first.stop_signal, "activate")
    assert not hasattr(first.stop_signal, "recover")
