"""Boundary tests for immutable, content-addressed reconciliation contracts."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from typing import Any

import pytest

from quant_agent.backtest import Side
from quant_agent.execution.order_drafts import OrderDraftBatch, OrderDraftGeneratorConfig
from quant_agent.execution.paper import (
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperOrderStatus,
)
from quant_agent.portfolio import AccountSnapshot, CashSnapshot
from quant_agent.reconciliation import (
    RECONCILIATION_ENGINE_VERSION,
    CashReconciliation,
    DuplicateClassification,
    DuplicateKeyKind,
    DuplicateReportGroup,
    FillReconciliation,
    ObservedExecutionEvidence,
    ObservedFillReport,
    ObservedOrderReport,
    PositionReconciliation,
    ReconciliationCheckStatus,
    ReconciliationDifferenceCode,
    ReconciliationDomain,
    ReconciliationEngine,
    ReconciliationFinding,
    ReconciliationPolicy,
    ReconciliationReportKind,
    ReconciliationRequest,
    ReconciliationResult,
    ReconciliationSeverity,
    ReconciliationStatus,
    ReconciliationStopAction,
    ReconciliationStopScope,
    ReconciliationStopSignal,
)
from quant_agent.regime.contracts import stable_hash

from .test_reconciliation_engine import (
    _execute_full,
    _execute_no_fill,
    _execute_rejected,
    _request,
    _snapshot,
)


def _clean_chain() -> tuple[ReconciliationRequest, ReconciliationResult]:
    fixture, receipt = _execute_full()
    request = _request(fixture, receipt)
    return request, ReconciliationEngine().reconcile(request)


def _finding(
    request: ReconciliationRequest,
    *,
    code: ReconciliationDifferenceCode = ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
    status: ReconciliationCheckStatus = ReconciliationCheckStatus.MISMATCH,
    severity: ReconciliationSeverity = ReconciliationSeverity.ERROR,
    entity_id: str = "contract-test-entity",
) -> ReconciliationFinding:
    return ReconciliationFinding.build(
        domain=ReconciliationDomain.INPUT,
        code=code,
        status=status,
        severity=severity,
        entity_id=entity_id,
        message="contract difference",
        decision_id=request.draft.decision_id,
        batch_hash=request.draft.batch_hash,
        expected_value="expected",
        observed_value="observed",
        delta="1",
        evidence_hashes=(request.observed.evidence_hash, request.receipt.receipt_hash),
    )


def _result_with(
    request: ReconciliationRequest,
    clean: ReconciliationResult,
    *,
    findings: tuple[ReconciliationFinding, ...] = (),
    duplicate_groups: tuple[DuplicateReportGroup, ...] = (),
) -> ReconciliationResult:
    return ReconciliationResult._build_from_engine(
        request=request,
        input_findings=findings,
        order_checks=clean.order_checks,
        fill_checks=clean.fill_checks,
        cash_check=clean.cash_check,
        position_checks=clean.position_checks,
        duplicate_groups=duplicate_groups,
    )


def _rehash_stop_signal(
    signal: ReconciliationStopSignal,
    **changes: object,
) -> ReconciliationStopSignal:
    values = asdict(signal)
    values.pop("signal_hash")
    values.update(changes)
    return ReconciliationStopSignal(**values, signal_hash=stable_hash(values))


def _rebuild_cash_check(
    check: CashReconciliation,
    **changes: object,
) -> CashReconciliation:
    values = {field.name: getattr(check, field.name) for field in fields(check)}
    for derived in ("status", "severity", "check_hash"):
        values.pop(derived)
    values.update(changes)
    return CashReconciliation.build(**values)


def _rebuild_fill_check(
    check: FillReconciliation,
    **changes: object,
) -> FillReconciliation:
    values = {field.name: getattr(check, field.name) for field in fields(check)}
    for derived in ("status", "severity", "check_hash"):
        values.pop(derived)
    values.update(changes)
    return FillReconciliation.build(**values)


def _rebuild_position_check(
    check: PositionReconciliation,
    **changes: object,
) -> PositionReconciliation:
    values = {field.name: getattr(check, field.name) for field in fields(check)}
    for derived in ("status", "severity", "check_hash"):
        values.pop(derived)
    values.update(changes)
    return PositionReconciliation.build(**values)


def _build_result_with_children(
    request: ReconciliationRequest,
    clean: ReconciliationResult,
    **changes: object,
) -> ReconciliationResult:
    values: dict[str, object] = {
        "input_findings": clean.input_findings,
        "order_checks": clean.order_checks,
        "fill_checks": clean.fill_checks,
        "cash_check": clean.cash_check,
        "position_checks": clean.position_checks,
        "duplicate_groups": clean.duplicate_groups,
    }
    values.update(changes)
    return ReconciliationResult.build(request=request, **values)  # type: ignore[arg-type]


def _request_with_extreme_receipt_decimal(
    request: ReconciliationRequest,
) -> tuple[OrderDraftBatch, PaperExecutionReceipt]:
    fixture, _ = _execute_full()
    source = fixture.snapshot
    extreme = Decimal("1e201")
    snapshot = AccountSnapshot.build(
        snapshot_id="account-extreme-reconciliation-boundary",
        account_id=source.account_id,
        runtime_mode=source.runtime_mode,
        as_of=source.as_of,
        valuation_at=source.valuation_at,
        data_version=source.data_version,
        currency=source.currency,
        cash=CashSnapshot(
            as_of=source.as_of,
            currency=source.currency,
            total_cash=extreme,
            available_cash=extreme,
            frozen_cash=Decimal(0),
        ),
        positions=source.positions,
        previous_snapshot_hash=source.previous_snapshot_hash,
        source_event_log_hash=source.source_event_log_hash,
    )
    draft = fixture.draft
    extreme_draft = OrderDraftBatch.build(
        decision_id=draft.decision_id,
        draft_as_of=draft.draft_as_of,
        data_version=draft.data_version,
        currency=draft.currency,
        account_snapshot_id=snapshot.snapshot_id,
        account_snapshot_hash=snapshot.content_hash,
        account_snapshot_as_of=snapshot.as_of,
        runtime_mode=draft.runtime_mode,
        portfolio_proposal_hash=draft.portfolio_proposal_hash,
        risk_request_hash=draft.risk_request_hash,
        risk_result_hash=draft.risk_result_hash,
        risk_status=draft.risk_status,
        risk_checked_at=draft.risk_checked_at,
        risk_engine_version=draft.risk_engine_version,
        risk_policy_version=draft.risk_policy_version,
        risk_policy_hash=draft.risk_policy_hash,
        config=OrderDraftGeneratorConfig(
            version=draft.generator_config_version,
            validity_seconds=draft.validity_seconds,
            funding_policy=draft.funding_policy,
        ),
        available_cash=extreme,
        lines=draft.lines,
        sizing_state_hashes=draft.state_hashes,
        sizing_market_rule_hashes=draft.market_rule_hashes,
    )
    account = fixture.engine.open_account(snapshot=snapshot)
    execution_request = PaperExecutionRequest.build(
        request_id="paper-request-extreme-reconciliation-boundary",
        idempotency_key="paper-key-extreme-reconciliation-boundary",
        account_id=account.account_id,
        expected_account_state_hash=account.state_hash,
        expected_account_snapshot_hash=account.source_snapshot_hash,
        batch_hash=extreme_draft.batch_hash,
        submitted_at=fixture.request.submitted_at,
        config=fixture.engine.config,
    )
    receipt = fixture.engine.execute(
        request=execution_request,
        draft=extreme_draft,
        account=account,
    )
    assert receipt.account_after.total_cash.adjusted() >= 200
    assert request.observed.account_snapshot.cash.total_cash.adjusted() < 200
    return extreme_draft, receipt


def test_observed_order_report_rejects_invalid_identity_time_quantity_and_status() -> None:
    request, _ = _clean_chain()
    report = request.observed.order_reports[0]
    quantity = report.requested_quantity
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": "2"}, "schema_version"),
        ({"source_system": "  "}, "source_system"),
        ({"client_order_id": ""}, "client_order_id"),
        ({"revision": 0}, "positive integer"),
        ({"revision": True}, "positive integer"),
        ({"event_time": datetime(2026, 1, 1)}, "timezone information"),
        ({"event_time": report.available_at + timedelta(microseconds=1)}, "cannot precede"),
        ({"instrument_type": "ETF"}, "instrument_type is invalid"),
        ({"side": "BUY"}, "side is invalid"),
        ({"status": "FILLED"}, "status is invalid"),
        ({"requested_quantity": Decimal(0)}, "positive whole Decimal"),
        ({"requested_quantity": Decimal("1.5")}, "positive whole Decimal"),
        ({"cumulative_filled_quantity": quantity + 1}, "cannot exceed"),
        ({"remaining_quantity": Decimal(1)}, "does not conserve"),
        (
            {
                "status": PaperOrderStatus.FILLED,
                "cumulative_filled_quantity": quantity - 1,
                "remaining_quantity": Decimal(1),
            },
            "FILLED order report",
        ),
        (
            {
                "status": PaperOrderStatus.PARTIALLY_FILLED,
                "cumulative_filled_quantity": Decimal(0),
                "remaining_quantity": quantity,
            },
            "proper partial fill",
        ),
        (
            {
                "status": PaperOrderStatus.ACCEPTED,
                "cumulative_filled_quantity": Decimal(1),
                "remaining_quantity": quantity - 1,
            },
            "cannot contain fills",
        ),
        (
            {
                "status": PaperOrderStatus.REJECTED,
                "cumulative_filled_quantity": Decimal(0),
                "remaining_quantity": quantity,
                "outcome_code": None,
            },
            "requires an outcome_code",
        ),
        (
            {
                "status": PaperOrderStatus.ACCEPTED,
                "cumulative_filled_quantity": Decimal(0),
                "remaining_quantity": quantity,
                "outcome_code": None,
            },
            "requires an outcome_code",
        ),
        ({"outcome_code": "unexpected"}, "filled order report cannot carry"),
    )

    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(report, **changes)


def test_observed_order_report_hashes_bind_semantics_and_envelope() -> None:
    request, _ = _clean_chain()
    report = request.observed.order_reports[0]

    with pytest.raises(ValueError, match="source_payload_hash"):
        replace(report, source_payload_hash="not-a-digest")
    with pytest.raises(ValueError, match="does not match report semantics"):
        replace(report, source_payload_hash=stable_hash({"forged": "order-content"}))
    with pytest.raises(ValueError, match="report_hash"):
        replace(report, report_hash="not-a-digest")
    with pytest.raises(ValueError, match="does not match its envelope"):
        replace(report, report_hash=stable_hash({"forged": "order-envelope"}))


def test_accepted_and_rejected_no_fill_reports_retain_outcome_code() -> None:
    for execution, expected_status in (
        (_execute_no_fill, PaperOrderStatus.ACCEPTED),
        (_execute_rejected, PaperOrderStatus.REJECTED),
    ):
        fixture, receipt = execution()
        evidence = _request(fixture, receipt).observed
        matching = tuple(item for item in evidence.order_reports if item.status is expected_status)
        assert matching
        assert all(item.outcome_code for item in matching)


def test_observed_fill_report_rejects_invalid_time_quantity_and_economics() -> None:
    request, _ = _clean_chain()
    report = request.observed.fill_reports[0]
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": "2"}, "schema_version"),
        ({"source_fill_id": " "}, "source_fill_id"),
        ({"client_fill_id": ""}, "client_fill_id"),
        ({"filled_at": datetime(2026, 1, 1)}, "timezone information"),
        ({"filled_at": report.available_at + timedelta(microseconds=1)}, "cannot precede"),
        ({"instrument_type": "ETF"}, "instrument_type is invalid"),
        ({"side": "BUY"}, "side is invalid"),
        ({"quantity": Decimal(0)}, "positive whole Decimal"),
        ({"quantity": Decimal("1.5")}, "positive whole Decimal"),
        ({"price": Decimal(0)}, "must be positive"),
        ({"gross_amount": Decimal(0)}, "must be positive"),
        ({"commission": Decimal("-0.01")}, "cannot be negative"),
        ({"gross_amount": report.gross_amount + 1}, "quantity times price"),
        ({"total_fee": report.total_fee + 1}, "fee components"),
        ({"cash_change": report.cash_change + 1}, "does not match side"),
        ({"price": Decimal("NaN")}, "finite Decimal"),
        ({"price": 1}, "finite Decimal"),
    )

    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(report, **changes)


def test_observed_fill_report_hashes_and_sell_cash_identity() -> None:
    request, _ = _clean_chain()
    report = request.observed.fill_reports[0]

    with pytest.raises(ValueError, match="source_payload_hash"):
        replace(report, source_payload_hash="bad")
    with pytest.raises(ValueError, match="does not match report semantics"):
        replace(report, source_payload_hash=stable_hash({"forged": "fill-content"}))
    with pytest.raises(ValueError, match="report_hash"):
        replace(report, report_hash="bad")
    with pytest.raises(ValueError, match="does not match its envelope"):
        replace(report, report_hash=stable_hash({"forged": "fill-envelope"}))

    sell = ObservedFillReport.build(
        source_system=report.source_system,
        account_id=report.account_id,
        report_id="sell-report",
        source_fill_id="sell-source-fill",
        source_order_id="sell-source-order",
        client_order_id="sell-client-order",
        client_fill_id="sell-client-fill",
        filled_at=report.filled_at,
        available_at=report.available_at,
        instrument_id=report.instrument_id,
        instrument_type=report.instrument_type,
        side=Side.SELL,
        currency=report.currency,
        quantity=Decimal(100),
        price=Decimal("10.25"),
        commission=Decimal(1),
        stamp_duty=Decimal("1.025"),
        transfer_fee=Decimal(0),
        other_fee=Decimal(0),
    )
    assert sell.gross_amount == Decimal(1025)
    assert sell.cash_change == sell.gross_amount - sell.total_fee


def test_evidence_build_canonicalizes_reports_and_hashes_source_cursor() -> None:
    request, _ = _clean_chain()
    evidence = request.observed
    reversed_evidence = ObservedExecutionEvidence.build(
        source_system=evidence.source_system,
        account_id=evidence.account_id,
        observed_at=evidence.observed_at,
        available_at=evidence.available_at,
        account_snapshot=evidence.account_snapshot,
        order_reports=tuple(reversed(evidence.order_reports)),
        fill_reports=tuple(reversed(evidence.fill_reports)),
        source_cursor=evidence.source_cursor,
    )
    changed_cursor = ObservedExecutionEvidence.build(
        source_system=evidence.source_system,
        account_id=evidence.account_id,
        observed_at=evidence.observed_at,
        available_at=evidence.available_at,
        account_snapshot=evidence.account_snapshot,
        order_reports=evidence.order_reports,
        fill_reports=evidence.fill_reports,
        source_cursor="paper-cursor-2",
    )

    assert reversed_evidence == evidence
    assert changed_cursor.evidence_hash != evidence.evidence_hash
    with pytest.raises(ValueError, match="canonical ordering"):
        replace(evidence, order_reports=tuple(reversed(evidence.order_reports)))
    with pytest.raises(ValueError, match="source_cursor"):
        replace(evidence, source_cursor=" ")
    with pytest.raises(ValueError, match="cannot precede"):
        replace(evidence, observed_at=evidence.available_at + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="evidence_hash"):
        replace(evidence, evidence_hash=stable_hash({"forged": "evidence"}))


def test_policy_rejects_invalid_lag_tolerances_and_stop_threshold() -> None:
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"version": " "}, "policy version"),
        ({"max_observation_lag_seconds": 0}, "within"),
        ({"max_observation_lag_seconds": True}, "within"),
        ({"max_observation_lag_seconds": 1.5}, "within"),
        ({"cash_tolerance": Decimal("-0.01")}, "cannot be negative"),
        ({"fee_tolerance": Decimal("NaN")}, "finite Decimal"),
        ({"price_tolerance": 0}, "finite Decimal"),
        ({"cost_tolerance": Decimal("Infinity")}, "finite Decimal"),
        ({"stop_on_severity": "CRITICAL"}, "ReconciliationSeverity"),
        ({"stop_on_severity": ReconciliationSeverity.INFO}, "cannot be INFO"),
    )

    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            ReconciliationPolicy(**changes)

    baseline = ReconciliationPolicy()
    changed = ReconciliationPolicy(cash_tolerance=Decimal("0.01"))
    assert baseline.policy_hash != changed.policy_hash


def test_request_hash_binds_every_input_and_never_grants_execution_authority() -> None:
    request, _ = _clean_chain()
    previous_hash = stable_hash({"previous": "reconciliation"})
    chained = ReconciliationRequest.build(
        request_id=request.request_id,
        idempotency_key=request.idempotency_key,
        reconciled_at=request.reconciled_at,
        draft=request.draft,
        receipt=request.receipt,
        observed=request.observed,
        policy=request.policy,
        previous_result_hash=previous_hash,
    )

    assert not request.is_executable
    assert chained.input_hash != request.input_hash
    with pytest.raises(ValueError, match="schema_version"):
        replace(request, schema_version="2")
    with pytest.raises(ValueError, match="request_id"):
        replace(request, request_id=" ")
    with pytest.raises(ValueError, match="timezone information"):
        replace(request, reconciled_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="previous_result_hash"):
        replace(request, previous_result_hash="bad")
    with pytest.raises(ValueError, match="input_hash"):
        replace(request, input_hash=stable_hash({"forged": "request"}))


def test_finding_identity_evidence_and_hash_are_canonical_and_tamper_evident() -> None:
    request, _ = _clean_chain()
    finding = _finding(request)
    rebuilt = ReconciliationFinding.build(
        domain=finding.domain,
        code=finding.code,
        status=finding.status,
        severity=finding.severity,
        entity_id=finding.entity_id,
        message=finding.message,
        decision_id=finding.decision_id,
        batch_hash=finding.batch_hash,
        expected_value=finding.expected_value,
        observed_value=finding.observed_value,
        delta=finding.delta,
        evidence_hashes=tuple(reversed(finding.evidence_hashes)),
    )
    assert rebuilt == finding

    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"domain": "INPUT"}, "domain is invalid"),
        ({"code": "CASH_TOTAL_MISMATCH"}, "code is invalid"),
        ({"status": ReconciliationCheckStatus.MATCH}, "cannot have MATCH"),
        ({"severity": "ERROR"}, "severity is invalid"),
        (
            {"evidence_hashes": (finding.evidence_hashes[0], finding.evidence_hashes[0])},
            "unique and sorted",
        ),
        ({"finding_id": "finding:forged"}, "does not match finding identity"),
        ({"finding_hash": stable_hash({"forged": "finding"})}, "does not match"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(finding, **changes)


def test_each_check_derives_status_severity_deltas_and_content_hash() -> None:
    _, result = _clean_chain()
    checks = (
        result.order_checks[0],
        result.fill_checks[0],
        result.cash_check,
        result.position_checks[0],
    )
    for check in checks:
        with pytest.raises(ValueError, match="status must be derived"):
            replace(check, status=ReconciliationCheckStatus.MISMATCH)
        with pytest.raises(ValueError, match="severity must be derived"):
            replace(check, severity=ReconciliationSeverity.ERROR)
        with pytest.raises(ValueError, match="does not match content"):
            replace(check, check_hash=stable_hash({"forged": type(check).__name__}))

    with pytest.raises(ValueError, match="whole Decimal"):
        replace(result.order_checks[0], planned_quantity=Decimal("1.5"))
    with pytest.raises(ValueError, match="quantity_delta is inconsistent"):
        replace(result.fill_checks[0], quantity_delta=Decimal(1))
    with pytest.raises(ValueError, match="total_delta is inconsistent"):
        replace(result.cash_check, total_delta=Decimal(1))
    with pytest.raises(ValueError, match="position delta"):
        replace(result.position_checks[0], total_delta=Decimal(1))
    with pytest.raises(ValueError, match="outcome_code"):
        replace(result.order_checks[0], expected_outcome_code=" ")
    for field_name in (
        "commission_delta",
        "stamp_duty_delta",
        "transfer_fee_delta",
        "other_fee_delta",
    ):
        with pytest.raises(ValueError, match="fee-component deltas are inconsistent"):
            replace(result.fill_checks[0], **{field_name: Decimal(1)})


def test_duplicate_groups_preserve_identical_and_conflicting_evidence_invariants() -> None:
    request, _ = _clean_chain()
    report = request.observed.order_reports[0]
    other_report_hash = stable_hash({"second-envelope": report.report_hash})
    other_content_hash = stable_hash({"conflict": report.source_payload_hash})
    common = {
        "report_kind": ReconciliationReportKind.ORDER,
        "key_kind": DuplicateKeyKind.SOURCE_IDENTITY,
        "source_system": report.source_system,
        "dedup_key": report.dedup_key,
        "report_ids": ("report-b", "report-a"),
        "report_hashes": (other_report_hash, report.report_hash),
    }
    identical = DuplicateReportGroup.build(
        **common,
        content_hashes=(report.source_payload_hash, report.source_payload_hash),
    )
    conflicting = DuplicateReportGroup.build(
        **common,
        content_hashes=(other_content_hash, report.source_payload_hash),
    )

    assert identical.classification is DuplicateClassification.IDENTICAL
    assert identical.canonical_report_hash == min(identical.report_hashes)
    assert identical.severity is ReconciliationSeverity.WARNING
    assert identical.difference_code is ReconciliationDifferenceCode.IDENTICAL_DUPLICATE_REPORT
    assert conflicting.classification is DuplicateClassification.CONFLICTING
    assert conflicting.canonical_report_hash is None
    assert conflicting.severity is ReconciliationSeverity.CRITICAL
    assert conflicting.difference_code is ReconciliationDifferenceCode.CONFLICTING_DUPLICATE_REPORT

    invalid_cases: tuple[tuple[DuplicateReportGroup, dict[str, Any], str], ...] = (
        (identical, {"occurrence_count": 1}, "at least two"),
        (identical, {"report_ids": ("report-a",)}, "counts must align"),
        (identical, {"report_ids": tuple(reversed(identical.report_ids))}, "must be sorted"),
        (identical, {"canonical_report_hash": None}, "canonical report"),
        (identical, {"severity": ReconciliationSeverity.CRITICAL}, "WARNING severity"),
        (conflicting, {"canonical_report_hash": conflicting.report_hashes[0]}, "cannot select"),
        (conflicting, {"severity": ReconciliationSeverity.WARNING}, "must be CRITICAL"),
        (identical, {"group_id": "duplicate:forged"}, "does not match its identity"),
        (identical, {"group_hash": stable_hash({"forged": "group"})}, "does not match"),
    )
    for group, changes, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            replace(group, **changes)


def test_stop_signal_is_canonical_non_executable_and_cannot_be_forged() -> None:
    request, _ = _clean_chain()
    finding = _finding(
        request,
        code=ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
        severity=ReconciliationSeverity.CRITICAL,
    )
    second_trigger = stable_hash({"second": "trigger"})
    required = ReconciliationStopSignal.build(
        account_id=request.receipt.account_after.account_id,
        policy_hash=request.policy.policy_hash,
        reconciliation_input_hash=request.input_hash,
        decision_id=request.draft.decision_id,
        batch_hash=request.draft.batch_hash,
        required=True,
        severity=ReconciliationSeverity.CRITICAL,
        reason_codes=(
            ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
            ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
        ),
        trigger_hashes=(second_trigger, finding.finding_hash),
        generated_at=request.reconciled_at,
    )
    empty = ReconciliationStopSignal.build(
        account_id=request.receipt.account_after.account_id,
        policy_hash=request.policy.policy_hash,
        reconciliation_input_hash=request.input_hash,
        decision_id=request.draft.decision_id,
        batch_hash=request.draft.batch_hash,
        required=False,
        severity=ReconciliationSeverity.CRITICAL,
        reason_codes=(ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,),
        trigger_hashes=(finding.finding_hash,),
        generated_at=request.reconciled_at,
    )

    assert not required.is_executable
    assert required.engine_version == RECONCILIATION_ENGINE_VERSION
    assert required.policy_hash == request.policy.policy_hash
    assert required.action is ReconciliationStopAction.STOP_NEW_ORDERS
    assert required.reason_codes == tuple(sorted(set(required.reason_codes), key=lambda x: x.value))
    assert required.trigger_hashes == tuple(sorted(set(required.trigger_hashes)))
    assert not empty.required
    assert empty.action is ReconciliationStopAction.NONE
    assert empty.severity is ReconciliationSeverity.INFO
    assert not empty.reason_codes and not empty.trigger_hashes

    invalid_cases: tuple[tuple[ReconciliationStopSignal, dict[str, Any], str], ...] = (
        (required, {"engine_version": "unknown"}, "engine_version is unknown"),
        (required, {"policy_hash": "bad"}, "policy_hash"),
        (required, {"scope": ReconciliationStopScope.GLOBAL}, "cannot emit a GLOBAL"),
        (required, {"action": ReconciliationStopAction.NONE}, "must stop new orders"),
        (required, {"reason_codes": ()}, "needs reasons and triggers"),
        (
            required,
            {"trigger_hashes": tuple(reversed(required.trigger_hashes))},
            "unique and sorted",
        ),
        (empty, {"action": ReconciliationStopAction.STOP_NEW_ORDERS}, "empty INFO/NONE"),
        (required, {"signal_hash": stable_hash({"forged": "stop"})}, "does not match"),
    )
    for signal, changes, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            replace(signal, **changes)


@pytest.mark.parametrize(
    ("status", "severity", "code", "expected_status", "stop_required"),
    (
        (
            ReconciliationCheckStatus.EXPECTED_VARIANCE,
            ReconciliationSeverity.WARNING,
            ReconciliationDifferenceCode.ORDER_PARTIAL,
            ReconciliationStatus.RECONCILED_WITH_VARIANCE,
            False,
        ),
        (
            ReconciliationCheckStatus.MISMATCH,
            ReconciliationSeverity.ERROR,
            ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
            ReconciliationStatus.MISMATCH,
            False,
        ),
        (
            ReconciliationCheckStatus.MISMATCH,
            ReconciliationSeverity.CRITICAL,
            ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
            ReconciliationStatus.MISMATCH,
            True,
        ),
        (
            ReconciliationCheckStatus.MISMATCH,
            ReconciliationSeverity.CRITICAL,
            ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
            ReconciliationStatus.UNRECONCILABLE,
            True,
        ),
    ),
)
def test_result_derives_status_max_severity_differences_and_stop(
    status: ReconciliationCheckStatus,
    severity: ReconciliationSeverity,
    code: ReconciliationDifferenceCode,
    expected_status: ReconciliationStatus,
    stop_required: bool,
) -> None:
    request, clean = _clean_chain()
    finding = _finding(request, code=code, status=status, severity=severity)
    result = _result_with(request, clean, findings=(finding,))

    assert result.status is expected_status
    assert result.max_severity is severity
    assert result.difference_hashes == (finding.finding_hash,)
    assert result.stop_signal.required is stop_required
    assert not result.is_executable
    if stop_required:
        assert result.stop_signal.trigger_hashes == (finding.finding_hash,)
        assert result.stop_signal.reason_codes == (code,)


def test_result_honors_policy_threshold_and_duplicate_severity() -> None:
    fixture, receipt = _execute_full()
    request = _request(
        fixture,
        receipt,
        policy=ReconciliationPolicy(stop_on_severity=ReconciliationSeverity.WARNING),
    )
    clean = ReconciliationEngine().reconcile(request)
    warning = _finding(
        request,
        code=ReconciliationDifferenceCode.ORDER_PARTIAL,
        status=ReconciliationCheckStatus.EXPECTED_VARIANCE,
        severity=ReconciliationSeverity.WARNING,
    )
    warning_result = _result_with(request, clean, findings=(warning,))
    assert warning_result.stop_signal.required
    assert warning_result.stop_signal.severity is ReconciliationSeverity.WARNING

    report = request.observed.order_reports[0]
    conflict = DuplicateReportGroup.build(
        report_kind=ReconciliationReportKind.ORDER,
        key_kind=DuplicateKeyKind.SOURCE_IDENTITY,
        source_system=report.source_system,
        dedup_key=report.dedup_key,
        report_ids=("a", "b"),
        report_hashes=(report.report_hash, stable_hash({"other": "report"})),
        content_hashes=(report.source_payload_hash, stable_hash({"other": "content"})),
    )
    conflict_result = _result_with(request, clean, duplicate_groups=(conflict,))
    assert conflict_result.status is ReconciliationStatus.UNRECONCILABLE
    assert conflict_result.max_severity is ReconciliationSeverity.CRITICAL
    assert conflict_result.difference_hashes == (conflict.group_hash,)
    assert conflict_result.stop_signal.trigger_hashes == (conflict.group_hash,)


def test_result_rejects_forged_derivations_bindings_and_hashes() -> None:
    request, result = _clean_chain()
    dummy_hash = stable_hash({"dummy": "difference"})
    forged_required = ReconciliationStopSignal.build(
        account_id=result.account_id,
        policy_hash=result.policy_hash,
        reconciliation_input_hash=result.input_hash,
        decision_id=result.decision_id,
        batch_hash=result.batch_hash,
        required=True,
        severity=ReconciliationSeverity.CRITICAL,
        reason_codes=(ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,),
        trigger_hashes=(dummy_hash,),
        generated_at=result.reconciled_at,
    )
    wrong_binding = ReconciliationStopSignal.build(
        account_id="other-account",
        policy_hash=result.policy_hash,
        reconciliation_input_hash=result.input_hash,
        decision_id=result.decision_id,
        batch_hash=result.batch_hash,
        required=False,
        severity=ReconciliationSeverity.INFO,
        reason_codes=(),
        trigger_hashes=(),
        generated_at=result.reconciled_at,
    )
    wrong_policy = _rehash_stop_signal(
        result.stop_signal,
        policy_hash=stable_hash({"wrong": "policy"}),
    )
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"status": ReconciliationStatus.MISMATCH}, "status must be derived"),
        ({"max_severity": ReconciliationSeverity.WARNING}, "max_severity must be derived"),
        ({"difference_hashes": (dummy_hash,)}, "must cover every difference"),
        ({"stop_signal": forged_required}, "must follow policy"),
        ({"stop_signal": wrong_binding}, "does not bind"),
        ({"stop_signal": wrong_policy}, "does not bind"),
        ({"reconciliation_id": "reconciliation:forged"}, "does not match request identity"),
        ({"result_hash": stable_hash({"forged": "result"})}, "does not match"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(result, **changes)

    severe_finding = _finding(
        request,
        severity=ReconciliationSeverity.CRITICAL,
    )
    severe = _result_with(request, result, findings=(severe_finding,))
    wrong_severity = _rehash_stop_signal(
        severe.stop_signal,
        severity=ReconciliationSeverity.ERROR,
    )
    wrong_trigger = _rehash_stop_signal(
        severe.stop_signal,
        trigger_hashes=(dummy_hash,),
    )
    wrong_reason = _rehash_stop_signal(
        severe.stop_signal,
        reason_codes=(ReconciliationDifferenceCode.FILL_CASH_MISMATCH,),
    )
    with pytest.raises(ValueError, match="severity does not match"):
        replace(severe, stop_signal=wrong_severity)
    with pytest.raises(ValueError, match="triggers do not cover"):
        replace(severe, stop_signal=wrong_trigger)
    with pytest.raises(ValueError, match="reasons do not cover"):
        replace(severe, stop_signal=wrong_reason)


@pytest.mark.parametrize("child_kind", ("order", "fill"))
def test_result_build_rejects_missing_check_for_a_receipt_order(child_kind: str) -> None:
    request, clean = _clean_chain()
    assert request.receipt.orders
    assert clean.order_checks and clean.fill_checks
    changes: dict[str, object]
    if child_kind == "order":
        omitted_order_id = clean.order_checks[0].order_id
        assert omitted_order_id in {item.order_id for item in request.receipt.orders}
        changes = {"order_checks": clean.order_checks[1:]}
    else:
        omitted_order_id = clean.fill_checks[0].order_id
        assert omitted_order_id in {item.order_id for item in request.receipt.orders}
        changes = {"fill_checks": clean.fill_checks[1:]}

    with pytest.raises(ValueError, match="receipt order"):
        _build_result_with_children(request, clean, **changes)


def test_result_build_rejects_missing_position_union_check() -> None:
    request, clean = _clean_chain()
    expected_union = {
        *(item.instrument_id for item in request.receipt.account_after.positions),
        *(item.instrument_id for item in request.observed.account_snapshot.positions),
    }
    assert expected_union == {item.instrument_id for item in clean.position_checks}
    assert clean.position_checks

    with pytest.raises(ValueError, match=r"position.*union|position.*instruments"):
        _build_result_with_children(
            request,
            clean,
            position_checks=clean.position_checks[1:],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"check_id": "cash:other-account"}, "cash.*account|cash.*identity"),
        ({"currency": "USD"}, "cash.*currency"),
    ),
)
def test_result_build_rejects_cash_check_for_wrong_identity_or_currency(
    changes: dict[str, object],
    message: str,
) -> None:
    request, clean = _clean_chain()
    forged = _rebuild_cash_check(clean.cash_check, **changes)

    with pytest.raises(ValueError, match=message):
        _build_result_with_children(request, clean, cash_check=forged)


def test_result_build_rejects_clean_cash_child_from_another_input() -> None:
    request, clean = _clean_chain()
    foreign_request = ReconciliationRequest.build(
        request_id="foreign-reconciliation-request",
        idempotency_key="foreign-reconciliation-key",
        reconciled_at=request.reconciled_at,
        draft=request.draft,
        receipt=request.receipt,
        observed=request.observed,
        policy=request.policy,
    )
    foreign = ReconciliationEngine().reconcile(foreign_request)
    assert foreign_request.input_hash != request.input_hash

    with pytest.raises(ValueError, match=r"cash.*input|cash.*request|cash.*bind"):
        _build_result_with_children(request, clean, cash_check=foreign.cash_check)


def test_policy_rejects_observation_lag_that_exceeds_timedelta_boundary() -> None:
    with pytest.raises(ValueError, match=r"within \[1, 86400\]"):
        ReconciliationPolicy(max_observation_lag_seconds=10**100)


def test_evidence_rejects_nested_snapshot_decimal_outside_numeric_boundary() -> None:
    request, _ = _clean_chain()
    evidence = request.observed
    extreme = Decimal("1e201")
    extreme_cash = CashSnapshot(
        as_of=request.receipt.processed_at,
        currency=request.receipt.account_after.currency,
        total_cash=extreme,
        available_cash=extreme,
        frozen_cash=Decimal(0),
    )
    extreme_snapshot = _snapshot(
        request.receipt,
        cash=extreme_cash,
        snapshot_id="observed-extreme-reconciliation-boundary",
    )

    with pytest.raises(ValueError, match="numeric boundary"):
        ObservedExecutionEvidence.build(
            source_system=evidence.source_system,
            account_id=evidence.account_id,
            observed_at=evidence.observed_at,
            available_at=evidence.available_at,
            account_snapshot=extreme_snapshot,
            order_reports=evidence.order_reports,
            fill_reports=evidence.fill_reports,
            source_cursor=evidence.source_cursor,
        )


def test_request_rejects_nested_receipt_decimal_outside_numeric_boundary() -> None:
    request, _ = _clean_chain()
    extreme_draft, extreme_receipt = _request_with_extreme_receipt_decimal(request)

    with pytest.raises(ValueError, match="numeric boundary"):
        ReconciliationRequest.build(
            request_id="reconciliation-request-extreme-decimal",
            idempotency_key="reconciliation-key-extreme-decimal",
            reconciled_at=request.reconciled_at,
            draft=extreme_draft,
            receipt=extreme_receipt,
            observed=request.observed,
            policy=request.policy,
        )


def test_public_result_build_rejects_deleted_input_finding() -> None:
    fixture, receipt = _execute_full()
    request = _request(fixture, receipt, evidence_account_id="different-observed-account")
    authoritative = ReconciliationEngine().reconcile(request)
    assert authoritative.input_findings

    with pytest.raises(ValueError, match="authoritative engine output"):
        _build_result_with_children(
            request,
            authoritative,
            input_findings=authoritative.input_findings[1:],
        )


def test_public_result_build_rejects_deleted_duplicate_group() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    request = _request(
        fixture,
        receipt,
        order_reports=(*base.order_reports, base.order_reports[0]),
    )
    authoritative = ReconciliationEngine().reconcile(request)
    assert authoritative.duplicate_groups

    with pytest.raises(ValueError, match="authoritative engine output"):
        _build_result_with_children(
            request,
            authoritative,
            duplicate_groups=authoritative.duplicate_groups[1:],
        )


def test_public_result_build_rejects_deleted_unexpected_order_check() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    unexpected = ObservedOrderReport.build(
        source_system=source.source_system,
        account_id=source.account_id,
        report_id="unexpected-order-contract-report",
        source_order_id="unexpected-source-order-contract",
        client_order_id="unexpected-client-order-contract",
        revision=1,
        event_time=source.event_time,
        available_at=source.available_at,
        instrument_id=source.instrument_id,
        instrument_type=source.instrument_type,
        side=source.side,
        status=source.status,
        requested_quantity=source.requested_quantity,
        cumulative_filled_quantity=source.cumulative_filled_quantity,
        remaining_quantity=source.remaining_quantity,
        outcome_code=source.outcome_code,
    )
    request = _request(
        fixture,
        receipt,
        order_reports=(*base.order_reports, unexpected),
    )
    authoritative = ReconciliationEngine().reconcile(request)
    unexpected_check = next(
        item
        for item in authoritative.order_checks
        if item.check_id == "unexpected-order:unexpected-client-order-contract"
    )

    with pytest.raises(ValueError, match="authoritative engine output"):
        _build_result_with_children(
            request,
            authoritative,
            order_checks=tuple(
                item for item in authoritative.order_checks if item != unexpected_check
            ),
        )


def test_public_result_build_rejects_deleted_unexpected_fill_check() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    unexpected = ObservedFillReport.build(
        source_system=source.source_system,
        account_id=source.account_id,
        report_id="unexpected-fill-contract-report",
        source_fill_id="unexpected-source-fill-contract",
        source_order_id="unexpected-source-order-contract",
        client_order_id="unexpected-client-order-contract",
        client_fill_id="unexpected-client-fill-contract",
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
    request = _request(
        fixture,
        receipt,
        fill_reports=(*base.fill_reports, unexpected),
    )
    authoritative = ReconciliationEngine().reconcile(request)
    unexpected_check = next(
        item
        for item in authoritative.fill_checks
        if item.order_id == "unexpected-client-order-contract"
    )

    with pytest.raises(ValueError, match="authoritative engine output"):
        _build_result_with_children(
            request,
            authoritative,
            fill_checks=tuple(
                item for item in authoritative.fill_checks if item != unexpected_check
            ),
        )


def test_request_rejects_fill_aggregate_that_crosses_numeric_boundary() -> None:
    fixture, receipt = _execute_full()
    request = _request(fixture, receipt)
    base = request.observed
    source = base.fill_reports[0]
    maximum_single_quantity = Decimal("9" * 200)
    with localcontext() as context:
        context.prec = 500
        aggregate = maximum_single_quantity + maximum_single_quantity
    assert len(maximum_single_quantity.as_tuple().digits) == 200
    assert len(aggregate.as_tuple().digits) == 201

    reports = tuple(
        ObservedFillReport.build(
            source_system=source.source_system,
            account_id=source.account_id,
            report_id=f"aggregate-boundary-report-{index}",
            source_fill_id=f"aggregate-boundary-source-fill-{index}",
            source_order_id="aggregate-boundary-source-order",
            client_order_id="aggregate-boundary-client-order",
            client_fill_id=f"aggregate-boundary-client-fill-{index}",
            filled_at=source.filled_at,
            available_at=source.available_at,
            instrument_id=source.instrument_id,
            instrument_type=source.instrument_type,
            side=Side.BUY,
            currency=source.currency,
            quantity=maximum_single_quantity,
            price=Decimal(1),
            commission=Decimal(0),
            stamp_duty=Decimal(0),
            transfer_fee=Decimal(0),
            other_fee=Decimal(0),
        )
        for index in range(2)
    )
    evidence = ObservedExecutionEvidence.build(
        source_system=base.source_system,
        account_id=base.account_id,
        observed_at=base.observed_at,
        available_at=base.available_at,
        account_snapshot=base.account_snapshot,
        order_reports=base.order_reports,
        fill_reports=(*base.fill_reports, *reports),
        source_cursor="aggregate-boundary-cursor",
    )

    with pytest.raises(ValueError, match=r"fill_aggregate.*numeric boundary"):
        ReconciliationRequest.build(
            request_id="aggregate-boundary-request",
            idempotency_key="aggregate-boundary-key",
            reconciled_at=request.reconciled_at,
            draft=request.draft,
            receipt=request.receipt,
            observed=evidence,
            policy=request.policy,
        )


def test_cash_reconciliation_revalidation_is_decimal_context_stable() -> None:
    request, result = _clean_chain()
    expected = Decimal("12345.678901")
    observed = Decimal("12345.802357")
    with localcontext() as context:
        context.prec = 100
        exact_delta = observed - expected
        finding = ReconciliationFinding.build(
            domain=ReconciliationDomain.CASH,
            code=ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
            status=ReconciliationCheckStatus.MISMATCH,
            severity=ReconciliationSeverity.ERROR,
            entity_id="cash:total-context-stability",
            message="cash total differs under a context-stability fixture",
            decision_id=request.draft.decision_id,
            batch_hash=request.draft.batch_hash,
            expected_value=str(expected),
            observed_value=str(observed),
            delta=str(exact_delta),
            evidence_hashes=(request.observed.evidence_hash,),
        )
        check = _rebuild_cash_check(
            result.cash_check,
            expected_total_cash=expected,
            observed_total_cash=observed,
            total_delta=exact_delta,
            mismatched_fields=("total_cash",),
            findings=(finding,),
        )

    with localcontext() as context:
        context.prec = 4
        assert observed - expected != exact_delta
        assert replace(check) == check


def test_fill_reconciliation_revalidation_is_decimal_context_stable() -> None:
    _, result = _clean_chain()
    expected = Decimal("12345.678901")
    observed = Decimal("12345.802357")
    with localcontext() as context:
        context.prec = 100
        exact_delta = observed - expected
        check = _rebuild_fill_check(
            result.fill_checks[0],
            expected_quantity=expected,
            observed_quantity=observed,
            quantity_delta=exact_delta,
            expected_gross=expected,
            observed_gross=observed,
            gross_delta=exact_delta,
            expected_total_fee=expected,
            observed_total_fee=observed,
            fee_delta=exact_delta,
            expected_commission=expected,
            observed_commission=observed,
            commission_delta=exact_delta,
            expected_stamp_duty=expected,
            observed_stamp_duty=observed,
            stamp_duty_delta=exact_delta,
            expected_transfer_fee=expected,
            observed_transfer_fee=observed,
            transfer_fee_delta=exact_delta,
            expected_other_fee=expected,
            observed_other_fee=observed,
            other_fee_delta=exact_delta,
            expected_cash_change=expected,
            observed_cash_change=observed,
            cash_delta=exact_delta,
            expected_vwap=Decimal("10.123456"),
            observed_vwap=Decimal("10.234567"),
            findings=(),
        )

    with localcontext() as context:
        context.prec = 4
        assert observed - expected != exact_delta
        assert replace(check) == check


def test_position_reconciliation_revalidation_is_decimal_context_stable() -> None:
    _, result = _clean_chain()
    expected = Decimal("12345.678901")
    observed = Decimal("12345.802357")
    with localcontext() as context:
        context.prec = 100
        exact_delta = observed - expected
        check = _rebuild_position_check(
            result.position_checks[0],
            expected_total=expected,
            observed_total=observed,
            total_delta=exact_delta,
            expected_available=expected,
            observed_available=observed,
            available_delta=exact_delta,
            expected_frozen=expected,
            observed_frozen=observed,
            frozen_delta=exact_delta,
            expected_unsettled=expected,
            observed_unsettled=observed,
            unsettled_delta=exact_delta,
            expected_cost_basis=expected,
            observed_cost_basis=observed,
            cost_delta=exact_delta,
            expected_average_cost=expected,
            observed_average_cost=observed,
            average_cost_delta=exact_delta,
            findings=(),
        )

    with localcontext() as context:
        context.prec = 4
        assert observed - expected != exact_delta
        assert replace(check) == check
