"""Adversarial branch coverage for independent execution reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext

import pytest

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.execution.paper import PaperOrderStatus
from quant_agent.portfolio import AccountSnapshot, CashSnapshot, PortfolioPositionSnapshot
from quant_agent.reconciliation import (
    RECONCILIATION_ENGINE_VERSION,
    DuplicateClassification,
    DuplicateKeyKind,
    ObservedFillReport,
    ObservedOrderReport,
    ReconciliationDifferenceCode,
    ReconciliationEngine,
    ReconciliationPolicy,
    ReconciliationResult,
    ReconciliationSeverity,
    ReconciliationStatus,
)
from quant_agent.regime.contracts import stable_hash

from .test_reconciliation_engine import (
    _execute_full,
    _execute_no_fill,
    _execute_rejected,
    _request,
    _snapshot,
)


def _codes(result: ReconciliationResult) -> set[ReconciliationDifferenceCode]:
    return {finding.code for finding in result.all_findings}


def _order_report(
    source: ObservedOrderReport,
    *,
    report_id: str | None = None,
    source_system: str | None = None,
    source_order_id: str | None = None,
    client_order_id: str | None = None,
    unlinked: bool = False,
    revision: int | None = None,
    event_time: datetime | None = None,
    available_at: datetime | None = None,
    instrument_id: str | None = None,
    instrument_type: TradableInstrumentType | None = None,
    side: Side | None = None,
    status: PaperOrderStatus | None = None,
    requested_quantity: Decimal | None = None,
    cumulative_filled_quantity: Decimal | None = None,
    remaining_quantity: Decimal | None = None,
    outcome_code: str | None = None,
) -> ObservedOrderReport:
    selected_status = status or source.status
    return ObservedOrderReport.build(
        source_system=source_system or source.source_system,
        account_id=source.account_id,
        report_id=report_id or f"{source.report_id}:adversarial",
        source_order_id=source_order_id or source.source_order_id,
        client_order_id=(None if unlinked else (client_order_id or source.client_order_id)),
        revision=revision or source.revision,
        event_time=event_time or source.event_time,
        available_at=available_at or source.available_at,
        instrument_id=instrument_id or source.instrument_id,
        instrument_type=instrument_type or source.instrument_type,
        side=side or source.side,
        status=selected_status,
        requested_quantity=requested_quantity or source.requested_quantity,
        cumulative_filled_quantity=(
            source.cumulative_filled_quantity
            if cumulative_filled_quantity is None
            else cumulative_filled_quantity
        ),
        remaining_quantity=(
            source.remaining_quantity if remaining_quantity is None else remaining_quantity
        ),
        outcome_code=source.outcome_code if outcome_code is None else outcome_code,
    )


def _fill_report(
    source: ObservedFillReport,
    *,
    report_id: str | None = None,
    source_system: str | None = None,
    source_fill_id: str | None = None,
    source_order_id: str | None = None,
    client_order_id: str | None = None,
    client_fill_id: str | None = None,
    unlinked: bool = False,
    unidentified: bool = False,
    filled_at: datetime | None = None,
    available_at: datetime | None = None,
    instrument_id: str | None = None,
    instrument_type: TradableInstrumentType | None = None,
    side: Side | None = None,
    currency: str | None = None,
    quantity: Decimal | None = None,
    price: Decimal | None = None,
    commission: Decimal | None = None,
    stamp_duty: Decimal | None = None,
    transfer_fee: Decimal | None = None,
    other_fee: Decimal | None = None,
) -> ObservedFillReport:
    return ObservedFillReport.build(
        source_system=source_system or source.source_system,
        account_id=source.account_id,
        report_id=report_id or f"{source.report_id}:adversarial",
        source_fill_id=source_fill_id or source.source_fill_id,
        source_order_id=source_order_id or source.source_order_id,
        client_order_id=(None if unlinked else (client_order_id or source.client_order_id)),
        client_fill_id=(None if unidentified else (client_fill_id or source.client_fill_id)),
        filled_at=filled_at or source.filled_at,
        available_at=available_at or source.available_at,
        instrument_id=instrument_id or source.instrument_id,
        instrument_type=instrument_type or source.instrument_type,
        side=side or source.side,
        currency=currency or source.currency,
        quantity=quantity or source.quantity,
        price=price or source.price,
        commission=source.commission if commission is None else commission,
        stamp_duty=source.stamp_duty if stamp_duty is None else stamp_duty,
        transfer_fee=source.transfer_fee if transfer_fee is None else transfer_fee,
        other_fee=source.other_fee if other_fee is None else other_fee,
    )


def _position(
    source: PortfolioPositionSnapshot,
    *,
    instrument_id: str | None = None,
    instrument_type: TradableInstrumentType | None = None,
    available_quantity: Decimal | None = None,
    frozen_quantity: Decimal | None = None,
    unsettled_quantity: Decimal | None = None,
    average_cost: Decimal | None = None,
) -> PortfolioPositionSnapshot:
    return PortfolioPositionSnapshot.build(
        instrument_id=instrument_id or source.instrument_id,
        instrument_type=instrument_type or source.instrument_type,
        available_quantity=(
            source.available_quantity if available_quantity is None else available_quantity
        ),
        frozen_quantity=(source.frozen_quantity if frozen_quantity is None else frozen_quantity),
        unsettled_quantity=(
            source.unsettled_quantity if unsettled_quantity is None else unsettled_quantity
        ),
        average_cost=source.average_cost if average_cost is None else average_cost,
        valuation_price=source.valuation_price,
        price_observed_at=source.price_observed_at,
        price_available_at=source.price_available_at,
        position_as_of=source.position_as_of,
        price_data_version=source.price_data_version,
        price_source_hash=source.price_source_hash,
        valuation_status=source.valuation_status,
        mark_policy_version=source.mark_policy_version,
        mark_policy_hash=source.mark_policy_hash,
    )


def _account_snapshot(
    source: AccountSnapshot,
    *,
    snapshot_id: str,
    runtime_mode: RuntimeMode | None = None,
    data_version: str | None = None,
    currency: str | None = None,
    previous_snapshot_hash: str | None = None,
) -> AccountSnapshot:
    selected_currency = currency or source.currency
    cash = CashSnapshot(
        as_of=source.as_of,
        currency=selected_currency,
        total_cash=source.cash.total_cash,
        available_cash=source.cash.available_cash,
        frozen_cash=source.cash.frozen_cash,
    )
    return AccountSnapshot.build(
        snapshot_id=snapshot_id,
        account_id=source.account_id,
        runtime_mode=runtime_mode or source.runtime_mode,
        as_of=source.as_of,
        valuation_at=source.valuation_at,
        data_version=data_version or source.data_version,
        currency=selected_currency,
        cash=cash,
        positions=source.positions,
        previous_snapshot_hash=(previous_snapshot_hash or source.previous_snapshot_hash),
        source_event_log_hash=source.source_event_log_hash,
    )


def test_unlinked_order_report_is_critical_even_when_economics_match() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    unlinked = _order_report(base.order_reports[0], unlinked=True)
    reports = (unlinked, *base.order_reports[1:])

    result = ReconciliationEngine().reconcile(_request(fixture, receipt, order_reports=reports))

    assert ReconciliationDifferenceCode.UNLINKED_ORDER_REPORT in _codes(result)
    assert ReconciliationDifferenceCode.ORDER_MISSING in _codes(result)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_unlinked_fill_report_is_critical_even_when_economics_match() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    unlinked = _fill_report(base.fill_reports[0], unlinked=True)
    reports = (unlinked, *base.fill_reports[1:])

    result = ReconciliationEngine().reconcile(_request(fixture, receipt, fill_reports=reports))

    assert ReconciliationDifferenceCode.UNLINKED_FILL_REPORT in _codes(result)
    assert ReconciliationDifferenceCode.FILL_MISSING in _codes(result)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_unexpected_order_report_is_not_silently_discarded() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    unexpected = _order_report(
        source,
        report_id="unexpected-order-report",
        source_order_id="unexpected-source-order",
        client_order_id="unexpected-client-order",
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, unexpected),
        )
    )

    assert ReconciliationDifferenceCode.ORDER_UNEXPECTED in _codes(result)
    check = next(item for item in result.order_checks if item.order_id == "unexpected-client-order")
    assert check.observed_requested_quantity == source.requested_quantity
    assert result.stop_signal.required


def test_order_identity_status_and_quantity_mismatches_are_all_visible() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    identity_source, status_source, quantity_source = base.order_reports[:3]
    identity = _order_report(
        identity_source,
        instrument_id="CN.SSE.999999",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.BUY if identity_source.side is Side.SELL else Side.SELL,
    )
    status = _order_report(status_source, status=PaperOrderStatus.CANCELED)
    changed_quantity = quantity_source.requested_quantity + Decimal(1)
    quantity = _order_report(
        quantity_source,
        requested_quantity=changed_quantity,
        cumulative_filled_quantity=changed_quantity,
        remaining_quantity=Decimal(0),
    )
    reports = (identity, status, quantity, *base.order_reports[3:])

    result = ReconciliationEngine().reconcile(_request(fixture, receipt, order_reports=reports))

    assert {
        ReconciliationDifferenceCode.ORDER_IDENTITY_MISMATCH,
        ReconciliationDifferenceCode.ORDER_STATUS_MISMATCH,
        ReconciliationDifferenceCode.ORDER_QUANTITY_MISMATCH,
    }.issubset(_codes(result))
    assert result.status is ReconciliationStatus.MISMATCH
    assert result.stop_signal.required


def test_order_outcome_reason_is_linked_and_mismatch_is_critical() -> None:
    fixture, receipt = _execute_no_fill()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    assert source.outcome_code == "ZERO_CAPACITY"
    changed = _order_report(source, outcome_code="FOREIGN_NO_FILL_REASON")

    result = ReconciliationEngine().reconcile(_request(fixture, receipt, order_reports=(changed,)))

    assert ReconciliationDifferenceCode.ORDER_REASON_MISMATCH in _codes(result)
    check = result.order_checks[0]
    assert check.expected_outcome_code == "ZERO_CAPACITY"
    assert check.observed_outcome_code == "FOREIGN_NO_FILL_REASON"
    assert result.max_severity is ReconciliationSeverity.CRITICAL
    assert result.stop_signal.required


def test_conflicting_order_duplicate_is_unreconcilable() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    conflict = _order_report(
        source,
        report_id=f"{source.report_id}:conflict",
        status=PaperOrderStatus.CANCELED,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, conflict),
        )
    )

    group = next(
        item
        for item in result.duplicate_groups
        if item.classification is DuplicateClassification.CONFLICTING
    )
    assert group.canonical_report_hash is None
    assert group.difference_code is ReconciliationDifferenceCode.CONFLICTING_DUPLICATE_REPORT
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_same_report_id_with_different_content_is_a_conflicting_duplicate() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    conflict = _order_report(
        source,
        report_id=source.report_id,
        status=PaperOrderStatus.CANCELED,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, conflict),
        )
    )

    group = next(
        item
        for item in result.duplicate_groups
        if item.classification is DuplicateClassification.CONFLICTING
    )
    assert group.report_ids == (source.report_id, source.report_id)
    assert len(set(group.report_hashes)) == 2
    assert group.canonical_report_hash is None
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_same_order_report_id_with_two_future_hashes_returns_a_stop_result() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    shared_report_id = "same-future-order-report"
    first = _order_report(
        source,
        report_id=shared_report_id,
        available_at=source.available_at + timedelta(seconds=10),
    )
    second = _order_report(
        source,
        report_id=shared_report_id,
        available_at=source.available_at + timedelta(seconds=11),
        status=PaperOrderStatus.CANCELED,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(first, second, *base.order_reports[1:]),
            reconciled_at=source.available_at + timedelta(seconds=1),
        )
    )

    group = next(item for item in result.duplicate_groups if shared_report_id in item.report_ids)
    future_findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.FUTURE_EVIDENCE
    ]
    assert group.key_kind is DuplicateKeyKind.REPORT_ID
    assert group.classification is DuplicateClassification.CONFLICTING
    assert len(future_findings) == 2
    assert len({item.entity_id for item in future_findings}) == 2
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_same_fill_report_id_with_two_future_hashes_returns_a_stop_result() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    shared_report_id = "same-future-fill-report"
    first = _fill_report(
        source,
        report_id=shared_report_id,
        available_at=source.available_at + timedelta(seconds=10),
    )
    second = _fill_report(
        source,
        report_id=shared_report_id,
        available_at=source.available_at + timedelta(seconds=11),
        price=source.price + Decimal("0.01"),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(first, second, *base.fill_reports[1:]),
            reconciled_at=source.available_at + timedelta(seconds=1),
        )
    )

    group = next(item for item in result.duplicate_groups if shared_report_id in item.report_ids)
    future_findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.FUTURE_EVIDENCE
    ]
    assert group.key_kind is DuplicateKeyKind.REPORT_ID
    assert group.classification is DuplicateClassification.CONFLICTING
    assert len(future_findings) == 2
    assert len({item.entity_id for item in future_findings}) == 2
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_duplicate_key_kinds_namespace_natural_legacy_sentinel_values() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source_identity_seed, report_id_seed = base.order_reports[:2]
    legacy_sentinel = DuplicateKeyKind.REPORT_ID.value
    source_first = _order_report(
        source_identity_seed,
        report_id="source-identity-envelope-1",
        source_order_id=legacy_sentinel,
    )
    source_second = _order_report(
        source_identity_seed,
        report_id="source-identity-envelope-2",
        source_order_id=legacy_sentinel,
    )
    report_first = _order_report(
        report_id_seed,
        report_id=legacy_sentinel,
    )
    report_second = _order_report(
        report_id_seed,
        report_id=legacy_sentinel,
        status=PaperOrderStatus.CANCELED,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(
                source_first,
                source_second,
                report_first,
                report_second,
                *base.order_reports[2:],
            ),
        )
    )

    source_group = next(
        item
        for item in result.duplicate_groups
        if item.key_kind is DuplicateKeyKind.SOURCE_IDENTITY
    )
    report_group = next(
        item for item in result.duplicate_groups if item.key_kind is DuplicateKeyKind.REPORT_ID
    )
    assert source_group.dedup_key[2] == legacy_sentinel
    assert report_group.dedup_key[2] == legacy_sentinel
    assert source_group.group_id != report_group.group_id
    assert source_group.group_hash != report_group.group_hash


def test_identical_duplicate_with_future_envelope_cannot_hide_in_canonicalization() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    future = _order_report(
        source,
        report_id=f"{source.report_id}:future-envelope",
        available_at=source.available_at + timedelta(seconds=10),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, future),
            reconciled_at=source.available_at + timedelta(seconds=1),
        )
    )

    group = next(
        item for item in result.duplicate_groups if source.report_hash in item.report_hashes
    )
    assert group.classification is DuplicateClassification.IDENTICAL
    assert len(set(group.content_hashes)) == 1
    assert ReconciliationDifferenceCode.FUTURE_EVIDENCE in _codes(result)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_identical_duplicate_with_stale_envelope_cannot_hide_in_canonicalization() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    stale = _fill_report(
        source,
        report_id=f"{source.report_id}:stale-envelope",
        available_at=source.filled_at,
    )
    policy = ReconciliationPolicy(max_observation_lag_seconds=1)

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(*base.fill_reports, stale),
            reconciled_at=source.available_at + timedelta(seconds=1),
            policy=policy,
        )
    )

    group = next(
        item for item in result.duplicate_groups if source.report_hash in item.report_hashes
    )
    assert group.classification is DuplicateClassification.IDENTICAL
    assert len(set(group.content_hashes)) == 1
    stale_findings = [
        finding
        for finding in result.input_findings
        if finding.code is ReconciliationDifferenceCode.STALE_EVIDENCE
    ]
    assert len(stale_findings) == 1
    assert stale.report_id in stale_findings[0].entity_id
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_multiple_source_orders_cannot_claim_one_client_order() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    second_source = _order_report(
        source,
        report_id=f"{source.report_id}:second-source",
        source_order_id=f"{source.source_order_id}:second-source",
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, second_source),
        )
    )

    findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.ORDER_IDENTITY_MISMATCH
    ]
    assert len(findings) == 1
    assert findings[0].order_id == source.client_order_id
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_fill_identity_mismatches_cover_source_and_business_identity() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    changed = _fill_report(
        source,
        source_order_id="foreign-source-order",
        client_fill_id="foreign-client-fill",
        instrument_id="CN.SSE.999999",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.BUY if source.side is Side.SELL else Side.SELL,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(changed, *base.fill_reports[1:]),
        )
    )

    identity_findings = [
        item
        for item in result.all_findings
        if item.code is ReconciliationDifferenceCode.FILL_IDENTITY_MISMATCH
    ]
    assert len(identity_findings) == 2
    assert result.stop_signal.required


def test_fill_quantity_mismatch_and_overfill_are_distinguished() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    quantity_source, overfill_source = base.fill_reports[:2]
    expected_overfill_order = next(
        item for item in receipt.orders if item.order_id == overfill_source.client_order_id
    )
    quantity = _fill_report(quantity_source, quantity=quantity_source.quantity - Decimal(1))
    overfill = _fill_report(
        overfill_source,
        quantity=expected_overfill_order.quantity + Decimal(1),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(quantity, overfill, *base.fill_reports[2:]),
        )
    )

    assert {
        ReconciliationDifferenceCode.FILL_QUANTITY_MISMATCH,
        ReconciliationDifferenceCode.FILL_OVERFILL,
    }.issubset(_codes(result))
    assert result.stop_signal.required


def test_duplicate_client_fill_identity_is_rejected_even_when_aggregates_match() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    assert source.quantity == Decimal(200)
    first = _fill_report(
        source,
        report_id=f"{source.report_id}:split-1",
        source_fill_id=f"{source.source_fill_id}:split-1",
        quantity=Decimal(100),
        commission=source.commission / 2,
        stamp_duty=source.stamp_duty / 2,
        transfer_fee=source.transfer_fee / 2,
        other_fee=source.other_fee / 2,
    )
    second = _fill_report(
        source,
        report_id=f"{source.report_id}:split-2",
        source_fill_id=f"{source.source_fill_id}:split-2",
        quantity=Decimal(100),
        commission=source.commission / 2,
        stamp_duty=source.stamp_duty / 2,
        transfer_fee=source.transfer_fee / 2,
        other_fee=source.other_fee / 2,
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(first, second, *base.fill_reports[1:]),
        )
    )

    check = next(item for item in result.fill_checks if item.order_id == source.client_order_id)
    assert check.quantity_delta == 0
    assert check.gross_delta == 0
    assert check.fee_delta == 0
    assert check.cash_delta == 0
    assert ReconciliationDifferenceCode.FILL_IDENTITY_MISMATCH in _codes(result)
    assert result.stop_signal.required


def test_fill_price_fee_gross_and_cash_differences_are_all_visible() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    changed = _fill_report(
        source,
        price=source.price + Decimal("0.01"),
        commission=source.commission + Decimal("0.01"),
        stamp_duty=source.stamp_duty + Decimal("0.02"),
        transfer_fee=source.transfer_fee + Decimal("0.03"),
        other_fee=source.other_fee + Decimal("0.04"),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(changed, *base.fill_reports[1:]),
        )
    )

    assert {
        ReconciliationDifferenceCode.FILL_GROSS_MISMATCH,
        ReconciliationDifferenceCode.FILL_PRICE_MISMATCH,
        ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
        ReconciliationDifferenceCode.FILL_CASH_MISMATCH,
    }.issubset(_codes(result))
    check = next(item for item in result.fill_checks if item.order_id == source.client_order_id)
    assert check.commission_delta == Decimal("0.01")
    assert check.stamp_duty_delta == Decimal("0.02")
    assert check.transfer_fee_delta == Decimal("0.03")
    assert check.other_fee_delta == Decimal("0.04")
    assert check.fee_delta == Decimal("0.10")
    assert result.max_severity is ReconciliationSeverity.CRITICAL
    assert result.stop_signal.required


def test_fee_component_swap_is_visible_when_total_fee_and_cash_are_unchanged() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    assert source.commission >= Decimal(1)
    changed = _fill_report(
        source,
        commission=source.commission - Decimal(1),
        other_fee=source.other_fee + Decimal(1),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(changed, *base.fill_reports[1:]),
        )
    )

    check = next(item for item in result.fill_checks if item.order_id == source.client_order_id)
    assert check.fee_delta == 0
    assert check.cash_delta == 0
    assert check.commission_delta == Decimal(-1)
    assert check.other_fee_delta == Decimal(1)
    component_findings = [
        item
        for item in check.findings
        if item.code is ReconciliationDifferenceCode.FILL_FEE_MISMATCH
    ]
    assert {item.entity_id.rsplit(":", 1)[-1] for item in component_findings} >= {
        "commission",
        "other-fee",
    }
    assert result.stop_signal.required


def test_fill_differences_inside_tolerance_remain_explicit_warnings() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    changed = _fill_report(
        source,
        price=source.price + Decimal("0.0001"),
        commission=source.commission + Decimal("0.0001"),
    )
    policy = ReconciliationPolicy(
        cash_tolerance=Decimal("0.1"),
        fee_tolerance=Decimal("0.001"),
        price_tolerance=Decimal("0.001"),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(changed, *base.fill_reports[1:]),
            policy=policy,
        )
    )

    visible_codes = {
        ReconciliationDifferenceCode.FILL_PRICE_MISMATCH,
        ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
    }
    assert visible_codes.issubset(_codes(result))
    assert all(
        finding.severity is ReconciliationSeverity.WARNING
        for finding in result.all_findings
        if finding.code in visible_codes
    )
    assert result.status is ReconciliationStatus.RECONCILED_WITH_VARIANCE
    assert not result.stop_signal.required


def test_unidentified_fill_keeps_sub_display_price_difference_critical() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    with localcontext() as context:
        context.prec = 100
        changed_price = source.price + Decimal("1e-60")
    changed = _fill_report(
        source,
        unidentified=True,
        price=changed_price,
    )
    policy = ReconciliationPolicy(
        price_tolerance=Decimal(0),
        cash_tolerance=Decimal("1e-50"),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(changed, *base.fill_reports[1:]),
            policy=policy,
        )
    )

    check = next(item for item in result.fill_checks if item.order_id == source.client_order_id)
    price_findings = [
        item
        for item in check.findings
        if item.code is ReconciliationDifferenceCode.FILL_PRICE_MISMATCH
    ]
    assert changed.client_fill_id is None
    assert check.gross_delta != 0
    assert abs(check.gross_delta) <= policy.cash_tolerance
    assert len(price_findings) == 1
    assert price_findings[0].severity is ReconciliationSeverity.CRITICAL
    assert result.status is ReconciliationStatus.MISMATCH
    assert result.stop_signal.required


def test_unknown_order_tiny_nonterminating_vwap_returns_unexpected_stop() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    source = base.fill_reports[0]
    unknown_order_id = "unknown-tiny-price-client-order"
    first = _fill_report(
        source,
        report_id="unknown-tiny-fill-report-1",
        source_fill_id="unknown-tiny-source-fill-1",
        source_order_id="unknown-tiny-source-order",
        client_order_id=unknown_order_id,
        client_fill_id="unknown-tiny-client-fill-1",
        quantity=Decimal(1),
        price=Decimal("1e-200"),
        commission=Decimal(0),
        stamp_duty=Decimal(0),
        transfer_fee=Decimal(0),
        other_fee=Decimal(0),
    )
    second = _fill_report(
        source,
        report_id="unknown-tiny-fill-report-2",
        source_fill_id="unknown-tiny-source-fill-2",
        source_order_id="unknown-tiny-source-order",
        client_order_id=unknown_order_id,
        client_fill_id="unknown-tiny-client-fill-2",
        quantity=Decimal(2),
        price=Decimal("2e-200"),
        commission=Decimal(0),
        stamp_duty=Decimal(0),
        transfer_fee=Decimal(0),
        other_fee=Decimal(0),
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            fill_reports=(*base.fill_reports, first, second),
        )
    )

    check = next(item for item in result.fill_checks if item.order_id == unknown_order_id)
    assert check.expected_quantity == 0
    assert check.observed_quantity == 3
    assert check.observed_gross == Decimal("5e-200")
    assert check.observed_vwap is not None
    assert ReconciliationDifferenceCode.FILL_UNEXPECTED in {item.code for item in check.findings}
    assert result.status is ReconciliationStatus.MISMATCH
    assert result.stop_signal.required


@pytest.mark.parametrize(
    ("threshold", "required"),
    (
        (ReconciliationSeverity.ERROR, False),
        (ReconciliationSeverity.WARNING, True),
    ),
)
def test_warning_stop_threshold_is_policy_driven(
    threshold: ReconciliationSeverity,
    required: bool,
) -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    duplicate = base.order_reports[0]
    policy = ReconciliationPolicy(stop_on_severity=threshold)

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, duplicate),
            policy=policy,
        )
    )

    assert result.max_severity is ReconciliationSeverity.WARNING
    assert result.stop_signal.required is required
    assert result.stop_signal.engine_version == RECONCILIATION_ENGINE_VERSION
    assert result.stop_signal.policy_hash == policy.policy_hash
    if required:
        assert result.stop_signal.severity is result.max_severity
        assert (
            ReconciliationDifferenceCode.IDENTICAL_DUPLICATE_REPORT
            in result.stop_signal.reason_codes
        )
    else:
        assert result.stop_signal.severity is ReconciliationSeverity.INFO
        assert not result.stop_signal.reason_codes


def test_missing_and_unexpected_positions_are_both_reported() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(receipt)
    missing_source = snapshot.positions[0]
    extra = _position(
        snapshot.positions[-1],
        instrument_id="CN.SSE.599999",
        instrument_type=TradableInstrumentType.STOCK,
    )
    positions = tuple(sorted((*snapshot.positions[1:], extra), key=lambda item: item.instrument_id))

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(
                receipt,
                positions=positions,
                snapshot_id="missing-and-unexpected-position",
            ),
        )
    )

    assert {
        ReconciliationDifferenceCode.POSITION_MISSING,
        ReconciliationDifferenceCode.POSITION_UNEXPECTED,
    }.issubset(_codes(result))
    missing_check = next(
        item
        for item in result.position_checks
        if item.instrument_id == missing_source.instrument_id
    )
    extra_check = next(
        item for item in result.position_checks if item.instrument_id == extra.instrument_id
    )
    assert missing_check.observed_total is None
    assert extra_check.expected_total is None
    assert result.stop_signal.required


def test_position_type_quantity_and_cost_mismatches_are_all_visible() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(receipt)
    type_source, quantity_source, cost_source = snapshot.positions[:3]
    wrong_type = _position(type_source, instrument_type=TradableInstrumentType.STOCK)
    wrong_quantity = _position(
        quantity_source,
        available_quantity=quantity_source.available_quantity + Decimal(1),
    )
    wrong_cost = _position(cost_source, average_cost=cost_source.average_cost + Decimal(1))
    positions = tuple(
        sorted((wrong_type, wrong_quantity, wrong_cost), key=lambda item: item.instrument_id)
    )

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(
                receipt,
                positions=positions,
                snapshot_id="position-type-quantity-cost",
            ),
        )
    )

    assert {
        ReconciliationDifferenceCode.POSITION_TYPE_MISMATCH,
        ReconciliationDifferenceCode.POSITION_QUANTITY_MISMATCH,
        ReconciliationDifferenceCode.POSITION_COST_MISMATCH,
    }.issubset(_codes(result))
    assert result.max_severity is ReconciliationSeverity.CRITICAL
    assert result.stop_signal.required


def test_position_cost_inside_tolerance_remains_visible_warning() -> None:
    fixture, receipt = _execute_full()
    snapshot = _snapshot(receipt)
    source = snapshot.positions[0]
    changed = _position(source, average_cost=source.average_cost + Decimal("0.0001"))
    positions = tuple(
        changed if item.instrument_id == source.instrument_id else item
        for item in snapshot.positions
    )
    policy = ReconciliationPolicy(cost_tolerance=Decimal("0.1"))

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            snapshot=_snapshot(
                receipt,
                positions=positions,
                snapshot_id="position-cost-within-tolerance",
            ),
            policy=policy,
        )
    )

    findings = [
        item
        for item in result.all_findings
        if item.code is ReconciliationDifferenceCode.POSITION_COST_MISMATCH
    ]
    assert len(findings) == 2
    assert {item.entity_id.rsplit(":", 1)[-1] for item in findings} == {
        "average_cost",
        "cost_basis",
    }
    assert all(item.severity is ReconciliationSeverity.WARNING for item in findings)
    assert result.status is ReconciliationStatus.RECONCILED_WITH_VARIANCE
    assert not result.stop_signal.required


def test_future_evidence_and_reports_are_rejected_at_reconciliation_time() -> None:
    fixture, receipt = _execute_full()
    available_at = receipt.processed_at + timedelta(seconds=10)
    reconciled_at = available_at - timedelta(seconds=1)

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            evidence_available_at=available_at,
            reconciled_at=reconciled_at,
        )
    )

    future_findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.FUTURE_EVIDENCE
    ]
    assert len(future_findings) == 1 + len(receipt.orders) + len(receipt.fills)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_rejected_final_order_report_cannot_predate_its_match_attempt() -> None:
    fixture, receipt = _execute_rejected()
    base = _request(fixture, receipt).observed
    source = base.order_reports[0]
    attempt = receipt.attempts[0]
    assert source.status is PaperOrderStatus.REJECTED
    predating = _order_report(
        source,
        event_time=attempt.attempted_at - timedelta(microseconds=1),
    )

    result = ReconciliationEngine().reconcile(
        _request(fixture, receipt, order_reports=(predating,))
    )

    boundary_findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH
        and item.entity_id.endswith(":event-lower-bound")
    ]
    assert len(boundary_findings) == 1
    assert boundary_findings[0].observed_value == str(predating.event_time)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_fill_report_build_is_independent_of_global_decimal_context() -> None:
    fixture, receipt = _execute_full()
    source = _request(fixture, receipt).observed.fill_reports[0]

    def build() -> ObservedFillReport:
        return ObservedFillReport.build(
            source_system=source.source_system,
            account_id=source.account_id,
            report_id="decimal-context-fill-report",
            source_fill_id="decimal-context-source-fill",
            source_order_id=source.source_order_id,
            client_order_id=source.client_order_id,
            client_fill_id="decimal-context-client-fill",
            filled_at=source.filled_at,
            available_at=source.available_at,
            instrument_id=source.instrument_id,
            instrument_type=source.instrument_type,
            side=source.side,
            currency=source.currency,
            quantity=Decimal("123456789"),
            price=Decimal("123456789.123456789123456789"),
            commission=Decimal("0.123456789123456789"),
            stamp_duty=Decimal("0.234567891234567891"),
            transfer_fee=Decimal("0.345678912345678912"),
            other_fee=Decimal("0.456789123456789123"),
        )

    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        low_precision = build()
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_UP
        high_precision = build()

    assert low_precision == high_precision
    assert low_precision.gross_amount == high_precision.gross_amount
    assert low_precision.total_fee == high_precision.total_fee
    assert low_precision.cash_change == high_precision.cash_change
    assert low_precision.source_payload_hash == high_precision.source_payload_hash
    assert low_precision.report_hash == high_precision.report_hash


def test_large_order_report_build_is_stable_under_low_global_precision() -> None:
    fixture, receipt = _execute_full()
    source = _request(fixture, receipt).observed.order_reports[0]
    requested = Decimal(f"1{'0' * 38}1")
    filled = Decimal(f"1{'0' * 39}")
    remaining = Decimal(1)

    def build() -> ObservedOrderReport:
        return ObservedOrderReport.build(
            source_system=source.source_system,
            account_id=source.account_id,
            report_id="large-decimal-order-report",
            source_order_id="large-decimal-source-order",
            client_order_id="large-decimal-client-order",
            revision=1,
            event_time=source.event_time,
            available_at=source.available_at,
            instrument_id=source.instrument_id,
            instrument_type=source.instrument_type,
            side=source.side,
            status=PaperOrderStatus.PARTIALLY_FILLED,
            requested_quantity=requested,
            cumulative_filled_quantity=filled,
            remaining_quantity=remaining,
        )

    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_DOWN
        low_precision = build()
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_UP
        high_precision = build()

    assert len(low_precision.requested_quantity.as_tuple().digits) == 40
    assert low_precision == high_precision
    assert low_precision.source_payload_hash == high_precision.source_payload_hash
    assert low_precision.report_hash == high_precision.report_hash


def test_report_source_mismatch_is_checked_for_orders_and_fills() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    order = _order_report(base.order_reports[0], source_system="rogue-source")
    fill = _fill_report(base.fill_reports[0], source_system="rogue-source")

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(order, *base.order_reports[1:]),
            fill_reports=(fill, *base.fill_reports[1:]),
        )
    )

    findings = [
        item
        for item in result.input_findings
        if item.code is ReconciliationDifferenceCode.SOURCE_SYSTEM_MISMATCH
    ]
    assert len(findings) == 2
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("runtime", ReconciliationDifferenceCode.RUNTIME_MODE_MISMATCH),
        ("data", ReconciliationDifferenceCode.DATA_VERSION_MISMATCH),
        ("currency", ReconciliationDifferenceCode.CURRENCY_MISMATCH),
        ("boundary", ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH),
    ),
)
def test_snapshot_runtime_data_currency_and_lineage_are_independently_bound(
    case: str,
    expected_code: ReconciliationDifferenceCode,
) -> None:
    fixture, receipt = _execute_full()
    source = _snapshot(receipt)
    kwargs: dict[str, object] = {}
    if case == "runtime":
        kwargs["runtime_mode"] = RuntimeMode.RESEARCH
    elif case == "data":
        kwargs["data_version"] = "unexpected-data-version"
    elif case == "currency":
        kwargs["currency"] = "USD"
    else:
        kwargs["previous_snapshot_hash"] = stable_hash({"wrong": "snapshot-boundary"})
    observed = _account_snapshot(source, snapshot_id=f"mismatch-{case}", **kwargs)  # type: ignore[arg-type]

    result = ReconciliationEngine().reconcile(_request(fixture, receipt, snapshot=observed))

    assert expected_code in _codes(result)
    assert result.status is ReconciliationStatus.UNRECONCILABLE
    assert result.stop_signal.required


def test_error_threshold_still_stops_on_critical_difference() -> None:
    fixture, receipt = _execute_full()
    base = _request(fixture, receipt).observed
    unexpected = _order_report(
        base.order_reports[0],
        source_order_id="critical-unexpected-source",
        client_order_id="critical-unexpected-client",
    )
    policy = ReconciliationPolicy(stop_on_severity=ReconciliationSeverity.ERROR)

    result = ReconciliationEngine().reconcile(
        _request(
            fixture,
            receipt,
            order_reports=(*base.order_reports, unexpected),
            policy=policy,
        )
    )

    assert result.max_severity is ReconciliationSeverity.CRITICAL
    assert result.stop_signal.required
    assert result.stop_signal.engine_version == result.engine_version
    assert result.stop_signal.policy_hash == result.policy_hash == policy.policy_hash
    assert result.stop_signal.severity is result.max_severity
    assert ReconciliationDifferenceCode.ORDER_UNEXPECTED in result.stop_signal.reason_codes
