"""Deterministic, read-only reconciliation over expected and independently observed evidence."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from enum import StrEnum
from itertools import pairwise
from typing import cast

from quant_agent.execution.paper import PaperFill, PaperOrder, PaperOrderStatus
from quant_agent.regime.contracts import stable_hash

from .contracts import (
    CashReconciliation,
    DuplicateKeyKind,
    DuplicateReportGroup,
    FillReconciliation,
    ObservedFillReport,
    ObservedOrderReport,
    OrderReconciliation,
    PositionReconciliation,
    ReconciliationCheckStatus,
    ReconciliationDifferenceCode,
    ReconciliationDomain,
    ReconciliationFinding,
    ReconciliationReportKind,
    ReconciliationRequest,
    ReconciliationResult,
    ReconciliationSeverity,
    _required_reconciliation_precision,
)

_ZERO = Decimal(0)
_DERIVED_RATIO_PRECISION = 200
_EXACT_COMPARISON_PRECISION = 832
_TERMINAL_ORDER_STATUSES = {
    PaperOrderStatus.FILLED,
    PaperOrderStatus.CANCELED,
    PaperOrderStatus.EXPIRED,
    PaperOrderStatus.REJECTED,
}


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    return str(value)


def _within(delta: Decimal, tolerance: Decimal) -> bool:
    return delta != 0 and abs(delta) <= tolerance


def _report_entity_prefix(
    kind: ReconciliationReportKind,
    report_id: str,
    report_hash: str,
) -> str:
    """Return a readable occurrence identity that cannot alias a conflicting envelope."""

    return f"{kind.value.lower()}-report:{report_id}:{report_hash}"


class ReconciliationEngine:
    """Compare one exact execution boundary without mutating execution or risk state."""

    def reconcile(self, request: ReconciliationRequest) -> ReconciliationResult:
        """Return all differences and an immutable downstream stop request."""

        with localcontext() as context:
            context.prec = self._required_precision(request)
            context.rounding = ROUND_HALF_EVEN
            return self._reconcile(request)

    def _reconcile(self, request: ReconciliationRequest) -> ReconciliationResult:
        """Run under an input-derived decimal context independent of process globals."""

        canonical_orders, order_duplicates = self._deduplicate_orders(
            request.observed.order_reports
        )
        canonical_fills, fill_duplicates = self._deduplicate_fills(request.observed.fill_reports)
        duplicate_groups = tuple(
            sorted((*order_duplicates, *fill_duplicates), key=lambda item: item.group_id)
        )
        input_findings = self._input_findings(
            request=request,
            order_reports=request.observed.order_reports,
            fill_reports=request.observed.fill_reports,
        )
        latest_orders, order_report_hashes, order_link_findings = self._latest_order_reports(
            request=request,
            reports=canonical_orders,
        )
        observed_fills, fill_link_findings = self._linked_fill_reports(
            request=request,
            reports=canonical_fills,
        )
        input_findings = tuple(
            sorted(
                (*input_findings, *order_link_findings, *fill_link_findings),
                key=lambda item: item.finding_hash,
            )
        )
        order_checks = self._order_checks(
            request=request,
            latest_reports=latest_orders,
            report_hashes=order_report_hashes,
        )
        fill_checks = self._fill_checks(
            request=request,
            observed_by_order=observed_fills,
            observed_orders=latest_orders,
        )
        cash_check = self._cash_check(request)
        position_checks = self._position_checks(request)
        return ReconciliationResult._build_from_engine(
            request=request,
            input_findings=input_findings,
            order_checks=order_checks,
            fill_checks=fill_checks,
            cash_check=cash_check,
            position_checks=position_checks,
            duplicate_groups=duplicate_groups,
        )

    @staticmethod
    def _required_precision(request: ReconciliationRequest) -> int:
        return _required_reconciliation_precision(request)

    @staticmethod
    def _deduplicate_orders(
        reports: tuple[ObservedOrderReport, ...],
    ) -> tuple[tuple[ObservedOrderReport, ...], tuple[DuplicateReportGroup, ...]]:
        report_ids: dict[tuple[str, str, str], list[ObservedOrderReport]] = defaultdict(list)
        for report in reports:
            report_ids[(report.source_system, report.account_id, report.report_id)].append(report)
        duplicates: list[DuplicateReportGroup] = []
        excluded_hashes: set[str] = set()
        for envelope_key in sorted(report_ids):
            values = report_ids[envelope_key]
            if len({item.report_hash for item in values}) <= 1:
                continue
            duplicates.append(
                DuplicateReportGroup.build(
                    report_kind=ReconciliationReportKind.ORDER,
                    key_kind=DuplicateKeyKind.REPORT_ID,
                    source_system=envelope_key[0],
                    dedup_key=envelope_key,
                    report_ids=tuple(item.report_id for item in values),
                    report_hashes=tuple(item.report_hash for item in values),
                    content_hashes=tuple(item.report_hash for item in values),
                )
            )
            excluded_hashes.update(item.report_hash for item in values)
        grouped: dict[tuple[str, str, str, str], list[ObservedOrderReport]] = defaultdict(list)
        for report in reports:
            if report.report_hash in excluded_hashes:
                continue
            grouped[report.dedup_key].append(report)
        canonical: list[ObservedOrderReport] = []
        for semantic_key in sorted(grouped):
            values = grouped[semantic_key]
            if len(values) == 1:
                canonical.append(values[0])
                continue
            group = DuplicateReportGroup.build(
                report_kind=ReconciliationReportKind.ORDER,
                key_kind=DuplicateKeyKind.SOURCE_IDENTITY,
                source_system=semantic_key[0],
                dedup_key=semantic_key,
                report_ids=tuple(item.report_id for item in values),
                report_hashes=tuple(item.report_hash for item in values),
                content_hashes=tuple(item.source_payload_hash for item in values),
            )
            duplicates.append(group)
            if group.canonical_report_hash is not None:
                canonical.append(min(values, key=lambda item: item.report_hash))
        return (
            tuple(sorted(canonical, key=lambda item: (*item.dedup_key, item.report_hash))),
            tuple(sorted(duplicates, key=lambda item: item.group_id)),
        )

    @staticmethod
    def _deduplicate_fills(
        reports: tuple[ObservedFillReport, ...],
    ) -> tuple[tuple[ObservedFillReport, ...], tuple[DuplicateReportGroup, ...]]:
        report_ids: dict[tuple[str, str, str], list[ObservedFillReport]] = defaultdict(list)
        for report in reports:
            report_ids[(report.source_system, report.account_id, report.report_id)].append(report)
        duplicates: list[DuplicateReportGroup] = []
        excluded_hashes: set[str] = set()
        for envelope_key in sorted(report_ids):
            values = report_ids[envelope_key]
            if len({item.report_hash for item in values}) <= 1:
                continue
            duplicates.append(
                DuplicateReportGroup.build(
                    report_kind=ReconciliationReportKind.FILL,
                    key_kind=DuplicateKeyKind.REPORT_ID,
                    source_system=envelope_key[0],
                    dedup_key=envelope_key,
                    report_ids=tuple(item.report_id for item in values),
                    report_hashes=tuple(item.report_hash for item in values),
                    content_hashes=tuple(item.report_hash for item in values),
                )
            )
            excluded_hashes.update(item.report_hash for item in values)
        grouped: dict[tuple[str, str, str], list[ObservedFillReport]] = defaultdict(list)
        for report in reports:
            if report.report_hash in excluded_hashes:
                continue
            grouped[report.dedup_key].append(report)
        canonical: list[ObservedFillReport] = []
        for semantic_key in sorted(grouped):
            values = grouped[semantic_key]
            if len(values) == 1:
                canonical.append(values[0])
                continue
            group = DuplicateReportGroup.build(
                report_kind=ReconciliationReportKind.FILL,
                key_kind=DuplicateKeyKind.SOURCE_IDENTITY,
                source_system=semantic_key[0],
                dedup_key=semantic_key,
                report_ids=tuple(item.report_id for item in values),
                report_hashes=tuple(item.report_hash for item in values),
                content_hashes=tuple(item.source_payload_hash for item in values),
            )
            duplicates.append(group)
            if group.canonical_report_hash is not None:
                canonical.append(min(values, key=lambda item: item.report_hash))
        return (
            tuple(sorted(canonical, key=lambda item: (*item.dedup_key, item.report_hash))),
            tuple(sorted(duplicates, key=lambda item: item.group_id)),
        )

    def _input_findings(
        self,
        *,
        request: ReconciliationRequest,
        order_reports: tuple[ObservedOrderReport, ...],
        fill_reports: tuple[ObservedFillReport, ...],
    ) -> tuple[ReconciliationFinding, ...]:
        draft = request.draft
        receipt = request.receipt
        account = receipt.account_after
        observed = request.observed
        snapshot = observed.account_snapshot
        findings: list[ReconciliationFinding] = []

        def critical(
            code: ReconciliationDifferenceCode,
            entity_id: str,
            message: str,
            *,
            expected: object | None = None,
            actual: object | None = None,
            evidence: tuple[str, ...] = (),
        ) -> None:
            findings.append(
                self._finding(
                    request=request,
                    domain=ReconciliationDomain.INPUT,
                    code=code,
                    status=ReconciliationCheckStatus.CONFLICT,
                    severity=ReconciliationSeverity.CRITICAL,
                    entity_id=entity_id,
                    message=message,
                    expected=expected,
                    actual=actual,
                    evidence=evidence,
                )
            )

        if draft.batch_hash != receipt.batch_hash:
            critical(
                ReconciliationDifferenceCode.DRAFT_RECEIPT_BATCH_MISMATCH,
                "expected:batch",
                "draft and execution receipt identify different batches",
                expected=draft.batch_hash,
                actual=receipt.batch_hash,
                evidence=(draft.batch_hash, receipt.receipt_hash),
            )
        mismatched_decisions = tuple(
            sorted(
                order.order_id for order in receipt.orders if order.decision_id != draft.decision_id
            )
        )
        if mismatched_decisions:
            critical(
                ReconciliationDifferenceCode.DECISION_MISMATCH,
                "expected:decision",
                "receipt orders do not all bind the draft decision",
                expected=draft.decision_id,
                actual=",".join(mismatched_decisions),
                evidence=(draft.batch_hash, receipt.receipt_hash),
            )
        before = receipt.account_before
        if (
            before.source_snapshot_id != draft.account_snapshot_id
            or before.source_snapshot_hash != draft.account_snapshot_hash
            or before.source_snapshot_as_of != draft.account_snapshot_as_of
        ):
            critical(
                ReconciliationDifferenceCode.SOURCE_SNAPSHOT_MISMATCH,
                "expected:source-snapshot",
                "paper execution does not start from the draft account snapshot",
                expected=draft.account_snapshot_hash,
                actual=before.source_snapshot_hash,
                evidence=(draft.batch_hash, receipt.receipt_hash),
            )
        if draft.runtime_mode is not account.runtime_mode:
            critical(
                ReconciliationDifferenceCode.RUNTIME_MODE_MISMATCH,
                "expected:runtime-mode",
                "draft and expected account runtime modes differ",
                expected=draft.runtime_mode,
                actual=account.runtime_mode,
                evidence=(draft.batch_hash, receipt.account_after_hash),
            )
        if draft.data_version != account.data_version:
            critical(
                ReconciliationDifferenceCode.DATA_VERSION_MISMATCH,
                "expected:data-version",
                "draft and expected account data versions differ",
                expected=draft.data_version,
                actual=account.data_version,
                evidence=(draft.batch_hash, receipt.account_after_hash),
            )
        if draft.currency != account.currency:
            critical(
                ReconciliationDifferenceCode.CURRENCY_MISMATCH,
                "expected:currency",
                "draft and expected account currencies differ",
                expected=draft.currency,
                actual=account.currency,
                evidence=(draft.batch_hash, receipt.account_after_hash),
            )

        expected_account_id = account.account_id
        expected_orders_by_id = {item.order_id: item for item in receipt.orders}
        expected_attempts_by_order = {item.order_id: item for item in receipt.attempts}
        maximum_report_lag = timedelta(seconds=request.policy.max_observation_lag_seconds)
        identity_values = {
            "evidence": observed.account_id,
            "snapshot": snapshot.account_id,
        }
        for label, actual in identity_values.items():
            if actual != expected_account_id:
                critical(
                    ReconciliationDifferenceCode.ACCOUNT_IDENTITY_MISMATCH,
                    f"observed:{label}:account",
                    f"observed {label} account does not match the expected account",
                    expected=expected_account_id,
                    actual=actual,
                    evidence=(observed.evidence_hash, snapshot.content_hash),
                )
        if snapshot.runtime_mode is not account.runtime_mode:
            critical(
                ReconciliationDifferenceCode.RUNTIME_MODE_MISMATCH,
                "observed:snapshot:runtime-mode",
                "observed snapshot runtime mode does not match the expected account",
                expected=account.runtime_mode,
                actual=snapshot.runtime_mode,
                evidence=(receipt.account_after_hash, snapshot.content_hash),
            )
        if snapshot.data_version != account.data_version:
            critical(
                ReconciliationDifferenceCode.DATA_VERSION_MISMATCH,
                "observed:snapshot:data-version",
                "observed snapshot data version does not match the expected account",
                expected=account.data_version,
                actual=snapshot.data_version,
                evidence=(receipt.account_after_hash, snapshot.content_hash),
            )
        if snapshot.currency != account.currency:
            critical(
                ReconciliationDifferenceCode.CURRENCY_MISMATCH,
                "observed:snapshot:currency",
                "observed snapshot currency does not match the expected account",
                expected=account.currency,
                actual=snapshot.currency,
                evidence=(receipt.account_after_hash, snapshot.content_hash),
            )
        if snapshot.as_of != receipt.processed_at or observed.observed_at != snapshot.as_of:
            critical(
                ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                "observed:snapshot:as-of",
                "first-version reconciliation requires the exact execution boundary",
                expected=receipt.processed_at,
                actual=snapshot.as_of,
                evidence=(receipt.receipt_hash, snapshot.content_hash),
            )
        if snapshot.previous_snapshot_hash != account.source_snapshot_hash:
            critical(
                ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                "observed:snapshot:previous",
                "observed snapshot does not extend the execution source snapshot",
                expected=account.source_snapshot_hash,
                actual=snapshot.previous_snapshot_hash,
                evidence=(receipt.account_after_hash, snapshot.content_hash),
            )
        if snapshot.source_event_log_hash != receipt.event_log_hash:
            critical(
                ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
                "observed:snapshot:event-log",
                "observed snapshot does not bind the execution event-log boundary",
                expected=receipt.event_log_hash,
                actual=snapshot.source_event_log_hash,
                evidence=(receipt.receipt_hash, snapshot.content_hash),
            )
        if observed.available_at > request.reconciled_at:
            critical(
                ReconciliationDifferenceCode.FUTURE_EVIDENCE,
                "observed:evidence:available-at",
                "evidence was not available at reconciliation time",
                expected=f"<= {request.reconciled_at.isoformat()}",
                actual=observed.available_at,
                evidence=(observed.evidence_hash,),
            )
        else:
            maximum_lag = timedelta(seconds=request.policy.max_observation_lag_seconds)
            if request.reconciled_at - observed.available_at > maximum_lag:
                critical(
                    ReconciliationDifferenceCode.STALE_EVIDENCE,
                    "observed:evidence:lag",
                    "evidence exceeds the configured observation lag",
                    expected=f"<= {request.policy.max_observation_lag_seconds}s",
                    actual=request.reconciled_at - observed.available_at,
                    evidence=(observed.evidence_hash,),
                )
        unique_order_reports = {item.report_hash: item for item in order_reports}
        for order_report in (unique_order_reports[key] for key in sorted(unique_order_reports)):
            report_entity = _report_entity_prefix(
                ReconciliationReportKind.ORDER,
                order_report.report_id,
                order_report.report_hash,
            )
            if order_report.source_system != observed.source_system:
                critical(
                    ReconciliationDifferenceCode.SOURCE_SYSTEM_MISMATCH,
                    f"{report_entity}:source",
                    "order report source system differs from its evidence envelope",
                    expected=observed.source_system,
                    actual=order_report.source_system,
                    evidence=(order_report.report_hash, observed.evidence_hash),
                )
            if order_report.account_id != expected_account_id:
                critical(
                    ReconciliationDifferenceCode.ACCOUNT_IDENTITY_MISMATCH,
                    f"{report_entity}:account",
                    "order report account does not match the expected account",
                    expected=expected_account_id,
                    actual=order_report.account_id,
                    evidence=(order_report.report_hash,),
                )
            if order_report.available_at > request.reconciled_at:
                critical(
                    ReconciliationDifferenceCode.FUTURE_EVIDENCE,
                    f"{report_entity}:available-at",
                    "order report was unavailable at reconciliation time",
                    expected=f"<= {request.reconciled_at.isoformat()}",
                    actual=order_report.available_at,
                    evidence=(order_report.report_hash,),
                )
            if order_report.available_at > observed.available_at:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:evidence-envelope",
                    "order report became available after its evidence envelope",
                    expected=f"<= {observed.available_at.isoformat()}",
                    actual=order_report.available_at,
                    evidence=(order_report.report_hash, observed.evidence_hash),
                )
            if order_report.event_time > snapshot.as_of:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:event-time",
                    "order report event is after the observed account boundary",
                    expected=f"<= {snapshot.as_of.isoformat()}",
                    actual=order_report.event_time,
                    evidence=(order_report.report_hash, snapshot.content_hash),
                )
            expected_order = (
                expected_orders_by_id.get(order_report.client_order_id)
                if order_report.client_order_id is not None
                else None
            )
            expected_order_attempt = (
                expected_attempts_by_order.get(order_report.client_order_id)
                if order_report.client_order_id is not None
                else None
            )
            order_time_floor = (
                max(expected_order.created_at, expected_order_attempt.attempted_at)
                if expected_order is not None and expected_order_attempt is not None
                else receipt.request_submitted_at
            )
            if order_report.event_time < order_time_floor:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:event-lower-bound",
                    "order report event predates its expected order boundary",
                    expected=f">= {order_time_floor.isoformat()}",
                    actual=order_report.event_time,
                    evidence=(order_report.report_hash, receipt.receipt_hash),
                )
            if (
                order_report.available_at <= request.reconciled_at
                and request.reconciled_at - order_report.available_at > maximum_report_lag
            ):
                critical(
                    ReconciliationDifferenceCode.STALE_EVIDENCE,
                    f"{report_entity}:lag",
                    "order report exceeds the configured observation lag",
                    expected=f"<= {request.policy.max_observation_lag_seconds}s",
                    actual=request.reconciled_at - order_report.available_at,
                    evidence=(order_report.report_hash,),
                )
        unique_fill_reports = {item.report_hash: item for item in fill_reports}
        for fill_report in (unique_fill_reports[key] for key in sorted(unique_fill_reports)):
            report_entity = _report_entity_prefix(
                ReconciliationReportKind.FILL,
                fill_report.report_id,
                fill_report.report_hash,
            )
            if fill_report.source_system != observed.source_system:
                critical(
                    ReconciliationDifferenceCode.SOURCE_SYSTEM_MISMATCH,
                    f"{report_entity}:source",
                    "fill report source system differs from its evidence envelope",
                    expected=observed.source_system,
                    actual=fill_report.source_system,
                    evidence=(fill_report.report_hash, observed.evidence_hash),
                )
            if fill_report.account_id != expected_account_id:
                critical(
                    ReconciliationDifferenceCode.ACCOUNT_IDENTITY_MISMATCH,
                    f"{report_entity}:account",
                    "fill report account does not match the expected account",
                    expected=expected_account_id,
                    actual=fill_report.account_id,
                    evidence=(fill_report.report_hash,),
                )
            if fill_report.currency != account.currency:
                critical(
                    ReconciliationDifferenceCode.CURRENCY_MISMATCH,
                    f"{report_entity}:currency",
                    "fill report currency does not match the expected account",
                    expected=account.currency,
                    actual=fill_report.currency,
                    evidence=(fill_report.report_hash,),
                )
            if fill_report.available_at > request.reconciled_at:
                critical(
                    ReconciliationDifferenceCode.FUTURE_EVIDENCE,
                    f"{report_entity}:available-at",
                    "fill report was unavailable at reconciliation time",
                    expected=f"<= {request.reconciled_at.isoformat()}",
                    actual=fill_report.available_at,
                    evidence=(fill_report.report_hash,),
                )
            if fill_report.available_at > observed.available_at:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:evidence-envelope",
                    "fill report became available after its evidence envelope",
                    expected=f"<= {observed.available_at.isoformat()}",
                    actual=fill_report.available_at,
                    evidence=(fill_report.report_hash, observed.evidence_hash),
                )
            if fill_report.filled_at > snapshot.as_of:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:filled-at",
                    "fill report event is after the observed account boundary",
                    expected=f"<= {snapshot.as_of.isoformat()}",
                    actual=fill_report.filled_at,
                    evidence=(fill_report.report_hash, snapshot.content_hash),
                )
            expected_attempt = (
                expected_attempts_by_order.get(fill_report.client_order_id)
                if fill_report.client_order_id is not None
                else None
            )
            fill_time_floor = (
                expected_attempt.attempted_at
                if expected_attempt is not None
                else receipt.request_submitted_at
            )
            if fill_report.filled_at < fill_time_floor:
                critical(
                    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
                    f"{report_entity}:event-lower-bound",
                    "fill report event predates its expected match-attempt boundary",
                    expected=f">= {fill_time_floor.isoformat()}",
                    actual=fill_report.filled_at,
                    evidence=(fill_report.report_hash, receipt.receipt_hash),
                )
            if (
                fill_report.available_at <= request.reconciled_at
                and request.reconciled_at - fill_report.available_at > maximum_report_lag
            ):
                critical(
                    ReconciliationDifferenceCode.STALE_EVIDENCE,
                    f"{report_entity}:lag",
                    "fill report exceeds the configured observation lag",
                    expected=f"<= {request.policy.max_observation_lag_seconds}s",
                    actual=request.reconciled_at - fill_report.available_at,
                    evidence=(fill_report.report_hash,),
                )
        return tuple(sorted(findings, key=lambda item: item.finding_hash))

    def _latest_order_reports(
        self,
        *,
        request: ReconciliationRequest,
        reports: tuple[ObservedOrderReport, ...],
    ) -> tuple[
        dict[str, ObservedOrderReport],
        dict[str, tuple[str, ...]],
        tuple[ReconciliationFinding, ...],
    ]:
        findings: list[ReconciliationFinding] = []
        linked: dict[str, list[ObservedOrderReport]] = defaultdict(list)
        histories: dict[tuple[str, str, str], list[ObservedOrderReport]] = defaultdict(list)
        for report in reports:
            report_entity = _report_entity_prefix(
                ReconciliationReportKind.ORDER,
                report.report_id,
                report.report_hash,
            )
            histories[(report.source_system, report.account_id, report.source_order_id)].append(
                report
            )
            if report.client_order_id is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.REPORT,
                        code=ReconciliationDifferenceCode.UNLINKED_ORDER_REPORT,
                        status=ReconciliationCheckStatus.CONFLICT,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"{report_entity}:unlinked",
                        message="order report has no explicit client order identity",
                        actual=report.source_order_id,
                        evidence=(report.report_hash,),
                    )
                )
                continue
            linked[report.client_order_id].append(report)
        for history_key, values in sorted(histories.items()):
            ordered = sorted(values, key=lambda item: item.revision)
            regression = False
            for previous, current in pairwise(ordered):
                immutable_identity_changed = (
                    previous.client_order_id != current.client_order_id
                    or previous.instrument_id != current.instrument_id
                    or previous.instrument_type is not current.instrument_type
                    or previous.side is not current.side
                    or previous.requested_quantity != current.requested_quantity
                )
                state_regressed = (
                    current.event_time < previous.event_time
                    or current.cumulative_filled_quantity < previous.cumulative_filled_quantity
                    or current.remaining_quantity > previous.remaining_quantity
                    or previous.status in _TERMINAL_ORDER_STATUSES
                )
                if immutable_identity_changed or state_regressed:
                    regression = True
                    break
            if regression:
                history_identity = stable_hash(
                    {
                        "source_system": history_key[0],
                        "account_id": history_key[1],
                        "source_order_id": history_key[2],
                    }
                )
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.REPORT,
                        code=ReconciliationDifferenceCode.ORDER_REPORT_REGRESSION,
                        status=ReconciliationCheckStatus.CONFLICT,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"source-order-history:{history_identity}",
                        message="order report revisions regress or change immutable identity",
                        evidence=tuple(item.report_hash for item in ordered),
                    )
                )
        latest: dict[str, ObservedOrderReport] = {}
        report_hashes: dict[str, tuple[str, ...]] = {}
        for client_order_id, values in sorted(linked.items()):
            source_ids = {item.source_order_id for item in values}
            if len(source_ids) > 1:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.REPORT,
                        code=ReconciliationDifferenceCode.ORDER_IDENTITY_MISMATCH,
                        status=ReconciliationCheckStatus.CONFLICT,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"client-order:{client_order_id}:sources",
                        message="multiple source order identities claim one client order",
                        expected=client_order_id,
                        actual=",".join(sorted(source_ids)),
                        order_id=client_order_id,
                        evidence=tuple(item.report_hash for item in values),
                    )
                )
            latest[client_order_id] = max(
                values,
                key=lambda item: (
                    item.revision,
                    item.event_time,
                    item.source_payload_hash,
                    item.report_hash,
                ),
            )
            report_hashes[client_order_id] = tuple(sorted({item.report_hash for item in values}))
        return latest, report_hashes, tuple(sorted(findings, key=lambda item: item.finding_hash))

    def _linked_fill_reports(
        self,
        *,
        request: ReconciliationRequest,
        reports: tuple[ObservedFillReport, ...],
    ) -> tuple[dict[str, tuple[ObservedFillReport, ...]], tuple[ReconciliationFinding, ...]]:
        findings: list[ReconciliationFinding] = []
        linked: dict[str, list[ObservedFillReport]] = defaultdict(list)
        for report in reports:
            report_entity = _report_entity_prefix(
                ReconciliationReportKind.FILL,
                report.report_id,
                report.report_hash,
            )
            if report.client_order_id is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.REPORT,
                        code=ReconciliationDifferenceCode.UNLINKED_FILL_REPORT,
                        status=ReconciliationCheckStatus.CONFLICT,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"{report_entity}:unlinked",
                        message="fill report has no explicit client order identity",
                        actual=report.source_fill_id,
                        fill_id=report.client_fill_id,
                        evidence=(report.report_hash,),
                    )
                )
                continue
            linked[report.client_order_id].append(report)
        return (
            {
                order_id: tuple(
                    sorted(values, key=lambda item: (item.source_fill_id, item.report_hash))
                )
                for order_id, values in sorted(linked.items())
            },
            tuple(sorted(findings, key=lambda item: item.finding_hash)),
        )

    def _order_checks(
        self,
        *,
        request: ReconciliationRequest,
        latest_reports: dict[str, ObservedOrderReport],
        report_hashes: dict[str, tuple[str, ...]],
    ) -> tuple[OrderReconciliation, ...]:
        draft_lines = {item.line_hash: item for item in request.draft.lines}
        expected_orders = {item.order_id: item for item in request.receipt.orders}
        attempts_by_order = {item.order_id: item for item in request.receipt.attempts}
        receipt_lines: dict[str, list[PaperOrder]] = defaultdict(list)
        for order in request.receipt.orders:
            receipt_lines[order.line_hash].append(order)
        checks: list[OrderReconciliation] = []
        represented_lines: set[str] = set()
        for order_id in sorted(expected_orders):
            order = expected_orders[order_id]
            line = draft_lines.get(order.line_hash)
            if line is not None:
                represented_lines.add(line.line_hash)
            observed = latest_reports.get(order_id)
            expected_attempt = attempts_by_order[order_id]
            expected_outcome_code = (
                expected_attempt.no_fill_reason.value
                if expected_attempt.no_fill_reason is not None
                else None
            )
            findings: list[ReconciliationFinding] = []
            identity_mismatches: list[str] = []
            if line is None:
                identity_mismatches.append("line_hash")
            else:
                comparisons = {
                    "instrument_id": line.instrument_id == order.instrument_id,
                    "instrument_type": line.instrument_type is order.instrument_type,
                    "side": line.side is order.side,
                    "quantity": line.quantity == order.quantity,
                    "full_liquidation": line.is_full_liquidation == order.is_full_liquidation,
                    "expires_at": request.draft.expires_at == order.expires_at,
                    "market_rule": (
                        line.market_rule_version == order.market_rule_version
                        and line.market_rule_hash == order.market_rule_hash
                    ),
                    "fee_rule": (
                        line.fee_rule_version == order.fee_rule_version
                        and line.fee_rule_hash == order.fee_rule_hash
                    ),
                    "slippage_model": (
                        line.slippage_model_version == order.slippage_model_version
                        and line.slippage_model_hash == order.slippage_model_hash
                    ),
                }
                identity_mismatches.extend(
                    field_name for field_name, matches in comparisons.items() if not matches
                )
            if order.batch_hash != request.draft.batch_hash:
                identity_mismatches.append("batch_hash")
            if order.decision_id != request.draft.decision_id:
                identity_mismatches.append("decision_id")
            if len(receipt_lines[order.line_hash]) != 1:
                identity_mismatches.append("line_cardinality")
            if identity_mismatches:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.ORDER,
                        code=ReconciliationDifferenceCode.ORDER_IDENTITY_MISMATCH,
                        status=ReconciliationCheckStatus.CONFLICT,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"order:{order_id}:expected-identity",
                        message=(
                            "draft-to-order identity mismatch: "
                            + ", ".join(sorted(set(identity_mismatches)))
                        ),
                        order_id=order_id,
                        line_hash=order.line_hash,
                        instrument_id=order.instrument_id,
                        evidence=(request.draft.batch_hash, order.order_hash),
                    )
                )
            if observed is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.ORDER,
                        code=ReconciliationDifferenceCode.ORDER_MISSING,
                        status=ReconciliationCheckStatus.MISSING,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"order:{order_id}:observed",
                        message="expected order has no unambiguous observed report",
                        expected=order.order_id,
                        order_id=order_id,
                        line_hash=order.line_hash,
                        instrument_id=order.instrument_id,
                        evidence=(order.order_hash, request.observed.evidence_hash),
                    )
                )
            else:
                observed_identity_mismatches = []
                if observed.instrument_id != order.instrument_id:
                    observed_identity_mismatches.append("instrument_id")
                if observed.instrument_type is not order.instrument_type:
                    observed_identity_mismatches.append("instrument_type")
                if observed.side is not order.side:
                    observed_identity_mismatches.append("side")
                if observed_identity_mismatches:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.ORDER,
                            code=ReconciliationDifferenceCode.ORDER_IDENTITY_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"order:{order_id}:observed-identity",
                            message=(
                                "observed order identity mismatch: "
                                + ", ".join(observed_identity_mismatches)
                            ),
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=(order.order_hash, observed.report_hash),
                        )
                    )
                quantity_mismatches = []
                if observed.requested_quantity != order.quantity:
                    quantity_mismatches.append("requested")
                if observed.cumulative_filled_quantity != order.filled_quantity:
                    quantity_mismatches.append("filled")
                if observed.remaining_quantity != order.remaining_quantity:
                    quantity_mismatches.append("remaining")
                if quantity_mismatches:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.ORDER,
                            code=ReconciliationDifferenceCode.ORDER_QUANTITY_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"order:{order_id}:quantities",
                            message=(
                                "observed order quantity mismatch: "
                                + ", ".join(quantity_mismatches)
                            ),
                            expected=(
                                f"{order.quantity}/{order.filled_quantity}/"
                                f"{order.remaining_quantity}"
                            ),
                            actual=(
                                f"{observed.requested_quantity}/"
                                f"{observed.cumulative_filled_quantity}/"
                                f"{observed.remaining_quantity}"
                            ),
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=(order.order_hash, observed.report_hash),
                        )
                    )
                if observed.status is not order.status:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.ORDER,
                            code=ReconciliationDifferenceCode.ORDER_STATUS_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"order:{order_id}:status",
                            message="observed order status differs from the expected lifecycle",
                            expected=order.status,
                            actual=observed.status,
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=(order.order_hash, observed.report_hash),
                        )
                    )
                if observed.outcome_code != expected_outcome_code:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.ORDER,
                            code=ReconciliationDifferenceCode.ORDER_REASON_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"order:{order_id}:outcome-code",
                            message="observed order outcome reason differs from expected",
                            expected=expected_outcome_code,
                            actual=observed.outcome_code,
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=(order.order_hash, observed.report_hash),
                        )
                    )
                if not any(item.severity is ReconciliationSeverity.CRITICAL for item in findings):
                    variance = self._expected_order_variance(request, order, observed)
                    if variance is not None:
                        findings.append(variance)
            checks.append(
                OrderReconciliation.build(
                    check_id=f"order:{order_id}",
                    input_hash=request.input_hash,
                    decision_id=request.draft.decision_id,
                    batch_hash=request.draft.batch_hash,
                    line_hash=order.line_hash,
                    order_id=order_id,
                    source_order_id=observed.source_order_id if observed else None,
                    instrument_id=order.instrument_id,
                    planned_side=line.side if line else order.side,
                    planned_quantity=line.quantity if line else order.quantity,
                    expected_status=order.status,
                    expected_outcome_code=expected_outcome_code,
                    expected_filled_quantity=order.filled_quantity,
                    expected_remaining_quantity=order.remaining_quantity,
                    observed_status=observed.status if observed else None,
                    observed_outcome_code=observed.outcome_code if observed else None,
                    observed_requested_quantity=(observed.requested_quantity if observed else None),
                    observed_filled_quantity=(
                        observed.cumulative_filled_quantity if observed else None
                    ),
                    observed_remaining_quantity=(observed.remaining_quantity if observed else None),
                    report_hashes=report_hashes.get(order_id, ()),
                    findings=tuple(findings),
                )
            )
        for line_hash in sorted(set(draft_lines) - represented_lines):
            line = draft_lines[line_hash]
            finding = self._finding(
                request=request,
                domain=ReconciliationDomain.ORDER,
                code=ReconciliationDifferenceCode.ORDER_MISSING,
                status=ReconciliationCheckStatus.MISSING,
                severity=ReconciliationSeverity.CRITICAL,
                entity_id=f"draft-line:{line_hash}:receipt-order",
                message="draft line has no expected receipt order",
                expected=line_hash,
                line_hash=line_hash,
                instrument_id=line.instrument_id,
                evidence=(request.draft.batch_hash, request.receipt.receipt_hash),
            )
            checks.append(
                OrderReconciliation.build(
                    check_id=f"draft-line:{line_hash}",
                    input_hash=request.input_hash,
                    decision_id=request.draft.decision_id,
                    batch_hash=request.draft.batch_hash,
                    line_hash=line_hash,
                    order_id=None,
                    source_order_id=None,
                    instrument_id=line.instrument_id,
                    planned_side=line.side,
                    planned_quantity=line.quantity,
                    expected_status=None,
                    expected_outcome_code=None,
                    expected_filled_quantity=None,
                    expected_remaining_quantity=None,
                    observed_status=None,
                    observed_outcome_code=None,
                    observed_requested_quantity=None,
                    observed_filled_quantity=None,
                    observed_remaining_quantity=None,
                    report_hashes=(),
                    findings=(finding,),
                )
            )
        for order_id in sorted(set(latest_reports) - set(expected_orders)):
            observed = latest_reports[order_id]
            finding = self._finding(
                request=request,
                domain=ReconciliationDomain.ORDER,
                code=ReconciliationDifferenceCode.ORDER_UNEXPECTED,
                status=ReconciliationCheckStatus.UNEXPECTED,
                severity=ReconciliationSeverity.CRITICAL,
                entity_id=f"order:{order_id}:unexpected",
                message="observed order has no expected receipt order",
                actual=order_id,
                order_id=order_id,
                instrument_id=observed.instrument_id,
                evidence=report_hashes[order_id],
            )
            checks.append(
                OrderReconciliation.build(
                    check_id=f"unexpected-order:{order_id}",
                    input_hash=request.input_hash,
                    decision_id=request.draft.decision_id,
                    batch_hash=request.draft.batch_hash,
                    line_hash=None,
                    order_id=order_id,
                    source_order_id=observed.source_order_id,
                    instrument_id=observed.instrument_id,
                    planned_side=None,
                    planned_quantity=None,
                    expected_status=None,
                    expected_outcome_code=None,
                    expected_filled_quantity=None,
                    expected_remaining_quantity=None,
                    observed_status=observed.status,
                    observed_outcome_code=observed.outcome_code,
                    observed_requested_quantity=observed.requested_quantity,
                    observed_filled_quantity=observed.cumulative_filled_quantity,
                    observed_remaining_quantity=observed.remaining_quantity,
                    report_hashes=report_hashes[order_id],
                    findings=(finding,),
                )
            )
        return tuple(sorted(checks, key=lambda item: item.check_id))

    def _expected_order_variance(
        self,
        request: ReconciliationRequest,
        order: PaperOrder,
        observed: ObservedOrderReport,
    ) -> ReconciliationFinding | None:
        if order.status is PaperOrderStatus.REJECTED:
            code = ReconciliationDifferenceCode.ORDER_REJECTED
            message = "expected rejected order was independently confirmed"
        elif order.status is PaperOrderStatus.PARTIALLY_FILLED:
            code = ReconciliationDifferenceCode.ORDER_PARTIAL
            message = "expected partial fill was independently confirmed"
        elif order.status in {
            PaperOrderStatus.ACCEPTED,
            PaperOrderStatus.CANCELED,
            PaperOrderStatus.EXPIRED,
        }:
            code = ReconciliationDifferenceCode.ORDER_NO_FILL
            message = "expected no-fill lifecycle was independently confirmed"
        else:
            return None
        return self._finding(
            request=request,
            domain=ReconciliationDomain.ORDER,
            code=code,
            status=ReconciliationCheckStatus.EXPECTED_VARIANCE,
            severity=ReconciliationSeverity.WARNING,
            entity_id=f"order:{order.order_id}:expected-variance",
            message=message,
            expected=order.status,
            actual=observed.status,
            order_id=order.order_id,
            line_hash=order.line_hash,
            instrument_id=order.instrument_id,
            evidence=(order.order_hash, observed.report_hash),
        )

    def _fill_checks(
        self,
        *,
        request: ReconciliationRequest,
        observed_by_order: dict[str, tuple[ObservedFillReport, ...]],
        observed_orders: dict[str, ObservedOrderReport],
    ) -> tuple[FillReconciliation, ...]:
        orders = {item.order_id: item for item in request.receipt.orders}
        expected_by_order: dict[str, list[PaperFill]] = defaultdict(list)
        for fill in request.receipt.fills:
            expected_by_order[fill.order_id].append(fill)
        checks: list[FillReconciliation] = []
        for order_id in sorted(set(orders) | set(observed_by_order)):
            order = orders.get(order_id)
            expected = tuple(expected_by_order.get(order_id, ()))
            observed = observed_by_order.get(order_id, ())
            expected_quantity = sum((item.quantity for item in expected), _ZERO)
            observed_quantity = sum((item.quantity for item in observed), _ZERO)
            expected_gross = sum((item.gross_amount for item in expected), _ZERO)
            observed_gross = sum((item.gross_amount for item in observed), _ZERO)
            expected_fee = sum((item.total_fee for item in expected), _ZERO)
            observed_fee = sum((item.total_fee for item in observed), _ZERO)
            expected_commission = sum((item.commission for item in expected), _ZERO)
            observed_commission = sum((item.commission for item in observed), _ZERO)
            expected_stamp_duty = sum((item.stamp_duty for item in expected), _ZERO)
            observed_stamp_duty = sum((item.stamp_duty for item in observed), _ZERO)
            expected_transfer_fee = sum((item.transfer_fee for item in expected), _ZERO)
            observed_transfer_fee = sum((item.transfer_fee for item in observed), _ZERO)
            expected_other_fee = sum((item.other_fee for item in expected), _ZERO)
            observed_other_fee = sum((item.other_fee for item in observed), _ZERO)
            expected_cash = sum((item.cash_change for item in expected), _ZERO)
            observed_cash = sum((item.cash_change for item in observed), _ZERO)
            expected_vwap = self._vwap(expected_quantity, expected_gross)
            observed_vwap = self._vwap(observed_quantity, observed_gross)
            findings: list[ReconciliationFinding] = []
            instrument_id = (
                order.instrument_id
                if order is not None
                else (observed[0].instrument_id if observed else "unknown")
            )
            line_hash = order.line_hash if order is not None else None
            evidence_hashes = tuple(
                sorted(
                    {
                        *(item.fill_hash for item in expected),
                        *(item.report_hash for item in observed),
                    }
                )
            )
            if order is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.FILL,
                        code=ReconciliationDifferenceCode.FILL_UNEXPECTED,
                        status=ReconciliationCheckStatus.UNEXPECTED,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"fill-order:{order_id}:unexpected",
                        message="observed fills refer to an unknown client order",
                        actual=order_id,
                        order_id=order_id,
                        instrument_id=instrument_id,
                        evidence=evidence_hashes,
                    )
                )
            else:
                identity_fields: set[str] = set()
                observed_order = observed_orders.get(order_id)
                for report in observed:
                    if report.instrument_id != order.instrument_id:
                        identity_fields.add("instrument_id")
                    if report.instrument_type is not order.instrument_type:
                        identity_fields.add("instrument_type")
                    if report.side is not order.side:
                        identity_fields.add("side")
                    if report.currency != request.receipt.account_after.currency:
                        identity_fields.add("currency")
                    if (
                        observed_order is None
                        or report.source_order_id != observed_order.source_order_id
                    ):
                        identity_fields.add("source_order_id")
                if identity_fields:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.FILL,
                            code=ReconciliationDifferenceCode.FILL_IDENTITY_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"fill-order:{order_id}:identity",
                            message=(
                                "observed fill identity mismatch: "
                                + ", ".join(sorted(identity_fields))
                            ),
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=evidence_hashes,
                        )
                    )
                expected_fill_ids = {item.fill_id for item in expected}
                identified_reports = tuple(
                    item for item in observed if item.client_fill_id is not None
                )
                observed_client_fill_ids = tuple(
                    cast(str, item.client_fill_id) for item in identified_reports
                )
                fill_mapping_conflict = bool(identified_reports) and (
                    len(identified_reports) != len(observed)
                    or len(set(observed_client_fill_ids)) != len(observed_client_fill_ids)
                    or set(observed_client_fill_ids) != expected_fill_ids
                )
                if fill_mapping_conflict:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.FILL,
                            code=ReconciliationDifferenceCode.FILL_IDENTITY_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"fill-order:{order_id}:fill-identities",
                            message=(
                                "observed client fill identities are not a one-to-one mapping"
                            ),
                            expected=",".join(sorted(expected_fill_ids)),
                            actual=",".join(sorted(observed_client_fill_ids)),
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=evidence_hashes,
                        )
                    )
                elif identified_reports:
                    expected_by_fill_id = {item.fill_id: item for item in expected}
                    observed_by_fill_id = {item.client_fill_id: item for item in identified_reports}
                    for fill_id in sorted(expected_by_fill_id):
                        observed_fill = observed_by_fill_id[fill_id]
                        self._individual_fill_comparisons(
                            findings=findings,
                            request=request,
                            order=order,
                            expected=expected_by_fill_id[fill_id],
                            observed=observed_fill,
                            evidence=(
                                expected_by_fill_id[fill_id].fill_hash,
                                observed_fill.report_hash,
                            ),
                        )
                if expected_quantity > 0 and observed_quantity == 0:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.FILL,
                            code=ReconciliationDifferenceCode.FILL_MISSING,
                            status=ReconciliationCheckStatus.MISSING,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"fill-order:{order_id}:missing",
                            message="expected fills have no observed fill reports",
                            expected=expected_quantity,
                            actual=observed_quantity,
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=evidence_hashes,
                        )
                    )
                elif expected_quantity == 0 and observed_quantity > 0:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.FILL,
                            code=ReconciliationDifferenceCode.FILL_UNEXPECTED,
                            status=ReconciliationCheckStatus.UNEXPECTED,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"fill-order:{order_id}:unexpected",
                            message="observed fills exist for an expected no-fill order",
                            expected=expected_quantity,
                            actual=observed_quantity,
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=evidence_hashes,
                        )
                    )
                elif expected_quantity != observed_quantity:
                    code = (
                        ReconciliationDifferenceCode.FILL_OVERFILL
                        if observed_quantity > order.quantity
                        else ReconciliationDifferenceCode.FILL_QUANTITY_MISMATCH
                    )
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.FILL,
                            code=code,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"fill-order:{order_id}:quantity",
                            message="observed aggregate fill quantity differs from expected",
                            expected=expected_quantity,
                            actual=observed_quantity,
                            delta=observed_quantity - expected_quantity,
                            order_id=order_id,
                            line_hash=order.line_hash,
                            instrument_id=order.instrument_id,
                            evidence=evidence_hashes,
                        )
                    )
                self._economic_difference(
                    findings=findings,
                    request=request,
                    code=ReconciliationDifferenceCode.FILL_GROSS_MISMATCH,
                    entity_id=f"fill-order:{order_id}:gross",
                    message="observed aggregate gross amount differs from expected",
                    expected=expected_gross,
                    observed=observed_gross,
                    tolerance=request.policy.cash_tolerance,
                    order=order,
                    evidence=evidence_hashes,
                )
                fee_components = (
                    ("commission", expected_commission, observed_commission),
                    ("stamp-duty", expected_stamp_duty, observed_stamp_duty),
                    ("transfer-fee", expected_transfer_fee, observed_transfer_fee),
                    ("other-fee", expected_other_fee, observed_other_fee),
                )
                for component, expected_component, observed_component in fee_components:
                    self._economic_difference(
                        findings=findings,
                        request=request,
                        code=ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                        entity_id=f"fill-order:{order_id}:fee:{component}",
                        message=f"observed aggregate {component} differs from expected",
                        expected=expected_component,
                        observed=observed_component,
                        tolerance=request.policy.fee_tolerance,
                        order=order,
                        evidence=evidence_hashes,
                    )
                if expected_vwap is not None and observed_vwap is not None:
                    self._vwap_difference(
                        findings=findings,
                        request=request,
                        expected_quantity=expected_quantity,
                        expected_gross=expected_gross,
                        observed_quantity=observed_quantity,
                        observed_gross=observed_gross,
                        expected_vwap=expected_vwap,
                        observed_vwap=observed_vwap,
                        tolerance=request.policy.price_tolerance,
                        order=order,
                        evidence=evidence_hashes,
                    )
                self._economic_difference(
                    findings=findings,
                    request=request,
                    code=ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                    entity_id=f"fill-order:{order_id}:fee",
                    message="observed aggregate fee differs from expected",
                    expected=expected_fee,
                    observed=observed_fee,
                    tolerance=request.policy.fee_tolerance,
                    order=order,
                    evidence=evidence_hashes,
                )
                self._economic_difference(
                    findings=findings,
                    request=request,
                    code=ReconciliationDifferenceCode.FILL_CASH_MISMATCH,
                    entity_id=f"fill-order:{order_id}:cash",
                    message="observed aggregate cash movement differs from expected",
                    expected=expected_cash,
                    observed=observed_cash,
                    tolerance=request.policy.cash_tolerance,
                    order=order,
                    evidence=evidence_hashes,
                )
            checks.append(
                FillReconciliation.build(
                    check_id=f"fill-order:{order_id}",
                    input_hash=request.input_hash,
                    decision_id=request.draft.decision_id,
                    batch_hash=request.draft.batch_hash,
                    line_hash=line_hash,
                    order_id=order_id,
                    instrument_id=instrument_id,
                    expected_fill_ids=tuple(sorted(item.fill_id for item in expected)),
                    observed_source_fill_ids=tuple(
                        sorted(item.source_fill_id for item in observed)
                    ),
                    expected_quantity=expected_quantity,
                    observed_quantity=observed_quantity,
                    quantity_delta=observed_quantity - expected_quantity,
                    expected_gross=expected_gross,
                    observed_gross=observed_gross,
                    gross_delta=observed_gross - expected_gross,
                    expected_total_fee=expected_fee,
                    observed_total_fee=observed_fee,
                    fee_delta=observed_fee - expected_fee,
                    expected_commission=expected_commission,
                    observed_commission=observed_commission,
                    commission_delta=observed_commission - expected_commission,
                    expected_stamp_duty=expected_stamp_duty,
                    observed_stamp_duty=observed_stamp_duty,
                    stamp_duty_delta=observed_stamp_duty - expected_stamp_duty,
                    expected_transfer_fee=expected_transfer_fee,
                    observed_transfer_fee=observed_transfer_fee,
                    transfer_fee_delta=observed_transfer_fee - expected_transfer_fee,
                    expected_other_fee=expected_other_fee,
                    observed_other_fee=observed_other_fee,
                    other_fee_delta=observed_other_fee - expected_other_fee,
                    expected_cash_change=expected_cash,
                    observed_cash_change=observed_cash,
                    cash_delta=observed_cash - expected_cash,
                    expected_vwap=expected_vwap,
                    observed_vwap=observed_vwap,
                    report_hashes=tuple(sorted(item.report_hash for item in observed)),
                    findings=tuple(findings),
                )
            )
        return tuple(sorted(checks, key=lambda item: item.check_id))

    def _individual_fill_comparisons(
        self,
        *,
        findings: list[ReconciliationFinding],
        request: ReconciliationRequest,
        order: PaperOrder,
        expected: PaperFill,
        observed: ObservedFillReport,
        evidence: tuple[str, ...],
    ) -> None:
        if expected.filled_at != observed.filled_at:
            findings.append(
                self._finding(
                    request=request,
                    domain=ReconciliationDomain.FILL,
                    code=ReconciliationDifferenceCode.FILL_IDENTITY_MISMATCH,
                    status=ReconciliationCheckStatus.MISMATCH,
                    severity=ReconciliationSeverity.CRITICAL,
                    entity_id=f"fill:{expected.fill_id}:filled-at",
                    message="observed fill time differs from the expected fill",
                    expected=expected.filled_at,
                    actual=observed.filled_at,
                    order_id=order.order_id,
                    line_hash=order.line_hash,
                    fill_id=expected.fill_id,
                    instrument_id=order.instrument_id,
                    evidence=evidence,
                )
            )
        if expected.quantity != observed.quantity:
            findings.append(
                self._finding(
                    request=request,
                    domain=ReconciliationDomain.FILL,
                    code=ReconciliationDifferenceCode.FILL_QUANTITY_MISMATCH,
                    status=ReconciliationCheckStatus.MISMATCH,
                    severity=ReconciliationSeverity.CRITICAL,
                    entity_id=f"fill:{expected.fill_id}:quantity",
                    message="observed individual fill quantity differs from expected",
                    expected=expected.quantity,
                    actual=observed.quantity,
                    delta=observed.quantity - expected.quantity,
                    order_id=order.order_id,
                    line_hash=order.line_hash,
                    fill_id=expected.fill_id,
                    instrument_id=order.instrument_id,
                    evidence=evidence,
                )
            )
        comparisons = (
            (
                "price",
                ReconciliationDifferenceCode.FILL_PRICE_MISMATCH,
                expected.price,
                observed.price,
                request.policy.price_tolerance,
            ),
            (
                "gross",
                ReconciliationDifferenceCode.FILL_GROSS_MISMATCH,
                expected.gross_amount,
                observed.gross_amount,
                request.policy.cash_tolerance,
            ),
            (
                "commission",
                ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                expected.commission,
                observed.commission,
                request.policy.fee_tolerance,
            ),
            (
                "stamp-duty",
                ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                expected.stamp_duty,
                observed.stamp_duty,
                request.policy.fee_tolerance,
            ),
            (
                "transfer-fee",
                ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                expected.transfer_fee,
                observed.transfer_fee,
                request.policy.fee_tolerance,
            ),
            (
                "other-fee",
                ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                expected.other_fee,
                observed.other_fee,
                request.policy.fee_tolerance,
            ),
            (
                "total-fee",
                ReconciliationDifferenceCode.FILL_FEE_MISMATCH,
                expected.total_fee,
                observed.total_fee,
                request.policy.fee_tolerance,
            ),
            (
                "cash",
                ReconciliationDifferenceCode.FILL_CASH_MISMATCH,
                expected.cash_change,
                observed.cash_change,
                request.policy.cash_tolerance,
            ),
        )
        for field_name, code, expected_value, observed_value, tolerance in comparisons:
            delta = observed_value - expected_value
            if delta == 0:
                continue
            within = _within(delta, tolerance)
            findings.append(
                self._finding(
                    request=request,
                    domain=ReconciliationDomain.FILL,
                    code=code,
                    status=(
                        ReconciliationCheckStatus.EXPECTED_VARIANCE
                        if within
                        else ReconciliationCheckStatus.MISMATCH
                    ),
                    severity=(
                        ReconciliationSeverity.WARNING
                        if within
                        else ReconciliationSeverity.CRITICAL
                    ),
                    entity_id=f"fill:{expected.fill_id}:{field_name}",
                    message=f"observed individual fill {field_name} differs from expected",
                    expected=expected_value,
                    actual=observed_value,
                    delta=delta,
                    order_id=order.order_id,
                    line_hash=order.line_hash,
                    fill_id=expected.fill_id,
                    instrument_id=order.instrument_id,
                    evidence=evidence,
                )
            )

    def _economic_difference(
        self,
        *,
        findings: list[ReconciliationFinding],
        request: ReconciliationRequest,
        code: ReconciliationDifferenceCode,
        entity_id: str,
        message: str,
        expected: Decimal,
        observed: Decimal,
        tolerance: Decimal,
        order: PaperOrder,
        evidence: tuple[str, ...],
    ) -> None:
        delta = observed - expected
        if delta == 0:
            return
        within = _within(delta, tolerance)
        findings.append(
            self._finding(
                request=request,
                domain=ReconciliationDomain.FILL,
                code=code,
                status=(
                    ReconciliationCheckStatus.EXPECTED_VARIANCE
                    if within
                    else ReconciliationCheckStatus.MISMATCH
                ),
                severity=(
                    ReconciliationSeverity.WARNING if within else ReconciliationSeverity.CRITICAL
                ),
                entity_id=entity_id,
                message=message,
                expected=expected,
                actual=observed,
                delta=delta,
                order_id=order.order_id,
                line_hash=order.line_hash,
                instrument_id=order.instrument_id,
                evidence=evidence,
            )
        )

    def _vwap_difference(
        self,
        *,
        findings: list[ReconciliationFinding],
        request: ReconciliationRequest,
        expected_quantity: Decimal,
        expected_gross: Decimal,
        observed_quantity: Decimal,
        observed_gross: Decimal,
        expected_vwap: Decimal,
        observed_vwap: Decimal,
        tolerance: Decimal,
        order: PaperOrder,
        evidence: tuple[str, ...],
    ) -> None:
        """Compare rational VWAPs exactly so display rounding cannot hide a difference."""

        with localcontext() as context:
            context.prec = _EXACT_COMPARISON_PRECISION
            numerator = (observed_gross * expected_quantity) - (expected_gross * observed_quantity)
            denominator = expected_quantity * observed_quantity
            threshold = tolerance * denominator
        if numerator == 0:
            return
        within = abs(numerator) <= threshold
        findings.append(
            self._finding(
                request=request,
                domain=ReconciliationDomain.FILL,
                code=ReconciliationDifferenceCode.FILL_PRICE_MISMATCH,
                status=(
                    ReconciliationCheckStatus.EXPECTED_VARIANCE
                    if within
                    else ReconciliationCheckStatus.MISMATCH
                ),
                severity=(
                    ReconciliationSeverity.WARNING if within else ReconciliationSeverity.CRITICAL
                ),
                entity_id=f"fill-order:{order.order_id}:vwap",
                message="observed fill VWAP differs from expected",
                expected=expected_vwap,
                actual=observed_vwap,
                delta=f"{numerator}/{denominator}",
                order_id=order.order_id,
                line_hash=order.line_hash,
                instrument_id=order.instrument_id,
                evidence=evidence,
            )
        )

    def _cash_check(self, request: ReconciliationRequest) -> CashReconciliation:
        expected = request.receipt.account_after
        observed = request.observed.account_snapshot.cash
        findings: list[ReconciliationFinding] = []
        mismatched: list[str] = []
        fields = (
            (
                "total_cash",
                expected.total_cash,
                observed.total_cash,
                ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
            ),
            (
                "available_cash",
                expected.available_cash,
                observed.available_cash,
                ReconciliationDifferenceCode.CASH_AVAILABLE_MISMATCH,
            ),
            (
                "frozen_cash",
                expected.frozen_cash,
                observed.frozen_cash,
                ReconciliationDifferenceCode.CASH_FROZEN_MISMATCH,
            ),
        )
        for field_name, expected_value, observed_value, code in fields:
            delta = observed_value - expected_value
            if delta == 0:
                continue
            mismatched.append(field_name)
            within = _within(delta, request.policy.cash_tolerance)
            findings.append(
                self._finding(
                    request=request,
                    domain=ReconciliationDomain.CASH,
                    code=code,
                    status=(
                        ReconciliationCheckStatus.EXPECTED_VARIANCE
                        if within
                        else ReconciliationCheckStatus.MISMATCH
                    ),
                    severity=(
                        ReconciliationSeverity.WARNING
                        if within
                        else ReconciliationSeverity.CRITICAL
                    ),
                    entity_id=f"cash:{field_name}",
                    message=f"observed {field_name} differs from expected paper cash",
                    expected=expected_value,
                    actual=observed_value,
                    delta=delta,
                    evidence=(expected.state_hash, request.observed.account_snapshot.content_hash),
                )
            )
        return CashReconciliation.build(
            check_id=f"cash:{expected.account_id}",
            input_hash=request.input_hash,
            decision_id=request.draft.decision_id,
            batch_hash=request.draft.batch_hash,
            account_id=expected.account_id,
            currency=expected.currency,
            expected_total_cash=expected.total_cash,
            observed_total_cash=observed.total_cash,
            total_delta=observed.total_cash - expected.total_cash,
            expected_available_cash=expected.available_cash,
            observed_available_cash=observed.available_cash,
            available_delta=observed.available_cash - expected.available_cash,
            expected_frozen_cash=expected.frozen_cash,
            observed_frozen_cash=observed.frozen_cash,
            frozen_delta=observed.frozen_cash - expected.frozen_cash,
            mismatched_fields=tuple(mismatched),
            findings=tuple(findings),
        )

    def _position_checks(
        self, request: ReconciliationRequest
    ) -> tuple[PositionReconciliation, ...]:
        expected = {item.instrument_id: item for item in request.receipt.account_after.positions}
        observed = {
            item.instrument_id: item for item in request.observed.account_snapshot.positions
        }
        orders_by_instrument: dict[str, list[str]] = defaultdict(list)
        fills_by_instrument: dict[str, list[str]] = defaultdict(list)
        for order in request.receipt.orders:
            orders_by_instrument[order.instrument_id].append(order.order_id)
        for fill in request.receipt.fills:
            fills_by_instrument[fill.instrument_id].append(fill.fill_id)
        checks: list[PositionReconciliation] = []
        for instrument_id in sorted(set(expected) | set(observed)):
            expected_position = expected.get(instrument_id)
            observed_position = observed.get(instrument_id)
            findings: list[ReconciliationFinding] = []
            evidence = (
                request.receipt.account_after_hash,
                request.observed.account_snapshot.content_hash,
            )
            if expected_position is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.POSITION,
                        code=ReconciliationDifferenceCode.POSITION_UNEXPECTED,
                        status=ReconciliationCheckStatus.UNEXPECTED,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"position:{instrument_id}:unexpected",
                        message="observed position does not exist in expected paper account",
                        actual=observed_position.total_quantity if observed_position else None,
                        instrument_id=instrument_id,
                        evidence=evidence,
                    )
                )
            elif observed_position is None:
                findings.append(
                    self._finding(
                        request=request,
                        domain=ReconciliationDomain.POSITION,
                        code=ReconciliationDifferenceCode.POSITION_MISSING,
                        status=ReconciliationCheckStatus.MISSING,
                        severity=ReconciliationSeverity.CRITICAL,
                        entity_id=f"position:{instrument_id}:missing",
                        message="expected paper position is absent from the observed snapshot",
                        expected=expected_position.total_quantity,
                        instrument_id=instrument_id,
                        evidence=evidence,
                    )
                )
            else:
                if expected_position.instrument_type is not observed_position.instrument_type:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.POSITION,
                            code=ReconciliationDifferenceCode.POSITION_TYPE_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"position:{instrument_id}:type",
                            message="observed position instrument type differs from expected",
                            expected=expected_position.instrument_type,
                            actual=observed_position.instrument_type,
                            instrument_id=instrument_id,
                            evidence=evidence,
                        )
                    )
                total_delta = observed_position.total_quantity - expected_position.total_quantity
                if total_delta != 0:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.POSITION,
                            code=ReconciliationDifferenceCode.POSITION_QUANTITY_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"position:{instrument_id}:total",
                            message="observed total position quantity differs from expected",
                            expected=expected_position.total_quantity,
                            actual=observed_position.total_quantity,
                            delta=total_delta,
                            instrument_id=instrument_id,
                            evidence=evidence,
                        )
                    )
                bucket_names = (
                    "available_quantity",
                    "frozen_quantity",
                    "unsettled_quantity",
                )
                bucket_mismatches = tuple(
                    field_name
                    for field_name in bucket_names
                    if getattr(expected_position, field_name)
                    != getattr(observed_position, field_name)
                )
                if bucket_mismatches:
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.POSITION,
                            code=ReconciliationDifferenceCode.POSITION_BUCKET_MISMATCH,
                            status=ReconciliationCheckStatus.MISMATCH,
                            severity=ReconciliationSeverity.CRITICAL,
                            entity_id=f"position:{instrument_id}:buckets",
                            message=(
                                "observed position quantity buckets differ: "
                                + ", ".join(bucket_mismatches)
                            ),
                            expected=(
                                f"{expected_position.available_quantity}/"
                                f"{expected_position.frozen_quantity}/"
                                f"{expected_position.unsettled_quantity}"
                            ),
                            actual=(
                                f"{observed_position.available_quantity}/"
                                f"{observed_position.frozen_quantity}/"
                                f"{observed_position.unsettled_quantity}"
                            ),
                            instrument_id=instrument_id,
                            evidence=evidence,
                        )
                    )
                cost_fields = (
                    ("cost_basis", expected_position.cost_basis, observed_position.cost_basis),
                    (
                        "average_cost",
                        expected_position.average_cost,
                        observed_position.average_cost,
                    ),
                )
                for field_name, expected_value, observed_value in cost_fields:
                    delta = observed_value - expected_value
                    if delta == 0:
                        continue
                    within = _within(delta, request.policy.cost_tolerance)
                    findings.append(
                        self._finding(
                            request=request,
                            domain=ReconciliationDomain.POSITION,
                            code=ReconciliationDifferenceCode.POSITION_COST_MISMATCH,
                            status=(
                                ReconciliationCheckStatus.EXPECTED_VARIANCE
                                if within
                                else ReconciliationCheckStatus.MISMATCH
                            ),
                            severity=(
                                ReconciliationSeverity.WARNING
                                if within
                                else ReconciliationSeverity.CRITICAL
                            ),
                            entity_id=f"position:{instrument_id}:{field_name}",
                            message=f"observed position {field_name} differs from expected",
                            expected=expected_value,
                            actual=observed_value,
                            delta=delta,
                            instrument_id=instrument_id,
                            evidence=evidence,
                        )
                    )
            checks.append(
                PositionReconciliation.build(
                    check_id=f"position:{instrument_id}",
                    input_hash=request.input_hash,
                    decision_id=request.draft.decision_id,
                    batch_hash=request.draft.batch_hash,
                    instrument_id=instrument_id,
                    expected_instrument_type=(
                        expected_position.instrument_type if expected_position else None
                    ),
                    observed_instrument_type=(
                        observed_position.instrument_type if observed_position else None
                    ),
                    expected_total=(
                        expected_position.total_quantity if expected_position else None
                    ),
                    observed_total=(
                        observed_position.total_quantity if observed_position else None
                    ),
                    total_delta=self._optional_delta(
                        expected_position.total_quantity if expected_position else None,
                        observed_position.total_quantity if observed_position else None,
                    ),
                    expected_available=(
                        expected_position.available_quantity if expected_position else None
                    ),
                    observed_available=(
                        observed_position.available_quantity if observed_position else None
                    ),
                    available_delta=self._optional_delta(
                        expected_position.available_quantity if expected_position else None,
                        observed_position.available_quantity if observed_position else None,
                    ),
                    expected_frozen=(
                        expected_position.frozen_quantity if expected_position else None
                    ),
                    observed_frozen=(
                        observed_position.frozen_quantity if observed_position else None
                    ),
                    frozen_delta=self._optional_delta(
                        expected_position.frozen_quantity if expected_position else None,
                        observed_position.frozen_quantity if observed_position else None,
                    ),
                    expected_unsettled=(
                        expected_position.unsettled_quantity if expected_position else None
                    ),
                    observed_unsettled=(
                        observed_position.unsettled_quantity if observed_position else None
                    ),
                    unsettled_delta=self._optional_delta(
                        expected_position.unsettled_quantity if expected_position else None,
                        observed_position.unsettled_quantity if observed_position else None,
                    ),
                    expected_cost_basis=(
                        expected_position.cost_basis if expected_position else None
                    ),
                    observed_cost_basis=(
                        observed_position.cost_basis if observed_position else None
                    ),
                    cost_delta=self._optional_delta(
                        expected_position.cost_basis if expected_position else None,
                        observed_position.cost_basis if observed_position else None,
                    ),
                    expected_average_cost=(
                        expected_position.average_cost if expected_position else None
                    ),
                    observed_average_cost=(
                        observed_position.average_cost if observed_position else None
                    ),
                    average_cost_delta=self._optional_delta(
                        expected_position.average_cost if expected_position else None,
                        observed_position.average_cost if observed_position else None,
                    ),
                    related_order_ids=tuple(sorted(orders_by_instrument[instrument_id])),
                    related_fill_ids=tuple(sorted(fills_by_instrument[instrument_id])),
                    findings=tuple(findings),
                )
            )
        return tuple(sorted(checks, key=lambda item: item.instrument_id))

    @staticmethod
    def _vwap(quantity: Decimal, gross: Decimal) -> Decimal | None:
        if quantity == 0:
            return None
        with localcontext() as context:
            context.prec = _DERIVED_RATIO_PRECISION
            context.rounding = ROUND_HALF_EVEN
            return gross / quantity

    @staticmethod
    def _optional_delta(expected: Decimal | None, observed: Decimal | None) -> Decimal | None:
        if expected is None or observed is None:
            return None
        return observed - expected

    @staticmethod
    def _finding(
        *,
        request: ReconciliationRequest,
        domain: ReconciliationDomain,
        code: ReconciliationDifferenceCode,
        status: ReconciliationCheckStatus,
        severity: ReconciliationSeverity,
        entity_id: str,
        message: str,
        expected: object | None = None,
        actual: object | None = None,
        delta: object | None = None,
        line_hash: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        instrument_id: str | None = None,
        evidence: tuple[str, ...] = (),
    ) -> ReconciliationFinding:
        return ReconciliationFinding.build(
            domain=domain,
            code=code,
            status=status,
            severity=severity,
            entity_id=entity_id,
            message=message,
            decision_id=request.draft.decision_id,
            batch_hash=request.draft.batch_hash,
            line_hash=line_hash,
            order_id=order_id,
            fill_id=fill_id,
            instrument_id=instrument_id,
            expected_value=_text(expected),
            observed_value=_text(actual),
            delta=_text(delta),
            evidence_hashes=evidence,
        )


__all__ = ["ReconciliationEngine"]
