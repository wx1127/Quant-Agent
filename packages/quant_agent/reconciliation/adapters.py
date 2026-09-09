"""Explicit adapters from paper events into the independent observation schema."""

from __future__ import annotations

from datetime import datetime

from quant_agent.execution.paper import (
    PaperExecutionReceipt,
    PaperFill,
    PaperMatchAttempt,
    PaperOrder,
)
from quant_agent.portfolio import AccountSnapshot

from .contracts import ObservedExecutionEvidence, ObservedFillReport, ObservedOrderReport


def observed_order_report_from_paper(
    *,
    order: PaperOrder,
    attempt: PaperMatchAttempt,
    account_id: str,
    available_at: datetime,
    source_system: str = "paper-simulator",
) -> ObservedOrderReport:
    """Normalize one paper order event without granting reconciliation authority."""

    if attempt.order_id != order.order_id:
        raise ValueError("paper attempt does not belong to the supplied order")
    if (
        attempt.status is not order.status
        or attempt.requested_quantity != order.quantity
        or attempt.filled_quantity != order.filled_quantity
    ):
        raise ValueError("paper attempt does not reconcile to the supplied order")
    outcome_code = attempt.no_fill_reason.value if attempt.no_fill_reason is not None else None
    return ObservedOrderReport.build(
        source_system=source_system,
        account_id=account_id,
        report_id=f"paper-order-report:{order.order_hash}",
        source_order_id=order.order_id,
        client_order_id=order.order_id,
        revision=1,
        event_time=attempt.attempted_at,
        available_at=available_at,
        instrument_id=order.instrument_id,
        instrument_type=order.instrument_type,
        side=order.side,
        status=order.status,
        requested_quantity=order.quantity,
        cumulative_filled_quantity=order.filled_quantity,
        remaining_quantity=order.remaining_quantity,
        outcome_code=outcome_code,
    )


def observed_fill_report_from_paper(
    *,
    fill: PaperFill,
    account_id: str,
    currency: str,
    available_at: datetime,
    source_system: str = "paper-simulator",
) -> ObservedFillReport:
    """Normalize one paper fill into an independently hash-bound report envelope."""

    return ObservedFillReport.build(
        source_system=source_system,
        account_id=account_id,
        report_id=f"paper-fill-report:{fill.fill_hash}",
        source_fill_id=fill.fill_id,
        source_order_id=fill.order_id,
        client_order_id=fill.order_id,
        client_fill_id=fill.fill_id,
        filled_at=fill.filled_at,
        available_at=available_at,
        instrument_id=fill.instrument_id,
        instrument_type=fill.instrument_type,
        side=fill.side,
        currency=currency,
        quantity=fill.quantity,
        price=fill.price,
        commission=fill.commission,
        stamp_duty=fill.stamp_duty,
        transfer_fee=fill.transfer_fee,
        other_fee=fill.other_fee,
    )


def observed_evidence_from_paper_receipt(
    *,
    receipt: PaperExecutionReceipt,
    account_snapshot: AccountSnapshot,
    available_at: datetime,
    source_system: str = "paper-simulator",
    source_cursor: str | None = None,
) -> ObservedExecutionEvidence:
    """Build normalized paper observations around a separately supplied account snapshot.

    This adapter is intended for paper-mode testing and replay.  The caller must still
    supply the observed account snapshot; reconciliation never derives that snapshot
    from the expected receipt.
    """

    attempts = {item.order_id: item for item in receipt.attempts}
    if set(attempts) != {item.order_id for item in receipt.orders}:
        raise ValueError("paper receipt must contain exactly one attempt per order")
    order_reports = tuple(
        observed_order_report_from_paper(
            order=order,
            attempt=attempts[order.order_id],
            account_id=receipt.account_after.account_id,
            available_at=available_at,
            source_system=source_system,
        )
        for order in receipt.orders
    )
    fill_reports = tuple(
        observed_fill_report_from_paper(
            fill=fill,
            account_id=receipt.account_after.account_id,
            currency=receipt.account_after.currency,
            available_at=available_at,
            source_system=source_system,
        )
        for fill in receipt.fills
    )
    return ObservedExecutionEvidence.build(
        source_system=source_system,
        account_id=receipt.account_after.account_id,
        observed_at=account_snapshot.as_of,
        available_at=available_at,
        account_snapshot=account_snapshot,
        order_reports=order_reports,
        fill_reports=fill_reports,
        source_cursor=source_cursor,
    )


__all__ = [
    "observed_evidence_from_paper_receipt",
    "observed_fill_report_from_paper",
    "observed_order_report_from_paper",
]
