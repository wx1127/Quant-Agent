"""Never-ignore reconciliation differences with incident severity."""

from dataclasses import dataclass
from enum import StrEnum

from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.execution.paper import PaperBatchResult
from quant_agent.portfolio.snapshots import AccountSnapshot


class ReconciliationSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class ReconciliationDifference:
    category: str
    severity: ReconciliationSeverity
    message: str
    order_id: str | None = None
    instrument_id: str | None = None
    expected: float | None = None
    actual: float | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    decision_id: str
    batch_hash: str
    matched: bool
    differences: tuple[ReconciliationDifference, ...]
    trigger_stop: bool


class ReconciliationEngine:
    def reconcile(
        self,
        batch: OrderDraftBatch,
        paper: PaperBatchResult,
        *,
        expected_account: AccountSnapshot,
        cash_tolerance: float = 0.01,
    ) -> ReconciliationResult:
        differences: list[ReconciliationDifference] = []
        if batch.batch_hash != paper.batch_hash:
            differences.append(
                ReconciliationDifference(
                    "BATCH",
                    ReconciliationSeverity.CRITICAL,
                    "paper result belongs to another batch",
                )
            )
        draft_by_id = {item.draft_id: item for item in batch.drafts}
        order_by_draft = {item.draft_id: item for item in paper.orders}
        for draft_id, draft in draft_by_id.items():
            paper_order = order_by_draft.get(draft_id)
            if paper_order is None:
                differences.append(
                    ReconciliationDifference(
                        "ORDER",
                        ReconciliationSeverity.CRITICAL,
                        "draft has no paper order",
                        order_id=draft_id,
                        instrument_id=draft.instrument_id,
                    )
                )
                continue
            filled = paper_order.order.filled_quantity
            if filled < draft.quantity:
                severity = (
                    ReconciliationSeverity.WARNING if filled > 0 else ReconciliationSeverity.INFO
                )
                differences.append(
                    ReconciliationDifference(
                        "FILL",
                        severity,
                        "draft is not fully filled",
                        order_id=paper_order.order.order_id,
                        instrument_id=draft.instrument_id,
                        expected=float(draft.quantity),
                        actual=float(filled),
                    )
                )
            if filled > draft.quantity:
                differences.append(
                    ReconciliationDifference(
                        "FILL",
                        ReconciliationSeverity.CRITICAL,
                        "filled quantity exceeds draft",
                        order_id=paper_order.order.order_id,
                        instrument_id=draft.instrument_id,
                        expected=float(draft.quantity),
                        actual=float(filled),
                    )
                )
        fill_ids = [item.fill_id for item in paper.fills]
        if len(set(fill_ids)) != len(fill_ids):
            differences.append(
                ReconciliationDifference(
                    "DUPLICATE_FILL",
                    ReconciliationSeverity.CRITICAL,
                    "duplicate fill report detected",
                )
            )
        reported_by_order: dict[str, int] = {}
        for fill in paper.fills:
            reported_by_order[fill.order_id] = (
                reported_by_order.get(fill.order_id, 0) + fill.quantity
            )
        for item in paper.orders:
            reported = reported_by_order.get(item.order.order_id, 0)
            if reported != item.order.filled_quantity:
                differences.append(
                    ReconciliationDifference(
                        "FILL_REPORT",
                        ReconciliationSeverity.CRITICAL,
                        "fill reports do not match order filled quantity",
                        order_id=item.order.order_id,
                        instrument_id=item.order.instrument_id,
                        expected=float(item.order.filled_quantity),
                        actual=float(reported),
                    )
                )
        if (
            abs(expected_account.available_cash - paper.account_snapshot.available_cash)
            > cash_tolerance
        ):
            differences.append(
                ReconciliationDifference(
                    "CASH",
                    ReconciliationSeverity.CRITICAL,
                    "cash balance differs from expected account",
                    expected=expected_account.available_cash,
                    actual=paper.account_snapshot.available_cash,
                )
            )
        expected_positions = {
            item.instrument_id: item.quantity for item in expected_account.holdings
        }
        actual_positions = {
            item.instrument_id: item.quantity for item in paper.account_snapshot.holdings
        }
        for instrument_id in sorted(set(expected_positions) | set(actual_positions)):
            expected = expected_positions.get(instrument_id, 0)
            actual = actual_positions.get(instrument_id, 0)
            if expected != actual:
                differences.append(
                    ReconciliationDifference(
                        "POSITION",
                        ReconciliationSeverity.CRITICAL,
                        "position quantity differs from expected account",
                        instrument_id=instrument_id,
                        expected=float(expected),
                        actual=float(actual),
                    )
                )
        trigger_stop = any(item.severity is ReconciliationSeverity.CRITICAL for item in differences)
        return ReconciliationResult(
            decision_id=batch.decision_id,
            batch_hash=batch.batch_hash,
            matched=not differences,
            differences=tuple(differences),
            trigger_stop=trigger_stop,
        )
