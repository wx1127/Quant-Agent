"""Paper-account validation loop with explicit approval and reconciliation gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class PaperState(StrEnum):
    CREATED = "created"
    APPROVED = "approved"
    FILLED = "filled"
    RECONCILED = "reconciled"


@dataclass(frozen=True, slots=True)
class PaperCycle:
    cycle_id: str
    draft_id: str
    state: PaperState
    fill_count: int
    reconciliation_ok: bool


class PaperValidation:
    def __init__(self) -> None:
        self._cycles: dict[str, PaperCycle] = {}

    def create(self, draft_id: str) -> PaperCycle:
        cycle = PaperCycle(str(uuid4()), draft_id, PaperState.CREATED, 0, False)
        self._cycles[cycle.cycle_id] = cycle
        return cycle

    def approve(self, cycle_id: str) -> PaperCycle:
        cycle = self._get(cycle_id)
        if cycle.state is not PaperState.CREATED:
            raise ValueError("paper cycle is not awaiting approval")
        return self._save(cycle, PaperState.APPROVED)

    def simulate_fill(self, cycle_id: str, fill_count: int) -> PaperCycle:
        cycle = self._get(cycle_id)
        if cycle.state is not PaperState.APPROVED:
            raise ValueError("paper cycle requires approval")
        if fill_count < 0:
            raise ValueError("fill_count cannot be negative")
        updated = PaperCycle(cycle.cycle_id, cycle.draft_id, PaperState.FILLED, fill_count, False)
        self._cycles[cycle_id] = updated
        return updated

    def reconcile(self, cycle_id: str, expected_fill_count: int) -> PaperCycle:
        cycle = self._get(cycle_id)
        if cycle.state is not PaperState.FILLED:
            raise ValueError("paper cycle requires simulated fills")
        ok = cycle.fill_count == expected_fill_count
        updated = PaperCycle(
            cycle.cycle_id, cycle.draft_id, PaperState.RECONCILED, cycle.fill_count, ok
        )
        self._cycles[cycle_id] = updated
        return updated

    def _get(self, cycle_id: str) -> PaperCycle:
        if cycle_id not in self._cycles:
            raise KeyError(cycle_id)
        return self._cycles[cycle_id]

    def _save(self, cycle: PaperCycle, state: PaperState) -> PaperCycle:
        updated = PaperCycle(
            cycle.cycle_id, cycle.draft_id, state, cycle.fill_count, cycle.reconciliation_ok
        )
        self._cycles[cycle.cycle_id] = updated
        return updated
