"""Shadow-run guard: evaluate workflows without external side effects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class ShadowAction(StrEnum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    AGENT = "agent"
    PAPER_SUBMIT = "paper_submit"
    EXTERNAL_WRITE = "external_write"


@dataclass(frozen=True, slots=True)
class ShadowEvent:
    event_id: str
    action: ShadowAction
    accepted: bool
    reason: str
    occurred_at: datetime


class ShadowRunner:
    """Run allowed analysis actions and fail closed on side-effect actions."""

    _allowed = frozenset({ShadowAction.RESEARCH, ShadowAction.BACKTEST, ShadowAction.AGENT})

    def run(self, action: ShadowAction) -> ShadowEvent:
        accepted = action in self._allowed
        return ShadowEvent(
            event_id=str(uuid4()),
            action=action,
            accepted=accepted,
            reason="shadow evaluation allowed" if accepted else "external side effect suppressed",
            occurred_at=datetime.now(UTC),
        )
