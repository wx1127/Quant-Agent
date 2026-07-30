"""Append-only audit event contracts and a local JSON Lines sink."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.observability.redaction import redact


class AuditEvent(BaseModel):
    """Immutable record of a security- or decision-relevant action."""

    model_config = ConfigDict(frozen=True)

    event_type: str
    actor_id: str
    action: str
    result: str
    request_id: str
    decision_id: str | None = None
    occurred_at: datetime = Field(default_factory=shanghai_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an aware datetime while keeping JSON serialization explicit."""

        return ensure_aware(value)


class AuditSink(Protocol):
    """Append-only audit destination."""

    def append(self, event: AuditEvent) -> None:
        """Persist one event."""


class JsonLinesAuditSink:
    """Local append-only sink for development and tests."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, event: AuditEvent) -> None:
        """Append a redacted event as one JSON object."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = redact(event.model_dump(mode="json"))
        with self._path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            audit_file.write("\n")
