"""Structured JSON logging with request and decision context."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from quant_agent.observability.redaction import redact

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_decision_id: ContextVar[str | None] = ContextVar("decision_id", default=None)


class JsonFormatter(logging.Formatter):
    """Small dependency-free JSON formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a redacted structured log record."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get(),
            "decision_id": _decision_id.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), ensure_ascii=False, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure the process root logger once with JSON output."""

    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)


@contextmanager
def log_context(*, request_id: str, decision_id: str | None = None) -> Iterator[None]:
    """Bind request-scoped identifiers and restore prior context afterward."""

    request_token = _request_id.set(request_id)
    decision_token = _decision_id.set(decision_id)
    try:
        yield
    finally:
        _request_id.reset(request_token)
        _decision_id.reset(decision_token)
