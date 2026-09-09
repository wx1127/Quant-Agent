"""Tests for logging context, redaction and audit persistence."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_agent.observability import (
    AuditEvent,
    JsonLinesAuditSink,
    configure_logging,
    log_context,
    redact,
)


def test_recursive_redaction() -> None:
    payload = {
        "api_key": "abc",
        "nested": {"brokerToken": "def", "safe": "value"},
        "items": [{"password": "ghi"}],
    }

    assert redact(payload) == {
        "api_key": "[REDACTED]",
        "nested": {"brokerToken": "[REDACTED]", "safe": "value"},
        "items": [{"password": "[REDACTED]"}],
    }


def test_logging_context_is_emitted_and_secret_message_is_not_structured(
    capsys: object,
) -> None:
    configure_logging("INFO")
    with log_context(request_id="req_1", decision_id="dec_1"):
        logging.getLogger("quant-agent-test").info("safe event")

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err)

    assert payload["request_id"] == "req_1"
    assert payload["decision_id"] == "dec_1"
    assert payload["message"] == "safe event"


def test_audit_sink_appends_redacted_json(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit/events.jsonl"
    sink = JsonLinesAuditSink(audit_path)
    event = AuditEvent(
        event_type="CONFIG_CHECK",
        actor_id="tester",
        action="load",
        result="allowed",
        request_id="req_1",
        occurred_at=datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        metadata={"token": "secret", "profile": "local"},
    )

    sink.append(event)

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["token"] == "[REDACTED]"
    assert payload["metadata"]["profile"] == "local"


def test_audit_sink_writes_complete_lines_under_thread_concurrency(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit/concurrent.jsonl"
    sink = JsonLinesAuditSink(audit_path)

    def append(index: int) -> None:
        sink.append(
            AuditEvent(
                event_type="AGENT_TOOL_AUTHORIZATION",
                actor_id="agent-worker-1",
                action="get_market_snapshot",
                result="DENIED",
                request_id=f"req_{index}",
                occurred_at=datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                metadata={"index": index},
            )
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        tuple(executor.map(append, range(100)))

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert len(payloads) == 100
    assert {item["request_id"] for item in payloads} == {f"req_{index}" for index in range(100)}


def test_audit_event_rejects_naive_time() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        AuditEvent(
            event_type="CONFIG_CHECK",
            actor_id="tester",
            action="load",
            result="allowed",
            request_id="req_1",
            occurred_at=datetime(2026, 7, 30, 15, 0),
        )
