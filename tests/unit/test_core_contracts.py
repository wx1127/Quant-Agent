"""Tests for stable cross-module domain contracts."""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_agent.core import (
    Money,
    Provenance,
    QuantAgentError,
    ToolResponse,
    new_decision_id,
    new_request_id,
)
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import ToolIssue
from quant_agent.core.time import ensure_aware


def test_money_rejects_float_and_rounds_decimal() -> None:
    with pytest.raises(ValidationError):
        Money(amount=1.1)
    with pytest.raises(ValidationError):
        Money(amount=True)

    value = Money(amount="12.345", currency="cny").rounded()

    assert value.amount == Decimal("12.35")
    assert value.currency == "CNY"


def test_money_rejects_invalid_currency() -> None:
    with pytest.raises(ValidationError, match="three-letter"):
        Money(amount="1", currency="yuan")


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        ensure_aware(datetime(2026, 7, 30, 15, 0))


def test_identifiers_have_stable_opaque_prefixes() -> None:
    request_id = new_request_id()
    decision_id = new_decision_id(datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")))

    assert str(request_id).startswith("req_")
    assert str(decision_id).startswith("dec_20260730_")


def test_typed_error_preserves_code_message_and_details() -> None:
    error = QuantAgentError(
        ErrorCode.DATA_INVALID,
        "invalid market snapshot",
        details={"data_version": "v1"},
    )

    assert error.code is ErrorCode.DATA_INVALID
    assert error.message == "invalid market snapshot"
    assert error.details == {"data_version": "v1"}


def test_tool_response_round_trip_and_consistency() -> None:
    response = ToolResponse[dict[str, int]](
        ok=True,
        request_id="req_test",
        as_of=datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        data={"count": 0},
        provenance=Provenance(service="test", version="0.1.0"),
    ).validate_consistency()

    restored = ToolResponse[dict[str, int]].model_validate_json(response.model_dump_json())

    assert restored == response


def test_successful_tool_response_cannot_contain_errors() -> None:
    response = ToolResponse[None](
        ok=True,
        request_id="req_test",
        as_of=datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        errors=(ToolIssue(code=ErrorCode.INTERNAL_ERROR, message="unexpected"),),
        provenance=Provenance(service="test", version="0.1.0"),
    )

    with pytest.raises(ValueError, match="successful response"):
        response.validate_consistency()


def test_failed_tool_response_requires_error() -> None:
    response = ToolResponse[None](
        ok=False,
        request_id="req_test",
        as_of=datetime(2026, 7, 30, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        provenance=Provenance(service="test", version="0.1.0"),
    )
    with pytest.raises(ValueError, match="at least one error"):
        response.validate_consistency()

    valid_failure = response.model_copy(
        update={
            "errors": (ToolIssue(code=ErrorCode.DATA_UNAVAILABLE, message="source unavailable"),)
        }
    )
    assert valid_failure.validate_consistency() == valid_failure
