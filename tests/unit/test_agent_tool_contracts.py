"""Strict Agent-tool definitions, schemas, contexts, and output boundaries."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools.contracts import (
    AgentTool,
    ToolArguments,
    ToolCallContext,
    ToolCapability,
    ToolCategory,
    ToolDenialReason,
    ToolDescriptor,
    ToolEffect,
    ToolExecutionContext,
    ToolExecutionFailed,
    ToolOutput,
    ToolOutputRejected,
    ToolPermissionDenied,
    ToolRegistrationError,
    validate_tool_name,
    validate_tool_response,
)
from quant_agent.config import RuntimeMode
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import Provenance, ToolIssue, ToolResponse
from quant_agent.core.time import SHANGHAI_TZ

AS_OF = datetime(2026, 9, 8, 15, 10, tzinfo=SHANGHAI_TZ)
REQUESTED_AT = AS_OF + timedelta(seconds=1)


class TradeSide(StrEnum):
    """Representative JSON-reachable enum used by tool contracts."""

    BUY = "BUY"
    SELL = "SELL"


class ExampleArguments(ToolArguments):
    symbol: str
    quantity: int
    side: TradeSide
    observed_at: datetime
    notional: Decimal
    dry_run: bool = False


class ExampleOutput(ToolOutput):
    accepted_quantity: int
    side: TradeSide
    processed_at: datetime
    notional: Decimal


def _handler(
    _context: ToolExecutionContext,
    arguments: ExampleArguments,
) -> ExampleOutput:
    return ExampleOutput(
        accepted_quantity=arguments.quantity,
        side=arguments.side,
        processed_at=arguments.observed_at,
        notional=arguments.notional,
    )


def _tool(**changes: Any) -> AgentTool[ExampleArguments, ExampleOutput]:
    values: dict[str, Any] = {
        "name": "submit_example",
        "version": "1.2.3",
        "description": "Submit one deterministic example.",
        "arguments_type": ExampleArguments,
        "output_type": ExampleOutput,
        "handler": _handler,
    }
    values.update(changes)
    return AgentTool(**values)


def _snapshot() -> DecisionSnapshot:
    strategy_ref = StrategySnapshotRef(
        strategy_name="etf-rotation",
        strategy_version="etf-rotation-v1",
        config_hash="1" * 64,
        parameter_version="parameters-v1",
        parameter_hash="2" * 64,
        registered_at=AS_OF - timedelta(days=1),
    )
    return DecisionSnapshot.build(
        decision_id="dec_20260908_0123456789abcdef0123456789abcdef",
        mode=RuntimeMode.PAPER,
        market="CN_A",
        as_of=AS_OF,
        data_version="market_20260908_eod_v1",
        data_content_hash="3" * 64,
        strategy_refs=(strategy_ref,),
        risk_policy_version="portfolio-risk-v1",
        risk_policy_hash="4" * 64,
        account_id="paper-account-1",
        account_snapshot_id="paper-account-1:20260908T151000",
        account_snapshot_hash="5" * 64,
        code_commit="6" * 40,
        code_artifact_hash="7" * 64,
        agent_version="quant-agent-harness-v1",
        model_version="model-release-v1",
    )


def _execution_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="req_contract_test",
        principal_id="agent:test",
        requested_at=REQUESTED_AT,
        decision_snapshot=_snapshot(),
    )


def _response(**changes: Any) -> ToolResponse[object]:
    context = _execution_context()
    values: dict[str, Any] = {
        "ok": True,
        "request_id": context.request_id,
        "decision_id": context.decision_snapshot.decision_id,
        "as_of": AS_OF,
        "data": {"accepted": True},
        "provenance": Provenance(
            service="contract-test",
            version="1.0.0",
            data_version=context.decision_snapshot.data_version,
        ),
    }
    values.update(changes)
    return ToolResponse[object](**values)


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "market_snapshot",
        "tool2",
        "a1_b2",
        "a" * 64,
    ],
)
def test_validate_tool_name_accepts_only_exact_lower_ascii_snake_case(name: str) -> None:
    assert validate_tool_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        None,
        1,
        "",
        "_tool",
        "tool_",
        "two__words",
        "Uppercase",
        "has-hyphen",
        "with space",
        " tool",
        "tool ",
        "t\nool",
        "工具",
        "a" * 65,
    ],
)
def test_validate_tool_name_rejects_invalid_or_normalizable_values(name: object) -> None:
    with pytest.raises(ValueError, match="lower-ASCII snake-case"):
        validate_tool_name(cast(str, name))


def test_validate_tool_name_rejects_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="lower-ASCII snake-case"):
        validate_tool_name(StringSubclass("get_market_snapshot"))


def test_tool_argument_and_output_bases_are_strict_closed_frozen_models() -> None:
    expected = {
        "extra": "forbid",
        "frozen": True,
        "revalidate_instances": "always",
        "strict": True,
        "validate_default": True,
    }

    for model_type in (ExampleArguments, ExampleOutput):
        assert {key: model_type.model_config.get(key) for key in expected} == expected

    arguments = ExampleArguments(
        symbol="000001.SZ",
        quantity=2,
        side=TradeSide.BUY,
        observed_at=AS_OF,
        notional=Decimal("123.45"),
    )
    output = ExampleOutput(
        accepted_quantity=2,
        side=TradeSide.BUY,
        processed_at=AS_OF,
        notional=Decimal("123.45"),
    )

    with pytest.raises(ValidationError, match="frozen"):
        arguments.quantity = 3
    with pytest.raises(ValidationError, match="frozen"):
        output.accepted_quantity = 3
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExampleArguments.model_validate(
            {
                **arguments.model_dump(),
                "unexpected": True,
            }
        )


def test_invalid_defaults_are_validated_in_argument_and_output_models() -> None:
    class InvalidDefaultArguments(ToolArguments):
        quantity: int = "2"  # type: ignore[assignment]

    class InvalidDefaultOutput(ToolOutput):
        accepted_quantity: int = "2"  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="int_type"):
        InvalidDefaultArguments()
    with pytest.raises(ValidationError, match="int_type"):
        InvalidDefaultOutput()


def test_argument_validation_uses_strict_json_but_keeps_json_native_types_reachable() -> None:
    tool = _tool()

    arguments = tool.validate_arguments(
        """{
            "symbol": "000001.SZ",
            "quantity": 2,
            "side": "BUY",
            "observed_at": "2026-09-08T15:10:00+08:00",
            "notional": "123.45"
        }"""
    )

    assert arguments == ExampleArguments(
        symbol="000001.SZ",
        quantity=2,
        side=TradeSide.BUY,
        observed_at=AS_OF,
        notional=Decimal("123.45"),
    )
    assert isinstance(arguments.side, TradeSide)
    assert isinstance(arguments.observed_at, datetime)
    assert arguments.observed_at.utcoffset() == timedelta(hours=8)
    assert isinstance(arguments.notional, Decimal)


@pytest.mark.parametrize(
    "payload",
    [
        '{"symbol":"000001.SZ","quantity":"2","side":"BUY",'
        '"observed_at":"2026-09-08T15:10:00+08:00","notional":"123.45"}',
        '{"symbol":"000001.SZ","quantity":true,"side":"BUY",'
        '"observed_at":"2026-09-08T15:10:00+08:00","notional":"123.45"}',
        '{"symbol":"000001.SZ","quantity":2,"side":"buy",'
        '"observed_at":"2026-09-08T15:10:00+08:00","notional":"123.45"}',
        '{"symbol":"000001.SZ","quantity":2,"side":"BUY",'
        '"observed_at":"not-a-time","notional":"123.45"}',
        '{"symbol":"000001.SZ","quantity":2,"side":"BUY",'
        '"observed_at":"2026-09-08T15:10:00+08:00","notional":"bad"}',
        '{"symbol":"000001.SZ","quantity":2,"side":"BUY",'
        '"observed_at":"2026-09-08T15:10:00+08:00","notional":"123.45",'
        '"unexpected":true}',
        "[]",
        "not-json",
    ],
)
def test_argument_validation_rejects_coercion_extras_wrong_shapes_and_bad_json(
    payload: str,
) -> None:
    with pytest.raises(ValidationError):
        _tool().validate_arguments(payload)


def test_argument_validation_requires_json_text_not_a_python_mapping() -> None:
    with pytest.raises(ValidationError):
        _tool().validate_arguments(  # type: ignore[arg-type]
            {
                "symbol": "000001.SZ",
                "quantity": 2,
                "side": "BUY",
                "observed_at": "2026-09-08T15:10:00+08:00",
                "notional": "123.45",
            }
        )


def test_output_validation_requires_exact_registered_type_and_revalidates_fields() -> None:
    tool = _tool()
    output = ExampleOutput(
        accepted_quantity=2,
        side=TradeSide.BUY,
        processed_at=AS_OF,
        notional=Decimal("123.45"),
    )

    validated = tool.validate_output_data(output)

    assert validated == output
    assert validated is not output
    assert isinstance(validated.side, TradeSide)
    assert isinstance(validated.processed_at, datetime)
    assert isinstance(validated.notional, Decimal)

    class DerivedOutput(ExampleOutput):
        pass

    class OtherOutput(ToolOutput):
        accepted_quantity: int

    wrong_values: tuple[object, ...] = (
        output.model_dump(),
        DerivedOutput(**output.model_dump()),
        OtherOutput(accepted_quantity=2),
        None,
    )
    for value in wrong_values:
        with pytest.raises(ToolOutputRejected, match="unsupported type"):
            tool.validate_output_data(value)


def test_output_validation_rejects_a_forged_or_post_construction_mutated_model() -> None:
    tool = _tool()
    forged = ExampleOutput.model_construct(
        accepted_quantity="2",
        side=TradeSide.BUY,
        processed_at=AS_OF,
        notional=Decimal("123.45"),
    )
    mutated = ExampleOutput(
        accepted_quantity=2,
        side=TradeSide.BUY,
        processed_at=AS_OF,
        notional=Decimal("123.45"),
    )
    object.__setattr__(mutated, "side", "BUY")

    for value in (forged, mutated):
        with pytest.raises(ToolOutputRejected, match="strict validation"):
            tool.validate_output_data(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "SubmitExample"),
        ("name", " submit_example"),
        ("name", "a" * 65),
        ("version", ""),
        ("version", "1/2"),
        ("version", "v" * 65),
        ("version", "latest"),
        ("version", "LATEST"),
        ("version", "default"),
        ("version", "current"),
        ("description", ""),
        ("description", " surrounding whitespace "),
        ("description", "line\nbreak"),
        ("description", "d" * 501),
        ("description", 1),
    ],
)
def test_agent_tool_rejects_invalid_identity_and_description(field: str, value: object) -> None:
    with pytest.raises(ToolRegistrationError, match="invalid Agent tool definition") as caught:
        _tool(**{field: value})

    assert isinstance(caught.value.__cause__, ValueError)


def test_agent_tool_accepts_bounded_immutable_versions_and_descriptions() -> None:
    assert _tool(version="1").version == "1"
    assert _tool(version="V" * 64).version == "V" * 64
    assert _tool(description="d" * 500).description == "d" * 500


def test_agent_tool_rejects_non_contract_or_weakened_argument_models() -> None:
    class PlainModel(BaseModel):
        quantity: int

    class ExtraArguments(ToolArguments):
        model_config = ConfigDict(extra="allow", frozen=True, strict=True)
        quantity: int

    class MutableArguments(ToolArguments):
        model_config = ConfigDict(extra="forbid", frozen=False, strict=True)
        quantity: int

    class CoercingArguments(ToolArguments):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=False)
        quantity: int

    for arguments_type in (object, PlainModel, ExtraArguments, MutableArguments, CoercingArguments):
        with pytest.raises(ToolRegistrationError, match="invalid Agent tool definition"):
            _tool(arguments_type=arguments_type)


def test_agent_tool_rejects_non_contract_or_weakened_output_models() -> None:
    class PlainModel(BaseModel):
        accepted_quantity: int

    class ExtraOutput(ToolOutput):
        model_config = ConfigDict(
            extra="allow",
            frozen=True,
            revalidate_instances="always",
            strict=True,
        )
        accepted_quantity: int

    class MutableOutput(ToolOutput):
        model_config = ConfigDict(
            extra="forbid",
            frozen=False,
            revalidate_instances="always",
            strict=True,
        )
        accepted_quantity: int

    class CoercingOutput(ToolOutput):
        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            revalidate_instances="always",
            strict=False,
        )
        accepted_quantity: int

    class NonRevalidatingOutput(ToolOutput):
        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            revalidate_instances="never",
            strict=True,
        )
        accepted_quantity: int

    output_types = (
        object,
        PlainModel,
        ExtraOutput,
        MutableOutput,
        CoercingOutput,
        NonRevalidatingOutput,
    )
    for output_type in output_types:
        with pytest.raises(ToolRegistrationError, match="invalid Agent tool definition"):
            _tool(output_type=output_type)


def test_agent_tool_rejects_non_callable_and_coroutine_handlers_at_registration() -> None:
    async def async_handler(
        _context: ToolExecutionContext,
        _arguments: ExampleArguments,
    ) -> ExampleOutput:
        return ExampleOutput(
            accepted_quantity=1,
            side=TradeSide.BUY,
            processed_at=AS_OF,
            notional=Decimal("1"),
        )

    for handler in (None, 1, async_handler):
        with pytest.raises(ToolRegistrationError, match="invalid Agent tool definition"):
            _tool(handler=handler)


def test_agent_tool_is_a_frozen_slotted_definition() -> None:
    tool = _tool()

    assert not hasattr(tool, "__dict__")
    with pytest.raises(FrozenInstanceError):
        tool.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tool.output_type = ToolOutput  # type: ignore[misc]


def test_agent_tool_invokes_a_synchronous_handler_and_rejects_coroutine_results() -> None:
    context = _execution_context()
    arguments = ExampleArguments(
        symbol="000001.SZ",
        quantity=2,
        side=TradeSide.BUY,
        observed_at=AS_OF,
        notional=Decimal("123.45"),
    )

    assert _tool().invoke(context, arguments) == _handler(context, arguments)

    created: list[Any] = []

    async def deferred() -> ExampleOutput:
        return _handler(context, arguments)

    def returns_coroutine(
        _context: ToolExecutionContext,
        _arguments: ExampleArguments,
    ) -> object:
        result = deferred()
        created.append(result)
        return result

    with pytest.raises(ToolExecutionFailed, match="async handler results"):
        _tool(handler=returns_coroutine).invoke(context, arguments)

    assert len(created) == 1
    assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED


def test_agent_tool_rejects_non_coroutine_awaitable_results() -> None:
    class AwaitableResult:
        def __await__(self) -> Any:
            yield None
            return None

    def returns_awaitable(
        _context: ToolExecutionContext,
        _arguments: ExampleArguments,
    ) -> object:
        return AwaitableResult()

    with pytest.raises(ToolExecutionFailed, match="async handler results"):
        _tool(handler=returns_awaitable).invoke(
            _execution_context(),
            ExampleArguments(
                symbol="000001.SZ",
                quantity=2,
                side=TradeSide.BUY,
                observed_at=AS_OF,
                notional=Decimal("123.45"),
            ),
        )


def test_schemas_and_descriptor_are_detached_and_hide_handler_internals() -> None:
    tool = _tool()

    first = tool.descriptor(
        category=ToolCategory.EXECUTION,
        capability=ToolCapability.PAPER_ORDER_SUBMIT,
        effect=ToolEffect.PAPER_EXECUTION_WRITE,
    )
    first.argument_schema["poison"] = True
    first.output_schema["poison"] = True
    properties = cast(dict[str, object], first.argument_schema["properties"])
    properties["symbol"] = {"type": "number"}

    second = tool.descriptor(
        category=ToolCategory.EXECUTION,
        capability=ToolCapability.PAPER_ORDER_SUBMIT,
        effect=ToolEffect.PAPER_EXECUTION_WRITE,
    )

    assert second.name == tool.name
    assert second.version == tool.version
    assert second.description == tool.description
    assert second.argument_schema == tool.argument_schema()
    assert second.output_schema == tool.output_schema()
    assert second.argument_schema["additionalProperties"] is False
    assert second.output_schema["additionalProperties"] is False
    assert "poison" not in second.argument_schema
    assert "poison" not in second.output_schema
    assert cast(dict[str, Any], second.argument_schema["properties"])["symbol"]["type"] == (
        "string"
    )
    assert set(second.model_dump()) == {
        "argument_schema",
        "capability",
        "category",
        "description",
        "effect",
        "name",
        "output_schema",
        "version",
    }


def test_tool_descriptor_is_strict_closed_and_frozen() -> None:
    descriptor = _tool().descriptor(
        category=ToolCategory.DATA,
        capability=ToolCapability.MARKET_DATA_READ,
        effect=ToolEffect.READ_ONLY,
    )

    with pytest.raises(ValidationError, match="frozen"):
        descriptor.name = "changed"
    with pytest.raises(ValidationError):
        ToolDescriptor.model_validate({**descriptor.model_dump(), "category": "DATA"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ToolDescriptor.model_validate({**descriptor.model_dump(), "handler": "hidden"})


def test_tool_freezes_both_schemas_and_adapters_against_model_rebuild_drift() -> None:
    class RebuildArguments(ToolArguments):
        quantity: int

    class RebuildOutput(ToolOutput):
        accepted_quantity: int

    def rebuild_handler(
        _context: ToolExecutionContext,
        arguments: RebuildArguments,
    ) -> RebuildOutput:
        return RebuildOutput(accepted_quantity=arguments.quantity)

    tool = AgentTool(
        name="rebuild_test",
        version="1",
        description="Freeze schemas and adapters at registration.",
        arguments_type=RebuildArguments,
        output_type=RebuildOutput,
        handler=rebuild_handler,
    )
    frozen_argument_schema = tool.argument_schema()
    frozen_output_schema = tool.output_schema()

    RebuildArguments.model_fields["quantity"].annotation = str
    RebuildOutput.model_fields["accepted_quantity"].annotation = str
    assert RebuildArguments.model_rebuild(force=True) is True
    assert RebuildOutput.model_rebuild(force=True) is True

    assert RebuildArguments.model_json_schema() != frozen_argument_schema
    assert RebuildOutput.model_json_schema() != frozen_output_schema
    assert tool.argument_schema() == frozen_argument_schema
    assert tool.output_schema() == frozen_output_schema
    assert tool.validate_arguments('{"quantity":2}').quantity == 2
    with pytest.raises(ValidationError):
        tool.validate_arguments('{"quantity":"2"}')
    assert (
        tool.validate_output_data(
            RebuildOutput.model_construct(accepted_quantity=2)
        ).accepted_quantity
        == 2
    )
    with pytest.raises(ToolOutputRejected, match="strict validation"):
        tool.validate_output_data(RebuildOutput.model_construct(accepted_quantity="2"))

    descriptor = tool.descriptor(
        category=ToolCategory.VALIDATION,
        capability=ToolCapability.MARKET_DATA_VALIDATE,
        effect=ToolEffect.READ_ONLY,
    )
    assert descriptor.argument_schema == frozen_argument_schema
    assert descriptor.output_schema == frozen_output_schema


def test_tool_call_context_contains_only_transport_identity_and_normalizes_utc() -> None:
    context = ToolCallContext(
        request_id="req_contract_test",
        decision_id="dec_contract_test",
        requested_at=REQUESTED_AT,
    )

    assert context.requested_at == REQUESTED_AT.astimezone(UTC)
    assert context.requested_at.tzinfo is UTC
    assert set(context.model_dump()) == {"request_id", "decision_id", "requested_at"}
    assert set(ToolCallContext.model_fields) == {"request_id", "decision_id", "requested_at"}
    for forbidden in ("mode", "account_id", "as_of", "approval_token"):
        assert forbidden not in context.model_dump()


@pytest.mark.parametrize("field", ["request_id", "decision_id"])
@pytest.mark.parametrize("value", ["", " ", " padded", "padded ", "line\nbreak", "x" * 129])
def test_tool_call_context_rejects_non_exact_identities(field: str, value: str) -> None:
    values: dict[str, object] = {
        "request_id": "req_contract_test",
        "decision_id": "dec_contract_test",
        "requested_at": REQUESTED_AT,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        ToolCallContext.model_validate(values)


def test_tool_call_context_is_strict_closed_frozen_and_timezone_aware() -> None:
    context = ToolCallContext(
        request_id="req_contract_test",
        decision_id="dec_contract_test",
        requested_at=REQUESTED_AT,
    )

    with pytest.raises(ValidationError, match="frozen"):
        context.request_id = "req_changed"
    with pytest.raises(ValidationError):
        ToolCallContext(
            request_id=1,  # type: ignore[arg-type]
            decision_id="dec_contract_test",
            requested_at=REQUESTED_AT,
        )
    with pytest.raises(ValidationError):
        ToolCallContext(
            request_id="req_contract_test",
            decision_id="dec_contract_test",
            requested_at=REQUESTED_AT.isoformat(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="timezone"):
        ToolCallContext(
            request_id="req_contract_test",
            decision_id="dec_contract_test",
            requested_at=REQUESTED_AT.replace(tzinfo=None),
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ToolCallContext.model_validate(
            {
                **context.model_dump(),
                "mode": RuntimeMode.PAPER,
                "account_id": "paper-account-1",
            }
        )


def test_execution_context_is_strict_closed_frozen_and_normalizes_utc() -> None:
    context = _execution_context()

    assert context.requested_at == REQUESTED_AT.astimezone(UTC)
    assert context.requested_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="frozen"):
        context.principal_id = "changed"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ToolExecutionContext.model_validate({**context.model_dump(), "mode": "PAPER"})
    with pytest.raises(ValidationError):
        ToolExecutionContext(
            request_id=" req_contract_test",
            principal_id="agent:test",
            requested_at=REQUESTED_AT,
            decision_snapshot=_snapshot(),
        )
    with pytest.raises(ValueError, match="timezone"):
        ToolExecutionContext(
            request_id="req_contract_test",
            principal_id="agent:test",
            requested_at=REQUESTED_AT.replace(tzinfo=None),
            decision_snapshot=_snapshot(),
        )


def test_validate_tool_response_accepts_only_the_exact_authorized_decision_binding() -> None:
    context = _execution_context()
    response = _response(as_of=AS_OF.astimezone(UTC))

    validated = validate_tool_response(response, context=context)

    assert validated == response
    assert validated is not response
    assert validated.provenance is not response.provenance
    assert validated.as_of.tzinfo is UTC


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "req_other"},
        {"decision_id": "dec_other"},
        {"as_of": AS_OF + timedelta(microseconds=1)},
        {
            "provenance": Provenance(
                service="contract-test",
                version="1.0.0",
                data_version="market_20260908_eod_v2",
            )
        },
    ],
)
def test_validate_tool_response_rejects_each_context_mismatch(changes: dict[str, object]) -> None:
    with pytest.raises(ToolOutputRejected, match="authorized decision context"):
        validate_tool_response(_response(**changes), context=_execution_context())


def test_validate_tool_response_rejects_wrong_types_and_inconsistent_status() -> None:
    context = _execution_context()

    with pytest.raises(ToolOutputRejected, match="unsupported response type"):
        validate_tool_response({"ok": True}, context=context)
    with pytest.raises(ToolOutputRejected, match="inconsistent response"):
        validate_tool_response(
            _response(errors=(ToolIssue(code=ErrorCode.INTERNAL_ERROR, message="unexpected"),)),
            context=context,
        )
    with pytest.raises(ToolOutputRejected, match="inconsistent response"):
        validate_tool_response(_response(ok=False, errors=()), context=context)
    with pytest.raises(ToolOutputRejected, match="malformed response fields"):
        validate_tool_response(
            _response(
                provenance=Provenance(
                    service="contract-test",
                    version="1.0.0",
                    data_version=None,
                )
            ),
            context=context,
        )


def test_invocation_denial_exposes_only_stable_structured_metadata() -> None:
    denial = ToolPermissionDenied(
        ToolDenialReason.CAPABILITY_NOT_GRANTED,
        mode=RuntimeMode.RESEARCH,
        tool_name="submit_paper_orders",
        validation_error_codes=("int_type",),
    )

    assert str(denial) == "tool invocation denied: CAPABILITY_NOT_GRANTED"
    assert denial.reason is ToolDenialReason.CAPABILITY_NOT_GRANTED
    assert denial.mode is RuntimeMode.RESEARCH
    assert denial.tool_name == "submit_paper_orders"
    assert denial.validation_error_codes == ("int_type",)
