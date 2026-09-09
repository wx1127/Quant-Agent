"""Fail-closed registration, discovery, and invocation of Agent-visible tools."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType
from typing import Any, NoReturn, Self, cast

from pydantic import ValidationError

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.config import RuntimeMode
from quant_agent.observability.audit import AuditEvent, AuditSink

from .contracts import (
    AgentTool,
    LiveExecutionAuthorizer,
    ToolArguments,
    ToolArgumentsRejected,
    ToolAuditUnavailable,
    ToolCallContext,
    ToolCapability,
    ToolDenialReason,
    ToolDescriptor,
    ToolExecutionContext,
    ToolExecutionFailed,
    ToolInvocationDenied,
    ToolNotRegistered,
    ToolOutputRejected,
    ToolPermissionDenied,
    ToolRegistrationError,
    ToolRegistryError,
    VerifiedDecisionResolver,
    validate_tool_name,
    validate_tool_response,
)
from .policy import (
    HUMAN_ONLY_TOOL_NAMES,
    MODE_TOOL_PERMISSIONS,
    TOOL_POLICY_VERSION,
    V1_TOOL_POLICIES,
    ToolPolicy,
)

_BUILD_TOKEN = object()
_MAX_ARGUMENT_BYTES = 65_536
_MAX_ARGUMENT_DEPTH = 12
_MAX_ARGUMENT_NODES = 10_000
_MAX_OBJECT_FIELDS = 256
_MAX_ARRAY_ITEMS = 1_000
_MAX_STRING_LENGTH = 10_000
_MAX_KEY_LENGTH = 128
_MAX_INTEGER_MAGNITUDE = (1 << 63) - 1
_CONTEXT_ARGUMENT_FIELDS = frozenset(
    {
        "account_id",
        "as_of",
        "data_version",
        "decision_id",
        "mode",
        "principal_id",
        "request_id",
        "requested_at",
        "runtime_mode",
    }
)
_SECRET_FIELD_FRAGMENTS = ("credential", "password", "secret")


class _PayloadError(ValueError):
    """Internal argument error carrying only a stable, non-sensitive code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _validate_principal(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 256
        or not value.isprintable()
    ):
        raise ToolRegistrationError("principal_id must be a bounded exact string")
    return value


def _validate_grants(capabilities: frozenset[ToolCapability], account_ids: frozenset[str]) -> None:
    if type(capabilities) is not frozenset or any(
        type(item) is not ToolCapability for item in capabilities
    ):
        raise ToolRegistrationError("granted_capabilities must be an exact capability frozenset")
    if type(account_ids) is not frozenset:
        raise ToolRegistrationError("granted_account_ids must be an exact string frozenset")
    for account_id in account_ids:
        if (
            type(account_id) is not str
            or not account_id
            or account_id != account_id.strip()
            or len(account_id) > 256
            or not account_id.isprintable()
        ):
            raise ToolRegistrationError("granted account IDs must be bounded exact strings")


def _schema_property_names(value: object) -> tuple[str, ...]:
    names: list[str] = []

    def visit(node: object) -> None:
        if type(node) is dict:
            mapping = node
            properties = mapping.get("properties")
            if type(properties) is dict:
                for key in properties:
                    if type(key) is str:
                        names.append(key)
            for child in mapping.values():
                visit(child)
        elif type(node) is list:
            for child in node:
                visit(child)

    visit(value)
    return tuple(names)


def _forbidden_schema_field(name: str, *, model_controlled: bool) -> bool:
    lowered = name.lower()
    secret_like = (
        lowered in {"api_key", "token"}
        or lowered.endswith(("_api_key", "_token"))
        or any(fragment in lowered for fragment in _SECRET_FIELD_FRAGMENTS)
    )
    return secret_like or (model_controlled and "approval" in lowered)


def _schema_is_constrained(node: object) -> bool:
    if type(node) is not dict:
        return False
    mapping = cast(dict[object, object], node)
    if any(key in mapping for key in ("$ref", "const", "enum", "type")):
        return True
    for combinator in ("allOf", "anyOf", "oneOf"):
        branches = mapping.get(combinator)
        if type(branches) is list and branches:
            return all(_schema_is_constrained(branch) for branch in branches)
    return False


def _validate_closed_schema(schema: dict[str, Any], *, model_controlled: bool) -> None:
    for name in _schema_property_names(schema):
        try:
            validate_tool_name(name)
        except (TypeError, ValueError):
            raise ToolRegistrationError(
                "tool schema fields and aliases must be lower-ASCII snake case"
            ) from None
        if (model_controlled and name in _CONTEXT_ARGUMENT_FIELDS) or _forbidden_schema_field(
            name, model_controlled=model_controlled
        ):
            raise ToolRegistrationError(
                "tool schema exposes a trusted-context, approval, or secret field"
            )

    def require_closed_models(node: object) -> None:
        if type(node) is dict:
            mapping = cast(dict[object, object], node)
            properties = mapping.get("properties")
            if type(properties) is dict:
                for property_schema in properties.values():
                    if not _schema_is_constrained(property_schema):
                        raise ToolRegistrationError(
                            "tool schema fields cannot use Any or unconstrained object types"
                        )
            if (mapping.get("type") == "object" or "properties" in mapping) and mapping.get(
                "additionalProperties"
            ) is not False:
                raise ToolRegistrationError("tool schemas cannot contain open or free-form objects")
            if mapping.get("type") == "array":
                items = mapping.get("items")
                prefix_items = mapping.get("prefixItems")
                constrained_items = _schema_is_constrained(items) or (
                    type(prefix_items) is list
                    and bool(prefix_items)
                    and all(_schema_is_constrained(item) for item in prefix_items)
                )
                if not constrained_items:
                    raise ToolRegistrationError("tool array items cannot be unconstrained")
            for combinator in ("allOf", "anyOf", "oneOf"):
                branches = mapping.get(combinator)
                if type(branches) is list and (
                    not branches or any(not _schema_is_constrained(branch) for branch in branches)
                ):
                    raise ToolRegistrationError(
                        "every tool schema union branch must be constrained"
                    )
            for child in mapping.values():
                require_closed_models(child)
        elif type(node) is list:
            for child in cast(list[object], node):
                require_closed_models(child)

    require_closed_models(schema)


def _validate_model_visible_schema(tool: AgentTool[Any, Any], policy: ToolPolicy) -> None:
    fields = tool.arguments_type.model_fields
    if policy.requires_idempotency_key:
        idempotency = fields.get("idempotency_key")
        if (
            idempotency is None
            or idempotency.annotation is not str
            or not idempotency.is_required()
        ):
            raise ToolRegistrationError(
                "write tools require a required exact-string idempotency_key field"
            )

    _validate_closed_schema(tool.argument_schema(), model_controlled=True)
    _validate_closed_schema(tool.output_schema(), model_controlled=False)


def _reject_json_constant(_: str) -> NoReturn:
    raise _PayloadError("non_finite_number")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _PayloadError("duplicate_json_key")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    ancestors: set[int] = set()
    nodes = 0

    def visit(node: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_ARGUMENT_NODES:
            raise _PayloadError("payload_too_complex")
        if depth > _MAX_ARGUMENT_DEPTH:
            raise _PayloadError("payload_too_deep")

        if node is None or type(node) is bool:
            return
        if type(node) is str:
            if len(node) > _MAX_STRING_LENGTH:
                raise _PayloadError("string_too_long")
            return
        if type(node) is int:
            if abs(node) > _MAX_INTEGER_MAGNITUDE:
                raise _PayloadError("integer_out_of_range")
            return
        if type(node) is float:
            if not math.isfinite(node):
                raise _PayloadError("non_finite_number")
            return
        if type(node) not in {dict, list}:
            raise _PayloadError("non_json_value")

        identity = id(node)
        if identity in ancestors:
            raise _PayloadError("cyclic_value")
        ancestors.add(identity)
        try:
            if type(node) is dict:
                mapping = cast(dict[object, object], node)
                if len(mapping) > _MAX_OBJECT_FIELDS:
                    raise _PayloadError("object_too_large")
                for key, child in mapping.items():
                    if (
                        type(key) is not str
                        or not key
                        or len(key) > _MAX_KEY_LENGTH
                        or not key.isprintable()
                    ):
                        raise _PayloadError("invalid_object_key")
                    visit(child, depth + 1)
            else:
                items = cast(list[object], node)
                if len(items) > _MAX_ARRAY_ITEMS:
                    raise _PayloadError("array_too_large")
                for child in items:
                    visit(child, depth + 1)
        finally:
            ancestors.remove(identity)

    visit(value, 0)


def _parse_arguments(raw: str | dict[str, object]) -> tuple[str, str]:
    try:
        if type(raw) is str:
            encoded = raw.encode("utf-8")
            if len(encoded) > _MAX_ARGUMENT_BYTES:
                raise _PayloadError("payload_too_large")
            payload = json.loads(
                raw,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        elif type(raw) is dict:
            payload = raw
        else:
            raise _PayloadError("top_level_object_required")

        if type(payload) is not dict:
            raise _PayloadError("top_level_object_required")
        _validate_json_tree(payload)
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(canonical.encode("utf-8")) > _MAX_ARGUMENT_BYTES:
            raise _PayloadError("payload_too_large")
        detached = json.loads(
            canonical,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except _PayloadError:
        raise
    except (json.JSONDecodeError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
        raise _PayloadError("invalid_json") from error

    if type(detached) is not dict:  # pragma: no cover - canonical object invariant
        raise _PayloadError("top_level_object_required")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical, digest


def _untrusted_name_digest(value: object) -> tuple[str, int]:
    if type(value) is str:
        raw = value.encode("utf-8", errors="surrogatepass")
    else:
        raw = f"<{type(value).__module__}.{type(value).__qualname__}>".encode()
    return hashlib.sha256(raw).hexdigest(), len(raw)


class AgentToolRegistryBuilder:
    """Trusted composition-only builder that seals permanently after one build."""

    def __init__(
        self,
        *,
        runtime_mode: RuntimeMode,
        principal_id: str,
        granted_capabilities: frozenset[ToolCapability],
        granted_account_ids: frozenset[str],
        decision_resolver: VerifiedDecisionResolver,
        audit_sink: AuditSink,
        live_execution_authorizer: LiveExecutionAuthorizer | None = None,
    ) -> None:
        if type(runtime_mode) is not RuntimeMode or runtime_mode is RuntimeMode.LIVE_AUTO:
            raise ToolRegistrationError("registry runtime mode must be an enabled RuntimeMode")
        _validate_principal(principal_id)
        _validate_grants(granted_capabilities, granted_account_ids)
        if not callable(getattr(decision_resolver, "resolve", None)):
            raise ToolRegistrationError("decision_resolver must implement resolve")
        if not callable(getattr(audit_sink, "append", None)):
            raise ToolRegistrationError("audit_sink must implement append")
        if live_execution_authorizer is not None and not callable(
            getattr(live_execution_authorizer, "authorization", None)
        ):
            raise ToolRegistrationError("live_execution_authorizer must implement authorization")

        self._runtime_mode = runtime_mode
        self._principal_id = principal_id
        self._granted_capabilities = granted_capabilities
        self._granted_account_ids = granted_account_ids
        self._decision_resolver = decision_resolver
        self._audit_sink = audit_sink
        self._live_execution_authorizer = live_execution_authorizer
        self._tools: dict[str, AgentTool[Any, Any]] = {}
        self._sealed = False
        self._lock = RLock()

    def register(self, tool: AgentTool[Any, Any]) -> Self:
        """Bind one explicitly constructed handler to an existing central policy."""

        with self._lock:
            if self._sealed:
                raise ToolRegistrationError("tool registry builder is already sealed")
            if not isinstance(tool, AgentTool):
                raise ToolRegistrationError("registry accepts only AgentTool definitions")
            if tool.name in HUMAN_ONLY_TOOL_NAMES:
                raise ToolRegistrationError("human-only operations cannot be Agent tools")
            policy = V1_TOOL_POLICIES.get(tool.name)
            if policy is None:
                raise ToolRegistrationError("tool has no central v1 policy")
            if tool.name in self._tools:
                raise ToolRegistrationError("duplicate Agent tool registration")
            _validate_model_visible_schema(tool, policy)
            self._tools[tool.name] = tool
        return self

    def build(self) -> AgentToolRegistry:
        """Seal the builder and return the immutable invocation surface."""

        with self._lock:
            if self._sealed:
                raise ToolRegistrationError("tool registry builder is already sealed")
            self._sealed = True
            tools = MappingProxyType(dict(self._tools))
        return AgentToolRegistry(
            _build_token=_BUILD_TOKEN,
            runtime_mode=self._runtime_mode,
            principal_id=self._principal_id,
            granted_capabilities=self._granted_capabilities,
            granted_account_ids=self._granted_account_ids,
            decision_resolver=self._decision_resolver,
            audit_sink=self._audit_sink,
            live_execution_authorizer=self._live_execution_authorizer,
            tools=tools,
        )


class AgentToolRegistry:
    """Frozen, default-deny Agent discovery and invocation boundary."""

    __slots__ = (
        "_audit_lock",
        "_audit_sink",
        "_decision_resolver",
        "_granted_account_ids",
        "_granted_capabilities",
        "_live_execution_authorizer",
        "_principal_id",
        "_runtime_mode",
        "_tools",
    )
    _runtime_mode: RuntimeMode
    _principal_id: str
    _granted_capabilities: frozenset[ToolCapability]
    _granted_account_ids: frozenset[str]
    _decision_resolver: VerifiedDecisionResolver
    _audit_sink: AuditSink
    _audit_lock: RLock
    _live_execution_authorizer: LiveExecutionAuthorizer | None
    _tools: Mapping[str, AgentTool[Any, Any]]

    def __init__(
        self,
        *,
        _build_token: object,
        runtime_mode: RuntimeMode,
        principal_id: str,
        granted_capabilities: frozenset[ToolCapability],
        granted_account_ids: frozenset[str],
        decision_resolver: VerifiedDecisionResolver,
        audit_sink: AuditSink,
        live_execution_authorizer: LiveExecutionAuthorizer | None,
        tools: Mapping[str, AgentTool[Any, Any]],
    ) -> None:
        if _build_token is not _BUILD_TOKEN:
            raise ToolRegistrationError("AgentToolRegistry must be created by its builder")
        object.__setattr__(self, "_runtime_mode", runtime_mode)
        object.__setattr__(self, "_principal_id", principal_id)
        object.__setattr__(self, "_granted_capabilities", granted_capabilities)
        object.__setattr__(self, "_granted_account_ids", granted_account_ids)
        object.__setattr__(self, "_decision_resolver", decision_resolver)
        object.__setattr__(self, "_audit_sink", audit_sink)
        object.__setattr__(self, "_audit_lock", RLock())
        object.__setattr__(self, "_live_execution_authorizer", live_execution_authorizer)
        object.__setattr__(self, "_tools", tools)

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Prevent runtime mutation after trusted construction."""

        raise AttributeError("AgentToolRegistry is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        """Prevent runtime removal after trusted construction."""

        raise AttributeError("AgentToolRegistry is immutable")

    @property
    def runtime_mode(self) -> RuntimeMode:
        """Return the server-fixed mode for diagnostics."""

        return self._runtime_mode

    @property
    def registered_tool_names(self) -> tuple[str, ...]:
        """Return a stable trusted-diagnostics view, not the Agent catalog."""

        return tuple(sorted(self._tools))

    def catalog(self, context: ToolCallContext) -> tuple[ToolDescriptor, ...]:
        """List only registered tools statically authorized for this exact decision."""

        execution_context = self._resolve_context(context, requested_tool_name=None)
        descriptors: list[ToolDescriptor] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            policy = V1_TOOL_POLICIES[name]
            if self._static_denial(policy, execution_context) is not None:
                continue
            descriptors.append(
                tool.descriptor(
                    category=policy.category,
                    capability=policy.capability,
                    effect=policy.effect,
                )
            )
        return tuple(descriptors)

    def invoke(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, object],
        context: ToolCallContext,
    ) -> object:
        """Authorize, audit, strictly validate, execute, and bind one tool result."""

        context = self._validate_call_context(context)
        try:
            validate_tool_name(tool_name)
        except (TypeError, ValueError):
            self._deny(
                ToolNotRegistered(
                    ToolDenialReason.INVALID_TOOL_NAME,
                    mode=self._runtime_mode,
                ),
                call_context=context,
                requested_tool_name=tool_name,
            )
        if tool_name in HUMAN_ONLY_TOOL_NAMES:
            self._deny(
                ToolNotRegistered(
                    ToolDenialReason.RESERVED_HUMAN_TOOL,
                    mode=self._runtime_mode,
                ),
                call_context=context,
                requested_tool_name=tool_name,
            )

        policy = V1_TOOL_POLICIES.get(tool_name)
        tool = self._tools.get(tool_name)
        if policy is None or tool is None:
            self._deny(
                ToolNotRegistered(
                    ToolDenialReason.TOOL_NOT_REGISTERED,
                    mode=self._runtime_mode,
                ),
                call_context=context,
                requested_tool_name=tool_name,
            )

        pre_snapshot_reason = self._pre_snapshot_denial(policy)
        if pre_snapshot_reason is not None:
            self._deny(
                ToolPermissionDenied(
                    pre_snapshot_reason,
                    mode=self._runtime_mode,
                    tool_name=tool_name,
                ),
                call_context=context,
                requested_tool_name=tool_name,
                policy=policy,
            )

        execution_context = self._resolve_context(
            context,
            requested_tool_name=tool_name,
            policy=policy,
        )
        static_reason = self._static_denial(policy, execution_context)
        if static_reason is not None:
            self._deny(
                ToolPermissionDenied(
                    static_reason,
                    mode=self._runtime_mode,
                    tool_name=tool_name,
                ),
                call_context=context,
                requested_tool_name=tool_name,
                execution_context=execution_context,
                policy=policy,
            )

        payload_error_code: str | None = None
        try:
            canonical_arguments, argument_digest = _parse_arguments(raw_arguments)
        except _PayloadError as error:
            payload_error_code = error.code
        if payload_error_code is not None:
            self._deny(
                ToolArgumentsRejected(
                    ToolDenialReason.INVALID_ARGUMENTS,
                    mode=self._runtime_mode,
                    tool_name=tool_name,
                    validation_error_codes=(payload_error_code,),
                ),
                call_context=context,
                requested_tool_name=tool_name,
                execution_context=execution_context,
                policy=policy,
                validation_error_codes=(payload_error_code,),
            )

        validation_codes: tuple[str, ...] | None = None
        try:
            arguments = tool.validate_arguments(canonical_arguments)
        except ValidationError as error:
            validation_codes = tuple(
                sorted(
                    {
                        str(item["type"])
                        for item in error.errors(include_input=False, include_url=False)
                    }
                )
            )
        except Exception:
            validation_codes = ("validator_failed",)
        if validation_codes is not None:
            self._deny(
                ToolArgumentsRejected(
                    ToolDenialReason.INVALID_ARGUMENTS,
                    mode=self._runtime_mode,
                    tool_name=tool_name,
                    validation_error_codes=validation_codes,
                ),
                call_context=context,
                requested_tool_name=tool_name,
                execution_context=execution_context,
                policy=policy,
                argument_digest=argument_digest,
                validation_error_codes=validation_codes,
            )

        if policy.requires_idempotency_key:
            idempotency_key = getattr(arguments, "idempotency_key", None)
            if (
                type(idempotency_key) is not str
                or not idempotency_key
                or idempotency_key != idempotency_key.strip()
                or len(idempotency_key) > 128
                or not idempotency_key.isprintable()
            ):
                self._deny(
                    ToolArgumentsRejected(
                        ToolDenialReason.IDEMPOTENCY_KEY_REQUIRED,
                        mode=self._runtime_mode,
                        tool_name=tool_name,
                    ),
                    call_context=context,
                    requested_tool_name=tool_name,
                    execution_context=execution_context,
                    policy=policy,
                    argument_digest=argument_digest,
                )

        if policy.requires_external_authorization:
            result = self._invoke_live_authorized(
                tool=tool,
                policy=policy,
                arguments=arguments,
                call_context=context,
                execution_context=execution_context,
                argument_digest=argument_digest,
            )
        else:
            self._append_authorization_event(
                result="ALLOWED",
                call_context=context,
                requested_tool_name=tool_name,
                execution_context=execution_context,
                policy=policy,
                argument_digest=argument_digest,
            )
            handler_failed = False
            try:
                result = tool.invoke(execution_context, arguments)
            except Exception:
                handler_failed = True
            if handler_failed:
                raise ToolExecutionFailed("registered tool execution failed") from None
        try:
            response = validate_tool_response(result, context=execution_context)
            if response.ok and response.data is None:
                raise ToolOutputRejected("successful tool response must contain typed data")
            if response.data is not None:
                validated_data = tool.validate_output_data(response.data)
                response = response.model_copy(update={"data": validated_data})
            return response
        except ToolOutputRejected:
            raise
        except Exception:
            pass
        raise ToolOutputRejected("registered tool output validation failed") from None

    def _invoke_live_authorized(
        self,
        *,
        tool: AgentTool[Any, Any],
        policy: ToolPolicy,
        arguments: ToolArguments,
        call_context: ToolCallContext,
        execution_context: ToolExecutionContext,
        argument_digest: str,
    ) -> object:
        """Keep the live authorization lease held across audit and execution."""

        authorizer = self._live_execution_authorizer
        if authorizer is None:  # pragma: no cover - static permission guard
            self._deny(
                ToolPermissionDenied(
                    ToolDenialReason.EXTERNAL_AUTHORIZATION_REQUIRED,
                    mode=self._runtime_mode,
                    tool_name=policy.name,
                ),
                call_context=call_context,
                requested_tool_name=policy.name,
                execution_context=execution_context,
                policy=policy,
                argument_digest=argument_digest,
            )

        handler_started = False
        result_marker = object()
        result: object = result_marker
        audit_failure: ToolAuditUnavailable | None = None
        authorization_failed = False
        execution_failed = False
        try:
            authorization = authorizer.authorization(execution_context, arguments)
            with authorization:
                try:
                    self._append_authorization_event(
                        result="ALLOWED",
                        call_context=call_context,
                        requested_tool_name=policy.name,
                        execution_context=execution_context,
                        policy=policy,
                        argument_digest=argument_digest,
                    )
                except ToolAuditUnavailable as error:
                    audit_failure = error
                    raise
                handler_started = True
                result = tool.invoke(execution_context, arguments)
        except ToolAuditUnavailable:
            if audit_failure is None:
                authorization_failed = not handler_started
                execution_failed = handler_started
        except Exception:
            authorization_failed = not handler_started
            execution_failed = handler_started
        if audit_failure is not None:
            raise audit_failure from None
        if authorization_failed:
            self._deny(
                ToolPermissionDenied(
                    ToolDenialReason.EXTERNAL_AUTHORIZATION_REQUIRED,
                    mode=self._runtime_mode,
                    tool_name=policy.name,
                ),
                call_context=call_context,
                requested_tool_name=policy.name,
                execution_context=execution_context,
                policy=policy,
                argument_digest=argument_digest,
            )
        if execution_failed:
            raise ToolExecutionFailed("registered live tool execution failed") from None
        if not handler_started or result is result_marker:
            raise ToolExecutionFailed("live authorization scope did not execute the handler")
        return result

    def _resolve_context(
        self,
        context: ToolCallContext,
        *,
        requested_tool_name: object,
        policy: ToolPolicy | None = None,
    ) -> ToolExecutionContext:
        detached_context = self._validate_call_context(context)

        resolution_failed = False
        try:
            resolved = self._decision_resolver.resolve(detached_context.decision_id)
            if type(resolved) is not DecisionSnapshot:
                raise TypeError("resolver did not return an exact DecisionSnapshot")
            snapshot = DecisionSnapshot.from_json(resolved.to_json())
        except Exception:
            resolution_failed = True
        if resolution_failed:
            self._deny(
                ToolPermissionDenied(
                    ToolDenialReason.DECISION_UNAVAILABLE,
                    mode=self._runtime_mode,
                ),
                call_context=detached_context,
                requested_tool_name=requested_tool_name,
                policy=policy,
            )

        if snapshot.decision_id != detached_context.decision_id or (
            snapshot.mode is not self._runtime_mode
        ):
            self._deny(
                ToolPermissionDenied(
                    ToolDenialReason.DECISION_CONTEXT_MISMATCH,
                    mode=self._runtime_mode,
                ),
                call_context=detached_context,
                requested_tool_name=requested_tool_name,
                snapshot=snapshot,
                policy=policy,
            )
        if snapshot.as_of > detached_context.requested_at:
            self._deny(
                ToolPermissionDenied(
                    ToolDenialReason.DECISION_FROM_FUTURE,
                    mode=self._runtime_mode,
                ),
                call_context=detached_context,
                requested_tool_name=requested_tool_name,
                snapshot=snapshot,
                policy=policy,
            )
        return ToolExecutionContext(
            request_id=detached_context.request_id,
            principal_id=self._principal_id,
            requested_at=detached_context.requested_at,
            decision_snapshot=snapshot,
        )

    @staticmethod
    def _validate_call_context(context: ToolCallContext) -> ToolCallContext:
        if type(context) is not ToolCallContext:
            raise ToolRegistryError("tool call context must be an exact ToolCallContext")
        validation_failed = False
        try:
            detached = ToolCallContext.model_validate_json(context.model_dump_json())
        except (TypeError, ValueError, ValidationError):
            validation_failed = True
        if validation_failed:
            raise ToolRegistryError("tool call context failed revalidation") from None
        return detached

    def _pre_snapshot_denial(self, policy: ToolPolicy) -> ToolDenialReason | None:
        if (
            policy.name not in MODE_TOOL_PERMISSIONS[self._runtime_mode]
            or self._runtime_mode not in policy.allowed_modes
        ):
            return ToolDenialReason.MODE_NOT_ALLOWED
        if policy.capability not in self._granted_capabilities:
            return ToolDenialReason.CAPABILITY_NOT_GRANTED
        if policy.requires_external_authorization and self._live_execution_authorizer is None:
            return ToolDenialReason.EXTERNAL_AUTHORIZATION_REQUIRED
        return None

    def _static_denial(
        self,
        policy: ToolPolicy,
        context: ToolExecutionContext,
    ) -> ToolDenialReason | None:
        pre_snapshot = self._pre_snapshot_denial(policy)
        if pre_snapshot is not None:
            return pre_snapshot
        if (
            policy.account_scoped
            and context.decision_snapshot.account_id not in self._granted_account_ids
        ):
            return ToolDenialReason.ACCOUNT_SCOPE_NOT_GRANTED
        return None

    def _deny(
        self,
        denial: ToolInvocationDenied,
        *,
        call_context: ToolCallContext,
        requested_tool_name: object,
        execution_context: ToolExecutionContext | None = None,
        snapshot: DecisionSnapshot | None = None,
        policy: ToolPolicy | None = None,
        argument_digest: str | None = None,
        validation_error_codes: tuple[str, ...] = (),
    ) -> NoReturn:
        self._append_authorization_event(
            result="DENIED",
            call_context=call_context,
            requested_tool_name=requested_tool_name,
            execution_context=execution_context,
            snapshot=snapshot,
            policy=policy,
            denial_reason=denial.reason,
            argument_digest=argument_digest,
            validation_error_codes=validation_error_codes,
            intended_denial=denial,
        )
        raise denial from None

    def _append_authorization_event(
        self,
        *,
        result: str,
        call_context: ToolCallContext,
        requested_tool_name: object,
        execution_context: ToolExecutionContext | None = None,
        snapshot: DecisionSnapshot | None = None,
        policy: ToolPolicy | None = None,
        denial_reason: ToolDenialReason | None = None,
        argument_digest: str | None = None,
        validation_error_codes: tuple[str, ...] = (),
        intended_denial: ToolInvocationDenied | None = None,
    ) -> None:
        resolved_snapshot = (
            execution_context.decision_snapshot if execution_context is not None else snapshot
        )
        metadata: dict[str, Any] = {
            "schema_version": "1",
            "policy_version": TOOL_POLICY_VERSION,
            "runtime_mode": self._runtime_mode.value,
        }
        if policy is not None:
            action = policy.name
            metadata.update(
                {
                    "tool_version": self._tools[policy.name].version,
                    "tool_category": policy.category.value,
                    "tool_capability": policy.capability.value,
                    "tool_effect": policy.effect.value,
                }
            )
        elif requested_tool_name is None:
            action = "catalog"
        else:
            action = "untrusted_tool_name"
            name_digest, name_size = _untrusted_name_digest(requested_tool_name)
            metadata["tool_name_sha256"] = name_digest
            metadata["tool_name_bytes"] = name_size
        if resolved_snapshot is not None:
            metadata["decision_snapshot_hash"] = resolved_snapshot.content_hash
        if denial_reason is not None:
            metadata["denial_reason"] = denial_reason.value
        if argument_digest is not None:
            metadata["argument_sha256"] = argument_digest
        if validation_error_codes:
            metadata["validation_error_codes"] = list(validation_error_codes)

        event = AuditEvent(
            event_type="AGENT_TOOL_AUTHORIZATION",
            actor_id=self._principal_id,
            action=action,
            result=result,
            request_id=call_context.request_id,
            decision_id=call_context.decision_id,
            metadata=metadata,
        )
        append_failed = False
        try:
            with self._audit_lock:
                self._audit_sink.append(event)
        except Exception:
            append_failed = True
        if append_failed:
            raise ToolAuditUnavailable(intended_denial) from None


__all__ = ["AgentToolRegistry", "AgentToolRegistryBuilder"]
