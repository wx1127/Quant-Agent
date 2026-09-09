"""Fail-closed, content-addressed contracts for independent risk services."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import cast

from quant_agent.core.time import ensure_aware

_SCHEMA_VERSION = "1"
_MACHINE_CODE = re.compile(r"[A-Z][A-Z0-9_.-]{0,127}")


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _machine_code(value: str, field_name: str) -> str:
    normalized = _non_empty(value, field_name)
    if _MACHINE_CODE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be an uppercase machine-readable code")
    return normalized


def _sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _exact_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be an exact Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


def _canonical_decimal(value: Decimal) -> str:
    _exact_decimal(value, "decimal")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _canonical(value: object) -> object:
    if isinstance(value, datetime):
        ensure_aware(value)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json(value: str, *, object_name: str) -> dict[str, object]:
    try:
        loaded: object = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{object_name} JSON is invalid") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{object_name} JSON must contain an object")
    return cast(dict[str, object], loaded)


def _require_keys(
    payload: dict[str, object], expected: frozenset[str], *, object_name: str
) -> None:
    if frozenset(payload) != expected:
        raise ValueError(f"{object_name} JSON fields do not match the contract")


def _string(payload: dict[str, object], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(payload: dict[str, object], field_name: str) -> str | None:
    value = payload[field_name]
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a string or null")


def _timestamp(payload: dict[str, object], field_name: str) -> datetime:
    value = _string(payload, field_name)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from error
    ensure_aware(parsed)
    return parsed


def _optional_decimal(payload: dict[str, object], field_name: str) -> Decimal | None:
    value = payload[field_name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an exact decimal string or null")
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f"{field_name} must be an exact decimal string or null") from error
    return _exact_decimal(result, field_name)


class RiskCheckStatus(StrEnum):
    """Externally stable outcome of a risk service call."""

    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"
    ERROR = "ERROR"


class RuleSeverity(StrEnum):
    """Whether a finding merely warns or vetoes the proposal."""

    WARNING = "WARNING"
    HARD_VIOLATION = "HARD_VIOLATION"


class RuleScope(StrEnum):
    """Machine-readable location at which a rule was evaluated."""

    PORTFOLIO = "PORTFOLIO"
    ACCOUNT = "ACCOUNT"
    INSTRUMENT = "INSTRUMENT"
    INDUSTRY = "INDUSTRY"


@dataclass(frozen=True, slots=True)
class RiskCheckRequest:
    """Exact data, account, proposal, valuation, and policy boundary to evaluate."""

    schema_version: str
    decision_id: str
    account_snapshot_id: str
    account_snapshot_hash: str
    account_snapshot_as_of: datetime
    portfolio_proposal_hash: str
    data_version: str
    valuation_version: str
    policy_version: str
    policy_hash: str
    risk_context_hash: str
    request_hash: str

    _JSON_KEYS = frozenset(
        {
            "schema_version",
            "decision_id",
            "account_snapshot_id",
            "account_snapshot_hash",
            "account_snapshot_as_of",
            "portfolio_proposal_hash",
            "data_version",
            "valuation_version",
            "policy_version",
            "policy_hash",
            "risk_context_hash",
            "request_hash",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("risk request schema_version must be '1'")
        for field_name in (
            "decision_id",
            "account_snapshot_id",
            "data_version",
            "valuation_version",
            "policy_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.account_snapshot_as_of)
        for value, field_name in (
            (self.account_snapshot_hash, "account_snapshot_hash"),
            (self.portfolio_proposal_hash, "portfolio_proposal_hash"),
            (self.policy_hash, "policy_hash"),
            (self.risk_context_hash, "risk_context_hash"),
            (self.request_hash, "request_hash"),
        ):
            _sha256(value, field_name)
        if self.request_hash != _stable_hash(self._content_payload()):
            raise ValueError("risk request request_hash does not match bound inputs")

    @classmethod
    def build(
        cls,
        *,
        decision_id: str,
        account_snapshot_id: str,
        account_snapshot_hash: str,
        account_snapshot_as_of: datetime,
        portfolio_proposal_hash: str,
        data_version: str,
        valuation_version: str,
        policy_version: str,
        policy_hash: str,
        risk_context_hash: str,
    ) -> RiskCheckRequest:
        """Build and hash one complete risk request boundary."""

        normalized_decision_id = _non_empty(decision_id, "decision_id")
        normalized_snapshot_id = _non_empty(account_snapshot_id, "account_snapshot_id")
        normalized_data_version = _non_empty(data_version, "data_version")
        normalized_valuation_version = _non_empty(valuation_version, "valuation_version")
        normalized_policy_version = _non_empty(policy_version, "policy_version")
        ensure_aware(account_snapshot_as_of)
        values: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "decision_id": normalized_decision_id,
            "account_snapshot_id": normalized_snapshot_id,
            "account_snapshot_hash": _sha256(account_snapshot_hash, "account_snapshot_hash"),
            "account_snapshot_as_of": account_snapshot_as_of,
            "portfolio_proposal_hash": _sha256(portfolio_proposal_hash, "portfolio_proposal_hash"),
            "data_version": normalized_data_version,
            "valuation_version": normalized_valuation_version,
            "policy_version": normalized_policy_version,
            "policy_hash": _sha256(policy_hash, "policy_hash"),
            "risk_context_hash": _sha256(risk_context_hash, "risk_context_hash"),
        }
        return cls(
            schema_version=_SCHEMA_VERSION,
            decision_id=normalized_decision_id,
            account_snapshot_id=normalized_snapshot_id,
            account_snapshot_hash=account_snapshot_hash,
            account_snapshot_as_of=account_snapshot_as_of,
            portfolio_proposal_hash=portfolio_proposal_hash,
            data_version=normalized_data_version,
            valuation_version=normalized_valuation_version,
            policy_version=normalized_policy_version,
            policy_hash=policy_hash,
            risk_context_hash=risk_context_hash,
            request_hash=_stable_hash(values),
        )

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "account_snapshot_id": self.account_snapshot_id,
            "account_snapshot_hash": self.account_snapshot_hash,
            "account_snapshot_as_of": self.account_snapshot_as_of,
            "portfolio_proposal_hash": self.portfolio_proposal_hash,
            "data_version": self.data_version,
            "valuation_version": self.valuation_version,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "risk_context_hash": self.risk_context_hash,
        }

    def _json_payload(self) -> dict[str, object]:
        return {**self._content_payload(), "request_hash": self.request_hash}

    def to_json(self) -> str:
        """Return canonical JSON without lossy numeric or local-time encodings."""

        return _canonical_json(self._json_payload())

    @classmethod
    def from_json(cls, value: str) -> RiskCheckRequest:
        """Decode an exact request and reject unknown, duplicate, or tampered fields."""

        return cls._from_payload(_load_json(value, object_name="risk request"))

    @classmethod
    def _from_payload(cls, payload: dict[str, object]) -> RiskCheckRequest:
        _require_keys(payload, cls._JSON_KEYS, object_name="risk request")
        return cls(
            schema_version=_string(payload, "schema_version"),
            decision_id=_string(payload, "decision_id"),
            account_snapshot_id=_string(payload, "account_snapshot_id"),
            account_snapshot_hash=_string(payload, "account_snapshot_hash"),
            account_snapshot_as_of=_timestamp(payload, "account_snapshot_as_of"),
            portfolio_proposal_hash=_string(payload, "portfolio_proposal_hash"),
            data_version=_string(payload, "data_version"),
            valuation_version=_string(payload, "valuation_version"),
            policy_version=_string(payload, "policy_version"),
            policy_hash=_string(payload, "policy_hash"),
            risk_context_hash=_string(payload, "risk_context_hash"),
            request_hash=_string(payload, "request_hash"),
        )


@dataclass(frozen=True, slots=True)
class RiskFinding:
    """One structured warning or hard violation, with no natural-language dependency."""

    rule_code: str
    rule_version: str
    severity: RuleSeverity
    scope: RuleScope
    scope_id: str | None = None
    instrument_id: str | None = None
    observed_value: Decimal | None = None
    limit_value: Decimal | None = None

    _JSON_KEYS = frozenset(
        {
            "rule_code",
            "rule_version",
            "severity",
            "scope",
            "scope_id",
            "instrument_id",
            "observed_value",
            "limit_value",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_code", _machine_code(self.rule_code, "rule_code"))
        object.__setattr__(self, "rule_version", _non_empty(self.rule_version, "rule_version"))
        if not isinstance(self.severity, RuleSeverity):
            raise ValueError("severity must be a RuleSeverity value")
        if not isinstance(self.scope, RuleScope):
            raise ValueError("scope must be a RuleScope value")
        if self.scope_id is not None:
            object.__setattr__(self, "scope_id", _non_empty(self.scope_id, "scope_id"))
        if self.instrument_id is not None:
            object.__setattr__(
                self, "instrument_id", _non_empty(self.instrument_id, "instrument_id")
            )
        if self.scope is RuleScope.INSTRUMENT and self.instrument_id is None:
            raise ValueError("instrument-scoped findings require instrument_id")
        if self.scope is RuleScope.INDUSTRY and self.scope_id is None:
            raise ValueError("industry-scoped findings require scope_id")
        if (self.observed_value is None) != (self.limit_value is None):
            raise ValueError("observed_value and limit_value must be present together")
        if self.observed_value is not None:
            _exact_decimal(self.observed_value, "observed_value")
            _exact_decimal(cast(Decimal, self.limit_value), "limit_value")

    def _payload(self) -> dict[str, object]:
        return {
            "rule_code": self.rule_code,
            "rule_version": self.rule_version,
            "severity": self.severity,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "instrument_id": self.instrument_id,
            "observed_value": self.observed_value,
            "limit_value": self.limit_value,
        }

    def _sort_key(self) -> tuple[str, ...]:
        return (
            self.rule_code,
            self.rule_version,
            self.severity.value,
            self.scope.value,
            self.scope_id or "",
            self.instrument_id or "",
            "" if self.observed_value is None else _canonical_decimal(self.observed_value),
            "" if self.limit_value is None else _canonical_decimal(self.limit_value),
        )

    def _identity_key(self) -> tuple[str, str, str, str]:
        return (
            self.rule_code,
            self.scope.value,
            self.scope_id or "",
            self.instrument_id or "",
        )

    @classmethod
    def _from_payload(cls, payload: dict[str, object]) -> RiskFinding:
        _require_keys(payload, cls._JSON_KEYS, object_name="risk finding")
        try:
            severity = RuleSeverity(_string(payload, "severity"))
            scope = RuleScope(_string(payload, "scope"))
        except ValueError as error:
            raise ValueError("risk finding contains an unknown enum value") from error
        return cls(
            rule_code=_string(payload, "rule_code"),
            rule_version=_string(payload, "rule_version"),
            severity=severity,
            scope=scope,
            scope_id=_optional_string(payload, "scope_id"),
            instrument_id=_optional_string(payload, "instrument_id"),
            observed_value=_optional_decimal(payload, "observed_value"),
            limit_value=_optional_decimal(payload, "limit_value"),
        )


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    """Deterministic risk outcome whose ERROR state is always fail-closed."""

    schema_version: str
    request: RiskCheckRequest
    status: RiskCheckStatus
    evaluated_at: datetime
    risk_engine_version: str
    findings: tuple[RiskFinding, ...]
    error_code: str | None
    result_hash: str

    _JSON_KEYS = frozenset(
        {
            "schema_version",
            "request",
            "status",
            "evaluated_at",
            "risk_engine_version",
            "findings",
            "error_code",
            "result_hash",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("risk result schema_version must be '1'")
        if not isinstance(self.request, RiskCheckRequest):
            raise ValueError("request must be a RiskCheckRequest")
        if not isinstance(self.status, RiskCheckStatus):
            raise ValueError("status must be a RiskCheckStatus value")
        ensure_aware(self.evaluated_at)
        if self.evaluated_at < self.request.account_snapshot_as_of:
            raise ValueError("evaluated_at cannot precede the bound account snapshot")
        object.__setattr__(
            self,
            "risk_engine_version",
            _non_empty(self.risk_engine_version, "risk_engine_version"),
        )
        if not isinstance(self.findings, tuple) or any(
            not isinstance(item, RiskFinding) for item in self.findings
        ):
            raise ValueError("findings must be an immutable tuple of RiskFinding values")
        if tuple(sorted(self.findings, key=RiskFinding._sort_key)) != self.findings:
            raise ValueError("risk findings must be deterministically sorted")
        identities = tuple(item._identity_key() for item in self.findings)
        if len(set(identities)) != len(identities):
            raise ValueError("risk findings must have unique rule and scope identities")
        warnings = tuple(item for item in self.findings if item.severity is RuleSeverity.WARNING)
        hard_violations = tuple(
            item for item in self.findings if item.severity is RuleSeverity.HARD_VIOLATION
        )
        if self.status is RiskCheckStatus.PASS and self.findings:
            raise ValueError("PASS results cannot contain findings")
        if self.status is RiskCheckStatus.WARN and (
            not warnings or hard_violations or len(warnings) != len(self.findings)
        ):
            raise ValueError("WARN results require warning findings only")
        if self.status is RiskCheckStatus.REJECT and not hard_violations:
            raise ValueError("REJECT results require at least one hard violation")
        if self.status is RiskCheckStatus.ERROR:
            if self.findings:
                raise ValueError("ERROR results cannot claim evaluated rule findings")
            if self.error_code is None:
                raise ValueError("ERROR results require an error_code")
        elif self.error_code is not None:
            raise ValueError("only ERROR results may contain an error_code")
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _machine_code(self.error_code, "error_code"))
        _sha256(self.result_hash, "result_hash")
        if self.result_hash != _stable_hash(self._content_payload()):
            raise ValueError("risk result result_hash does not match bound outcome")

    @classmethod
    def build(
        cls,
        *,
        request: RiskCheckRequest,
        status: RiskCheckStatus,
        evaluated_at: datetime,
        risk_engine_version: str,
        findings: tuple[RiskFinding, ...] = (),
        error_code: str | None = None,
    ) -> RiskCheckResult:
        """Build a sorted, content-addressed outcome and enforce status semantics."""

        if not isinstance(findings, tuple):
            raise ValueError("findings must be an immutable tuple")
        ordered_findings = tuple(sorted(findings, key=RiskFinding._sort_key))
        normalized_engine_version = _non_empty(risk_engine_version, "risk_engine_version")
        normalized_error_code = (
            None if error_code is None else _machine_code(error_code, "error_code")
        )
        values: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "request": request._json_payload(),
            "status": status,
            "evaluated_at": evaluated_at,
            "risk_engine_version": normalized_engine_version,
            "findings": [item._payload() for item in ordered_findings],
            "error_code": normalized_error_code,
        }
        return cls(
            schema_version=_SCHEMA_VERSION,
            request=request,
            status=status,
            evaluated_at=evaluated_at,
            risk_engine_version=normalized_engine_version,
            findings=ordered_findings,
            error_code=normalized_error_code,
            result_hash=_stable_hash(values),
        )

    @classmethod
    def service_error(
        cls,
        *,
        request: RiskCheckRequest,
        evaluated_at: datetime,
        risk_engine_version: str,
        error_code: str,
    ) -> RiskCheckResult:
        """Create an explicit service error that can never authorize the proposal."""

        return cls.build(
            request=request,
            status=RiskCheckStatus.ERROR,
            evaluated_at=evaluated_at,
            risk_engine_version=risk_engine_version,
            findings=(),
            error_code=error_code,
        )

    @property
    def warnings(self) -> tuple[RiskFinding, ...]:
        """Return warning findings without weakening any hard decision."""

        return tuple(item for item in self.findings if item.severity is RuleSeverity.WARNING)

    @property
    def violations(self) -> tuple[RiskFinding, ...]:
        """Return hard violations using the domain term expected by downstream services."""

        return self.hard_violations

    @property
    def hard_violations(self) -> tuple[RiskFinding, ...]:
        """Return findings that independently veto the proposal."""

        return tuple(item for item in self.findings if item.severity is RuleSeverity.HARD_VIOLATION)

    @property
    def allows_execution(self) -> bool:
        """Whether downstream code may proceed; service errors fail closed."""

        return self.status in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}

    @property
    def is_rejected(self) -> bool:
        """Treat both rule rejection and service failure as rejection."""

        return not self.allows_execution

    @property
    def passed(self) -> bool:
        """Compatibility view: warnings pass, hard violations and errors do not."""

        return self.allows_execution

    @property
    def checked_policy_version(self) -> str:
        """Expose the version proven by the embedded request."""

        return self.request.policy_version

    @property
    def checked_policy_hash(self) -> str:
        """Expose the policy content hash proven by the embedded request."""

        return self.request.policy_hash

    @property
    def checked_at(self) -> datetime:
        """Expose the timezone-aware evaluation time under conventional naming."""

        return self.evaluated_at

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request": self.request._json_payload(),
            "status": self.status,
            "evaluated_at": self.evaluated_at,
            "risk_engine_version": self.risk_engine_version,
            "findings": [item._payload() for item in self.findings],
            "error_code": self.error_code,
        }

    def to_json(self) -> str:
        """Return canonical JSON for audit storage and cross-process transport."""

        return _canonical_json({**self._content_payload(), "result_hash": self.result_hash})

    @classmethod
    def from_json(cls, value: str) -> RiskCheckResult:
        """Decode and revalidate the request, findings, outcome, and root hash."""

        payload = _load_json(value, object_name="risk result")
        _require_keys(payload, cls._JSON_KEYS, object_name="risk result")
        request_payload = payload["request"]
        findings_payload = payload["findings"]
        if not isinstance(request_payload, dict):
            raise ValueError("risk result request must be an object")
        if not isinstance(findings_payload, list):
            raise ValueError("risk result findings must be an array")
        findings: list[RiskFinding] = []
        for item in findings_payload:
            if not isinstance(item, dict):
                raise ValueError("each risk finding must be an object")
            findings.append(RiskFinding._from_payload(cast(dict[str, object], item)))
        try:
            status = RiskCheckStatus(_string(payload, "status"))
        except ValueError as error:
            raise ValueError("risk result contains an unknown status") from error
        return cls(
            schema_version=_string(payload, "schema_version"),
            request=RiskCheckRequest._from_payload(cast(dict[str, object], request_payload)),
            status=status,
            evaluated_at=_timestamp(payload, "evaluated_at"),
            risk_engine_version=_string(payload, "risk_engine_version"),
            findings=tuple(findings),
            error_code=_optional_string(payload, "error_code"),
            result_hash=_string(payload, "result_hash"),
        )


RiskResultStatus = RiskCheckStatus
RiskSeverity = RuleSeverity
RiskDecision = RiskCheckResult
RiskRequest = RiskCheckRequest
RiskRuleSeverity = RuleSeverity


__all__ = [
    "RiskCheckRequest",
    "RiskCheckResult",
    "RiskCheckStatus",
    "RiskDecision",
    "RiskFinding",
    "RiskRequest",
    "RiskResultStatus",
    "RiskRuleSeverity",
    "RiskSeverity",
    "RuleScope",
    "RuleSeverity",
]
