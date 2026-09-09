"""Boundary and rejection tests for the independent risk-service contracts."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.risk import (
    RiskCheckRequest,
    RiskCheckResult,
    RiskCheckStatus,
    RiskFinding,
    RuleScope,
    RuleSeverity,
)

AS_OF = datetime(2026, 8, 28, 18, tzinfo=SHANGHAI_TZ)
EVALUATED_AT = AS_OF + timedelta(seconds=5)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _request() -> RiskCheckRequest:
    return RiskCheckRequest.build(
        decision_id="decision-1",
        account_snapshot_id="snapshot-1",
        account_snapshot_hash=_hash("account"),
        account_snapshot_as_of=AS_OF,
        portfolio_proposal_hash=_hash("proposal"),
        data_version="data-v1",
        valuation_version="valuation-v1",
        policy_version="policy-v1",
        policy_hash=_hash("policy"),
        risk_context_hash=_hash("risk-context"),
    )


def _warning(
    *,
    rule_code: str = "TURNOVER_WARNING",
    scope: RuleScope = RuleScope.PORTFOLIO,
    scope_id: str | None = None,
    observed_value: Decimal | None = Decimal("0.2"),
    limit_value: Decimal | None = Decimal("0.1"),
) -> RiskFinding:
    return RiskFinding(
        rule_code=rule_code,
        rule_version="rule-v1",
        severity=RuleSeverity.WARNING,
        scope=scope,
        scope_id=scope_id,
        observed_value=observed_value,
        limit_value=limit_value,
    )


def _hard_violation(*, rule_version: str = "rule-v1") -> RiskFinding:
    return RiskFinding(
        rule_code="MAX_WEIGHT",
        rule_version=rule_version,
        severity=RuleSeverity.HARD_VIOLATION,
        scope=RuleScope.INSTRUMENT,
        instrument_id="CN.SSE.600000",
        observed_value=Decimal("0.08"),
        limit_value=Decimal("0.05"),
    )


def _result(
    *,
    status: RiskCheckStatus = RiskCheckStatus.PASS,
    findings: tuple[RiskFinding, ...] = (),
    error_code: str | None = None,
) -> RiskCheckResult:
    return RiskCheckResult.build(
        request=_request(),
        status=status,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=findings,
        error_code=error_code,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("decision_id", None),
        ("account_snapshot_id", " "),
        ("data_version", ""),
        ("valuation_version", 7),
        ("policy_version", "\t"),
    ],
)
def test_request_build_rejects_non_string_and_empty_identifiers(
    field_name: str, invalid_value: object
) -> None:
    values: dict[str, object] = {
        "decision_id": "decision-1",
        "account_snapshot_id": "snapshot-1",
        "account_snapshot_hash": _hash("account"),
        "account_snapshot_as_of": AS_OF,
        "portfolio_proposal_hash": _hash("proposal"),
        "data_version": "data-v1",
        "valuation_version": "valuation-v1",
        "policy_version": "policy-v1",
        "policy_hash": _hash("policy"),
        "risk_context_hash": _hash("risk-context"),
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        RiskCheckRequest.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("account_snapshot_hash", None),
        ("portfolio_proposal_hash", "f" * 63),
        ("policy_hash", "F" * 64),
        ("risk_context_hash", "f" * 63),
    ],
)
def test_request_build_rejects_invalid_hashes(field_name: str, invalid_value: object) -> None:
    values: dict[str, object] = {
        "decision_id": "decision-1",
        "account_snapshot_id": "snapshot-1",
        "account_snapshot_hash": _hash("account"),
        "account_snapshot_as_of": AS_OF,
        "portfolio_proposal_hash": _hash("proposal"),
        "data_version": "data-v1",
        "valuation_version": "valuation-v1",
        "policy_version": "policy-v1",
        "policy_hash": _hash("policy"),
        "risk_context_hash": _hash("risk-context"),
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        RiskCheckRequest.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_json", ["{", "[]", "null", "true"])
def test_request_from_json_rejects_invalid_or_non_object_documents(invalid_json: str) -> None:
    with pytest.raises(ValueError, match="risk request JSON"):
        RiskCheckRequest.from_json(invalid_json)


def test_request_from_json_rejects_runtime_type_and_field_type_errors() -> None:
    with pytest.raises(ValueError, match="risk request JSON is invalid"):
        RiskCheckRequest.from_json(None)  # type: ignore[arg-type]

    payload = json.loads(_request().to_json())
    payload["decision_id"] = 12
    with pytest.raises(ValueError, match="decision_id must be a string"):
        RiskCheckRequest.from_json(json.dumps(payload))

    payload = json.loads(_request().to_json())
    payload["account_snapshot_as_of"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        RiskCheckRequest.from_json(json.dumps(payload))

    payload = json.loads(_request().to_json())
    payload["schema_version"] = "2"
    with pytest.raises(ValueError, match="schema_version"):
        RiskCheckRequest.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rule_code": None}, "rule_code must be a string"),
        ({"rule_code": "lower_case"}, "machine-readable"),
        ({"rule_version": " "}, "rule_version"),
        ({"severity": "WARNING"}, "severity must be a RuleSeverity"),
        ({"scope": "PORTFOLIO"}, "scope must be a RuleScope"),
        ({"scope_id": " "}, "scope_id"),
        ({"instrument_id": ""}, "instrument_id"),
        ({"observed_value": Decimal("NaN")}, "finite"),
        ({"limit_value": Decimal("Infinity")}, "finite"),
    ],
)
def test_finding_constructor_rejects_invalid_runtime_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_warning(), **changes)


def test_finding_round_trip_covers_nullable_values_industry_scope_and_zero() -> None:
    finding = _warning(
        rule_code="INDUSTRY_NOTICE",
        scope=RuleScope.INDUSTRY,
        scope_id="BANKS",
        observed_value=None,
        limit_value=None,
    )
    result = _result(status=RiskCheckStatus.WARN, findings=(finding,))

    decoded = RiskCheckResult.from_json(result.to_json())

    assert decoded.findings == (finding,)
    assert decoded.findings[0].scope_id == "BANKS"
    assert decoded.findings[0].observed_value is None
    zero = _warning(observed_value=Decimal("-0.000"), limit_value=Decimal("0.0"))
    zero_result = _result(status=RiskCheckStatus.WARN, findings=(zero,))
    assert '"observed_value":"0"' in zero_result.to_json()
    assert '"limit_value":"0"' in zero_result.to_json()


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("severity", "NOTICE", "unknown enum value"),
        ("scope", "BOOK", "unknown enum value"),
        ("scope_id", 1, "string or null"),
        ("instrument_id", [], "string or null"),
        ("observed_value", 0.2, "exact decimal string"),
        ("observed_value", "not-decimal", "exact decimal string"),
        ("limit_value", "NaN", "finite"),
    ],
)
def test_result_from_json_rejects_invalid_finding_fields(
    field_name: str, invalid_value: object, message: str
) -> None:
    payload = json.loads(_result(status=RiskCheckStatus.WARN, findings=(_warning(),)).to_json())
    payload["findings"][0][field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        RiskCheckResult.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "2"}, "schema_version"),
        ({"request": object()}, "request must be a RiskCheckRequest"),
        ({"status": "PASS"}, "status must be a RiskCheckStatus"),
        ({"findings": (_warning(), object())}, "immutable tuple of RiskFinding"),
        ({"result_hash": "x"}, "SHA-256"),
    ],
)
def test_result_constructor_rejects_invalid_runtime_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_result(), **changes)


def test_result_constructor_rejects_unsorted_findings() -> None:
    result = _result(
        status=RiskCheckStatus.REJECT,
        findings=(_warning(), _hard_violation()),
    )
    assert result.findings == (_hard_violation(), _warning())

    with pytest.raises(ValueError, match="deterministically sorted"):
        replace(result, findings=tuple(reversed(result.findings)))


def test_result_status_and_error_code_combinations_fail_closed() -> None:
    request = _request()

    with pytest.raises(ValueError, match="WARN"):
        _result(status=RiskCheckStatus.WARN)
    with pytest.raises(ValueError, match="cannot claim"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.ERROR,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=(_warning(),),
            error_code="SERVICE_FAILED",
        )
    with pytest.raises(ValueError, match="only ERROR"):
        _result(status=RiskCheckStatus.PASS, error_code="UNEXPECTED_ERROR")
    with pytest.raises(ValueError, match="machine-readable"):
        RiskCheckResult.service_error(
            request=request,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            error_code="not machine readable",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("request", [], "request must be an object"),
        ("findings", {}, "findings must be an array"),
        ("findings", ["not-an-object"], "risk finding must be an object"),
        ("status", "SKIPPED", "unknown status"),
        ("error_code", 42, "string or null"),
    ],
)
def test_result_from_json_rejects_invalid_structure_and_status(
    field_name: str, invalid_value: object, message: str
) -> None:
    payload = json.loads(_result().to_json())
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        RiskCheckResult.from_json(json.dumps(payload))


def test_result_from_json_rejects_invalid_documents_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="risk result JSON is invalid"):
        RiskCheckResult.from_json("{")
    with pytest.raises(ValueError, match="must contain an object"):
        RiskCheckResult.from_json("[]")

    payload = json.loads(_result().to_json())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        RiskCheckResult.from_json(json.dumps(payload))


def test_result_properties_cover_warning_rejection_and_service_error_views() -> None:
    warning = _warning()
    hard = _hard_violation()
    rejected = _result(status=RiskCheckStatus.REJECT, findings=(warning, hard))
    failed = _result(status=RiskCheckStatus.ERROR, error_code="SERVICE_FAILED")

    assert rejected.warnings == (warning,)
    assert rejected.violations == (hard,)
    assert not rejected.passed
    assert failed.is_rejected
    assert not failed.passed
    assert failed.warnings == ()
    assert failed.hard_violations == ()
