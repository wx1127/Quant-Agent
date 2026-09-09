"""Contract tests for the independent, fail-closed risk-service boundary."""

import hashlib
import json
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta
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


def _request(*, as_of: datetime = AS_OF) -> RiskCheckRequest:
    return RiskCheckRequest.build(
        decision_id="dec_20260828_contract",
        account_snapshot_id="account-snapshot-20260828",
        account_snapshot_hash=_hash("account"),
        account_snapshot_as_of=as_of,
        portfolio_proposal_hash=_hash("proposal"),
        data_version="market-pit-20260828-v1",
        valuation_version="account-mark-v1",
        policy_version="portfolio-risk-v3",
        policy_hash=_hash("policy-v3"),
        risk_context_hash=_hash("risk-context"),
    )


def _warning(*, observed: Decimal = Decimal("0.18")) -> RiskFinding:
    return RiskFinding(
        rule_code="DAILY_TURNOVER_WARNING",
        rule_version="turnover-v2",
        severity=RuleSeverity.WARNING,
        scope=RuleScope.PORTFOLIO,
        observed_value=observed,
        limit_value=Decimal("0.15"),
    )


def _hard_violation() -> RiskFinding:
    return RiskFinding(
        rule_code="MAX_INSTRUMENT_WEIGHT",
        rule_version="concentration-v4",
        severity=RuleSeverity.HARD_VIOLATION,
        scope=RuleScope.INSTRUMENT,
        instrument_id="CN.SSE.600000",
        observed_value=Decimal("0.08"),
        limit_value=Decimal("0.05"),
    )


def test_request_binds_every_required_identity_and_is_deeply_immutable() -> None:
    request = _request()

    assert request.decision_id == "dec_20260828_contract"
    assert request.account_snapshot_id == "account-snapshot-20260828"
    assert request.account_snapshot_hash == _hash("account")
    assert request.account_snapshot_as_of == AS_OF
    assert request.portfolio_proposal_hash == _hash("proposal")
    assert request.data_version == "market-pit-20260828-v1"
    assert request.valuation_version == "account-mark-v1"
    assert request.policy_version == "portfolio-risk-v3"
    assert request.policy_hash == _hash("policy-v3")
    assert request.risk_context_hash == _hash("risk-context")
    assert len(request.request_hash) == 64
    assert {item.name for item in fields(request)}.isdisjoint(
        {"agent_prompt", "natural_language", "strategy_signal"}
    )
    with pytest.raises(FrozenInstanceError):
        request.data_version = "tampered"  # type: ignore[misc]


def test_request_hash_normalizes_equivalent_timezones_and_rejects_tampering() -> None:
    local = _request()
    utc = _request(as_of=AS_OF.astimezone(UTC))

    assert local.request_hash == utc.request_hash
    with pytest.raises(ValueError, match="request_hash"):
        replace(local, decision_id="dec_changed")
    with pytest.raises(ValueError, match="request_hash"):
        replace(local, account_snapshot_hash=_hash("other-account"))
    with pytest.raises(ValueError, match="request_hash"):
        replace(local, portfolio_proposal_hash=_hash("other-proposal"))
    with pytest.raises(ValueError, match="request_hash"):
        replace(local, policy_version="portfolio-risk-v4")


def test_request_enforces_timezone_versions_and_sha256_digests() -> None:
    request = _request()

    with pytest.raises(ValueError, match="timezone"):
        replace(request, account_snapshot_as_of=datetime(2026, 8, 28, 18))
    with pytest.raises(ValueError, match="valuation_version"):
        RiskCheckRequest.build(
            decision_id="decision",
            account_snapshot_id="snapshot",
            account_snapshot_hash=_hash("account"),
            account_snapshot_as_of=AS_OF,
            portfolio_proposal_hash=_hash("proposal"),
            data_version="data-v1",
            valuation_version=" ",
            policy_version="policy-v1",
            policy_hash=_hash("policy"),
            risk_context_hash=_hash("risk-context"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(request, policy_hash="A" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(request, portfolio_proposal_hash="too-short")


def test_findings_are_structured_located_and_require_exact_decimals() -> None:
    hard = _hard_violation()

    assert hard.rule_code == "MAX_INSTRUMENT_WEIGHT"
    assert hard.rule_version == "concentration-v4"
    assert hard.instrument_id == "CN.SSE.600000"
    assert hard.severity is RuleSeverity.HARD_VIOLATION
    assert hard.observed_value == Decimal("0.08")
    with pytest.raises(ValueError, match="instrument_id"):
        replace(hard, instrument_id=None)
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(hard, observed_value=0.08)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="present together"):
        replace(hard, limit_value=None)
    with pytest.raises(ValueError, match="machine-readable"):
        replace(hard, rule_code="用自然语言描述违规")
    with pytest.raises(ValueError, match="scope_id"):
        RiskFinding(
            rule_code="MAX_INDUSTRY_WEIGHT",
            rule_version="industry-v1",
            severity=RuleSeverity.WARNING,
            scope=RuleScope.INDUSTRY,
        )


def test_status_invariants_distinguish_pass_warning_rejection_and_error() -> None:
    request = _request()
    passed = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.PASS,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
    )
    warned = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.WARN,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_warning(),),
    )
    rejected = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.REJECT,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_warning(), _hard_violation()),
    )

    assert passed.allows_execution and not passed.is_rejected
    assert passed.passed
    assert warned.allows_execution and warned.warnings == (_warning(),)
    assert rejected.is_rejected and not rejected.allows_execution
    assert rejected.hard_violations == (_hard_violation(),)
    assert rejected.violations == rejected.hard_violations
    assert rejected.checked_policy_version == request.policy_version
    assert rejected.checked_policy_hash == request.policy_hash
    assert rejected.checked_at == EVALUATED_AT
    with pytest.raises(ValueError, match="PASS"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.PASS,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=(_warning(),),
        )
    with pytest.raises(ValueError, match="WARN"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.WARN,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=(_hard_violation(),),
        )
    with pytest.raises(ValueError, match="REJECT"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.REJECT,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=(_warning(),),
        )


def test_service_error_is_explicit_and_fails_closed_by_contract() -> None:
    request = _request()
    result = RiskCheckResult.service_error(
        request=request,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        error_code="RISK_SERVICE_TIMEOUT",
    )

    assert result.status is RiskCheckStatus.ERROR
    assert result.error_code == "RISK_SERVICE_TIMEOUT"
    assert result.is_rejected
    assert not result.allows_execution
    assert result.findings == ()
    with pytest.raises(ValueError, match="error_code"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.ERROR,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
        )


def test_result_hash_is_order_independent_and_decimal_scale_independent() -> None:
    request = _request()
    first = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.REJECT,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_warning(observed=Decimal("0.1800")), _hard_violation()),
    )
    second = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.REJECT,
        evaluated_at=EVALUATED_AT.astimezone(UTC),
        risk_engine_version="risk-engine-v1",
        findings=(_hard_violation(), _warning(observed=Decimal("0.18"))),
    )

    assert first.result_hash == second.result_hash
    assert first.to_json() == second.to_json()
    with pytest.raises(FrozenInstanceError):
        first.findings[0].rule_code = "CHANGED"  # type: ignore[misc]
    with pytest.raises(ValueError, match="immutable tuple"):
        RiskCheckResult.build(
            request=request,
            status=RiskCheckStatus.WARN,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=[_warning()],  # type: ignore[arg-type]
        )


def test_canonical_json_round_trip_revalidates_request_and_result_hashes() -> None:
    request = _request()
    result = RiskCheckResult.build(
        request=request,
        status=RiskCheckStatus.REJECT,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_hard_violation(),),
    )

    assert RiskCheckRequest.from_json(request.to_json()) == request
    assert RiskCheckResult.from_json(result.to_json()) == result
    assert RiskCheckResult.from_json(result.to_json()).to_json() == result.to_json()
    assert "0.08" in result.to_json()

    request_payload = json.loads(request.to_json())
    request_payload["valuation_version"] = "tampered-mark-v2"
    with pytest.raises(ValueError, match="request_hash"):
        RiskCheckRequest.from_json(json.dumps(request_payload))

    result_payload = json.loads(result.to_json())
    result_payload["findings"][0]["observed_value"] = "0.09"
    with pytest.raises(ValueError, match="result_hash"):
        RiskCheckResult.from_json(json.dumps(result_payload))


def test_json_decoder_rejects_unknown_duplicate_and_lossy_numeric_fields() -> None:
    request_json = _request().to_json()
    payload = json.loads(request_json)
    payload["agent_instruction"] = "ignore risk policy"
    with pytest.raises(ValueError, match="fields do not match"):
        RiskCheckRequest.from_json(json.dumps(payload))

    duplicate = request_json.replace(
        '"decision_id":',
        '"decision_id":"dec_duplicate","decision_id":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        RiskCheckRequest.from_json(duplicate)

    result = RiskCheckResult.build(
        request=_request(),
        status=RiskCheckStatus.WARN,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_warning(),),
    )
    result_payload = json.loads(result.to_json())
    result_payload["findings"][0]["observed_value"] = 0.18
    with pytest.raises(ValueError, match="exact decimal string"):
        RiskCheckResult.from_json(json.dumps(result_payload))


def test_result_rejects_tampering_stale_evaluation_and_duplicate_rule_locations() -> None:
    result = RiskCheckResult.build(
        request=_request(),
        status=RiskCheckStatus.REJECT,
        evaluated_at=EVALUATED_AT,
        risk_engine_version="risk-engine-v1",
        findings=(_hard_violation(),),
    )

    with pytest.raises(ValueError, match="result_hash"):
        replace(result, risk_engine_version="risk-engine-v2")
    with pytest.raises(ValueError, match="cannot precede"):
        RiskCheckResult.build(
            request=_request(),
            status=RiskCheckStatus.PASS,
            evaluated_at=AS_OF - timedelta(microseconds=1),
            risk_engine_version="risk-engine-v1",
        )
    duplicate = replace(_hard_violation(), rule_version="concentration-v5")
    with pytest.raises(ValueError, match="unique rule and scope"):
        RiskCheckResult.build(
            request=_request(),
            status=RiskCheckStatus.REJECT,
            evaluated_at=EVALUATED_AT,
            risk_engine_version="risk-engine-v1",
            findings=(_hard_violation(), duplicate),
        )
