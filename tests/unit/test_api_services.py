from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from apps.api.core.auth import AuthService, Principal, Role
from apps.api.core.services import (
    OrderApprovalService,
    OrderDraftRecord,
    ResearchCatalog,
    ResearchRecord,
)

from quant_agent.core.errors import ErrorCode, QuantAgentError
from quant_agent.risk.kill_switch import KillSwitch

NOW = datetime(2026, 7, 30, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


def draft(draft_id: str = "draft-1") -> OrderDraftRecord:
    return OrderDraftRecord(
        draft_id,
        "paper-1",
        "dec_20260730_test",
        f"hash-{draft_id}",
        NOW + timedelta(hours=1),
        ({"instrument_id": "000001.SZ", "side": "BUY", "quantity": 100},),
        5.0,
        ("fixture warning",),
    )


def test_auth_service_keeps_tokens_hashed_and_rejects_invalid_registration() -> None:
    auth = AuthService()
    principal = Principal("viewer", frozenset({Role.VIEWER}))
    with pytest.raises(ValueError, match="at least"):
        auth.register_token("short", principal)
    auth.register_token("valid-token-123456789", principal)
    with pytest.raises(ValueError, match="already"):
        auth.register_token("valid-token-123456789", principal)
    assert auth.authenticate("valid-token-123456789") == principal
    with pytest.raises(QuantAgentError) as error:
        auth.authenticate("wrong-token-123456789")
    assert error.value.code is ErrorCode.UNAUTHORIZED


def test_research_catalog_validates_kinds_duplicates_and_unavailable_regime() -> None:
    catalog = ResearchCatalog()
    with pytest.raises(QuantAgentError) as unavailable:
        catalog.regime()
    assert unavailable.value.code is ErrorCode.DATA_UNAVAILABLE
    record = ResearchRecord("theme-a", NOW, "market_v1", {"state": "CONFIRMED"})
    with pytest.raises(ValueError, match="unsupported"):
        catalog.publish("unknown", [record])
    with pytest.raises(ValueError, match="unique"):
        catalog.publish("themes", [record, record])
    catalog.publish("themes", [record])
    with pytest.raises(QuantAgentError) as missing_collection:
        catalog.list("unknown", offset=0, limit=10, filters={})
    assert missing_collection.value.code is ErrorCode.NOT_FOUND


def test_approval_token_has_independent_ttl_and_invalid_configuration_fails() -> None:
    with pytest.raises(ValueError, match="signing secret"):
        OrderApprovalService(b"short", KillSwitch(), lambda item: {})
    with pytest.raises(ValueError, match="TTL"):
        OrderApprovalService(
            b"x" * 32,
            KillSwitch(),
            lambda item: {},
            approval_ttl=timedelta(minutes=30),
        )
    service = OrderApprovalService(b"x" * 32, KillSwitch(), lambda item: {"ok": True})
    service.put_draft(draft())
    approval = service.approve(
        "draft-1",
        approver_id="approver",
        approved_at=NOW,
        request_id="req-1",
    )
    assert approval.expires_at == NOW + timedelta(minutes=5)
    with pytest.raises(QuantAgentError) as expired:
        service.submit(
            "draft-1",
            approval_token=approval.token,
            idempotency_key="idem-1",
            submitted_at=NOW + timedelta(minutes=6),
            submitted_by="trader",
            request_id="req-2",
        )
    assert expired.value.code is ErrorCode.APPROVAL_EXPIRED
