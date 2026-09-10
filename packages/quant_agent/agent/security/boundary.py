"""Deterministic user/external-content boundary for Agent orchestration."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from typing import Final

from quant_agent.agent.content_security import ContentDecision, ContentSanitizer
from quant_agent.agent.runtime import AgentRunSnapshot
from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.agent.tools.contracts import ToolEffect, validate_tool_name
from quant_agent.agent.tools.policy import allowed_v1_tool_names, get_v1_tool_policy
from quant_agent.core.time import ensure_aware

from .contracts import (
    AgentInputEnvelope,
    ExternalContentInput,
    InputDisposition,
    SecurityFinding,
    SecurityFindingCode,
    SecuritySource,
    UntrustedContentBlock,
    stable_security_hash,
)
from .text import SensitiveTextGuard

_MODE_OVERRIDE = re.compile(
    r"(?i)(?:切换|更改|改成|设为|启用|开启).{0,30}(?:模式|实盘|自动交易|"
    r"live[_ -]?auto|live[_ -]?assisted|paper|research|backtest)"
    r"|\b(?:switch|change|set|enable)\b.{0,30}\b(?:runtime[ _-]?mode|"
    r"live[_ -]?auto|live[_ -]?assisted|paper|research|backtest)\b"
)
_PRIVILEGE = re.compile(
    r"(?i)(?:批准订单|审批订单|修改风控|调整风控阈值|关闭停机|禁用停机|恢复停机|绕过风控|"
    r"approve[ _-]?order|change[ _-]?risk|disable[ _-]?(?:kill|safety)|"
    r"recover[ _-]?kill|bypass[ _-]?risk)"
)
_OVERRIDE = re.compile(
    r"(?i)\b(?:ignore|disregard|forget|override|bypass)\b.{0,50}\b(?:system|developer|instruction|policy|rule|prompt)\b"
    r"|(?:忽略|无视|绕过|覆盖).{0,30}(?:系统|指令|提示|规则|策略)"
)
_SECRET_REQUEST = re.compile(
    r"(?i)(?:输出|显示|泄露|给我|reveal|show|print|send).{0,40}(?:密钥|令牌|密码|凭证|"
    r"api[ _-]?key|token|secret|password)"
)
_EXECUTION_EFFECTS: Final[frozenset[ToolEffect]] = frozenset(
    {
        ToolEffect.ARTIFACT_WRITE,
        ToolEffect.PAPER_EXECUTION_WRITE,
        ToolEffect.LIVE_EXTERNAL_WRITE,
    }
)


class AgentSecurityBoundaryError(ValueError):
    """Safe boundary error without user or external text."""


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _authority_findings(text: str) -> tuple[SecurityFindingCode, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    codes: list[SecurityFindingCode] = []
    if _MODE_OVERRIDE.search(normalized) or any(
        token in _compact(normalized)
        for token in ("切换模式", "启用实盘", "enableliveauto", "switchruntimemode")
    ):
        codes.append(SecurityFindingCode.USER_MODE_OVERRIDE)
    if _PRIVILEGE.search(normalized):
        codes.append(SecurityFindingCode.USER_PRIVILEGE_ESCALATION)
    if _OVERRIDE.search(normalized):
        codes.append(SecurityFindingCode.USER_INSTRUCTION_OVERRIDE)
    if _SECRET_REQUEST.search(normalized):
        codes.append(SecurityFindingCode.USER_SECRET_REQUEST)
    return tuple(dict.fromkeys(codes))


def _finding(
    code: SecurityFindingCode,
    source: SecuritySource,
    content_id: str,
) -> SecurityFinding:
    return SecurityFinding(code=code, source=source, content_id=content_id)


class AgentSecurityBoundary:
    """Build a server-owned prompt context with no authority in external text."""

    def __init__(
        self,
        *,
        sanitizer: ContentSanitizer | None = None,
        text_guard: SensitiveTextGuard | None = None,
    ) -> None:
        self._sanitizer = sanitizer or ContentSanitizer()
        self._text_guard = text_guard or SensitiveTextGuard()

    def prepare(
        self,
        *,
        decision: DecisionSnapshot,
        run: AgentRunSnapshot,
        user_request: str,
        external_content: tuple[ExternalContentInput, ...] = (),
        available_tool_names: tuple[str, ...] = (),
        prepared_at: datetime,
    ) -> AgentInputEnvelope:
        """Return a bounded envelope; mode and tools never come from user text."""

        if type(decision) is not DecisionSnapshot or type(run) is not AgentRunSnapshot:
            raise TypeError("decision and run must be exact trusted snapshot types")
        try:
            decision = DecisionSnapshot.from_json(decision.to_json())
            run = AgentRunSnapshot.from_json(run.to_json())
        except (TypeError, ValueError) as error:
            raise AgentSecurityBoundaryError(
                "trusted security inputs failed integrity validation"
            ) from error
        if (
            run.decision_id != decision.decision_id
            or run.decision_snapshot_hash != decision.content_hash
            or run.runtime_mode is not decision.mode
        ):
            raise AgentSecurityBoundaryError("run does not bind the decision snapshot")
        if type(user_request) is not str or not user_request.strip() or len(user_request) > 20_000:
            raise ValueError("user_request must be non-empty and bounded")
        if type(external_content) is not tuple or any(
            type(item) is not ExternalContentInput for item in external_content
        ):
            raise TypeError("external_content must be an exact tuple of ExternalContentInput")
        now = ensure_aware(prepared_at).astimezone(UTC)
        protected = (decision.account_id, decision.account_snapshot_id)
        findings: list[SecurityFinding] = []
        user_id = f"user:{_text_hash(user_request)}"
        authority = _authority_findings(user_request)
        redacted_user = self._text_guard.redact(user_request, protected_values=protected)
        if redacted_user.findings:
            findings.append(
                _finding(SecurityFindingCode.USER_SECRET_REDACTED, SecuritySource.USER, user_id)
            )
        findings.extend(_finding(code, SecuritySource.USER, user_id) for code in authority)
        if authority:
            disposition = InputDisposition.REJECTED
            user_intent = None
        elif redacted_user.findings:
            disposition = InputDisposition.ACCEPTED_WITH_REDACTIONS
            user_intent = redacted_user.text
        else:
            disposition = InputDisposition.ACCEPTED
            user_intent = redacted_user.text

        blocks: list[UntrustedContentBlock] = []
        for item in external_content:
            source_hash = _text_hash(item.text)
            event_id = _text_hash(f"{run.run_id}:{item.source_id}:{source_hash}")
            sanitized = self._sanitizer.sanitize(
                event_id=event_id,
                raw_payload_id=item.raw_payload_id,
                text=item.text,
            )
            safe_text = None
            if (
                sanitized.decision is ContentDecision.ACCEPTED
                and sanitized.sanitized_text is not None
            ):
                safe_result = self._text_guard.redact(
                    sanitized.sanitized_text, protected_values=protected
                )
                safe_text = safe_result.text
                if safe_result.findings:
                    findings.append(
                        _finding(
                            SecurityFindingCode.EXTERNAL_CONTENT_REDACTED,
                            SecuritySource.EXTERNAL_CONTENT,
                            item.source_id,
                        )
                    )
            else:
                findings.append(
                    _finding(
                        SecurityFindingCode.EXTERNAL_CONTENT_QUARANTINED,
                        SecuritySource.EXTERNAL_CONTENT,
                        item.source_id,
                    )
                )
            blocks.append(
                UntrustedContentBlock(
                    source_id=item.source_id,
                    raw_payload_id=item.raw_payload_id,
                    source_content_hash=sanitized.source_content_hash,
                    sanitizer_result_hash=sanitized.result_hash,
                    decision=sanitized.decision,
                    safe_text=safe_text,
                )
            )

        tools = self._validate_tools(run, available_tool_names)
        execution_tools = tuple(
            name
            for name in tools
            if (policy := get_v1_tool_policy(name)) is not None
            and policy.effect in _EXECUTION_EFFECTS
        )
        if blocks and execution_tools:
            tools = tuple(
                name
                for name in tools
                if (policy := get_v1_tool_policy(name)) is not None
                and policy.effect not in _EXECUTION_EFFECTS
            )
            findings.append(
                _finding(
                    SecurityFindingCode.EXECUTION_TOOLS_SUPPRESSED,
                    SecuritySource.SYSTEM,
                    run.run_id,
                )
            )
        values = {
            "schema_version": "1",
            "policy_version": "agent-security-policy-v1",
            "run_id": run.run_id,
            "decision_id": decision.decision_id,
            "runtime_mode": run.runtime_mode,
            "goal": run.goal,
            "run_state": run.state,
            "user_request_hash": _text_hash(user_request),
            "user_intent": user_intent,
            "user_disposition": disposition,
            "external_content": tuple(blocks),
            "available_tool_names": tools,
            "findings": tuple(findings),
            "prepared_at": now,
        }
        return AgentInputEnvelope.model_validate(
            {**values, "envelope_hash": stable_security_hash(values)}, strict=True
        )

    @staticmethod
    def _validate_tools(run: AgentRunSnapshot, names: tuple[str, ...]) -> tuple[str, ...]:
        if type(names) is not tuple:
            raise TypeError("available_tool_names must be an exact tuple")
        result: list[str] = []
        allowed = allowed_v1_tool_names(run.runtime_mode)
        for name in names:
            try:
                validate_tool_name(name)
            except (TypeError, ValueError) as error:
                raise AgentSecurityBoundaryError("available tool name is invalid") from error
            policy = get_v1_tool_policy(name)
            if name not in allowed or policy is None:
                raise AgentSecurityBoundaryError("available tool is outside the server allow-list")
            result.append(name)
        return tuple(sorted(set(result)))


__all__ = ["AgentSecurityBoundary", "AgentSecurityBoundaryError"]
