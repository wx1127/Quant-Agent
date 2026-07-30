"""Prompt-injection, mode-escalation, argument and output guards."""

import re
from dataclasses import dataclass
from typing import Any

from quant_agent.agent.content_security import ContentSecurityResult, ExternalContentSanitizer
from quant_agent.config.models import RuntimeMode
from quant_agent.observability.redaction import redact

_MODE_ESCALATION = re.compile(
    r"(?i)(切换|改成|进入|switch|change).{0,20}(LIVE_AUTO|LIVE_ASSISTED|PAPER|实盘|自动交易)"
)
_CREDENTIAL_VALUE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{12,}|bearer\s+[a-z0-9._-]{12,}|"
    r"(?:api[_-]?key|token|password)\s*[:=]\s*\S+)"
)
_ACCOUNT_NUMBER = re.compile(r"(?<!\d)\d{8,24}(?!\d)")


@dataclass(frozen=True, slots=True)
class UserInstructionAssessment:
    allowed: bool
    configured_mode: RuntimeMode
    reasons: tuple[str, ...]


class AgentSecurityGuard:
    def __init__(self, *, configured_mode: RuntimeMode, allowed_instruments: set[str]) -> None:
        self._mode = configured_mode
        self._allowed_instruments = frozenset(allowed_instruments)
        self._sanitizer = ExternalContentSanitizer()

    def isolate_external_content(self, content: str) -> ContentSecurityResult:
        return self._sanitizer.sanitize(content)

    def assess_user_instruction(self, content: str) -> UserInstructionAssessment:
        reasons = ("runtime mode is server-controlled",) if _MODE_ESCALATION.search(content) else ()
        return UserInstructionAssessment(not reasons, self._mode, reasons)

    def validate_tool_arguments(self, arguments: dict[str, Any]) -> None:
        forbidden = {"tool", "tool_name", "command", "system_prompt", "approval_token"}
        lowered = {str(key).casefold() for key in arguments}
        overlap = forbidden & lowered
        if overlap:
            raise ValueError(f"forbidden tool argument fields: {sorted(overlap)}")
        instrument = arguments.get("instrument_id")
        if instrument is not None and instrument not in self._allowed_instruments:
            raise ValueError("instrument is outside the server-qualified universe")

    @staticmethod
    def filter_output(value: Any) -> Any:
        redacted = redact(value)
        if isinstance(redacted, str):
            redacted = _CREDENTIAL_VALUE.sub("[REDACTED]", redacted)
            redacted = _ACCOUNT_NUMBER.sub(_mask_account, redacted)
        if isinstance(redacted, dict):
            return {key: AgentSecurityGuard.filter_output(item) for key, item in redacted.items()}
        if isinstance(redacted, (list, tuple)):
            return [AgentSecurityGuard.filter_output(item) for item in redacted]
        return redacted


def _mask_account(match: re.Match[str]) -> str:
    value = match.group()
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
