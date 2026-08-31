"""Deterministic quarantine boundary for untrusted announcements and news text."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from html.parser import HTMLParser

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_agent.core.time import ensure_aware
from quant_agent.data.sync.hashing import JsonObject, canonical_hash
from quant_agent.observability.audit import AuditEvent

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PROMPT_ROLE_MARKER",
        re.compile(
            r"(?im)(?:^|\n)\s*(?:system|developer|assistant|tool)\s*:|<\|(?:system|developer|assistant|tool)\|>"
        ),
    ),
    (
        "INSTRUCTION_OVERRIDE",
        re.compile(
            r"(?is)\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}\b(?:instruction|prompt|policy|system|rule)\b"
            r"|(?:忽略|无视|覆盖|绕过).{0,30}(?:指令|提示词|系统|规则|策略)"
        ),
    ),
    (
        "TOOL_OR_ORDER_INSTRUCTION",
        re.compile(
            r"(?is)\b(?:call|invoke|execute|run|submit|approve)\b.{0,60}\b(?:tool|function|command|order|trade)\b"
            r"|(?:调用|执行|提交|批准).{0,30}(?:工具|函数|命令|订单|交易)"
        ),
    ),
    (
        "SECRET_EXTRACTION",
        re.compile(
            r"(?is)\b(?:reveal|print|send|exfiltrate|show)\b.{0,60}"
            r"\b(?:secret|token|password|api[ _-]?key|credential)\b"
            r"|(?:泄露|输出|发送|显示).{0,30}(?:密钥|令牌|密码|凭证)"
        ),
    ),
)


class ContentDecision(StrEnum):
    """Whether normalized text may enter deterministic fact extraction."""

    ACCEPTED = "ACCEPTED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class ContentFindingSeverity(StrEnum):
    """Stable severity for audit and downstream policy."""

    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


class ContentFinding(BaseModel):
    """A text-free finding so malicious content never enters logs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: ContentFindingSeverity
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized or not normalized.replace("_", "").isalnum():
            raise ValueError("finding code must be an uppercase identifier")
        return normalized

    @model_validator(mode="after")
    def validate_span(self) -> ContentFinding:
        if self.end < self.start:
            raise ValueError("finding end cannot precede start")
        return self


class ContentSecurityPolicy(BaseModel):
    """Versioned, non-disableable limits for the first sanitizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "content-security-v1"
    max_characters: int = Field(default=20_000, ge=1, le=200_000)
    max_lines: int = Field(default=500, ge=1, le=5_000)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content security policy version must be non-empty")
        return normalized

    @property
    def config_hash(self) -> str:
        return canonical_hash(
            {
                "max_characters": self.max_characters,
                "max_lines": self.max_lines,
                "pattern_set": "prompt-injection-v1",
                "version": self.version,
            }
        )


class SanitizedContent(BaseModel):
    """Auditable result that never carries quarantined source text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    raw_payload_id: int = Field(gt=0)
    source_content_hash: str
    normalized_text_hash: str
    policy_version: str
    policy_config_hash: str
    decision: ContentDecision
    sanitized_text: str | None
    original_length: int = Field(ge=0)
    normalized_length: int = Field(ge=0)
    truncated: bool
    findings: tuple[ContentFinding, ...]
    result_hash: str

    @field_validator(
        "event_id",
        "source_content_hash",
        "normalized_text_hash",
        "policy_config_hash",
        "result_hash",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _SHA256.fullmatch(normalized) is None:
            raise ValueError("content identities must be SHA-256 hexadecimal digests")
        return normalized

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy_version must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_decision_and_identity(self) -> SanitizedContent:
        if tuple(sorted(self.findings, key=lambda item: (item.start, item.end, item.code))) != (
            self.findings
        ):
            raise ValueError("content findings must be in deterministic order")
        blocking = any(item.severity is ContentFindingSeverity.BLOCKING for item in self.findings)
        if self.decision is ContentDecision.ACCEPTED:
            if not self.sanitized_text or blocking:
                raise ValueError("accepted content requires non-blocking sanitized text")
            if len(self.sanitized_text) != self.normalized_length:
                raise ValueError("normalized_length must match sanitized_text")
            actual_text_hash = hashlib.sha256(self.sanitized_text.encode("utf-8")).hexdigest()
            if actual_text_hash != self.normalized_text_hash:
                raise ValueError("normalized_text_hash does not match sanitized_text")
        elif self.sanitized_text is not None:
            raise ValueError("quarantined or rejected content cannot carry sanitized text")
        if self.decision is ContentDecision.QUARANTINED and not blocking:
            raise ValueError("quarantined content requires a blocking finding")
        payload: JsonObject = {
            "decision": self.decision.value,
            "event_id": self.event_id,
            "findings": [
                {
                    "code": item.code,
                    "end": item.end,
                    "severity": item.severity.value,
                    "start": item.start,
                }
                for item in self.findings
            ],
            "normalized_length": self.normalized_length,
            "normalized_text_hash": self.normalized_text_hash,
            "original_length": self.original_length,
            "policy_config_hash": self.policy_config_hash,
            "policy_version": self.policy_version,
            "raw_payload_id": self.raw_payload_id,
            "sanitized_text": self.sanitized_text,
            "source_content_hash": self.source_content_hash,
            "truncated": self.truncated,
        }
        if canonical_hash(payload) != self.result_hash:
            raise ValueError("result_hash does not match the sanitized content result")
        return self

    @property
    def usable_for_facts(self) -> bool:
        return self.decision is ContentDecision.ACCEPTED


class _SafeTextExtractor(HTMLParser):
    """Extract display text while dropping active HTML content entirely."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.blocked_depth = 0
        self.removed_active_content = False

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "iframe", "object"}:
            self.blocked_depth += 1
            self.removed_active_content = True
        elif not self.blocked_depth and normalized == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "iframe", "object"} and self.blocked_depth:
            self.blocked_depth -= 1
        elif not self.blocked_depth and normalized in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.blocked_depth:
            self.parts.append(data)


class ContentSanitizer:
    """Normalize safe text or quarantine it without interpreting business facts."""

    def __init__(self, policy: ContentSecurityPolicy | None = None) -> None:
        self._policy = policy or ContentSecurityPolicy()

    @property
    def policy(self) -> ContentSecurityPolicy:
        return self._policy

    def sanitize(
        self,
        *,
        event_id: str,
        raw_payload_id: int,
        text: str,
        expected_source_hash: str | None = None,
    ) -> SanitizedContent:
        """Return deterministic text or a text-free quarantine result."""

        normalized_event_id = _hash_value(event_id, "event_id")
        if raw_payload_id <= 0:
            raise ValueError("raw_payload_id must be positive")
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_source_hash is not None and source_hash != _hash_value(
            expected_source_hash,
            "expected_source_hash",
        ):
            raise ValueError("source text hash does not match the expected raw content")

        normalized, active_content_removed = _normalize_text(text)
        original_length = len(text)
        findings = _detect_findings(normalized)
        if active_content_removed:
            findings.append(
                ContentFinding(
                    code="ACTIVE_HTML_CONTENT",
                    severity=ContentFindingSeverity.BLOCKING,
                    start=0,
                    end=0,
                )
            )

        limited, truncated, limit_findings = self._apply_limits(normalized)
        findings.extend(limit_findings)
        findings = sorted(findings, key=lambda item: (item.start, item.end, item.code))
        normalized_hash = hashlib.sha256(limited.encode("utf-8")).hexdigest()
        blocking = any(item.severity is ContentFindingSeverity.BLOCKING for item in findings)
        if not limited:
            decision = ContentDecision.REJECTED
            sanitized_text = None
        elif blocking:
            decision = ContentDecision.QUARANTINED
            sanitized_text = None
        else:
            decision = ContentDecision.ACCEPTED
            sanitized_text = limited

        payload = _result_payload(
            event_id=normalized_event_id,
            raw_payload_id=raw_payload_id,
            source_content_hash=source_hash,
            normalized_text_hash=normalized_hash,
            policy=self._policy,
            decision=decision,
            sanitized_text=sanitized_text,
            original_length=original_length,
            normalized_length=len(limited),
            truncated=truncated,
            findings=tuple(findings),
        )
        return SanitizedContent(
            event_id=normalized_event_id,
            raw_payload_id=raw_payload_id,
            source_content_hash=source_hash,
            normalized_text_hash=normalized_hash,
            policy_version=self._policy.version,
            policy_config_hash=self._policy.config_hash,
            decision=decision,
            sanitized_text=sanitized_text,
            original_length=original_length,
            normalized_length=len(limited),
            truncated=truncated,
            findings=tuple(findings),
            result_hash=canonical_hash(payload),
        )

    def audit_event(
        self,
        result: SanitizedContent,
        *,
        request_id: str,
        occurred_at: datetime,
    ) -> AuditEvent:
        """Build a text-free append-only audit event."""

        return AuditEvent(
            event_type="CONTENT_SECURITY",
            actor_id="content-sanitizer",
            action="sanitize_external_content",
            result=result.decision.value,
            request_id=request_id,
            decision_id=result.event_id,
            occurred_at=ensure_aware(occurred_at),
            metadata={
                "finding_codes": [item.code for item in result.findings],
                "normalized_text_hash": result.normalized_text_hash,
                "policy_config_hash": result.policy_config_hash,
                "raw_payload_id": result.raw_payload_id,
                "result_hash": result.result_hash,
                "source_content_hash": result.source_content_hash,
                "truncated": result.truncated,
            },
        )

    def _apply_limits(
        self,
        text: str,
    ) -> tuple[str, bool, list[ContentFinding]]:
        findings: list[ContentFinding] = []
        lines = text.splitlines()
        truncated = len(lines) > self._policy.max_lines
        if truncated:
            lines = lines[: self._policy.max_lines]
            findings.append(
                ContentFinding(
                    code="LINE_LIMIT",
                    severity=ContentFindingSeverity.WARNING,
                    start=len("\n".join(lines)),
                    end=len(text),
                )
            )
        limited = "\n".join(lines)
        if len(limited) > self._policy.max_characters:
            findings.append(
                ContentFinding(
                    code="CHARACTER_LIMIT",
                    severity=ContentFindingSeverity.WARNING,
                    start=self._policy.max_characters,
                    end=len(limited),
                )
            )
            limited = limited[: self._policy.max_characters].rstrip()
            truncated = True
        return limited, truncated, findings


def _hash_value(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return normalized


def _normalize_text(value: str) -> tuple[str, bool]:
    extractor = _SafeTextExtractor()
    extractor.feed(unicodedata.normalize("NFKC", value))
    extractor.close()
    joined = "".join(extractor.parts).replace("\r\n", "\n").replace("\r", "\n")
    safe_characters = "".join(
        character
        for character in joined
        if character in {"\n", "\t"} or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )
    lines = (_SPACE.sub(" ", line).strip() for line in safe_characters.split("\n"))
    normalized = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return normalized, extractor.removed_active_content


def _detect_findings(text: str) -> list[ContentFinding]:
    findings: list[ContentFinding] = []
    for code, pattern in _INSTRUCTION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            findings.append(
                ContentFinding(
                    code=code,
                    severity=ContentFindingSeverity.BLOCKING,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return findings


def _result_payload(
    *,
    event_id: str,
    raw_payload_id: int,
    source_content_hash: str,
    normalized_text_hash: str,
    policy: ContentSecurityPolicy,
    decision: ContentDecision,
    sanitized_text: str | None,
    original_length: int,
    normalized_length: int,
    truncated: bool,
    findings: tuple[ContentFinding, ...],
) -> JsonObject:
    return {
        "decision": decision.value,
        "event_id": event_id,
        "findings": [
            {
                "code": item.code,
                "end": item.end,
                "severity": item.severity.value,
                "start": item.start,
            }
            for item in findings
        ],
        "normalized_length": normalized_length,
        "normalized_text_hash": normalized_text_hash,
        "original_length": original_length,
        "policy_config_hash": policy.config_hash,
        "policy_version": policy.version,
        "raw_payload_id": raw_payload_id,
        "sanitized_text": sanitized_text,
        "source_content_hash": source_content_hash,
        "truncated": truncated,
    }


__all__ = [
    "ContentDecision",
    "ContentFinding",
    "ContentFindingSeverity",
    "ContentSanitizer",
    "ContentSecurityPolicy",
    "SanitizedContent",
]
