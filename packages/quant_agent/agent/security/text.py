"""Deterministic secret, credential, and account-identifier filtering."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

_REDACTED: Final = "[REDACTED]"
_NAMED_SECRET = re.compile(
    r"(?i)\b(?:api[ _-]?key|access[ _-]?token|refresh[ _-]?token|token|secret|password|"
    r"authorization|credential)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;]{4,}"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PROVIDER_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{16})\b"
)
_SECRET_REFERENCE = re.compile(r"(?i)\b(?:env|vault)://[A-Za-z_][A-Za-z0-9_.-]*")
_LONG_ACCOUNT_NUMBER = re.compile(r"(?<!\d)\d{12,32}(?!\d)")


class SensitiveFindingCode(StrEnum):
    """Text-free reason a value was removed or publication was blocked."""

    NAMED_SECRET = "NAMED_SECRET"
    BEARER_TOKEN = "BEARER_TOKEN"
    JWT = "JWT"
    PROVIDER_TOKEN = "PROVIDER_TOKEN"
    SECRET_REFERENCE = "SECRET_REFERENCE"
    PROTECTED_IDENTIFIER = "PROTECTED_IDENTIFIER"
    LONG_ACCOUNT_NUMBER = "LONG_ACCOUNT_NUMBER"


class SensitiveTextResult(BaseModel):
    """Safe text and non-sensitive finding codes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str
    findings: tuple[SensitiveFindingCode, ...] = ()


def iter_string_values(value: object) -> Iterable[str]:
    """Yield strings recursively without interpreting object keys as content."""

    if isinstance(value, BaseModel):
        yield from iter_string_values(value.model_dump(mode="python"))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_string_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from iter_string_values(item)
    elif type(value) is str:
        yield value


class SensitiveTextGuard:
    """Redact inbound text and reject sensitive outbound objects."""

    def redact(
        self,
        text: str,
        *,
        protected_values: tuple[str, ...] = (),
    ) -> SensitiveTextResult:
        """Return a copy with recognized sensitive values removed."""

        if type(text) is not str:
            raise TypeError("text must be an exact string")
        safe = unicodedata.normalize("NFKC", text)
        findings: list[SensitiveFindingCode] = []
        patterns = (
            (_NAMED_SECRET, SensitiveFindingCode.NAMED_SECRET),
            (_BEARER, SensitiveFindingCode.BEARER_TOKEN),
            (_JWT, SensitiveFindingCode.JWT),
            (_PROVIDER_TOKEN, SensitiveFindingCode.PROVIDER_TOKEN),
            (_SECRET_REFERENCE, SensitiveFindingCode.SECRET_REFERENCE),
            (_LONG_ACCOUNT_NUMBER, SensitiveFindingCode.LONG_ACCOUNT_NUMBER),
        )
        for pattern, code in patterns:
            safe, count = pattern.subn(_REDACTED, safe)
            if count:
                findings.append(code)
        for protected in protected_values:
            if type(protected) is not str or len(protected) < 8:
                continue
            safe, count = re.subn(re.escape(protected), _REDACTED, safe, flags=re.IGNORECASE)
            if count:
                findings.append(SensitiveFindingCode.PROTECTED_IDENTIFIER)
        return SensitiveTextResult(text=safe, findings=tuple(dict.fromkeys(findings)))

    def assert_safe(
        self,
        value: object,
        *,
        protected_values: tuple[str, ...] = (),
    ) -> None:
        """Reject if any string in an outbound object would require redaction."""

        for text in iter_string_values(value):
            result = self.redact(text, protected_values=protected_values)
            if result.findings:
                raise SensitiveOutputBlocked(result.findings)


class SensitiveOutputBlocked(ValueError):
    """Safe exception containing codes but never the rejected text."""

    def __init__(self, findings: tuple[SensitiveFindingCode, ...]) -> None:
        super().__init__("sensitive Agent output was blocked")
        self.findings = findings


__all__ = [
    "SensitiveFindingCode",
    "SensitiveOutputBlocked",
    "SensitiveTextGuard",
    "SensitiveTextResult",
    "iter_string_values",
]
