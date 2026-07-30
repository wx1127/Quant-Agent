"""Treat all external text as untrusted data, never as executable instructions."""

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContentSecurityResult:
    raw_hash: str
    sanitized_text: str
    untrusted: bool
    instruction_detected: bool
    truncated: bool
    audit_reasons: tuple[str, ...]


class ExternalContentSanitizer:
    """Remove instruction-like spans and enforce deterministic text limits."""

    _patterns = (
        re.compile(r"(?i)ignore\s+(all\s+)?(previous|system)\s+instructions?"),
        re.compile(r"忽略(?:以上|之前|系统)(?:所有)?(?:规则|指令|提示)"),
        re.compile(r"(?i)(call|invoke|run)\s+(the\s+)?tool"),
        re.compile(r"(?:调用|执行|启动)(?:任意|所有|以下)?(?:工具|命令|交易)"),
        re.compile(r"(?:立即|马上)?(?:买入|卖出)[^\n。]{0,40}"),
    )

    def __init__(self, *, max_length: int = 20_000) -> None:
        if max_length < 100:
            raise ValueError("max_length must be at least 100")
        self._max_length = max_length

    def sanitize(self, content: str) -> ContentSecurityResult:
        raw_hash = hashlib.sha256(content.encode()).hexdigest()
        normalized = content.replace("\x00", "").replace("\r\n", "\n")
        reasons: list[str] = []
        detected = False
        for pattern in self._patterns:
            normalized, count = pattern.subn("[已隔离的指令性文本]", normalized)
            if count:
                detected = True
                reasons.append(f"instruction pattern isolated: {pattern.pattern}")
        truncated = len(normalized) > self._max_length
        if truncated:
            normalized = normalized[: self._max_length]
            reasons.append("content truncated to configured limit")
        if "\x00" in content:
            reasons.append("null bytes removed")
        return ContentSecurityResult(
            raw_hash=raw_hash,
            sanitized_text=normalized,
            untrusted=True,
            instruction_detected=detected,
            truncated=truncated,
            audit_reasons=tuple(reasons),
        )
