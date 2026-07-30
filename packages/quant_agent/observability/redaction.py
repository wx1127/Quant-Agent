"""Conservative recursive redaction for logs and audit metadata."""

from collections.abc import Mapping, Sequence
from typing import Any

_SENSITIVE_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REDACTED = "[REDACTED]"


def _is_sensitive(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in _SENSITIVE_PARTS)


def redact(value: Any) -> Any:
    """Return a recursively redacted copy of common JSON-compatible objects."""

    if isinstance(value, Mapping):
        return {
            key: _REDACTED if _is_sensitive(key) else redact(item) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    return value
