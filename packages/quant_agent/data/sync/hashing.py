"""Canonical JSON identities for synchronization scope and provider requests."""

import hashlib
import json
import re
from collections.abc import Mapping
from typing import cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_SECRET_KEY_NORMALIZER = re.compile(r"[^a-z0-9]")
_SECRET_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "llmapikey",
        "marketdatatoken",
        "password",
        "refreshtoken",
        "secret",
        "token",
    }
)


def canonical_json(value: JsonValue) -> str:
    """Serialize a JSON value deterministically and reject non-standard floats."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def clone_json_object(value: Mapping[str, JsonValue]) -> JsonObject:
    """Return a detached, JSON-normalized object suitable for an ORM JSON column."""

    encoded = canonical_json(dict(value))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guaranteed by the input type
        raise TypeError("JSON object clone did not produce an object")
    return cast(JsonObject, decoded)


def canonical_hash(value: JsonValue) -> str:
    """Return the SHA-256 identity of canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_scope_hash(scope: Mapping[str, JsonValue]) -> str:
    """Return a stable identity for a logical checkpoint scope."""

    return canonical_hash(dict(scope))


def _is_secret_key(key: str) -> bool:
    normalized = _SECRET_KEY_NORMALIZER.sub("", key.casefold())
    return normalized in _SECRET_KEYS


def _sanitize(value: JsonValue, *, pagination_scope: bool = False) -> JsonValue:
    if isinstance(value, dict):
        sanitized: JsonObject = {}
        for key, item in value.items():
            normalized = _SECRET_KEY_NORMALIZER.sub("", key.casefold())
            pagination_token = pagination_scope and normalized == "token"
            if _is_secret_key(key) and not pagination_token:
                continue
            sanitized[key] = _sanitize(
                item,
                pagination_scope=normalized in {"cursor", "pagecursor", "pagination"},
            )
        return sanitized
    if isinstance(value, list):
        return [_sanitize(item, pagination_scope=pagination_scope) for item in value]
    return value


def sanitized_request_params(params: Mapping[str, JsonValue]) -> JsonObject:
    """Remove credential fields recursively while preserving pagination tokens."""

    sanitized = _sanitize(dict(params))
    if not isinstance(sanitized, dict):  # pragma: no cover - guaranteed by the input type
        raise TypeError("sanitized request parameters must be an object")
    return sanitized


def canonical_request_hash(
    *,
    provider: str,
    endpoint: str,
    params: Mapping[str, JsonValue],
) -> str:
    """Hash a logical provider request without allowing credentials into its identity."""

    if not provider or not endpoint:
        raise ValueError("provider and endpoint must be non-empty")
    identity: JsonObject = {
        "provider": provider,
        "endpoint": endpoint,
        "params": sanitized_request_params(params),
    }
    return canonical_hash(identity)
