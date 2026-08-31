"""Unit tests for credential-free canonical synchronization identities."""

import pytest

from quant_agent.data.sync.hashing import (
    canonical_hash,
    canonical_request_hash,
    canonical_scope_hash,
    sanitized_request_params,
)


def test_scope_hash_is_stable_across_mapping_order() -> None:
    first = {"market": "CN", "filters": {"type": "STOCK", "active": True}}
    second = {"filters": {"active": True, "type": "STOCK"}, "market": "CN"}

    assert canonical_scope_hash(first) == canonical_scope_hash(second)


def test_request_hash_excludes_credentials_but_keeps_pagination_cursor() -> None:
    first = {
        "token": "first-secret",
        "offset": 0,
        "next_token": "provider-cursor-1",
        "headers": {"Authorization": "Bearer first", "X-Trace": "trace"},
    }
    second = {
        "token": "second-secret",
        "offset": 0,
        "next_token": "provider-cursor-1",
        "headers": {"Authorization": "Bearer second", "X-Trace": "trace"},
    }
    next_page = {**second, "next_token": "provider-cursor-2"}

    first_hash = canonical_request_hash(provider="tushare", endpoint="daily", params=first)
    second_hash = canonical_request_hash(provider="tushare", endpoint="daily", params=second)

    assert first_hash == second_hash
    assert first_hash != canonical_request_hash(
        provider="tushare",
        endpoint="daily",
        params=next_page,
    )
    assert sanitized_request_params(first) == {
        "offset": 0,
        "next_token": "provider-cursor-1",
        "headers": {"X-Trace": "trace"},
    }


def test_request_sanitizing_descends_into_lists_and_requires_request_identity() -> None:
    assert sanitized_request_params(
        {
            "requests": [{"token": "secret", "offset": 0}, {"offset": 1}],
            "cursor": {"token": "page-2"},
        }
    ) == {
        "requests": [{"offset": 0}, {"offset": 1}],
        "cursor": {"token": "page-2"},
    }
    with pytest.raises(ValueError, match="provider and endpoint"):
        canonical_request_hash(provider="", endpoint="daily", params={})


def test_canonical_hash_rejects_non_standard_nan() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_hash({"invalid": float("nan")})
