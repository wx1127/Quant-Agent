"""Canonical numeric identities shared by versioned feature configurations."""

from decimal import Decimal


def canonical_decimal(value: Decimal, *, field_name: str = "decimal") -> str:
    """Return one scale-independent text identity and reject NaN or infinity."""

    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")
