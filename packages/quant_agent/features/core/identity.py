"""Canonical numeric identities shared by versioned feature configurations."""

from decimal import Decimal


def canonical_decimal(value: Decimal, *, field_name: str = "decimal") -> str:
    """Return one scale-independent text identity and reject NaN or infinity."""

    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    sign, raw_digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} has an invalid exponent")
    if not any(raw_digits):
        return "0"
    digits = list(raw_digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    point = len(coefficient) + exponent
    if point <= 0:
        rendered = f"0.{('0' * -point)}{coefficient}"
    elif point >= len(coefficient):
        rendered = f"{coefficient}{'0' * (point - len(coefficient))}"
    else:
        rendered = f"{coefficient[:point]}.{coefficient[point:]}"
    return f"-{rendered}" if sign else rendered
