"""Decimal-based monetary value objects."""

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict, field_validator

_CENT = Decimal("0.01")


class Money(BaseModel):
    """Currency amount that avoids binary floating-point arithmetic."""

    model_config = ConfigDict(frozen=True)

    amount: Decimal
    currency: str = "CNY"

    @field_validator("amount", mode="before")
    @classmethod
    def parse_amount(cls, value: object) -> Decimal:
        """Parse strings and integers while rejecting float ambiguity."""

        if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
            raise ValueError("Money amount must be a Decimal, integer, or string")
        return Decimal(value) if not isinstance(value, Decimal) else value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize and validate ISO-like currency codes."""

        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return normalized

    def rounded(self) -> "Money":
        """Return a value rounded to cents for display or settlement."""

        return Money(
            amount=self.amount.quantize(_CENT, rounding=ROUND_HALF_UP),
            currency=self.currency,
        )
