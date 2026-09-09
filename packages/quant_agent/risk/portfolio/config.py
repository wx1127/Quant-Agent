"""Strict TOML loader for immutable portfolio risk thresholds."""

from __future__ import annotations

import tomllib
from dataclasses import fields
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .contracts import PortfolioRiskPolicy

_SECTION = "portfolio_risk"
_DECIMAL_FIELDS = frozenset(
    field.name for field in fields(PortfolioRiskPolicy) if field.name != "version"
)
_EXPECTED_FIELDS = frozenset({"version", *_DECIMAL_FIELDS})


def load_portfolio_risk_policy(path: str | Path) -> PortfolioRiskPolicy:
    """Load an exact policy; decimal thresholds must be quoted strings."""

    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    if set(payload) != {_SECTION} or not isinstance(payload[_SECTION], dict):
        raise ValueError("risk policy TOML must contain only [portfolio_risk]")
    values = payload[_SECTION]
    if set(values) != _EXPECTED_FIELDS:
        raise ValueError("portfolio risk policy fields do not match the contract")
    version = values["version"]
    if not isinstance(version, str):
        raise ValueError("portfolio risk policy version must be a string")
    decimals: dict[str, Decimal] = {}
    for field_name in _DECIMAL_FIELDS:
        raw = values[field_name]
        if not isinstance(raw, str):
            raise ValueError(f"{field_name} must be a quoted exact decimal")
        try:
            decimals[field_name] = Decimal(raw)
        except InvalidOperation as error:
            raise ValueError(f"{field_name} must be a quoted exact decimal") from error
    return PortfolioRiskPolicy(version=version, **decimals)


__all__ = ["load_portfolio_risk_policy"]
