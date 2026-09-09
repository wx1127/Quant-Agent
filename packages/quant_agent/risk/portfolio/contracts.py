"""Versioned policy and point-in-time context for independent portfolio risk checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.regime.contracts import stable_hash

PORTFOLIO_RISK_ENGINE_VERSION = "portfolio-risk-engine-v1"


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _exact_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be an exact Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


def _sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class PortfolioRiskInputError(ValueError):
    """Raised internally when a risk request is not bound to its exact proposal inputs."""


class PortfolioRiskRuleCode(StrEnum):
    """Stable machine codes emitted by the first portfolio risk policy."""

    MAX_TOTAL_EXPOSURE = "MAX_TOTAL_EXPOSURE"
    REGIME_RISK_BUDGET = "REGIME_RISK_BUDGET"
    MAX_ETF_EXPOSURE = "MAX_ETF_EXPOSURE"
    MAX_STOCK_EXPOSURE = "MAX_STOCK_EXPOSURE"
    MAX_SINGLE_ETF_WEIGHT = "MAX_SINGLE_ETF_WEIGHT"
    MAX_SINGLE_STOCK_WEIGHT = "MAX_SINGLE_STOCK_WEIGHT"
    MAX_STOCK_INDUSTRY_WEIGHT = "MAX_STOCK_INDUSTRY_WEIGHT"
    TURNOVER_WARNING = "TURNOVER_WARNING"
    MAX_ONE_WAY_TURNOVER = "MAX_ONE_WAY_TURNOVER"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    DRAWDOWN_STOP_NEW_RISK = "DRAWDOWN_STOP_NEW_RISK"


@dataclass(frozen=True, slots=True)
class PortfolioRiskPolicy:
    """Immutable limits; tighter values can only reduce the set of passing proposals."""

    version: str = "portfolio-risk-policy-v1"
    maximum_total_weight: Decimal = Decimal("0.95")
    maximum_etf_weight: Decimal = Decimal("0.70")
    maximum_stock_weight: Decimal = Decimal("0.25")
    maximum_single_etf_weight: Decimal = Decimal("0.25")
    maximum_single_stock_weight: Decimal = Decimal("0.10")
    maximum_stock_industry_weight: Decimal = Decimal("0.20")
    turnover_warning_threshold: Decimal = Decimal("0.20")
    maximum_one_way_turnover: Decimal = Decimal("0.35")
    drawdown_warning_threshold: Decimal = Decimal("0.08")
    drawdown_stop_new_risk_threshold: Decimal = Decimal("0.12")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "policy version"))
        values = (
            self.maximum_total_weight,
            self.maximum_etf_weight,
            self.maximum_stock_weight,
            self.maximum_single_etf_weight,
            self.maximum_single_stock_weight,
            self.maximum_stock_industry_weight,
            self.turnover_warning_threshold,
            self.maximum_one_way_turnover,
            self.drawdown_warning_threshold,
            self.drawdown_stop_new_risk_threshold,
        )
        for value in values:
            _exact_decimal(value, "portfolio risk threshold")
            if value < 0 or value > 1:
                raise ValueError("portfolio risk thresholds must be within 0..1")
        hard_caps = (
            self.maximum_total_weight,
            self.maximum_etf_weight,
            self.maximum_stock_weight,
            self.maximum_single_etf_weight,
            self.maximum_single_stock_weight,
            self.maximum_stock_industry_weight,
            self.maximum_one_way_turnover,
            self.drawdown_stop_new_risk_threshold,
        )
        if any(value <= 0 for value in hard_caps):
            raise ValueError("hard portfolio risk thresholds must be positive")
        if not (
            self.maximum_single_etf_weight <= self.maximum_etf_weight <= self.maximum_total_weight
        ):
            raise ValueError("ETF limits must be ordered from instrument to total")
        if not (
            self.maximum_single_stock_weight
            <= self.maximum_stock_industry_weight
            <= self.maximum_stock_weight
            <= self.maximum_total_weight
        ):
            raise ValueError("stock limits must be ordered from instrument to portfolio")
        if self.turnover_warning_threshold >= self.maximum_one_way_turnover:
            raise ValueError("turnover warning threshold must be below its hard limit")
        if self.drawdown_warning_threshold >= self.drawdown_stop_new_risk_threshold:
            raise ValueError("drawdown warning threshold must be below stop-new-risk")

    @property
    def policy_hash(self) -> str:
        """Hash every threshold and the policy version."""

        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PortfolioRiskContext:
    """Auditable drawdown and regime-budget facts bound to one account snapshot."""

    account_snapshot_hash: str
    as_of: datetime
    valuation_version: str
    current_equity: Decimal
    equity_high_watermark: Decimal
    high_watermark_at: datetime
    drawdown_source_hash: str
    regime_risk_budget: Decimal
    regime_result_hash: str
    context_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        ensure_aware(self.high_watermark_at)
        if self.high_watermark_at > self.as_of:
            raise ValueError("equity high-watermark cannot be observed in the future")
        object.__setattr__(
            self,
            "valuation_version",
            _non_empty(self.valuation_version, "valuation version"),
        )
        for value, field_name in (
            (self.account_snapshot_hash, "account_snapshot_hash"),
            (self.drawdown_source_hash, "drawdown_source_hash"),
            (self.regime_result_hash, "regime_result_hash"),
            (self.context_hash, "context_hash"),
        ):
            _sha256(value, field_name)
        _exact_decimal(self.current_equity, "current_equity")
        _exact_decimal(self.equity_high_watermark, "equity_high_watermark")
        _exact_decimal(self.regime_risk_budget, "regime_risk_budget")
        if self.current_equity <= 0:
            raise ValueError("current_equity must be positive")
        if self.equity_high_watermark < self.current_equity:
            raise ValueError("equity high-watermark cannot be below current equity")
        if self.regime_risk_budget < 0 or self.regime_risk_budget > 1:
            raise ValueError("regime risk budget must be within 0..1")
        if self.context_hash != stable_hash(self._content_payload()):
            raise ValueError("portfolio risk context_hash does not match its facts")

    @classmethod
    def build(
        cls,
        *,
        account_snapshot_hash: str,
        as_of: datetime,
        valuation_version: str,
        current_equity: Decimal,
        equity_high_watermark: Decimal,
        high_watermark_at: datetime,
        drawdown_source_hash: str,
        regime_risk_budget: Decimal,
        regime_result_hash: str,
    ) -> PortfolioRiskContext:
        values: dict[str, object] = {
            "account_snapshot_hash": account_snapshot_hash,
            "as_of": as_of,
            "valuation_version": valuation_version,
            "current_equity": current_equity,
            "equity_high_watermark": equity_high_watermark,
            "high_watermark_at": high_watermark_at,
            "drawdown_source_hash": drawdown_source_hash,
            "regime_risk_budget": regime_risk_budget,
            "regime_result_hash": regime_result_hash,
        }
        return cls(
            account_snapshot_hash=account_snapshot_hash,
            as_of=as_of,
            valuation_version=valuation_version,
            current_equity=current_equity,
            equity_high_watermark=equity_high_watermark,
            high_watermark_at=high_watermark_at,
            drawdown_source_hash=drawdown_source_hash,
            regime_risk_budget=regime_risk_budget,
            regime_result_hash=regime_result_hash,
            context_hash=stable_hash(values),
        )

    @property
    def current_drawdown(self) -> Decimal:
        """Return peak-to-current drawdown using deterministic high precision."""

        with localcontext() as context:
            context.prec = 50
            return (self.equity_high_watermark - self.current_equity) / (self.equity_high_watermark)

    def _content_payload(self) -> dict[str, object]:
        return {
            "account_snapshot_hash": self.account_snapshot_hash,
            "as_of": self.as_of,
            "valuation_version": self.valuation_version,
            "current_equity": self.current_equity,
            "equity_high_watermark": self.equity_high_watermark,
            "high_watermark_at": self.high_watermark_at,
            "drawdown_source_hash": self.drawdown_source_hash,
            "regime_risk_budget": self.regime_risk_budget,
            "regime_result_hash": self.regime_result_hash,
        }


__all__ = [
    "PORTFOLIO_RISK_ENGINE_VERSION",
    "PortfolioRiskContext",
    "PortfolioRiskInputError",
    "PortfolioRiskPolicy",
    "PortfolioRiskRuleCode",
]
