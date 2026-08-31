"""Versioned contracts for industry ranking and contribution attribution."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.data.domain import IndustryMembership
from quant_agent.features.core.identity import canonical_decimal


class InsufficientIndustryData(ValueError):
    """Raised when the benchmark cannot support every configured horizon."""


class IndustryStrengthStatus(StrEnum):
    """Whether an industry has enough tradeable members and history to be ranked."""

    READY = "READY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class PointInTimeIndustryMembership:
    """One membership revision with the time it became usable by research."""

    membership: IndustryMembership
    level: int
    available_at: datetime
    revision: str

    def __post_init__(self) -> None:
        if self.level not in (1, 2, 3):
            raise ValueError("industry membership level must be 1, 2, or 3")
        ensure_aware(self.available_at)
        if not self.revision.strip():
            raise ValueError("industry membership revision must be non-empty")


@dataclass(frozen=True, slots=True)
class IndustryStrengthConfig:
    """Versioned horizons, eligibility floors, and transparent scoring weights."""

    version: str = "industry-strength-v1"
    horizons: tuple[int, ...] = (5, 20, 60)
    horizon_weights: tuple[Decimal, ...] = (
        Decimal("0.2"),
        Decimal("0.3"),
        Decimal("0.5"),
    )
    minimum_members: int = 3
    minimum_member_coverage: Decimal = Decimal("0.80")
    new_high_window: int = 60
    turnover_lookback: int = 5
    downside_window: int = 20
    relative_weight: Decimal = Decimal("0.50")
    breadth_weight: Decimal = Decimal("0.20")
    turnover_weight: Decimal = Decimal("0.10")
    new_high_weight: Decimal = Decimal("0.10")
    resilience_weight: Decimal = Decimal("0.10")
    score_scale: Decimal = Decimal("200")

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("industry strength version must be non-empty")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("industry horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("industry horizons must be unique and ascending")
        if len(self.horizons) != len(self.horizon_weights):
            raise ValueError("each industry horizon requires one weight")
        decimal_values = (
            *self.horizon_weights,
            self.relative_weight,
            self.breadth_weight,
            self.turnover_weight,
            self.new_high_weight,
            self.resilience_weight,
            self.score_scale,
            self.minimum_member_coverage,
        )
        for value in decimal_values:
            canonical_decimal(value, field_name="industry strength numeric parameter")
        if (
            any(value < 0 for value in self.horizon_weights)
            or sum(self.horizon_weights, Decimal(0)) <= 0
        ):
            raise ValueError("horizon weights must be non-negative with a positive sum")
        if (
            min(
                self.minimum_members,
                self.new_high_window,
                self.turnover_lookback,
                self.downside_window,
            )
            <= 0
        ):
            raise ValueError("industry eligibility and auxiliary windows must be positive")
        component_weights = (
            self.relative_weight,
            self.breadth_weight,
            self.turnover_weight,
            self.new_high_weight,
            self.resilience_weight,
        )
        if any(value < 0 for value in component_weights):
            raise ValueError("industry score weights cannot be negative")
        if sum(component_weights, Decimal(0)) != Decimal(1):
            raise ValueError("industry score weights must sum to one")
        if self.score_scale <= 0:
            raise ValueError("industry score_scale must be positive")
        if not Decimal(0) < self.minimum_member_coverage <= Decimal(1):
            raise ValueError("minimum_member_coverage must be within (0, 1]")

    @property
    def config_hash(self) -> str:
        payload = {
            "breadth_weight": canonical_decimal(self.breadth_weight),
            "downside_window": self.downside_window,
            "horizon_weights": [canonical_decimal(value) for value in self.horizon_weights],
            "horizons": list(self.horizons),
            "minimum_members": self.minimum_members,
            "minimum_member_coverage": canonical_decimal(self.minimum_member_coverage),
            "new_high_weight": canonical_decimal(self.new_high_weight),
            "new_high_window": self.new_high_window,
            "relative_weight": canonical_decimal(self.relative_weight),
            "resilience_weight": canonical_decimal(self.resilience_weight),
            "score_scale": canonical_decimal(self.score_scale),
            "turnover_lookback": self.turnover_lookback,
            "turnover_weight": canonical_decimal(self.turnover_weight),
            "version": self.version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class HorizonRelativeReturn:
    """Compounded industry and benchmark returns over one aligned horizon."""

    horizon: int
    industry_return: Decimal
    benchmark_return: Decimal
    relative_return: Decimal
    aligned_sessions: int

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.aligned_sessions != self.horizon:
            raise ValueError("relative-return horizon must be fully aligned")
        if not all(
            value.is_finite()
            for value in (self.industry_return, self.benchmark_return, self.relative_return)
        ):
            raise ValueError("relative returns must be finite")


@dataclass(frozen=True, slots=True)
class MemberContribution:
    """Current-session member return, equal-weight contribution, and turnover share."""

    instrument_id: str
    current_return: Decimal
    return_contribution: Decimal
    turnover_share: Decimal

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("member contribution instrument_id must be non-empty")
        if not all(
            value.is_finite()
            for value in (self.current_return, self.return_contribution, self.turnover_share)
        ):
            raise ValueError("member contributions must be finite")
        if not Decimal(0) <= self.turnover_share <= Decimal(1):
            raise ValueError("turnover_share must be between zero and one")


@dataclass(frozen=True, slots=True)
class IndustryStrengthResult:
    """One industry feature row, including reasons when it is not rankable."""

    industry_id: str
    status: IndustryStrengthStatus
    rank: int | None
    score: Decimal | None
    horizon_returns: tuple[HorizonRelativeReturn, ...]
    current_member_count: int
    traded_member_count: int
    comparable_return_count: int
    current_member_coverage: Decimal | None
    advancing_count: int
    breadth_ratio: Decimal | None
    new_high_count: int
    new_high_denominator: int
    new_high_ratio: Decimal | None
    turnover_growth: Decimal | None
    downside_resilience: Decimal | None
    contributions: tuple[MemberContribution, ...]
    reason: str | None

    def __post_init__(self) -> None:
        if not self.industry_id.strip():
            raise ValueError("industry_id must be non-empty")
        counts = (
            self.current_member_count,
            self.traded_member_count,
            self.comparable_return_count,
            self.advancing_count,
            self.new_high_count,
            self.new_high_denominator,
        )
        if any(value < 0 for value in counts):
            raise ValueError("industry feature counts cannot be negative")
        if not (
            self.comparable_return_count <= self.traded_member_count <= self.current_member_count
        ):
            raise ValueError("industry member counts must be monotonic")
        if self.advancing_count > self.comparable_return_count:
            raise ValueError("advancing_count cannot exceed comparable returns")
        if self.new_high_count > self.new_high_denominator:
            raise ValueError("new_high_count cannot exceed its denominator")
        for ratio in (
            self.current_member_coverage,
            self.breadth_ratio,
            self.new_high_ratio,
        ):
            if ratio is not None and not Decimal(0) <= ratio <= Decimal(1):
                raise ValueError("industry ratios must be between zero and one")
        if self.status is IndustryStrengthStatus.READY:
            if self.rank is None or self.rank <= 0:
                raise ValueError("ready industry requires a positive rank")
            if self.score is None or not self.score.is_finite():
                raise ValueError("ready industry requires a finite score")
            if self.reason is not None:
                raise ValueError("ready industry cannot include an insufficiency reason")
            if self.current_member_coverage is None:
                raise ValueError("ready industry requires current member coverage")
            if len(self.contributions) != self.comparable_return_count:
                raise ValueError("ready contributions must cover every comparable return")
        else:
            if self.rank is not None or self.score is not None:
                raise ValueError("insufficient industry cannot be ranked or scored")
            if not self.reason:
                raise ValueError("insufficient industry requires a reason")


@dataclass(frozen=True, slots=True)
class IndustryStrengthSnapshot:
    """Reproducible ranking bound to classification and data versions."""

    session_date: date
    as_of: datetime
    data_version: str
    classification_version: str
    industry_level: int
    feature_version: str
    config_hash: str
    industries: tuple[IndustryStrengthResult, ...]
    cache_key: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        versions = (self.data_version, self.classification_version, self.feature_version)
        if any(not value.strip() for value in versions):
            raise ValueError("industry snapshot versions must be non-empty")
        if len(self.config_hash) != 64 or len(self.cache_key) != 64:
            raise ValueError("industry snapshot hashes must be SHA-256 digests")
        if not self.industries:
            raise ValueError("industry snapshot requires requested industries")
        if self.industry_level not in (1, 2, 3):
            raise ValueError("industry_level must be 1, 2, or 3")
