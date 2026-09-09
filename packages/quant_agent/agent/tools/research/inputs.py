"""Trusted input boundary for deterministic Agent research tools.

The model never constructs these values.  A composition root supplies one
``ResearchInputSource`` backed by verified, point-in-time data, while the
research toolset validates every returned identity against its frozen decision.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import ConfigDict

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.data.quality import QualityContext
from quant_agent.data.snapshots import SnapshotManifest
from quant_agent.features.fundamental_quality import FundamentalQualitySnapshot
from quant_agent.features.industry_strength import IndustryStrengthSnapshot
from quant_agent.features.market_breadth import MarketBreadthSnapshot
from quant_agent.features.market_trend import MarketTrendSnapshot
from quant_agent.features.stock_strength import StockStrengthSnapshot
from quant_agent.features.tradeability import TradeabilitySnapshot
from quant_agent.leaders.candidates import CandidateSupplementalSignal


class ResearchInputError(RuntimeError):
    """Base class for failures at the trusted research-input boundary."""


class ResearchInputUnavailable(ResearchInputError):
    """Raised when a required verified input does not exist for the decision."""


class ResearchInputInvalid(ResearchInputError):
    """Raised when stored research input cannot satisfy its trusted contract."""


@dataclass(frozen=True, slots=True)
class RegimeFeaturePair:
    """Aligned trend and breadth features for one historical market session."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    trend: MarketTrendSnapshot
    breadth: MarketBreadthSnapshot


@dataclass(frozen=True, slots=True)
class DataBoundQualityContext:
    """Quality-rule inputs bound to the immutable dataset they describe."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    data_version: str
    data_content_hash: str
    context: QualityContext


@dataclass(frozen=True, slots=True)
class AccountBoundLeaderInputs:
    """Leader features computed for one exact, immutable account snapshot.

    Tradeability can depend on account-sized target positions, so these feature
    triples must remain bound to the same account identity as the decision.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    account_id: str
    account_snapshot_id: str
    account_snapshot_hash: str
    stock_strength: tuple[StockStrengthSnapshot, ...]
    tradeability: tuple[TradeabilitySnapshot, ...]
    fundamentals: tuple[FundamentalQualitySnapshot, ...]


class ResearchInputSource(Protocol):
    """Read-only provider of authoritative inputs for one frozen decision."""

    def load_verified_manifest(self, decision: DecisionSnapshot) -> SnapshotManifest:
        """Return the content-verified dataset manifest bound to ``decision``."""

        ...

    def load_quality_context(self, decision: DecisionSnapshot) -> DataBoundQualityContext:
        """Return quality-rule inputs and their immutable dataset identity."""

        ...

    def load_regime_history(self, decision: DecisionSnapshot) -> tuple[RegimeFeaturePair, ...]:
        """Return complete chronological regime features through the decision time."""

        ...

    def load_industry_history(
        self, decision: DecisionSnapshot
    ) -> tuple[IndustryStrengthSnapshot, ...]:
        """Return complete chronological industry features through the decision time."""

        ...

    def load_leader_inputs(self, decision: DecisionSnapshot) -> AccountBoundLeaderInputs:
        """Return same-account feature triples used for leader and candidate ranking."""

        ...

    def load_candidate_signals(
        self, decision: DecisionSnapshot
    ) -> tuple[CandidateSupplementalSignal, ...]:
        """Return optional, point-in-time supplemental signals for candidates."""

        ...
