"""Point-in-time announcement and trusted-event evidence."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware


class EventSourceType(StrEnum):
    EXCHANGE = "exchange"
    COMPANY = "company"
    GOVERNMENT = "government"
    LICENSED_NEWS = "licensed_news"


class EventDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class RawEvent:
    """Raw external record retained even when parsing is inconclusive."""

    source_type: EventSourceType
    source_url: str
    published_at: datetime
    title: str
    content: str
    entities: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.published_at)
        if not self.source_url or not self.content:
            raise ValueError("event source URL and content are required")

    @property
    def content_hash(self) -> str:
        """Hash source content independently of transport URL."""

        return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EventEvidence:
    """Structured auxiliary evidence with complete provenance."""

    source_type: EventSourceType
    source_url: str
    published_at: datetime
    entities: tuple[str, ...]
    event_type: str
    direction: EventDirection
    confidence: float
    excerpt_hash: str
    raw_content_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.published_at)
        if not 0 <= self.confidence <= 1:
            raise ValueError("event confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Raw event plus optional structured interpretation."""

    raw: RawEvent
    evidence: EventEvidence | None
    parse_note: str


@dataclass(frozen=True, slots=True)
class EventRule:
    """Deterministic phrase rule for first-version event extraction."""

    event_type: str
    direction: EventDirection
    phrases: tuple[str, ...]
    confidence: float


class EventEvidenceEngine:
    """Filter by decision time, deduplicate and conservatively structure events."""

    def __init__(self, rules: tuple[EventRule, ...] | None = None) -> None:
        self._rules = rules or (
            EventRule("earnings_growth", EventDirection.POSITIVE, ("业绩预增", "扭亏为盈"), 0.85),
            EventRule("major_contract", EventDirection.POSITIVE, ("重大合同", "中标"), 0.75),
            EventRule("regulatory_risk", EventDirection.NEGATIVE, ("立案调查", "行政处罚"), 0.90),
            EventRule("earnings_warning", EventDirection.NEGATIVE, ("业绩预亏", "大幅亏损"), 0.85),
        )

    def collect(self, events: list[RawEvent], *, as_of: datetime) -> tuple[EventRecord, ...]:
        """Return unique records legally available at the decision time."""

        ensure_aware(as_of)
        seen: set[str] = set()
        records: list[EventRecord] = []
        for raw in sorted(events, key=lambda item: (item.published_at, item.content_hash)):
            if raw.published_at > as_of or raw.content_hash in seen:
                continue
            seen.add(raw.content_hash)
            text = f"{raw.title}\n{raw.content}"
            matches = [
                rule for rule in self._rules if any(phrase in text for phrase in rule.phrases)
            ]
            if len(matches) != 1 or not raw.entities:
                records.append(
                    EventRecord(raw, None, "ambiguous or unmapped event; raw text retained")
                )
                continue
            rule = matches[0]
            excerpt_hash = hashlib.sha256(text.encode()).hexdigest()
            records.append(
                EventRecord(
                    raw=raw,
                    evidence=EventEvidence(
                        source_type=raw.source_type,
                        source_url=raw.source_url,
                        published_at=raw.published_at,
                        entities=raw.entities,
                        event_type=rule.event_type,
                        direction=rule.direction,
                        confidence=rule.confidence,
                        excerpt_hash=excerpt_hash,
                        raw_content_hash=raw.content_hash,
                    ),
                    parse_note="matched one deterministic event rule",
                )
            )
        return tuple(records)
