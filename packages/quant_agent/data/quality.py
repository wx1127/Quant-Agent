"""Composable data-quality rules with fail-closed snapshot gating."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from sqlalchemy.orm import Session

from quant_agent.core.time import shanghai_now
from quant_agent.data.models import DataQualityResultRow


class QualitySeverity(StrEnum):
    """Severity determines whether a snapshot may be released."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class BarQualityRecord:
    """Raw values inspected before validated domain construction."""

    instrument_id: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal


@dataclass(frozen=True, slots=True)
class FactorQualityRecord:
    """Adjustment factor input used by quality rules."""

    instrument_id: str
    trade_date: date
    factor: Decimal


@dataclass(frozen=True, slots=True)
class SourceCloseRecord:
    """Comparable unadjusted close from one independent data source."""

    instrument_id: str
    trade_date: date
    close: Decimal
    source: str


@dataclass(frozen=True, slots=True)
class QualityContext:
    """Dataset view shared by registered rules."""

    bars: tuple[BarQualityRecord, ...] = ()
    factors: tuple[FactorQualityRecord, ...] = ()
    source_closes: tuple[SourceCloseRecord, ...] = ()
    expected_instrument_ids: tuple[str, ...] = ()
    expected_open_dates: tuple[date, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One rule result suitable for persistence and reports."""

    rule_id: str
    severity: QualitySeverity
    entity_key: str | None
    message: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Complete quality result for one data version."""

    data_version: str
    observed_at: datetime
    issues: tuple[QualityIssue, ...]

    @property
    def qualified(self) -> bool:
        """Return whether no blocking issue exists."""

        return not any(
            issue.severity in {QualitySeverity.ERROR, QualitySeverity.CRITICAL}
            for issue in self.issues
        )


class QualityRule(Protocol):
    """Rule contract."""

    rule_id: str

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]: ...


class OhlcvValidityRule:
    """Detect illegal OHLC relationships and negative activity."""

    rule_id = "OHLCV_VALIDITY"

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]:
        issues: list[QualityIssue] = []
        for bar in context.bars:
            key = f"{bar.instrument_id}:{bar.trade_date}"
            if min(bar.open, bar.high, bar.low, bar.close) <= 0:
                issues.append(
                    QualityIssue(
                        self.rule_id,
                        QualitySeverity.ERROR,
                        key,
                        "OHLC contains a non-positive price",
                    )
                )
            elif bar.high < max(bar.open, bar.low, bar.close) or bar.low > min(
                bar.open,
                bar.high,
                bar.close,
            ):
                issues.append(
                    QualityIssue(
                        self.rule_id,
                        QualitySeverity.ERROR,
                        key,
                        "OHLC relationship is invalid",
                    )
                )
            if bar.volume < 0 or bar.turnover < 0:
                issues.append(
                    QualityIssue(
                        self.rule_id,
                        QualitySeverity.ERROR,
                        key,
                        "volume or turnover is negative",
                    )
                )
        return issues


class DuplicateBarRule:
    """Detect duplicate instrument/date bars before database insertion."""

    rule_id = "DUPLICATE_DAILY_BAR"

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]:
        keys = [(bar.instrument_id, bar.trade_date) for bar in context.bars]
        return [
            QualityIssue(
                self.rule_id,
                QualitySeverity.ERROR,
                f"{instrument_id}:{trade_date}",
                f"duplicate daily bar count={count}",
            )
            for (instrument_id, trade_date), count in Counter(keys).items()
            if count > 1
        ]


class CalendarCompletenessRule:
    """Detect missing bars for the explicitly expected universe/calendar."""

    rule_id = "CALENDAR_COMPLETENESS"

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]:
        if not context.expected_instrument_ids or not context.expected_open_dates:
            return ()
        actual = {(bar.instrument_id, bar.trade_date) for bar in context.bars}
        expected = {
            (instrument_id, trade_date)
            for instrument_id in context.expected_instrument_ids
            for trade_date in context.expected_open_dates
        }
        return [
            QualityIssue(
                self.rule_id,
                QualitySeverity.ERROR,
                f"{instrument_id}:{trade_date}",
                "expected daily bar is missing",
            )
            for instrument_id, trade_date in sorted(expected - actual)
        ]


class AdjustmentFactorRule:
    """Detect invalid and unusually discontinuous adjustment factors."""

    rule_id = "ADJUSTMENT_FACTOR_CONTINUITY"

    def __init__(self, relative_jump_threshold: Decimal = Decimal("0.5")) -> None:
        self._threshold = relative_jump_threshold

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]:
        issues: list[QualityIssue] = []
        grouped: dict[str, list[FactorQualityRecord]] = {}
        for factor in context.factors:
            grouped.setdefault(factor.instrument_id, []).append(factor)
        for instrument_id, factors in grouped.items():
            previous: Decimal | None = None
            for factor in sorted(factors, key=lambda item: item.trade_date):
                key = f"{instrument_id}:{factor.trade_date}"
                if factor.factor <= 0:
                    issues.append(
                        QualityIssue(
                            self.rule_id,
                            QualitySeverity.ERROR,
                            key,
                            "adjustment factor must be positive",
                        )
                    )
                elif (
                    previous is not None
                    and abs(factor.factor / previous - Decimal(1)) > self._threshold
                ):
                    issues.append(
                        QualityIssue(
                            self.rule_id,
                            QualitySeverity.WARNING,
                            key,
                            "adjustment factor has an unusually large jump",
                        )
                    )
                previous = factor.factor if factor.factor > 0 else previous
        return issues


class CrossSourceCloseRule:
    """Fail a snapshot when independent unadjusted closes materially disagree."""

    rule_id = "CROSS_SOURCE_CLOSE_DEVIATION"

    def __init__(self, relative_tolerance: Decimal = Decimal("0.02")) -> None:
        if relative_tolerance < 0:
            raise ValueError("relative_tolerance cannot be negative")
        self._tolerance = relative_tolerance

    def evaluate(self, context: QualityContext) -> Sequence[QualityIssue]:
        grouped: dict[tuple[str, date], dict[str, Decimal]] = {}
        for item in context.source_closes:
            grouped.setdefault((item.instrument_id, item.trade_date), {})[item.source] = item.close
        issues: list[QualityIssue] = []
        for (instrument_id, trade_date), values_by_source in sorted(grouped.items()):
            values = tuple(values_by_source.values())
            if len(values) < 2:
                continue
            lowest = min(values)
            highest = max(values)
            if lowest <= 0:
                issues.append(
                    QualityIssue(
                        self.rule_id,
                        QualitySeverity.ERROR,
                        f"{instrument_id}:{trade_date}",
                        "cross-source close must be positive",
                    )
                )
                continue
            relative_deviation = highest / lowest - Decimal(1)
            if relative_deviation > self._tolerance:
                issues.append(
                    QualityIssue(
                        self.rule_id,
                        QualitySeverity.ERROR,
                        f"{instrument_id}:{trade_date}",
                        (f"cross-source close deviation exceeds tolerance: {relative_deviation}"),
                    )
                )
        return issues


class QualityEngine:
    """Run registered rules and optionally persist every result."""

    def __init__(
        self,
        rules: Iterable[QualityRule] | None = None,
    ) -> None:
        self._rules = tuple(
            rules
            or (
                OhlcvValidityRule(),
                DuplicateBarRule(),
                CalendarCompletenessRule(),
                AdjustmentFactorRule(),
                CrossSourceCloseRule(),
            )
        )

    def run(
        self,
        data_version: str,
        context: QualityContext,
        *,
        session: Session | None = None,
        observed_at: datetime | None = None,
    ) -> QualityReport:
        """Evaluate all rules and persist issues plus successful rule outcomes."""

        timestamp = observed_at or shanghai_now()
        issues = tuple(issue for rule in self._rules for issue in rule.evaluate(context))
        report = QualityReport(
            data_version=data_version,
            observed_at=timestamp,
            issues=issues,
        )
        if session is not None:
            issues_by_rule: dict[str, list[QualityIssue]] = {
                rule.rule_id: [] for rule in self._rules
            }
            for issue in issues:
                issues_by_rule[issue.rule_id].append(issue)
            for rule_id, rule_issues in issues_by_rule.items():
                if not rule_issues:
                    session.add(
                        DataQualityResultRow(
                            data_version=data_version,
                            rule_id=rule_id,
                            severity=QualitySeverity.INFO.value,
                            passed=True,
                            entity_key=None,
                            message="rule passed",
                            observed_at=timestamp,
                        )
                    )
                else:
                    for issue in rule_issues:
                        session.add(
                            DataQualityResultRow(
                                data_version=data_version,
                                rule_id=rule_id,
                                severity=issue.severity.value,
                                passed=False,
                                entity_key=issue.entity_key,
                                message=issue.message,
                                observed_at=timestamp,
                            )
                        )
            session.flush()
        return report
