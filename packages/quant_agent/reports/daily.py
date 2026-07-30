"""Structured, evidence-complete daily research reports."""

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from quant_agent.core.time import ensure_aware


@dataclass(frozen=True, slots=True)
class DailyReportItem:
    item_id: str
    title: str
    state: str
    metrics: tuple[tuple[str, str], ...]
    support_evidence: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    risks: tuple[str, ...]
    invalidations: tuple[str, ...]
    source_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.support_evidence or not self.counter_evidence:
            raise ValueError("daily report items require support and counter evidence")
        if not self.source_refs:
            raise ValueError("daily report metrics require structured source references")


@dataclass(frozen=True, slots=True)
class DailyReport:
    report_id: str
    as_of: datetime
    data_versions: tuple[str, ...]
    market: DailyReportItem
    themes: tuple[DailyReportItem, ...]
    leaders: tuple[DailyReportItem, ...]
    candidates: tuple[DailyReportItem, ...]
    portfolio_risks: tuple[DailyReportItem, ...]
    changes_from_previous: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not self.data_versions:
            raise ValueError("daily report requires data versions")


class DailyReportRenderer:
    """Render only supplied structured values; no numeric inference occurs here."""

    def markdown(self, report: DailyReport) -> str:
        lines = [
            f"# 每日量化研究报告 {report.report_id}",
            "",
            f"- 数据截止：{report.as_of.isoformat()}",
            f"- 数据版本：{', '.join(report.data_versions)}",
            "",
            "## 市场阶段",
            "",
            *self._markdown_item(report.market),
        ]
        for title, items in (
            ("主线", report.themes),
            ("龙头", report.leaders),
            ("候选", report.candidates),
            ("组合风险", report.portfolio_risks),
        ):
            lines.extend(("", f"## {title}", ""))
            if not items:
                lines.append("证据不足，暂无可验证结果。")
            for item in items:
                lines.extend(self._markdown_item(item))
        lines.extend(("", "## 与前一交易日变化", ""))
        lines.extend(f"- {item}" for item in report.changes_from_previous)
        lines.extend(("", "## 警告", ""))
        lines.extend(f"- {item}" for item in report.warnings)
        return "\n".join(lines).strip() + "\n"

    def html(self, report: DailyReport) -> str:
        sections = [
            self._html_section("市场阶段", (report.market,)),
            self._html_section("主线", report.themes),
            self._html_section("龙头", report.leaders),
            self._html_section("候选", report.candidates),
            self._html_section("组合风险", report.portfolio_risks),
        ]
        changes = "".join(f"<li>{html.escape(item)}</li>" for item in report.changes_from_previous)
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in report.warnings)
        return (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            "<title>每日量化研究报告</title></head><body>"
            f"<h1>每日量化研究报告 {html.escape(report.report_id)}</h1>"
            f"<p>数据截止：<time>{html.escape(report.as_of.isoformat())}</time></p>"
            f"<p>数据版本：{html.escape(', '.join(report.data_versions))}</p>"
            f"{''.join(sections)}"
            f"<section><h2>与前一交易日变化</h2><ul>{changes}</ul></section>"
            f'<section class="risk-warning"><h2>警告</h2><ul>{warnings}</ul></section>'
            "</body></html>"
        )

    @staticmethod
    def _markdown_item(item: DailyReportItem) -> list[str]:
        lines = [f"### {item.title}", "", f"- 状态：{item.state}"]
        lines.extend(f"- {name}：{value}" for name, value in item.metrics)
        lines.extend(
            (
                f"- 支持证据：{'；'.join(item.support_evidence)}",
                f"- 反对证据：{'；'.join(item.counter_evidence)}",
                f"- 风险：{'；'.join(item.risks)}",
                f"- 失效条件：{'；'.join(item.invalidations)}",
                f"- 来源：{', '.join(item.source_refs)}",
            )
        )
        return lines

    @staticmethod
    def _html_section(title: str, items: tuple[DailyReportItem, ...]) -> str:
        cards = []
        for item in items:
            metrics = "".join(
                f"<li>{html.escape(name)}：{html.escape(value)}</li>"
                for name, value in item.metrics
            )
            cards.append(
                "<article>"
                f"<h3>{html.escape(item.title)}</h3>"
                f"<p>状态：{html.escape(item.state)}</p><ul>{metrics}</ul>"
                f"<p>支持证据：{html.escape('；'.join(item.support_evidence))}</p>"
                f"<p>反对证据：{html.escape('；'.join(item.counter_evidence))}</p>"
                f"<p>风险：{html.escape('；'.join(item.risks))}</p>"
                f"<p>失效条件：{html.escape('；'.join(item.invalidations))}</p>"
                f"<p>来源：{html.escape(', '.join(item.source_refs))}</p>"
                "</article>"
            )
        if not cards:
            cards.append("<p>证据不足，暂无可验证结果。</p>")
        return f"<section><h2>{html.escape(title)}</h2>{''.join(cards)}</section>"


def report_to_dict(report: DailyReport) -> dict[str, Any]:
    """Return a JSON-compatible structure without invoking either renderer."""

    from dataclasses import asdict

    return asdict(report)
