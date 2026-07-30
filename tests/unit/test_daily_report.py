from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_agent.reports.daily import (
    DailyReport,
    DailyReportItem,
    DailyReportRenderer,
    report_to_dict,
)

NOW = datetime(2026, 7, 30, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


def item(item_id: str = "market") -> DailyReportItem:
    return DailyReportItem(
        item_id=item_id,
        title="<市场阶段>",
        state="RANGE_STRONG",
        metrics=(("阶段分", "68.2"), ("置信标签", "中等")),
        support_evidence=("市场宽度保持",),
        counter_evidence=("成交尚未放大",),
        risks=("阶段分类不是上涨概率",),
        invalidations=("市场宽度连续走弱",),
        source_refs=("tool:detect_market_regime#/data/score",),
    )


def report() -> DailyReport:
    return DailyReport(
        report_id="daily-20260730",
        as_of=NOW,
        data_versions=("market_v1", "risk_v1"),
        market=item(),
        themes=(item("theme-a"),),
        leaders=(),
        candidates=(),
        portfolio_risks=(item("risk"),),
        changes_from_previous=("主线 A 从 EMERGING 进入 CONFIRMED",),
        warnings=("报告不构成收益承诺",),
    )


def test_daily_report_renders_markdown_html_and_preserves_structured_numbers() -> None:
    renderer = DailyReportRenderer()
    markdown = renderer.markdown(report())
    rendered_html = renderer.html(report())
    assert "数据截止" in markdown
    assert "阶段分：68.2" in markdown
    assert "支持证据" in markdown and "反对证据" in markdown
    assert "与前一交易日变化" in markdown
    assert "&lt;市场阶段&gt;" in rendered_html
    assert "<市场阶段>" not in rendered_html
    assert "68.2" in rendered_html
    assert report_to_dict(report())["data_versions"] == ("market_v1", "risk_v1")


def test_daily_report_rejects_missing_evidence_versions_and_naive_time() -> None:
    with pytest.raises(ValueError, match="support and counter"):
        replace(item(), counter_evidence=())
    with pytest.raises(ValueError, match="data versions"):
        replace(report(), data_versions=())
    with pytest.raises(ValueError, match="timezone information"):
        replace(report(), as_of=NOW.replace(tzinfo=None))
