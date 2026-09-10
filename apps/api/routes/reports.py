"""Versioned daily research report endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.app import Principal, _principal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class DailyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    report_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    as_of: datetime
    data_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    market_summary: str = Field(min_length=1)
    mainlines: tuple[dict[str, object], ...] = ()
    leaders: tuple[dict[str, object], ...] = ()
    candidates: tuple[dict[str, object], ...] = ()
    risks: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class ReportProvider(Protocol):
    def get(self, report_date: str) -> DailyReport | None: ...


class InMemoryReportProvider:
    def __init__(self, reports: tuple[DailyReport, ...] = ()) -> None:
        self._reports = {report.report_date: report for report in reports}

    def get(self, report_date: str) -> DailyReport | None:
        return self._reports.get(report_date)


def build_reports_router(provider: ReportProvider | None = None) -> APIRouter:
    backend = provider or InMemoryReportProvider()
    router = APIRouter(prefix="/reports", tags=["reports"])

    @router.get("/daily/{report_date}", response_model=DailyReport)
    async def daily_report(
        report_date: str,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> DailyReport:
        del principal
        report = backend.get(report_date)
        if report is None:
            raise HTTPException(status_code=404, detail="daily report not found")
        return report

    return router
