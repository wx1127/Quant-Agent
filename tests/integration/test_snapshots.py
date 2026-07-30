"""Integration tests for immutable Parquet snapshots and DuckDB queries."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.data.models import DatasetVersionRow
from quant_agent.data.quality import (
    QualityIssue,
    QualityReport,
    QualitySeverity,
)
from quant_agent.data.snapshots import SnapshotStore

TZ = ZoneInfo("Asia/Shanghai")


def _quality(version: str, *, passed: bool = True) -> QualityReport:
    issues = (
        ()
        if passed
        else (
            QualityIssue(
                rule_id="TEST",
                severity=QualitySeverity.ERROR,
                entity_key=None,
                message="failed",
            ),
        )
    )
    return QualityReport(
        data_version=version,
        observed_at=datetime(2026, 7, 30, 17, tzinfo=TZ),
        issues=issues,
    )


def test_snapshot_is_immutable_verifiable_and_queryable(
    tmp_path: Path,
    db_session: Session,
) -> None:
    store = SnapshotStore(tmp_path)
    table = pa.table(
        {
            "instrument_id": ["CN.SZ.000001", "CN.SZ.000002"],
            "close": [10.0, 20.0],
        }
    )

    manifest = store.create(
        "market_20260730_v1",
        {"daily_bars": table},
        quality_report=_quality("market_20260730_v1"),
        metadata={"as_of": "2026-07-30T17:00:00+08:00"},
        session=db_session,
    )
    repeated = store.create(
        "market_20260730_v1",
        {"daily_bars": table},
        quality_report=_quality("market_20260730_v1"),
        metadata={"as_of": "2026-07-30T17:00:00+08:00"},
    )

    assert repeated == manifest
    assert store.verify("market_20260730_v1") is True
    assert store.read_table("market_20260730_v1", "daily_bars").num_rows == 2
    result = store.query(
        "market_20260730_v1",
        "SELECT instrument_id FROM daily_bars WHERE close > 15",
    )
    assert result.to_pydict() == {"instrument_id": ["CN.SZ.000002"]}
    stored = db_session.scalar(
        select(DatasetVersionRow).where(DatasetVersionRow.data_version == "market_20260730_v1")
    )
    assert stored is not None
    assert stored.content_hash == manifest.content_hash


def test_existing_snapshot_version_rejects_different_content(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.create(
        "v1",
        {"daily_bars": pa.table({"value": [1]})},
        quality_report=_quality("v1"),
    )

    with pytest.raises(ValueError, match="different table content"):
        store.create(
            "v1",
            {"daily_bars": pa.table({"value": [2]})},
            quality_report=_quality("v1"),
        )


def test_snapshot_publication_is_blocked_by_quality_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="quality failed"):
        SnapshotStore(tmp_path).create(
            "bad",
            {"daily_bars": pa.table({"value": [1]})},
            quality_report=_quality("bad", passed=False),
        )


def test_snapshot_rejects_unsafe_table_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        SnapshotStore(tmp_path).create(
            "v1",
            {"bad-name": pa.table({"value": [1]})},
            quality_report=_quality("v1"),
        )
