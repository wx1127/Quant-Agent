"""Read-only synchronization status queries for operators and local diagnostics."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select

from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.data.database import Database
from quant_agent.data.models import DataSyncCheckpointRow, DataSyncRunRow


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def query_sync_status(
    *,
    database_url: str,
    provider: str | None = None,
    dataset: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return recent runs and current checkpoints without mutating synchronization state."""

    if limit <= 0 or limit > 200:
        raise ValueError("status limit must be between 1 and 200")
    run_query: Select[tuple[DataSyncRunRow]] = select(DataSyncRunRow)
    checkpoint_query: Select[tuple[DataSyncCheckpointRow]] = select(DataSyncCheckpointRow)
    if provider is not None:
        run_query = run_query.where(DataSyncRunRow.provider == provider)
        checkpoint_query = checkpoint_query.where(DataSyncCheckpointRow.provider == provider)
    if dataset is not None:
        run_query = run_query.where(DataSyncRunRow.dataset == dataset)
        checkpoint_query = checkpoint_query.where(DataSyncCheckpointRow.dataset == dataset)
    run_query = run_query.order_by(DataSyncRunRow.created_at.desc()).limit(limit)
    checkpoint_query = checkpoint_query.order_by(
        DataSyncCheckpointRow.provider,
        DataSyncCheckpointRow.dataset,
        DataSyncCheckpointRow.scope_hash,
    )

    database = Database(database_url)
    with database.session() as session:
        runs = tuple(session.scalars(run_query))
        checkpoints = tuple(session.scalars(checkpoint_query))
    return {
        "filters": {
            "dataset": dataset,
            "limit": limit,
            "provider": provider,
        },
        "runs": [
            {
                "checkpoint_after": row.checkpoint_after,
                "checkpoint_before": row.checkpoint_before,
                "created_at": _timestamp(row.created_at),
                "dataset": row.dataset,
                "error_code": row.error_code,
                "finished_at": _timestamp(row.finished_at),
                "mode": row.mode,
                "provider": row.provider,
                "run_id": row.run_id,
                "scope": row.scope,
                "started_at": _timestamp(row.started_at),
                "state": row.state,
                "target_watermark": row.target_watermark,
            }
            for row in runs
        ],
        "checkpoints": [
            {
                "active_partition": row.active_partition,
                "committed_watermark": row.committed_watermark,
                "dataset": row.dataset,
                "is_leased": row.lease_owner is not None,
                "last_success_run_id": row.last_success_run_id,
                "provider": row.provider,
                "resume_cursor": row.resume_cursor,
                "revision": row.revision,
                "scope": row.scope,
                "updated_at": _timestamp(row.updated_at),
            }
            for row in checkpoints
        ],
    }
