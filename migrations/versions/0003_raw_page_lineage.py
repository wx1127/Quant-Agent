"""Add sanitized request metadata and raw-page replay lineage.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LINEAGE_COLUMNS = {
    "request_hash",
    "request_params",
    "fetched_at",
    "schema_version",
    "size_bytes",
}


def upgrade() -> None:
    """Extend immutable raw responses with replay-safe request metadata."""

    inspector = sa.inspect(op.get_bind())
    existing_columns = {column["name"] for column in inspector.get_columns("raw_payload")}
    existing_lineage_columns = existing_columns.intersection(_LINEAGE_COLUMNS)
    checkpoint_columns = {
        column["name"] for column in inspector.get_columns("data_sync_checkpoint")
    }
    has_progress_run = "progress_run_id" in checkpoint_columns
    # 0001 historically used live Base metadata, so a clean replay with current
    # models already has every column. Real deployed 0002 databases have none.
    if existing_lineage_columns == _LINEAGE_COLUMNS:
        if not has_progress_run:
            _add_checkpoint_progress_run()
        return
    if existing_lineage_columns:
        raise RuntimeError(
            "partial raw-page lineage schema exists before revision 0003: "
            f"{sorted(existing_lineage_columns)}"
        )

    op.add_column(
        "raw_payload",
        sa.Column("request_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "raw_payload",
        sa.Column("request_params", sa.JSON(), nullable=True),
    )
    op.add_column(
        "raw_payload",
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "raw_payload",
        sa.Column("schema_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "raw_payload",
        sa.Column("size_bytes", sa.Integer(), nullable=True),
    )

    raw_payload = sa.table(
        "raw_payload",
        sa.column("request_fingerprint", sa.String(length=64)),
        sa.column("request_hash", sa.String(length=64)),
        sa.column("request_params", sa.JSON()),
        sa.column("ingested_at", sa.DateTime(timezone=True)),
        sa.column("fetched_at", sa.DateTime(timezone=True)),
        sa.column("schema_version", sa.String(length=64)),
        sa.column("size_bytes", sa.Integer()),
    )
    op.execute(
        raw_payload.update().values(
            request_hash=raw_payload.c.request_fingerprint,
            request_params={},
            fetched_at=raw_payload.c.ingested_at,
            schema_version="legacy-v1",
            size_bytes=0,
        )
    )
    with op.batch_alter_table("raw_payload") as batch:
        batch.alter_column(
            "request_hash",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch.alter_column(
            "request_params",
            existing_type=sa.JSON(),
            nullable=False,
        )
        batch.alter_column(
            "fetched_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )
        batch.alter_column(
            "schema_version",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch.alter_column(
            "size_bytes",
            existing_type=sa.Integer(),
            nullable=False,
        )
    op.create_index(
        "ix_raw_payload_request_hash",
        "raw_payload",
        ["provider", "endpoint", "request_hash"],
        unique=False,
    )
    if not has_progress_run:
        _add_checkpoint_progress_run()


def _add_checkpoint_progress_run() -> None:
    with op.batch_alter_table("data_sync_checkpoint") as batch:
        batch.add_column(sa.Column("progress_run_id", sa.String(length=64), nullable=True))
        batch.create_foreign_key(
            "fk_data_sync_checkpoint_progress_run",
            "data_sync_run",
            ["progress_run_id"],
            ["run_id"],
        )


def downgrade() -> None:
    """Remove replay metadata while preserving legacy raw responses."""

    with op.batch_alter_table("data_sync_checkpoint") as batch:
        batch.drop_constraint(
            "fk_data_sync_checkpoint_progress_run",
            type_="foreignkey",
        )
        batch.drop_column("progress_run_id")
    op.drop_index("ix_raw_payload_request_hash", table_name="raw_payload")
    with op.batch_alter_table("raw_payload") as batch:
        batch.drop_column("size_bytes")
        batch.drop_column("schema_version")
        batch.drop_column("fetched_at")
        batch.drop_column("request_params")
        batch.drop_column("request_hash")
