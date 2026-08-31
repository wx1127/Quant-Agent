"""Add resumable incremental synchronization state.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create fixed run, checkpoint, and page schemas."""

    # Revision 0001 historically called ``Base.metadata.create_all`` instead of
    # declaring a fixed schema. A clean replay with newer application models can
    # therefore expose all three tables before this revision runs. Existing 0001
    # deployments expose none of them. Accept only those two coherent states.
    sync_tables = {"data_sync_run", "data_sync_checkpoint", "data_sync_page"}
    existing_sync_tables = sync_tables.intersection(sa.inspect(op.get_bind()).get_table_names())
    if existing_sync_tables == sync_tables:
        return
    if existing_sync_tables:
        raise RuntimeError(
            "partial incremental-sync schema exists before revision 0002: "
            f"{sorted(existing_sync_tables)}"
        )

    op.create_table(
        "data_sync_run",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("scope_hash", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=True),
        sa.Column("requested_from", sa.JSON(), nullable=True),
        sa.Column("target_watermark", sa.JSON(), nullable=False),
        sa.Column("checkpoint_before", sa.JSON(), nullable=False),
        sa.Column("checkpoint_after", sa.JSON(), nullable=True),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("code_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_data_sync_run_idempotency"),
    )
    op.create_index(
        "ix_data_sync_run_dataset_state",
        "data_sync_run",
        ["provider", "dataset", "state"],
        unique=False,
    )

    op.create_table(
        "data_sync_checkpoint",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("scope_hash", sa.String(length=64), nullable=False),
        sa.Column("committed_watermark", sa.JSON(), nullable=False),
        sa.Column("active_partition", sa.JSON(), nullable=True),
        sa.Column("resume_cursor", sa.JSON(), nullable=True),
        sa.Column("last_success_run_id", sa.String(length=64), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["last_success_run_id"],
            ["data_sync_run.run_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "dataset",
            "scope_hash",
            name="uq_data_sync_checkpoint_scope",
        ),
    )
    op.create_index(
        "ix_data_sync_checkpoint_lease",
        "data_sync_checkpoint",
        ["lease_expires_at"],
        unique=False,
    )

    op.create_table(
        "data_sync_page",
        sa.Column("page_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("partition", sa.JSON(), nullable=False),
        sa.Column("partition_hash", sa.String(length=64), nullable=False),
        sa.Column("page_ordinal", sa.Integer(), nullable=False),
        sa.Column("cursor_in", sa.JSON(), nullable=True),
        sa.Column("cursor_out", sa.JSON(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("raw_payload_id", sa.Integer(), nullable=True),
        sa.Column("received_count", sa.Integer(), nullable=False),
        sa.Column("accepted_count", sa.Integer(), nullable=False),
        sa.Column("rejected_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["raw_payload_id"], ["raw_payload.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["data_sync_run.run_id"]),
        sa.PrimaryKeyConstraint("page_id"),
        sa.UniqueConstraint(
            "run_id",
            "partition_hash",
            "page_ordinal",
            name="uq_data_sync_page_ordinal",
        ),
        sa.UniqueConstraint(
            "run_id",
            "request_hash",
            name="uq_data_sync_page_request",
        ),
    )
    op.create_index(
        "ix_data_sync_page_run_state",
        "data_sync_page",
        ["run_id", "state"],
        unique=False,
    )


def downgrade() -> None:
    """Remove synchronization state without touching the P1 foundation."""

    op.drop_index("ix_data_sync_page_run_state", table_name="data_sync_page")
    op.drop_table("data_sync_page")
    op.drop_index("ix_data_sync_checkpoint_lease", table_name="data_sync_checkpoint")
    op.drop_table("data_sync_checkpoint")
    op.drop_index("ix_data_sync_run_dataset_state", table_name="data_sync_run")
    op.drop_table("data_sync_run")
