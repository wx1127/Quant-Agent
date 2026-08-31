"""Add point-in-time external event evidence.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create immutable event revisions and their instrument mappings."""

    op.create_table(
        "event_evidence",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("headline", sa.String(length=512), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_record_id", sa.String(length=128), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_path", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint(
            "source",
            "source_record_id",
            "version",
            name="uq_event_evidence_source_revision",
        ),
    )
    op.create_index(
        "ix_event_evidence_point_in_time",
        "event_evidence",
        ["available_at", "event_type"],
        unique=False,
    )
    op.create_table(
        "event_instrument_link",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["event_evidence.event_id"]),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("event_id", "instrument_id"),
    )
    op.create_index(
        "ix_event_instrument_lookup",
        "event_instrument_link",
        ["instrument_id", "event_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove event evidence without touching raw payloads or instruments."""

    op.drop_index("ix_event_instrument_lookup", table_name="event_instrument_link")
    op.drop_table("event_instrument_link")
    op.drop_index("ix_event_evidence_point_in_time", table_name="event_evidence")
    op.drop_table("event_evidence")
