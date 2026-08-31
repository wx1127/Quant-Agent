"""Add index weights and raw-to-curated record lineage.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CURATED_TABLES = {"index_constituent_weight", "curated_record_lineage"}


def upgrade() -> None:
    """Create fixed curated weight and source-lineage schemas."""

    # Before revision 0001 was frozen, it imported live ORM metadata. A database
    # first initialized with newer application models can therefore already have
    # both tables. Accept that coherent historical state, but reject partial DDL.
    existing_tables = _CURATED_TABLES.intersection(sa.inspect(op.get_bind()).get_table_names())
    if existing_tables == _CURATED_TABLES:
        return
    if existing_tables:
        raise RuntimeError(
            f"partial curated-lineage schema exists before revision 0004: {sorted(existing_tables)}"
        )

    op.create_table(
        "index_constituent_weight",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("index_instrument_id", sa.String(length=64), nullable=False),
        sa.Column("constituent_instrument_id", sa.String(length=64), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("weight_percent", sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "index_instrument_id <> constituent_instrument_id",
            name="ck_index_constituent_distinct_instruments",
        ),
        sa.CheckConstraint(
            "weight_percent >= 0 AND weight_percent <= 100",
            name="ck_index_constituent_weight_percent",
        ),
        sa.ForeignKeyConstraint(
            ["constituent_instrument_id"],
            ["instrument.instrument_id"],
        ),
        sa.ForeignKeyConstraint(
            ["index_instrument_id"],
            ["instrument.instrument_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "index_instrument_id",
            "constituent_instrument_id",
            "trade_date",
            "source",
            "version",
            name="uq_index_constituent_weight_version",
        ),
    )
    op.create_index(
        "ix_index_constituent_weight_date",
        "index_constituent_weight",
        ["index_instrument_id", "trade_date"],
        unique=False,
    )

    op.create_table(
        "curated_record_lineage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("raw_payload_id", sa.Integer(), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_key", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["raw_payload_id"], ["raw_payload.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "raw_payload_id",
            "dataset",
            "entity_type",
            "entity_key",
            "content_hash",
            name="uq_curated_record_lineage_association",
        ),
    )
    op.create_index(
        "ix_curated_record_lineage_entity",
        "curated_record_lineage",
        ["dataset", "entity_type", "entity_key", "content_hash"],
        unique=False,
    )


def downgrade() -> None:
    """Remove curated lineage tables without touching raw or P1 data."""

    op.drop_index(
        "ix_curated_record_lineage_entity",
        table_name="curated_record_lineage",
    )
    op.drop_table("curated_record_lineage")
    op.drop_index(
        "ix_index_constituent_weight_date",
        table_name="index_constituent_weight",
    )
    op.drop_table("index_constituent_weight")
