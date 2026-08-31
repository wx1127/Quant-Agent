"""Create the frozen P1 data-foundation schema.

Revision ID: 0001
Revises:
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the original thirteen P1 tables from fixed DDL."""

    op.create_table(
        "instrument",
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=False),
        sa.Column("instrument_type", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("listed_on", sa.Date(), nullable=False),
        sa.Column("delisted_on", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("instrument_id"),
    )
    op.create_index(
        "ix_instrument_active_dates",
        "instrument",
        ["listed_on", "delisted_on"],
        unique=False,
    )

    op.create_table(
        "trading_calendar",
        sa.Column("market", sa.String(length=16), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("is_open", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("market", "trade_date"),
    )

    op.create_table(
        "raw_payload",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_fingerprint"),
    )

    op.create_table(
        "data_quality_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("data_version", sa.String(length=64), nullable=False),
        sa.Column("rule_id", sa.String(length=96), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("entity_key", sa.String(length=192), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_quality_version_passed",
        "data_quality_result",
        ["data_version", "passed"],
        unique=False,
    )

    op.create_table(
        "dataset_version",
        sa.Column("data_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("data_version"),
        sa.UniqueConstraint("content_hash"),
    )

    op.create_table(
        "instrument_alias",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "symbol",
            "exchange",
            "effective_from",
            name="uq_instrument_alias_effective",
        ),
    )
    op.create_index(
        "ix_instrument_alias_lookup",
        "instrument_alias",
        ["symbol", "exchange", "effective_from", "effective_to"],
        unique=False,
    )

    op.create_table(
        "instrument_status",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "status",
            "effective_from",
            "version",
            name="uq_instrument_status_version",
        ),
    )
    op.create_index(
        "ix_instrument_status_dates",
        "instrument_status",
        ["instrument_id", "effective_from", "effective_to"],
        unique=False,
    )

    op.create_table(
        "market_bar_daily",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("high", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("low", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("close", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("volume", sa.Numeric(precision=28, scale=4), nullable=False),
        sa.Column("turnover", sa.Numeric(precision=28, scale=4), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "trade_date",
            "source",
            "version",
            name="uq_daily_bar_version",
        ),
    )
    op.create_index(
        "ix_daily_bar_instrument_date",
        "market_bar_daily",
        ["instrument_id", "trade_date"],
        unique=False,
    )

    op.create_table(
        "adjustment_factor",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("factor", sa.Numeric(precision=28, scale=10), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "trade_date",
            "source",
            "version",
            name="uq_adjustment_factor_version",
        ),
    )

    op.create_table(
        "corporate_action",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("announced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cash_amount", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("share_ratio", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "action_type",
            "ex_date",
            "source",
            "version",
            name="uq_corporate_action_version",
        ),
    )

    op.create_table(
        "financial_statement_point_in_time",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("report_period", sa.Date(), nullable=False),
        sa.Column("metric_name", sa.String(length=96), nullable=False),
        sa.Column("metric_value", sa.Numeric(precision=30, scale=8), nullable=False),
        sa.Column("announced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_revision", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "report_period",
            "metric_name",
            "source",
            "provider_revision",
            name="uq_fundamental_revision",
        ),
    )
    op.create_index(
        "ix_fundamental_point_in_time",
        "financial_statement_point_in_time",
        ["instrument_id", "metric_name", "available_at"],
        unique=False,
    )

    op.create_table(
        "industry_classification",
        sa.Column("industry_id", sa.String(length=96), nullable=False),
        sa.Column("classification", sa.String(length=32), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.String(length=96), nullable=True),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["industry_classification.industry_id"],
        ),
        sa.PrimaryKeyConstraint("industry_id"),
        sa.UniqueConstraint(
            "classification",
            "code",
            "version",
            name="uq_industry_classification_version",
        ),
    )

    op.create_table(
        "industry_membership_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("industry_id", sa.String(length=96), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["industry_id"],
            ["industry_classification.industry_id"],
        ),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.instrument_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "industry_id",
            "effective_from",
            "version",
            name="uq_industry_membership_version",
        ),
    )
    op.create_index(
        "ix_industry_membership_dates",
        "industry_membership_history",
        ["instrument_id", "effective_from", "effective_to"],
        unique=False,
    )


def downgrade() -> None:
    """Drop only the frozen P1 schema in reverse dependency order."""

    op.drop_table("industry_membership_history")
    op.drop_table("industry_classification")
    op.drop_table("financial_statement_point_in_time")
    op.drop_table("corporate_action")
    op.drop_table("adjustment_factor")
    op.drop_table("market_bar_daily")
    op.drop_table("instrument_status")
    op.drop_table("instrument_alias")
    op.drop_table("dataset_version")
    op.drop_table("data_quality_result")
    op.drop_table("raw_payload")
    op.drop_table("trading_calendar")
    op.drop_table("instrument")
