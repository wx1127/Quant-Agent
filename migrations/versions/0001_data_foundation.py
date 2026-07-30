"""Create the P1 data-foundation schema.

Revision ID: 0001
Revises:
Create Date: 2026-07-30
"""

from collections.abc import Sequence

from alembic import op

from quant_agent.data import models as data_models  # noqa: F401
from quant_agent.data.database import Base

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create every table in the initial versioned data schema."""

    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    """Drop the initial schema in reverse dependency order."""

    Base.metadata.drop_all(bind=op.get_bind())
