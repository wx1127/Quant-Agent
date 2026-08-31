"""Alembic environment for Quant Agent database migrations."""

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from quant_agent.data import models as data_models  # noqa: F401
from quant_agent.data.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the migration URL, preferring an explicit environment override."""

    database_url = os.environ.get("QUANT_AGENT_DATABASE_URL")
    if database_url is None:
        database_url = config.get_main_option("sqlalchemy.url")
    if not database_url:
        raise RuntimeError("database URL is not configured")
    return database_url


def _configure_database_url() -> str:
    """Apply the resolved URL without exposing credentials in migration output."""

    database_url = _database_url()
    # Alembic uses ConfigParser interpolation, so percent-encoded URL values
    # must be escaped when assigned programmatically.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return database_url


def _ensure_sqlite_parent(database_url: str) -> None:
    """Create the parent directory for a file-backed SQLite database."""

    url = make_url(database_url)
    if (
        url.get_backend_name() != "sqlite"
        or url.database in {None, "", ":memory:"}
        or url.database.startswith("file:")
    ):
        return
    Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


database_url = _configure_database_url()


def run_migrations_offline() -> None:
    """Run migrations without creating an Engine."""

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live connection."""

    _ensure_sqlite_parent(database_url)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
