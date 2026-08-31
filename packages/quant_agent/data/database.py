"""SQLAlchemy database boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry


class Base(DeclarativeBase):
    """Shared declarative base for versioned database tables."""


def _enable_sqlite_foreign_keys(
    dbapi_connection: DBAPIConnection,
    _connection_record: ConnectionPoolEntry,
) -> None:
    """Enable SQLite's opt-in foreign-key enforcement for every connection."""

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _begin_deferred_sqlite_transaction(connection: Connection) -> None:
    """Start a real SQLite transaction and defer its FK checks to commit."""

    connection.exec_driver_sql("BEGIN")
    connection.exec_driver_sql("PRAGMA defer_foreign_keys=ON")


class Database:
    """Small engine/session owner suitable for services and tests."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        if url.startswith(("env://", "vault://")):
            raise ValueError("database URL reference must be resolved before use")
        parsed_url = make_url(url)
        if parsed_url.get_backend_name() == "sqlite":
            database_path = parsed_url.database
            if (
                database_path is not None
                and database_path != ":memory:"
                and not database_path.startswith("file:")
            ):
                Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        is_sqlite = parsed_url.get_backend_name() == "sqlite"
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine: Engine = create_engine(url, echo=echo, connect_args=connect_args)
        if is_sqlite:
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
            event.listen(self.engine, "begin", _begin_deferred_sqlite_transaction)
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        """Create the current schema for local tests.

        Production deployments must use Alembic migrations instead.
        """

        from quant_agent.data import models as data_models  # noqa: F401

        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional session with rollback on error."""

        db_session = self._session_factory()
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()
