"""Alembic migration inspection and forward-only upgrade boundary."""

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine


@dataclass(frozen=True, slots=True)
class MigrationInspection:
    current_revisions: tuple[str, ...]
    target_revisions: tuple[str, ...]
    upgrade_required: bool
    compatible: bool


class MigrationGuard:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._config_path = self._project_root / "alembic.ini"
        if not self._config_path.is_file():
            raise FileNotFoundError("alembic.ini is required")

    def inspect(self, database_url: str) -> MigrationInspection:
        config = self._config(database_url)
        script = ScriptDirectory.from_config(config)
        targets = tuple(sorted(script.get_heads()))
        if len(targets) != 1:
            return MigrationInspection((), targets, True, False)
        engine = create_engine(database_url)
        try:
            with engine.connect() as connection:
                current = tuple(sorted(MigrationContext.configure(connection).get_current_heads()))
        finally:
            engine.dispose()
        known = {revision.revision for revision in script.walk_revisions()}
        compatible = all(revision in known for revision in current)
        return MigrationInspection(
            current,
            targets,
            current != targets,
            compatible,
        )

    def upgrade(self, database_url: str) -> MigrationInspection:
        before = self.inspect(database_url)
        if not before.compatible:
            raise ValueError("database revision is not part of the release migration graph")
        command.upgrade(self._config(database_url), "head")
        after = self.inspect(database_url)
        if not after.compatible or after.upgrade_required:
            raise RuntimeError("database did not reach the expected migration head")
        return after

    def _config(self, database_url: str) -> Config:
        config = Config(str(self._config_path))
        config.set_main_option("script_location", str(self._project_root / "migrations"))
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
        return config
