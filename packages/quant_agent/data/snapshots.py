"""Immutable Parquet snapshots with content manifests and DuckDB queries."""

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from quant_agent.core.time import shanghai_now
from quant_agent.data.models import DatasetVersionRow
from quant_agent.data.quality import QualityReport

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class SnapshotFile(BaseModel):
    """One immutable Parquet artifact."""

    model_config = ConfigDict(frozen=True)

    name: str
    relative_path: str
    row_count: int
    sha256: str


class SnapshotManifest(BaseModel):
    """Content-addressed dataset snapshot manifest."""

    model_config = ConfigDict(frozen=True)

    data_version: str
    created_at: str
    content_hash: str
    files: tuple[SnapshotFile, ...]
    metadata: dict[str, Any]


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _manifest_hash(
    data_version: str,
    files: tuple[SnapshotFile, ...],
    metadata: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "data_version": data_version,
            "files": [item.model_dump() for item in files],
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SnapshotStore:
    """Write-once dataset store rooted at a controlled directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def create(
        self,
        data_version: str,
        tables: Mapping[str, pa.Table],
        *,
        quality_report: QualityReport,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
    ) -> SnapshotManifest:
        """Write qualified tables and atomically publish an immutable manifest."""

        if quality_report.data_version != data_version:
            raise ValueError("quality report version does not match snapshot version")
        if not quality_report.qualified:
            raise ValueError("data quality failed; snapshot publication is blocked")
        if not tables:
            raise ValueError("snapshot requires at least one table")
        for name in tables:
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError(f"unsafe snapshot table name: {name}")

        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / data_version
        if target.exists():
            existing = self.load_manifest(data_version)
            expected_metadata = metadata or {}
            if existing.metadata != expected_metadata:
                raise ValueError("snapshot version already exists with different metadata")
            existing_names = {item.name for item in existing.files}
            if existing_names != set(tables):
                raise ValueError("snapshot version already exists with different tables")
            for name, table in tables.items():
                if not self.read_table(data_version, name).equals(table):
                    raise ValueError("snapshot version already exists with different table content")
            return existing

        temp_root = Path(tempfile.mkdtemp(prefix=f".{data_version}-", dir=self._root))
        try:
            files: list[SnapshotFile] = []
            for name in sorted(tables):
                file_name = f"{name}.parquet"
                file_path = temp_root / file_name
                pq.write_table(tables[name], file_path, compression="zstd")
                files.append(
                    SnapshotFile(
                        name=name,
                        relative_path=file_name,
                        row_count=tables[name].num_rows,
                        sha256=_file_hash(file_path),
                    )
                )
            frozen_files = tuple(files)
            frozen_metadata = metadata or {}
            manifest = SnapshotManifest(
                data_version=data_version,
                created_at=shanghai_now().isoformat(),
                content_hash=_manifest_hash(
                    data_version,
                    frozen_files,
                    frozen_metadata,
                ),
                files=frozen_files,
                metadata=frozen_metadata,
            )
            (temp_root / "manifest.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            temp_root.replace(target)
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

        if session is not None:
            session.add(
                DatasetVersionRow(
                    data_version=data_version,
                    status="QUALIFIED",
                    content_hash=manifest.content_hash,
                    manifest=manifest.model_dump(mode="json"),
                )
            )
            session.flush()
        return manifest

    def load_manifest(self, data_version: str) -> SnapshotManifest:
        """Load and validate a published manifest."""

        manifest_path = self._root / data_version / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"snapshot not found: {data_version}")
        return SnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def verify(self, data_version: str) -> bool:
        """Verify every file hash and the manifest content hash."""

        manifest = self.load_manifest(data_version)
        snapshot_dir = self._root / data_version
        for item in manifest.files:
            path = snapshot_dir / item.relative_path
            if not path.is_file() or _file_hash(path) != item.sha256:
                return False
        return manifest.content_hash == _manifest_hash(
            manifest.data_version,
            manifest.files,
            manifest.metadata,
        )

    def read_table(self, data_version: str, name: str) -> pa.Table:
        """Read one named Parquet table."""

        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe snapshot table name: {name}")
        manifest = self.load_manifest(data_version)
        known = {item.name: item.relative_path for item in manifest.files}
        if name not in known:
            raise KeyError(name)
        return pq.read_table(self._root / data_version / known[name])

    def query(self, data_version: str, sql: str) -> pa.Table:
        """Query snapshot tables through isolated in-memory DuckDB views."""

        manifest = self.load_manifest(data_version)
        connection = duckdb.connect(database=":memory:")
        try:
            for item in manifest.files:
                if not _SAFE_NAME.fullmatch(item.name):
                    raise ValueError(f"unsafe snapshot table name: {item.name}")
                path = str((self._root / data_version / item.relative_path).resolve())
                connection.register(item.name, pq.read_table(path))
            return connection.execute(sql).to_arrow_table()
        finally:
            connection.close()
