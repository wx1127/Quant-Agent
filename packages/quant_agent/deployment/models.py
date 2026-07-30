"""Immutable release manifests and audit-preserving deployment plans."""

import re
from dataclasses import dataclass
from datetime import datetime

from quant_agent.core.time import ensure_aware

_IMAGE_DIGEST = re.compile(r"^[a-z0-9._/-]+@sha256:[a-f0-9]{64}$")
_GIT_COMMIT = re.compile(r"^[a-f0-9]{7,64}$")


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    release_id: str
    image_ref: str
    git_commit: str
    schema_revision: str
    config_version: str
    audit_store_id: str
    production_kill_switch_default: bool
    created_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.created_at)
        if not self.release_id or not self.schema_revision or not self.config_version:
            raise ValueError("release identity, schema and config versions are required")
        if not _IMAGE_DIGEST.fullmatch(self.image_ref):
            raise ValueError("release image must use an immutable repository@sha256 digest")
        if not _GIT_COMMIT.fullmatch(self.git_commit):
            raise ValueError("release git commit must be a hexadecimal revision")
        if not self.audit_store_id:
            raise ValueError("audit store identity is required")
        if not self.production_kill_switch_default:
            raise ValueError("production release must start with Kill Switch active")


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    action: str
    from_release_id: str | None
    to_release_id: str
    steps: tuple[str, ...]
    database_downgrade: bool
    audit_store_id: str
    production_kill_switch_active: bool


class ReleasePlanner:
    """Plan code deployment/rollback without deleting audit data or downgrading schema."""

    def deploy(
        self,
        current: ReleaseManifest | None,
        target: ReleaseManifest,
    ) -> DeploymentPlan:
        if current is not None and current.audit_store_id != target.audit_store_id:
            raise ValueError("deployment cannot replace the append-only audit store")
        return DeploymentPlan(
            action="DEPLOY",
            from_release_id=current.release_id if current else None,
            to_release_id=target.release_id,
            steps=(
                "verify image digest and release manifest",
                "inspect current database revision",
                "apply forward-only migration",
                "start candidate with production Kill Switch active",
                "verify liveness and readiness",
                "record release audit event",
            ),
            database_downgrade=False,
            audit_store_id=target.audit_store_id,
            production_kill_switch_active=True,
        )

    def rollback(
        self,
        current: ReleaseManifest,
        target: ReleaseManifest,
    ) -> DeploymentPlan:
        if current.audit_store_id != target.audit_store_id:
            raise ValueError("rollback cannot replace or truncate the audit store")
        if current.schema_revision != target.schema_revision:
            raise ValueError(
                "automatic rollback requires schema-compatible images; "
                "database downgrade is forbidden"
            )
        return DeploymentPlan(
            action="ROLLBACK",
            from_release_id=current.release_id,
            to_release_id=target.release_id,
            steps=(
                "activate global Kill Switch",
                "retain current database schema",
                "retain append-only audit volume",
                "start prior immutable image digest",
                "verify liveness, readiness and Kill Switch",
                "record rollback audit event",
            ),
            database_downgrade=False,
            audit_store_id=current.audit_store_id,
            production_kill_switch_active=True,
        )
