"""Inspect or apply the repository's forward-only Alembic migration."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from quant_agent.deployment.migrations import MigrationGuard


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--upgrade", action="store_true")
    arguments = parser.parse_args()
    database_url = os.getenv("QUANT_AGENT_DATABASE_URL")
    if not database_url:
        parser.error("QUANT_AGENT_DATABASE_URL is required")
    project_root = Path(__file__).resolve().parents[2]
    guard = MigrationGuard(project_root)
    result = guard.upgrade(database_url) if arguments.upgrade else guard.inspect(database_url)
    print(json.dumps(asdict(result), indent=2))
    if not result.compatible:
        return 1
    if arguments.check and result.upgrade_required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
