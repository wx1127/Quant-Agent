"""Command-line entry point for local Quant Agent workflows."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pydantic import SecretStr

from quant_agent.config import QuantAgentSettings
from quant_agent.data.domain import InstrumentType
from quant_agent.data.sync import SyncMode
from quant_agent.pipelines import (
    TushareDataset,
    TushareSyncSpec,
    query_sync_status,
    run_demo_pipeline,
    run_research_demo,
    run_tushare_sync,
    sync_result_dict,
)

_DEFAULT_DATABASE_URL = "sqlite:///./artifacts/quant_agent.db"
_DEFAULT_STORAGE_PATH = "./artifacts/snapshots"


class CliInputError(ValueError):
    """A validation error whose curated message is safe to show to the operator."""


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo",
        help="run the deterministic provider-to-snapshot verification pipeline",
    )
    demo.add_argument("--config", type=Path, help="optional environment TOML file")
    demo.add_argument("--database-url", help="override the local SQLite database URL")
    demo.add_argument("--storage-path", type=Path, help="override snapshot storage path")
    demo.add_argument("--data-version", default="demo_20260730_v1")
    demo.set_defaults(handler=_run_demo)

    data = subparsers.add_parser("data", help="inspect and run data synchronization")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    status = data_commands.add_parser("status", help="show sync runs and checkpoints")
    status.add_argument("--config", type=Path, help="optional environment TOML file")
    status.add_argument("--database-url", help="override the configured database URL")
    status.add_argument("--provider", help="filter by provider")
    status.add_argument("--dataset", help="filter by dataset")
    status.add_argument("--limit", type=int, default=20)
    status.set_defaults(handler=_run_data_status)
    sync = data_commands.add_parser("sync", help="run one raw-first Tushare dataset sync")
    sync.add_argument("--config", type=Path, help="optional environment TOML file")
    sync.add_argument("--database-url", help="override the configured database URL")
    sync.add_argument(
        "--dataset",
        choices=[value.value for value in TushareDataset],
        required=True,
    )
    sync.add_argument("--start", type=_iso_date, required=True)
    sync.add_argument("--end", type=_iso_date, required=True)
    sync.add_argument(
        "--instrument-type",
        choices=[value.value for value in InstrumentType],
        default=InstrumentType.STOCK.value,
    )
    sync.add_argument(
        "--market",
        help="calendar market for calendar/bars, or provider market for instrument master",
    )
    sync.add_argument(
        "--instrument-id",
        action="append",
        default=[],
        help="repeatable stable ID filter for daily/adjustment/announcement datasets",
    )
    sync.add_argument(
        "--mode",
        choices=[value.value for value in SyncMode],
        default=SyncMode.INCREMENTAL.value,
    )
    sync.add_argument("--idempotency-key")
    sync.add_argument(
        "--token-env",
        default="MARKET_DATA_TOKEN",
        help="environment variable containing the token; literal token arguments are unsupported",
    )
    sync.set_defaults(handler=_run_data_sync)

    research = subparsers.add_parser("research", help="run deterministic quantitative analysis")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    research_demo = research_commands.add_parser(
        "demo",
        help="run the synthetic point-in-time market and industry analysis",
    )
    research_demo.add_argument("--data-version", default="research_demo_v1")
    research_demo.set_defaults(handler=_run_research_demo)
    return parser


def _run_demo(args: argparse.Namespace) -> int:
    if args.config is not None:
        settings = QuantAgentSettings.load(args.config)
        configured_database_url = settings.resolve_database_url().get_secret_value()
        configured_storage_path = settings.storage.research_storage_path
    else:
        configured_database_url = os.environ.get(
            "QUANT_AGENT_DATABASE_URL",
            _DEFAULT_DATABASE_URL,
        )
        configured_storage_path = os.environ.get(
            "QUANT_AGENT_RESEARCH_STORAGE_PATH",
            _DEFAULT_STORAGE_PATH,
        )
    database_url = args.database_url or configured_database_url
    storage_path = args.storage_path or Path(configured_storage_path)
    result = run_demo_pipeline(
        database_url=database_url,
        storage_path=storage_path,
        data_version=args.data_version,
    )
    print(json.dumps({"ok": True, **result.to_dict()}, ensure_ascii=False, sort_keys=True))
    return 0


def _run_data_status(args: argparse.Namespace) -> int:
    if args.config is not None:
        settings = QuantAgentSettings.load(args.config)
        configured_database_url = settings.resolve_database_url().get_secret_value()
    else:
        configured_database_url = os.environ.get(
            "QUANT_AGENT_DATABASE_URL",
            _DEFAULT_DATABASE_URL,
        )
    result = query_sync_status(
        database_url=args.database_url or configured_database_url,
        provider=args.provider,
        dataset=args.dataset,
        limit=args.limit,
    )
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, sort_keys=True))
    return 0


def _run_data_sync(args: argparse.Namespace) -> int:
    try:
        spec = TushareSyncSpec(
            dataset=TushareDataset(args.dataset),
            start=args.start,
            end=args.end,
            instrument_type=InstrumentType(args.instrument_type),
            market=args.market,
            instrument_ids=tuple(args.instrument_id),
            mode=SyncMode(args.mode),
            idempotency_key=args.idempotency_key,
        )
    except ValueError as error:
        raise CliInputError(str(error)) from error
    if args.config is not None:
        settings = QuantAgentSettings.load(args.config)
        configured_database_url = settings.resolve_database_url().get_secret_value()
        try:
            token = settings.resolve_market_data_token()
        except ValueError as error:
            raise CliInputError("configured market data token could not be resolved") from error
    else:
        configured_database_url = os.environ.get(
            "QUANT_AGENT_DATABASE_URL",
            _DEFAULT_DATABASE_URL,
        )
        token_value = os.environ.get(args.token_env)
        if not token_value:
            raise CliInputError(
                f"market data token environment variable {args.token_env!r} is not set"
            )
        token = SecretStr(token_value)
    result = run_tushare_sync(
        spec=spec,
        database_url=args.database_url or configured_database_url,
        market_data_token=token,
    )
    print(
        json.dumps(
            {
                "dataset": spec.dataset.value,
                "ok": True,
                **sync_result_dict(result),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_research_demo(args: argparse.Namespace) -> int:
    result = run_research_demo(data_version=args.data_version)
    print(json.dumps({"ok": True, **result.to_dict()}, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run one command, and return a process exit code."""

    args = _parser().parse_args(argv)
    handler = args.handler
    if not callable(handler):
        raise TypeError("CLI command handler is not callable")
    try:
        return int(handler(args))
    except CliInputError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "INVALID_INPUT",
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": type(error).__name__,
                    "message": "command failed safely",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
