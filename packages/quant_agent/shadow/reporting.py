"""Machine-readable and operator-readable shadow-run status reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from quant_agent.shadow.models import ShadowAcceptance, ShadowSessionConfig


def write_shadow_report(
    output_dir: str | Path,
    config: ShadowSessionConfig,
    acceptance: ShadowAcceptance,
    *,
    generated_on: date,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": asdict(config),
        "acceptance": asdict(acceptance),
        "generated_on": generated_on.isoformat(),
    }
    json_path = root / "shadow-status.json"
    markdown_path = root / "shadow-status.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                f"# Shadow Run {config.session_id}",
                "",
                f"- status: `{acceptance.status}`",
                (
                    f"- consecutive days: `{acceptance.consecutive_trading_days}"
                    f"/{acceptance.required_trading_days}`"
                ),
                f"- data success: `{acceptance.data_success_rate:.2%}`",
                f"- report success: `{acceptance.report_success_rate:.2%}`",
                f"- pipeline success: `{acceptance.pipeline_success_rate:.2%}`",
                (f"- future-data violations: `{acceptance.future_data_violations}`"),
                (f"- executable orders emitted: `{acceptance.executable_orders_emitted}`"),
                (f"- unreconciled trading days: `{len(acceptance.unreconciled_trading_days)}`"),
                (f"- open P0/P1 incidents: `{len(acceptance.open_high_incident_ids)}`"),
                "",
                "## Blocking reasons",
                "",
                *([f"- {reason}" for reason in acceptance.reasons] or ["- none"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path
