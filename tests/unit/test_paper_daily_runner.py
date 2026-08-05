import json
from datetime import date
from pathlib import Path

import pytest
from scripts.paper.run_daily import _pending


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_explicit_paper_override_supersedes_original_pending_without_overwrite(
    tmp_path: Path,
) -> None:
    trading_date = date(2026, 8, 6)
    original = {"signal_date": "2026-08-05", "execute_on": "2026-08-06"}
    override = {
        "signal_date": "2026-08-05",
        "execute_on": "2026-08-06",
        "mode": "PAPER",
        "manual_override": True,
        "superseded_signal_dates": ["2026-08-05"],
        "batch": {"drafts": [{"side": "SELL"}]},
    }
    _write(tmp_path / "pending" / "2026-08-05.json", original)
    _write(tmp_path / "pending_overrides" / "2026-08-06.json", override)

    assert _pending(tmp_path, trading_date) == override
    assert json.loads(
        (tmp_path / "pending" / "2026-08-05.json").read_text(encoding="utf-8")
    ) == original


def test_pending_override_requires_explicit_paper_authorization(tmp_path: Path) -> None:
    _write(
        tmp_path / "pending_overrides" / "2026-08-06.json",
        {
            "signal_date": "2026-08-05",
            "execute_on": "2026-08-06",
            "manual_override": True,
            "superseded_signal_dates": [],
        },
    )

    with pytest.raises(ValueError, match="explicitly authorized PAPER"):
        _pending(tmp_path, date(2026, 8, 6))
