"""One-command E2E acceptance runner: python -m tests.e2e.runner."""

import json
from dataclasses import asdict
from pathlib import Path

from quant_agent.validation.environment import E2EEnvironment


def main() -> int:
    fixture = Path(__file__).parent / "fixtures" / "market_day.json"
    output = Path("artifacts/e2e")
    try:
        first = E2EEnvironment(fixture).run()
        second = E2EEnvironment(fixture).run()
        if first != second or not first.reconciled:
            raise AssertionError("E2E flow is not deterministic or reconciliation failed")
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2),
            encoding="utf-8",
        )
        return 1
    print(json.dumps(asdict(first), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
