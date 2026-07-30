"""One-command golden replay runner: python -m tests.replay.runner."""

import json
from dataclasses import asdict

from quant_agent.validation.golden import SystemGoldenReplay


def main() -> int:
    report = SystemGoldenReplay().run()
    print(
        json.dumps(
            {
                "passed": report.passed,
                "results": [asdict(item) for item in report.results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
