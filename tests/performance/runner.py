"""One-command performance acceptance runner: python -m tests.performance.runner."""

import json

from apps.api.app import create_app
from fastapi.testclient import TestClient

from quant_agent.validation.performance import PerformanceHarness


def _data_update() -> list[tuple[str, float]]:
    return [(f"stock-{index:04d}", round(10 + index / 100, 2)) for index in range(6_000)]


def _ranking() -> list[float]:
    return sorted(((index * 17 % 1000) / 1000 for index in range(6_000)), reverse=True)


def main() -> int:
    harness = PerformanceHarness()
    with TestClient(create_app()) as client:
        report = harness.measure(
            {
                "data-update": (_data_update, 250.0),
                "feature-and-ranking": (_ranking, 200.0),
                "api-health": (lambda: client.get("/health").raise_for_status(), 100.0),
            }
        )
    payload = {
        "passed": report.passed,
        "latency": {item.name: harness.summarize(item) for item in report.latency},
        "retained_memory_bytes": report.retained_memory_bytes,
        "memory_limit_bytes": report.memory_limit_bytes,
        "capacity": {
            "rows": report.capacity.rows,
            "estimated_gib": report.capacity.estimated_gib,
            "trading_days": report.capacity.trading_days,
        },
        "after_close_budget_hours": report.after_close_budget_hours,
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
