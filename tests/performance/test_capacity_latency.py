from apps.api.app import create_app
from fastapi.testclient import TestClient

from quant_agent.validation.performance import CapacityProjection, PerformanceHarness


def _data_update_workload() -> list[dict[str, float | str]]:
    return [
        {"instrument_id": f"stock-{index:04d}", "close": round(10 + index / 100, 2)}
        for index in range(6_000)
    ]


def _feature_and_rank_workload() -> list[tuple[str, float]]:
    values = [(f"stock-{index:04d}", (index * 17 % 1000) / 1000) for index in range(6_000)]
    return sorted(values, key=lambda item: (-item[1], item[0]))[:100]


def test_daily_workloads_meet_latency_memory_and_capacity_targets() -> None:
    with TestClient(create_app()) as client:
        report = PerformanceHarness().measure(
            {
                "data-update": (_data_update_workload, 250.0),
                "feature-and-ranking": (_feature_and_rank_workload, 200.0),
                "api-health": (lambda: client.get("/health").raise_for_status(), 100.0),
            },
            repeats=20,
        )
    assert report.passed
    assert report.capacity.rows == 1_512_000
    assert report.capacity.estimated_gib > 0


def test_capacity_requires_at_least_one_trading_year() -> None:
    assert CapacityProjection(6_000, 252, 512).covers_one_year
    assert not CapacityProjection(6_000, 251, 512).covers_one_year
