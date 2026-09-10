"""Small dependency-free metrics registry for local and test deployments."""

from __future__ import annotations

from threading import Lock
from time import monotonic


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = 0
        self._errors = 0
        self._total_seconds = 0.0

    def observe_request(self, elapsed: float, error: bool = False) -> None:
        with self._lock:
            self._requests += 1
            self._errors += int(error)
            self._total_seconds += elapsed

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            requests = self._requests
            return {
                "api_requests_total": requests,
                "api_errors_total": self._errors,
                "api_request_seconds_total": round(self._total_seconds, 6),
                "api_request_seconds_average": (
                    round(self._total_seconds / requests, 6) if requests else 0.0
                ),
            }


metrics = MetricsRegistry()


def timer() -> float:
    return monotonic()
