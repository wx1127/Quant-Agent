"""Low-cardinality operational metrics for data, services and trading safety."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from quant_agent.core.time import shanghai_now


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    component: str
    operation: str
    requests: int
    successes: int
    total_latency_seconds: float
    maximum_latency_seconds: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.requests if self.requests else 1.0


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    generated_at: datetime
    data_completeness: tuple[tuple[str, float], ...]
    services: tuple[ServiceHealth, ...]
    risk_service_available: bool
    risk_rejections_total: int
    order_submissions_total: int
    duplicate_order_risks_total: int
    reconciliation_differences: tuple[tuple[str, int], ...]
    kill_switch_active: bool

    def reconciliation_count(self, severity: str) -> int:
        return dict(self.reconciliation_differences).get(severity.upper(), 0)


@dataclass(slots=True)
class _MutableServiceHealth:
    requests: int = 0
    successes: int = 0
    total_latency_seconds: float = 0.0
    maximum_latency_seconds: float = 0.0


class MonitoringRegistry:
    """Thread-safe in-process registry with bounded, operator-defined labels."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._data_completeness: dict[str, float] = {}
        self._services: dict[tuple[str, str], _MutableServiceHealth] = {}
        self._risk_service_available = True
        self._risk_rejections_total = 0
        self._order_submissions_total = 0
        self._duplicate_order_risks_total = 0
        self._reconciliation_differences: dict[str, int] = {}
        self._kill_switch_active = False

    def record_data_completeness(self, pipeline: str, ratio: float) -> None:
        _validate_label(pipeline)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("data completeness ratio must be between zero and one")
        with self._lock:
            self._data_completeness[pipeline] = ratio

    def record_service(
        self,
        component: str,
        operation: str,
        *,
        success: bool,
        latency_seconds: float,
    ) -> None:
        _validate_label(component)
        _validate_label(operation)
        if latency_seconds < 0:
            raise ValueError("service latency cannot be negative")
        with self._lock:
            health = self._services.setdefault((component, operation), _MutableServiceHealth())
            health.requests += 1
            health.successes += int(success)
            health.total_latency_seconds += latency_seconds
            health.maximum_latency_seconds = max(health.maximum_latency_seconds, latency_seconds)

    def record_risk_check(self, *, available: bool, passed: bool) -> None:
        with self._lock:
            self._risk_service_available = available
            if available and not passed:
                self._risk_rejections_total += 1

    def record_order_submission(self, *, duplicate_risk: bool = False) -> None:
        with self._lock:
            if duplicate_risk:
                self._duplicate_order_risks_total += 1
            else:
                self._order_submissions_total += 1

    def record_reconciliation(self, severities: tuple[str, ...]) -> None:
        with self._lock:
            for severity in severities:
                normalized = severity.upper()
                if normalized not in {"INFO", "WARNING", "CRITICAL"}:
                    raise ValueError("unsupported reconciliation severity")
                self._reconciliation_differences[normalized] = (
                    self._reconciliation_differences.get(normalized, 0) + 1
                )

    def set_kill_switch(self, active: bool) -> None:
        with self._lock:
            self._kill_switch_active = active

    def snapshot(self, *, generated_at: datetime | None = None) -> MonitoringSnapshot:
        with self._lock:
            services = tuple(
                ServiceHealth(
                    component=component,
                    operation=operation,
                    requests=item.requests,
                    successes=item.successes,
                    total_latency_seconds=item.total_latency_seconds,
                    maximum_latency_seconds=item.maximum_latency_seconds,
                )
                for (component, operation), item in sorted(self._services.items())
            )
            return MonitoringSnapshot(
                generated_at=generated_at or shanghai_now(),
                data_completeness=tuple(sorted(self._data_completeness.items())),
                services=services,
                risk_service_available=self._risk_service_available,
                risk_rejections_total=self._risk_rejections_total,
                order_submissions_total=self._order_submissions_total,
                duplicate_order_risks_total=self._duplicate_order_risks_total,
                reconciliation_differences=tuple(sorted(self._reconciliation_differences.items())),
                kill_switch_active=self._kill_switch_active,
            )

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP quant_agent_data_completeness_ratio Completed expected data ratio.",
            "# TYPE quant_agent_data_completeness_ratio gauge",
        ]
        lines.extend(
            f'quant_agent_data_completeness_ratio{{pipeline="{pipeline}"}} {ratio:.6f}'
            for pipeline, ratio in snapshot.data_completeness
        )
        lines.extend(
            [
                "# HELP quant_agent_service_requests_total Service request count.",
                "# TYPE quant_agent_service_requests_total counter",
            ]
        )
        for service in snapshot.services:
            labels = f'component="{service.component}",operation="{service.operation}"'
            lines.append(f"quant_agent_service_requests_total{{{labels}}} {service.requests}")
            lines.append(f"quant_agent_service_success_total{{{labels}}} {service.successes}")
            lines.append(
                "quant_agent_service_latency_seconds_sum"
                f"{{{labels}}} {service.total_latency_seconds:.6f}"
            )
            lines.append(
                "quant_agent_service_latency_seconds_max"
                f"{{{labels}}} {service.maximum_latency_seconds:.6f}"
            )
        lines.extend(
            [
                f"quant_agent_risk_service_available {int(snapshot.risk_service_available)}",
                f"quant_agent_risk_rejections_total {snapshot.risk_rejections_total}",
                f"quant_agent_order_submissions_total {snapshot.order_submissions_total}",
                (f"quant_agent_duplicate_order_risks_total {snapshot.duplicate_order_risks_total}"),
                f"quant_agent_kill_switch_active {int(snapshot.kill_switch_active)}",
            ]
        )
        for severity, count in snapshot.reconciliation_differences:
            lines.append(
                f'quant_agent_reconciliation_differences_total{{severity="{severity}"}} {count}'
            )
        return "\n".join(lines) + "\n"


def _validate_label(value: str) -> None:
    if not value or len(value) > 80:
        raise ValueError("monitoring labels must contain between 1 and 80 characters")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-/{}"
        for character in value
    ):
        raise ValueError("monitoring labels contain unsupported characters")
