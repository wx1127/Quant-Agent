"""Deterministic P0-P3 alert evaluation, routing, deduplication and delivery."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Protocol

from quant_agent.core.time import shanghai_now
from quant_agent.observability.monitoring import MonitoringSnapshot
from quant_agent.observability.redaction import redact


class AlertSeverity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


@dataclass(frozen=True, slots=True)
class AlertPolicyConfig:
    minimum_data_completeness: float = 0.98
    minimum_service_success_rate: float = 0.99
    maximum_service_latency_seconds: float = 2.0
    minimum_service_requests: int = 10
    elevated_risk_rejections: int = 5

    @classmethod
    def from_toml(cls, path: str | Path) -> AlertPolicyConfig:
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)["thresholds"]
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AlertEvent:
    fingerprint: str
    policy_id: str
    severity: AlertSeverity
    title: str
    summary: str
    source: str
    runbook: str
    diagnostics: tuple[tuple[str, str], ...]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class AlertDelivery:
    event: AlertEvent
    targets: tuple[str, ...]
    delivered_at: datetime


class AlertSink(Protocol):
    def send(self, delivery: AlertDelivery) -> None: ...


class InMemoryAlertSink:
    def __init__(self) -> None:
        self.deliveries: list[AlertDelivery] = []

    def send(self, delivery: AlertDelivery) -> None:
        self.deliveries.append(delivery)


class JsonLinesAlertSink:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = RLock()

    def send(self, delivery: AlertDelivery) -> None:
        payload = redact(asdict(delivery))
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")


class AlertPolicyEngine:
    def __init__(self, config: AlertPolicyConfig | None = None) -> None:
        self._config = config or AlertPolicyConfig()

    def evaluate(self, snapshot: MonitoringSnapshot) -> tuple[AlertEvent, ...]:
        alerts: list[AlertEvent] = []
        if snapshot.duplicate_order_risks_total:
            alerts.append(
                _event(
                    "duplicate-order-risk",
                    AlertSeverity.P0,
                    "检测到重复下单风险",
                    "幂等键绑定到不同订单草案; 已拒绝提交。",
                    "order-gateway",
                    "docs/07-monitoring-alerting-runbook.md#p0-重复下单风险",
                    {"count": str(snapshot.duplicate_order_risks_total)},
                    snapshot.generated_at,
                )
            )
        critical = snapshot.reconciliation_count("CRITICAL")
        if critical:
            alerts.append(
                _event(
                    "critical-reconciliation-difference",
                    AlertSeverity.P0,
                    "账实核对存在严重差异",
                    "订单、成交、现金或持仓核对出现严重差异。",
                    "reconciliation",
                    "docs/07-monitoring-alerting-runbook.md#p0-严重账实差异",
                    {"critical_differences": str(critical)},
                    snapshot.generated_at,
                )
            )
        if not snapshot.risk_service_available:
            alerts.append(
                _event(
                    "risk-service-unavailable",
                    AlertSeverity.P1,
                    "风控服务不可用",
                    "风控检查不可用; 交易变更必须保持拒绝状态。",
                    "risk-service",
                    "docs/07-monitoring-alerting-runbook.md#p1-风控不可用",
                    {"fail_closed": "true"},
                    snapshot.generated_at,
                )
            )
        if snapshot.kill_switch_active:
            alerts.append(
                _event(
                    "kill-switch-active",
                    AlertSeverity.P1,
                    "Kill Switch 已开启",
                    "全局或账户交易门禁处于开启状态; 需要 RiskAdmin 复核。",
                    "risk-control",
                    "docs/07-monitoring-alerting-runbook.md#p1-kill-switch-开启",
                    {"active": "true"},
                    snapshot.generated_at,
                )
            )
        for pipeline, ratio in snapshot.data_completeness:
            if ratio < self._config.minimum_data_completeness:
                alerts.append(
                    _event(
                        f"data-incomplete:{pipeline}",
                        AlertSeverity.P2,
                        "数据完成度低于阈值",
                        "预期数据尚未完整到达; 依赖任务不得继续。",
                        "data-pipeline",
                        "docs/07-monitoring-alerting-runbook.md#p2-数据完成度不足",
                        {"pipeline": pipeline, "ratio": f"{ratio:.4f}"},
                        snapshot.generated_at,
                    )
                )
        for service in snapshot.services:
            if service.requests < self._config.minimum_service_requests:
                continue
            diagnostics = {
                "component": service.component,
                "operation": service.operation,
            }
            if service.success_rate < self._config.minimum_service_success_rate:
                alerts.append(
                    _event(
                        f"service-success:{service.component}:{service.operation}",
                        AlertSeverity.P2,
                        "服务成功率低于阈值",
                        "服务成功率下降; 需要检查最近错误与依赖状态。",
                        "service-runtime",
                        "docs/07-monitoring-alerting-runbook.md#p2-服务健康度",
                        {
                            **diagnostics,
                            "success_rate": f"{service.success_rate:.4f}",
                        },
                        snapshot.generated_at,
                    )
                )
            if service.maximum_latency_seconds > self._config.maximum_service_latency_seconds:
                alerts.append(
                    _event(
                        f"service-latency:{service.component}:{service.operation}",
                        AlertSeverity.P2,
                        "服务延迟超过阈值",
                        "服务最大延迟超过预算; 需要检查任务积压与依赖。",
                        "service-runtime",
                        "docs/07-monitoring-alerting-runbook.md#p2-服务健康度",
                        {
                            **diagnostics,
                            "maximum_latency_seconds": (f"{service.maximum_latency_seconds:.4f}"),
                        },
                        snapshot.generated_at,
                    )
                )
        warning = snapshot.reconciliation_count("WARNING")
        if warning:
            alerts.append(
                _event(
                    "warning-reconciliation-difference",
                    AlertSeverity.P2,
                    "账实核对存在待处理差异",
                    "发现部分成交或其他非严重差异; 需要运营复核。",
                    "reconciliation",
                    "docs/07-monitoring-alerting-runbook.md#p2-一般账实差异",
                    {"warning_differences": str(warning)},
                    snapshot.generated_at,
                )
            )
        if snapshot.risk_rejections_total >= self._config.elevated_risk_rejections:
            alerts.append(
                _event(
                    "elevated-risk-rejections",
                    AlertSeverity.P3,
                    "风控拒绝数量升高",
                    "风控拒绝达到观察阈值; 请在日常运营中复核原因分布。",
                    "risk-service",
                    "docs/07-monitoring-alerting-runbook.md#p3-风控拒绝升高",
                    {"count": str(snapshot.risk_rejections_total)},
                    snapshot.generated_at,
                )
            )
        return tuple(alerts)


class AlertRouter:
    def __init__(
        self,
        routes: dict[AlertSeverity, tuple[str, ...]] | None = None,
    ) -> None:
        self._routes = routes or {
            AlertSeverity.P0: ("incident-primary", "risk-lead"),
            AlertSeverity.P1: ("risk-oncall", "platform-oncall"),
            AlertSeverity.P2: ("data-platform", "operations"),
            AlertSeverity.P3: ("operations-review",),
        }
        if set(self._routes) != set(AlertSeverity):
            raise ValueError("alert routes must explicitly cover P0 through P3")
        if any(not targets for targets in self._routes.values()):
            raise ValueError("every alert severity requires at least one target")

    def targets(self, severity: AlertSeverity) -> tuple[str, ...]:
        return self._routes[severity]


class AlertDispatcher:
    def __init__(
        self,
        sink: AlertSink,
        *,
        router: AlertRouter | None = None,
        cooldown: timedelta = timedelta(minutes=15),
    ) -> None:
        if cooldown < timedelta(0):
            raise ValueError("alert cooldown cannot be negative")
        self._sink = sink
        self._router = router or AlertRouter()
        self._cooldown = cooldown
        self._last_delivery: dict[str, datetime] = {}
        self._lock = RLock()

    def dispatch(
        self,
        events: tuple[AlertEvent, ...],
        *,
        delivered_at: datetime | None = None,
    ) -> tuple[AlertDelivery, ...]:
        moment = delivered_at or shanghai_now()
        deliveries: list[AlertDelivery] = []
        with self._lock:
            for event in events:
                previous = self._last_delivery.get(event.fingerprint)
                if previous is not None and moment - previous < self._cooldown:
                    continue
                delivery = AlertDelivery(
                    event=event,
                    targets=self._router.targets(event.severity),
                    delivered_at=moment,
                )
                self._sink.send(delivery)
                self._last_delivery[event.fingerprint] = moment
                deliveries.append(delivery)
        return tuple(deliveries)


def _event(
    policy_id: str,
    severity: AlertSeverity,
    title: str,
    summary: str,
    source: str,
    runbook: str,
    diagnostics: dict[str, str],
    occurred_at: datetime,
) -> AlertEvent:
    safe_diagnostics = tuple(
        sorted((key, _sanitize_text(value)) for key, value in diagnostics.items())
    )
    fingerprint_source = json.dumps(
        [policy_id, severity, safe_diagnostics],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return AlertEvent(
        fingerprint=hashlib.sha256(fingerprint_source.encode()).hexdigest(),
        policy_id=policy_id,
        severity=severity,
        title=_sanitize_text(title),
        summary=_sanitize_text(summary),
        source=source,
        runbook=runbook,
        diagnostics=safe_diagnostics,
        occurred_at=occurred_at,
    )


_SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*|password\s*[=:]\s*|"
    r"secret\s*[=:]\s*|token\s*[=:]\s*)\S+"
)


def _sanitize_text(value: str) -> str:
    return _SECRET_PATTERN.sub(r"\1[REDACTED]", value)
