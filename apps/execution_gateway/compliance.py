"""Broker and compliance boundary; real trading is opt-in and fail-closed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionEnvironment(StrEnum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class ComplianceChecklist:
    account_verified: bool = False
    broker_agreement_verified: bool = False
    risk_disclosure_accepted: bool = False
    kill_switch_tested: bool = False
    approval_policy_version: str = "unbound"

    def live_ready(self) -> bool:
        return all(
            (
                self.account_verified,
                self.broker_agreement_verified,
                self.risk_disclosure_accepted,
                self.kill_switch_tested,
                self.approval_policy_version != "unbound",
            )
        )


class BrokerExecutionGate:
    def __init__(self, environment: ExecutionEnvironment, checklist: ComplianceChecklist) -> None:
        self.environment = environment
        self.checklist = checklist

    def authorize(self) -> None:
        if self.environment is not ExecutionEnvironment.LIVE:
            return
        if not self.checklist.live_ready():
            raise PermissionError("live execution compliance checklist is incomplete")

