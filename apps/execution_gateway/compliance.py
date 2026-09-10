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


@dataclass(frozen=True, slots=True)
class SmallCapitalAcceptance:
    capital: float
    minimum_capital: float = 1000.0
    max_position_fraction: float = 0.1
    daily_loss_fraction: float = 0.02
    kill_switch_ready: bool = False
    rollback_ready: bool = False

    def validate(self, proposed_position: float, daily_loss: float) -> None:
        if self.capital < self.minimum_capital:
            raise PermissionError("capital is below small-capital acceptance minimum")
        if proposed_position < 0 or proposed_position > self.capital * self.max_position_fraction:
            raise PermissionError("proposed position exceeds small-capital position limit")
        if daily_loss > self.capital * self.daily_loss_fraction:
            raise PermissionError("daily loss exceeds small-capital loss limit")
        if not self.kill_switch_ready:
            raise PermissionError("kill switch is not ready")
        if not self.rollback_ready:
            raise PermissionError("rollback is not ready")
