"""Fail-closed shadow-running evidence and acceptance controls."""

from quant_agent.shadow.io import load_shadow_config, load_shadow_day, load_trading_calendar
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.models import (
    IncidentSeverity,
    ShadowAcceptance,
    ShadowAcceptanceStatus,
    ShadowDayEvidence,
    ShadowIncident,
    ShadowSessionConfig,
)
from quant_agent.shadow.session import ShadowRunEvaluator

__all__ = [
    "IncidentSeverity",
    "ShadowAcceptance",
    "ShadowAcceptanceStatus",
    "ShadowDayEvidence",
    "ShadowEvidenceLedger",
    "ShadowIncident",
    "ShadowRunEvaluator",
    "ShadowSessionConfig",
    "load_shadow_config",
    "load_shadow_day",
    "load_trading_calendar",
]
