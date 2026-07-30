"""Validated request contracts for API mutations."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_agent.core.time import ensure_aware


class BacktestCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_id: str = Field(min_length=1, max_length=80)
    parameter_version: str = Field(min_length=1, max_length=80)
    data_version: str = Field(min_length=1, max_length=120)


class PortfolioProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str = Field(min_length=1, max_length=80)
    decision_id: str = Field(pattern=r"^dec_[A-Za-z0-9_]+$")
    as_of: datetime
    data_version: str = Field(min_length=1)
    target: dict[str, Any]

    @field_validator("as_of")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)


class RiskCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str = Field(min_length=1, max_length=80)
    decision_id: str = Field(pattern=r"^dec_[A-Za-z0-9_]+$")
    target: dict[str, Any]


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_token: str = Field(min_length=20)


class AgentMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=8_000)
    conversation_id: str = Field(min_length=1, max_length=96)
