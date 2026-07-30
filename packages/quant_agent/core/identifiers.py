"""Opaque identifiers used across module boundaries."""

from datetime import datetime
from typing import NewType
from uuid import uuid4

from quant_agent.core.time import ensure_aware, shanghai_now

RequestId = NewType("RequestId", str)
DecisionId = NewType("DecisionId", str)


def new_request_id() -> RequestId:
    """Return a caller-safe opaque request identifier."""

    return RequestId(f"req_{uuid4().hex}")


def new_decision_id(at: datetime | None = None) -> DecisionId:
    """Return a decision identifier with a readable local date prefix."""

    decision_time = ensure_aware(at or shanghai_now())
    return DecisionId(f"dec_{decision_time:%Y%m%d}_{uuid4().hex}")
