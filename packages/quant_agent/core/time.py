"""Timezone-safe datetime helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def ensure_aware(value: datetime) -> datetime:
    """Reject timestamps without timezone information."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include timezone information")
    return value


def shanghai_now() -> datetime:
    """Return the current time in the market timezone."""

    return datetime.now(tz=SHANGHAI_TZ)
