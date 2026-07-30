"""Container health check with production Kill Switch verification."""

import json
import os
import urllib.request


def main() -> int:
    url = os.getenv("QUANT_AGENT_HEALTH_URL", "http://127.0.0.1:8000/health/ready")
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.load(response)
    except Exception:
        return 1
    if response.status != 200 or payload.get("status") != "ready":
        return 1
    if (
        os.getenv("QUANT_AGENT_APP_ENV") == "production"
        and payload.get("kill_switch_active") is not True
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
