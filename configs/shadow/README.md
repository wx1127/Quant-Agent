# Shadow run configuration

`shadow_20260803_v1.toml` defines the planned P8-T07 session. The checked-in
calendar is planning evidence derived from the official SSE closure schedule,
not sufficient evidence for a counted run. Before day one, use
`scripts/shadow/bootstrap_calendar.py` with `MARKET_DATA_TOKEN` to freeze the
provider calendar into runtime storage and retain its hash.

Secrets must only enter through environment variables. Runtime ledgers,
snapshots, reports and alert JSONL files belong under ignored `data/`,
`artifacts/` or `logs/` directories.
