# Shadow run configuration

`shadow_20260701_historical_v1.toml` and `calendar_20260701_20d.json` define the
completed point-in-time historical P8-T07 session. The July calendar is checked
against non-empty real daily-bar responses during replay. `shadow_20260803_v1.toml`
and its calendar remain as the superseded real-time plan and must not be mixed
with the historical evidence ledger.

Secrets must only enter through environment variables or a read-only secret file.
Runtime ledgers, snapshots, reports and alert JSONL files belong under ignored
`data/`, `artifacts/` or `logs/` directories.
