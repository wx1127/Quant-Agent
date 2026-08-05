# P8 Daily Profit Report and Next-Day Drafts

## Purpose

Every successful P8-T08 PAPER day must produce a close-based profit report and the
next trading day's draft orders after the market close data is available.

This report is for PAPER validation only. It must not connect to a live broker, must
not submit live orders, and must not switch runtime mode away from `PAPER`.

## Daily Report Contract

The daily report is written to:

```text
data/paper/p8-t08/reports/<trading_date>.json
```

The report must include:

- `close_report`: close-marked equity, daily PnL, daily return, cash, holdings value,
  and per-position unrealized PnL.
- `filled_orders`: the simulated fills executed for the current trading day.
- `next_day_order_draft`: the draft batch eligible only for the next trading day.
- `reconciliation`: account reconciliation after close marking.
- `fees` and `slippage`: simulated trading costs.
- `alerts` and `manual_interventions`: daily operational evidence.
- `next_day_order_draft.excluded_buy_candidates`: candidates blocked by buy
  exclusion rules.

Stock-facing sections must include `name` and keep `instrument_id` as an audit key.
User-facing reports should display `name` first; `instrument_id` is retained for
traceability and debugging.

## Valuation Rule

If a previous close signal creates a next-day order, the simulated fill still occurs
after the next day's open using the configured PaperBroker slippage rule.

After fills are processed, the account is marked again using the current trading
day's close price. Therefore the daily report reflects close-based floating PnL,
not only commissions and slippage.

## Name Snapshot

The daily runner fetches or reuses the Tushare `stock_basic` instrument snapshot and
writes it to:

```text
data/paper/p8-t08/snapshots/instruments-<trading_date>.json
```

This snapshot supplies stock names for fills, holdings, candidates, and next-day
drafts. If a name is unavailable, the system falls back to `instrument_id` without
blocking the PAPER run.

## Buy Exclusion Rules

P8 PAPER buy drafts must exclude:

- Stocks whose daily price limit is above 10%:
  - ChiNext: `CN.SZ.300*` and `CN.SZ.301*`.
  - STAR Market: `CN.SH.688*` and `CN.SH.689*`.
  - Beijing Stock Exchange: `CN.BJ.*`.
- Hong Kong stocks: instruments whose exchange segment is `HK`.

These are hard buy filters. If the strongest candidates are all excluded, the
next-day buy draft may be empty. Existing PAPER holdings are not rewritten or
backfilled; the rule applies to newly generated buy drafts.

## Acceptance

- Daily report includes close-marked PnL and next-day draft orders.
- Fills, holdings, candidates, and drafts include stock names.
- Stocks with daily price limits above 10% and Hong Kong candidates are listed in
  `excluded_buy_candidates` and do not enter buy drafts.
- Same-day close data is not used to create same-day fills.
- Pending drafts with display-only `name` fields can still be read and executed.
- Tests, lint, and type checks pass.
