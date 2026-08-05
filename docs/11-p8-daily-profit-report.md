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
- Stocks whose actual close-to-close daily return has an absolute value above 10%.

These filters run before candidate scoring, sorting, and Top-10 truncation. An
excluded stock therefore has no candidate score or rank. Exclusions are retained
separately in `market.candidate_exclusions` and the next-day draft audit section.
If the strongest eligible set is empty, the next-day buy draft may be empty.
Existing PAPER evidence is not rewritten or backfilled.

P8 mainlines use the industry field frozen with the current Tushare instrument
snapshot. Exchange boards are not valid mainlines. A missing industry snapshot is
a failed PAPER day and must not silently fall back to exchange-board segments.

## Exit Rule Layer

Exit decisions use only data available at the current close and create sell drafts
for the next trading day. The rule version is `p8-paper-exit-rules-v1`.

The first version uses these deterministic thresholds:

- Single-stock stop loss: close return from average cost is at or below -8%.
- Fixed take profit: close return from average cost is at or above +20%.
- Trailing stop: peak return reached +10% and close has fallen at least 8% from the peak.
- Moving-average exit: close crosses from above to below MA5 or MA10.
- Mainline exit: the holding's industry is no longer in the confirmed mainline set.
- Large gap-down control: current open is at least 5% below the previous close.
- High-volume sell-off: daily return is at or below -7% while volume is at least
  1.5 times the preceding five-session average.
- Maximum holding period: 20 evaluated trading days.
- Candidate rotation: the holding is no longer in the selected eligible candidate set.

Risk exits have priority over candidate rotation. Every signal records all matched
reasons and the highest-priority primary rule. A stock bought that morning remains
frozen for same-day selling under T+1, but a next-trading-day sell draft is valid.
Such a signal is marked `READY_AFTER_T_PLUS_ONE_RELEASE`; the next-day executor must
release the frozen quantity before simulation and must never create a same-day fill.

Position state stores entry date, evaluated trading-day count, peak close, and last
evaluation date in the pending draft. State inferred for a legacy holding is marked
explicitly and is never presented as an observed historical entry date.

## Acceptance

- Daily report includes close-marked PnL and next-day draft orders.
- Fills, holdings, candidates, and drafts include stock names.
- Stocks with daily price limits above 10% and Hong Kong candidates are listed in
  `excluded_buy_candidates` and do not enter buy drafts.
- Stocks with an absolute daily return above 10% are removed before scoring and ranking.
- Exit evidence covers all configured rules, priorities, and T+1 next-day release.
- Same-day close data is not used to create same-day fills.
- Pending drafts with display-only `name` fields can still be read and executed.
- Tests, lint, and type checks pass.
