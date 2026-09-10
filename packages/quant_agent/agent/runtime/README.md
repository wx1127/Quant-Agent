# Explicit Agent Runtime

This package is the model-facing workflow boundary for P6-T05. It sits above
`AgentToolRegistry`: the registry still owns mode, capability, account, schema,
audit, idempotency, and live-authorization checks, while `AgentRuntime` adds
run state, prerequisite artifacts, deadlines, cancellation, and deterministic
trace replay.

The intended call path is:

```text
model adapter
  -> AgentRuntime.catalog / AgentRuntime.invoke_tool
  -> AgentToolRegistry.catalog / AgentToolRegistry.invoke
  -> deterministic domain handler
```

Do not give a model the underlying registry. Hiding a tool from a prompt is not
an authorization boundary; `invoke_tool` repeats the stage and artifact checks
before dispatch.

## Workflow profiles

`AgentRunGoal` is selected by trusted application code and cannot be changed by
tool arguments:

- `MARKET_RESEARCH`
- `BACKTEST_REPORT`
- `PORTFOLIO_REPORT`
- `ORDER_DRAFT`
- `PAPER_EXECUTION`
- `LIVE_ASSISTED_EXECUTION`

Research and backtest runs may finish as `REPORTED` without portfolio risk.
Trading runs must pass the following gates:

```text
SNAPSHOT_READY
  -> DATA_VALIDATED
  -> ANALYZED
  -> PORTFOLIO_READY
  -> RISK_CHECKED
  -> DRAFT_READY
  -> EXECUTING                  # PAPER
  -> RECONCILING
  -> COMPLETED | INCIDENT

DRAFT_READY
  -> PENDING_APPROVAL
  -> APPROVED
  -> EXECUTING                  # LIVE_ASSISTED only
```

Risk `REJECT` ends the run as `REJECTED`; risk `ERROR` or broken data bindings
end it as `DATA_INVALID`. The model cannot call an approval tool. A live run is
unlocked only by `record_external_approval()` with evidence bound to the exact
decision and draft batch, and the lower registry still requires its independent
one-use `LiveExecutionAuthorizer` lease.

`PAPER` does not require human approval. The domain draft still says
`requires_human_approval=True` because it is never itself executable; the
runtime's mode-specific transition controls whether the next submission is a
paper write or a separately approved live write.

## Tool and artifact gates

The visible catalog is the intersection of:

1. the registry's fixed authorization result;
2. the current goal/state allow-list;
3. verified prerequisite artifacts already pinned by the run.

Successful responses pin both a domain content hash and the exact model-visible
response hash. Important links include data content, account snapshot, target
portfolio, risk result, draft batch, execution receipt, and reconciliation
result. Paper before/after account projections also retain their frozen source
snapshot identity. Later calls must use the run's locked `batch_hash`; a model
cannot switch to another draft from the same account and decision. The first
dispatched write call also locks its hashed idempotency identity; a rejected
pre-dispatch attempt cannot poison a later valid retry.

Only hashes and bounded identifiers enter runtime events. Raw arguments,
approval secrets, credentials, exception text, and full tool payloads are not
stored in this trace.

## Limits, timeout, and cancellation

`AgentRunLimits` freezes the total model tool budget, invalid-call budget,
independent reconciliation budget, total wall-clock timeout, and approval
timeout. Invalid stage calls and registry argument/permission failures consume
budget so a model cannot loop indefinitely on rejected requests.

Tool handlers are synchronous, so timeout and cancellation are cooperative at
dispatch boundaries; Python cannot safely kill a write handler halfway through
a broker or paper transaction. Before an execution-write reservation, cancellation
enters `CANCELLED` and the inclusive deadline (`now >= deadline`) enters
`TIMED_OUT`. Once submission may have happened, neither signal may claim the
order was cancelled. A receipt must enter `RECONCILING`; the independent
reconciliation budget remains usable even after the model deadline. Unknown
execution state, unrecoverable reconciliation, a required stop signal, or
budget exhaustion after execution enters `INCIDENT`.

## Replay and persistence boundary

Every event is immutable, strict, content-addressed, and chained through
`previous_event_hash`. `replay_agent_run(events)` validates sequence, identity,
state edges, tool reservations/outcomes, service-equivalent error projections,
limits, deadlines, approval freshness and binding, prerequisite artifacts, and
terminal absorption before reconstructing the exact `AgentRunSnapshot`. Replay
never invokes a tool and therefore never repeats a side effect. The repository
accepts a candidate only when this independently derived snapshot exactly equals
the supplied one.

`InMemoryAgentRunRepository` provides defensive JSON copies and atomic CAS for
tests and single-process composition. It is intentionally not durable across a
restart, and process-local CAS cannot close the crash window between an external
write and runtime-event persistence. P6-T09 remains responsible for a durable
outbox/exactly-once boundary, persistent cross-process traces, raw payload
retention/redaction, multi-store aggregation, and full artifact difference
replay.

When reconciliation returns `stop_signal.required=True`, the run always enters
`INCIDENT`. An optional `ReconciliationIncidentHandler` may resolve the full
authoritative `ReconciliationResult`, verify its result hash, and call the
existing `KillSwitchService` idempotently. The runtime never fabricates an
activation request from a possibly truncated model projection, and it exposes
no Agent API for deactivating or recovering a kill switch.
