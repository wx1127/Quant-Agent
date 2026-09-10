# Agent answer boundary

This package turns model-authored answer drafts into immutable, evidence-bound
`AgentAnswer` records. It is a publication boundary, not an order-execution
interface.

## Trust split

- `AgentAnswerDraft` is model-controlled. It may propose facts, inferences,
  counter-evidence, risks, invalidation conditions, and non-executable reader
  actions.
- `AgentAnswerPublisher` receives the authoritative `DecisionSnapshot`, the
  replay-validated `AgentRunSnapshot`, and the exact tool responses retained by
  the application layer.
- The publisher injects `run_id`, `decision_id`, decision hash, runtime mode,
  goal, run state, data cutoff, and generation time. A draft cannot supply or
  override these fields.
- `AgentAnswer` includes a content hash and supports strict deterministic JSON
  round trips.

## Evidence references

An `EvidenceReference` binds all of the following:

1. runtime request and tool name;
2. complete tool-response hash;
3. RFC 6901 path below `/data`;
4. hash of the value at that path;
5. immutable data version.

Publication replays these checks against a successful or replayed runtime event
and the exact response body. A reference to a failed call, another decision,
another cutoff, another data version, a changed response, or a missing JSON path
is not publishable.

Use `build_evidence_reference(...)` only with the actual response returned by
the runtime. Application code must retain those response bodies until P6-T09
adds durable trace storage; the runtime event chain intentionally retains hashes
and artifacts rather than arbitrary complete response payloads.

## Numeric claims

Numbers are carried by `AgentMetric`, never embedded in narrative prose. The
metric's display value must exactly equal the referenced numeric JSON scalar.
This prevents a model from citing a real response while changing the number in
its answer. Instrument identifiers, dates, thresholds, percentages, and scores
should likewise be represented as structured fields or metrics instead of being
typed into statements.

## Failure and language policy

- Missing, stale, failed, mismatched, or unverifiable evidence produces a fixed
  `INSUFFICIENT_EVIDENCE` answer with `NO_ACTION`; unverified claims are dropped.
- Tool warning and error codes are copied from runtime events into the answer.
  Tool failure is therefore not interpreted as absence of risk or signal.
- Guarantee language such as `稳赚`, `确定上涨`, `必买`, `guaranteed profit`,
  and obfuscated spacing or Unicode variants is rejected.
- Every inference must name its source facts and have complete counter-evidence,
  a risk section, and an invalidation condition.
- Allowed actions are reader/control-plane steps only. No submit, buy, sell, or
  mode-switch action exists in the answer schema. Draft review and approval
  requests remain gated by verified runtime state.
