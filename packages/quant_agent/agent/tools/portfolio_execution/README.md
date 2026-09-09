# Portfolio execution Agent tools

This package exposes the P5 portfolio, portfolio-risk, order-draft, paper-execution,
and reconciliation engines through the P6 Agent registry. It deliberately adds
orchestration and trust boundaries; it does not reimplement those domain engines.

The tool chain is:

```text
DecisionSnapshot
  -> get_portfolio_snapshot
  -> build_target_portfolio
  -> check_portfolio_risk
  -> create_order_draft / get_order_draft
  -> submit_paper_orders (PAPER only)
  -> reconcile_account
```

Every handler receives account, mode, decision, time, and data-version context from
the verified registry rather than model arguments. The pipeline revalidates all
trusted source objects, checks content-hash links at every transition, and permits
draft creation only after a `PASS` or `WARN` risk result. A `REJECT` or `ERROR`
result is returned intact and cannot create an artifact.

`create_order_draft` writes through an injected `DraftArtifactStore` and requires
an idempotency key. `InMemoryDraftArtifactStore` is thread-safe and useful for
tests or a single-process local run; production composition must inject a durable,
transactional implementation. The toolset never silently constructs a store.

`submit_paper_orders` is built only when a `PaperOrderGateway` is injected. The
provided `ServiceBackedPaperOrderGateway` always delegates to
`PaperExecutionService`, preserving Kill Switch locking, repository CAS, batch
consumption, and receipt idempotency. It never invokes `PaperExecutionEngine`
directly. The tool is centrally authorized only in `PAPER`; no live broker path is
present in this package.

`reconcile_account` requires independently observed execution evidence from the
input source. It returns the full reconciliation result and stop signal but remains
read-only: activating a Kill Switch is a later trusted-harness responsibility.

Composition roots must inject:

- a `PortfolioExecutionInputSource` backed by authoritative, point-in-time data;
- a durable `DraftArtifactStore` outside local/test use;
- immutable portfolio, risk, market-rule, fee, and slippage configurations;
- for paper submission, a repository-backed gateway with a real new-order gate;
- independent observed evidence for reconciliation.

No tool accepts account IDs, runtime mode, decision IDs, approval state, policy
selection, market rules, or execution configuration from the model.
