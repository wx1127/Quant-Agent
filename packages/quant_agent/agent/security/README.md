# Agent security boundary

`AgentSecurityBoundary` builds a server-owned `AgentInputEnvelope` for one
validated decision and runtime snapshot. The user request is treated as intent,
not as authority: mode changes, approval/control-plane operations, rule bypasses,
and secret-extraction requests are rejected with text-free findings.

External news, announcements, and web text are passed through the existing
`ContentSanitizer` and wrapped as `UntrustedContentBlock`. The block explicitly
has `instruction_authority=False`; quarantined content never carries its raw
text. When any external block is present, artifact writes, paper submission,
and live submission tools are removed from the model-visible set. The runtime
registry remains the final authorization boundary.

`SensitiveTextGuard` redacts inbound credentials and rejects outbound strings
containing API keys, bearer/JWT/provider tokens, secret references, long account
numbers, or the current decision account identifiers. Findings carry only stable
codes, never the sensitive fragment. `AgentAnswerPublisher` invokes this guard
before creating a published answer.

The package deliberately does not store raw prompt content or secret values in
security findings. Durable cross-process prompt/response retention and external
audit anchoring remain later P6-T09 work.
