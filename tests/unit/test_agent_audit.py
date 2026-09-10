from quant_agent.agent.audit import AuditChain, compare_replay


def test_audit_chain_verifies_and_replay_is_deterministic() -> None:
    first = AuditChain()
    first.append("plan", {"tool": "research"})
    first.append("answer", {"evidence": ["e1"]})
    second = AuditChain()
    second.append("plan", {"tool": "research"})
    second.append("answer", {"evidence": ["e1"]})
    assert first.verify() is True
    assert compare_replay(first.events, second.events) == ()


def test_audit_chain_detects_tampering_and_replay_difference() -> None:
    chain = AuditChain()
    chain.append("plan", {"tool": "research"})
    chain._events[0] = chain.events[0].__class__(
        0, "plan", {"tool": "other"}, "GENESIS", chain.events[0].event_hash
    )
    assert chain.verify() is False
    other = AuditChain()
    other.append("answer", {"evidence": []})
    assert compare_replay(chain.events, other.events) == ("event_type:0", "payload:0")
