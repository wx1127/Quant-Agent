from pathlib import Path

from quant_agent.validation.environment import E2EEnvironment

FIXTURE = Path(__file__).parent / "fixtures" / "market_day.json"


def test_fixed_full_flow_is_repeatable_and_reconciled() -> None:
    first = E2EEnvironment(FIXTURE).run()
    second = E2EEnvironment(FIXTURE).run()
    assert first == second
    assert first.content_hash == second.content_hash
    assert first.reconciled
    assert len(first.order_ids) == len(first.fill_ids) == 1
