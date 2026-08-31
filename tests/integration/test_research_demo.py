"""Integration tests for the deterministic quantitative research demonstration."""

import json

import pytest

from quant_agent.cli import main
from quant_agent.pipelines import run_research_demo


def test_research_demo_is_reproducible_and_quantitative() -> None:
    first = run_research_demo()
    repeated = run_research_demo()

    assert repeated == first
    assert len(first.result_hash) == 64
    assert first.industry_ranking[0][0] == 1
    assert first.industry_ranking[0][1] == "DEMO:L3:GROWTH"
    assert first.industry_ranking[0][2] > first.industry_ranking[1][2]
    assert first.confidence_meaning == "RULE_EVIDENCE_CONSISTENCY"


def test_research_demo_cli_outputs_stable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["research", "demo", "--data-version", "acceptance-v1"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["ok"] is True
    assert payload["data_version"] == "acceptance-v1"
    assert payload["industry_ranking"][0]["rank"] == 1
    assert len(payload["result_hash"]) == 64
