"""Tests for fail-closed environment configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_agent.config import QuantAgentSettings, RuntimeMode

ROOT = Path(__file__).resolve().parents[2]


def test_local_config_is_research_and_live_auto_disabled() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")

    assert settings.runtime.mode is RuntimeMode.RESEARCH
    assert settings.runtime.allow_live_auto is False


def test_live_auto_cannot_be_enabled() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["runtime"]["mode"] = "LIVE_AUTO"
    payload["runtime"]["allow_live_auto"] = True

    with pytest.raises(ValidationError, match="LIVE_AUTO"):
        QuantAgentSettings.model_validate(payload)


def test_plaintext_secret_is_rejected() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["secrets"]["llm_api_key_ref"] = "plaintext-key"

    with pytest.raises(ValidationError, match="env:// or vault://"):
        QuantAgentSettings.model_validate(payload)


def test_empty_secret_reference_is_rejected() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["secrets"]["llm_api_key_ref"] = "env://"

    with pytest.raises(ValidationError, match="cannot be empty"):
        QuantAgentSettings.model_validate(payload)


def test_local_environment_cannot_use_non_research_mode() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["runtime"]["mode"] = "PAPER"

    with pytest.raises(ValidationError, match="must use RESEARCH"):
        QuantAgentSettings.model_validate(payload)


def test_unknown_log_level_is_rejected() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["logging"]["level"] = "VERBOSE"

    with pytest.raises(ValidationError, match="unsupported log level"):
        QuantAgentSettings.model_validate(payload)


@pytest.mark.parametrize("environment", ["test", "replay", "paper", "production"])
def test_all_committed_environment_configs_are_valid(environment: str) -> None:
    settings = QuantAgentSettings.from_toml(ROOT / f"configs/environments/{environment}.toml")

    assert settings.runtime.allow_live_auto is False
