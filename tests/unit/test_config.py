"""Tests for fail-closed environment configuration."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

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


def test_empty_storage_value_is_rejected() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")
    payload = settings.model_dump()
    payload["storage"]["database_url"] = "   "

    with pytest.raises(ValidationError, match="cannot be empty"):
        QuantAgentSettings.model_validate(payload)


@pytest.mark.parametrize("environment", ["test", "replay", "paper", "production"])
def test_all_committed_environment_configs_are_valid(environment: str) -> None:
    settings = QuantAgentSettings.from_toml(ROOT / f"configs/environments/{environment}.toml")

    assert settings.runtime.allow_live_auto is False


def test_load_applies_only_supported_environment_overrides() -> None:
    settings = QuantAgentSettings.load(
        ROOT / "configs/environments/local.toml",
        environ={
            "QUANT_AGENT_LOG_LEVEL": "debug",
            "QUANT_AGENT_DATABASE_URL": "sqlite:///override.db",
            "UNRELATED_VALUE": "ignored",
        },
    )

    assert settings.logging.level == "DEBUG"
    assert settings.storage.database_url == "env://QUANT_AGENT_DATABASE_URL"
    assert (
        settings.resolve_database_url(
            environ={"QUANT_AGENT_DATABASE_URL": "sqlite:///override.db"}
        ).get_secret_value()
        == "sqlite:///override.db"
    )
    assert "UNRELATED_VALUE" not in settings.model_dump_json()
    assert "sqlite:///override.db" not in settings.model_dump_json()


def test_load_parses_boolean_overrides_and_still_rejects_live_auto() -> None:
    with pytest.raises(ValidationError, match="LIVE_AUTO"):
        QuantAgentSettings.load(
            ROOT / "configs/environments/local.toml",
            environ={"QUANT_AGENT_ALLOW_LIVE_AUTO": "true"},
        )

    with pytest.raises(ValueError, match="must be a boolean"):
        QuantAgentSettings.load(
            ROOT / "configs/environments/local.toml",
            environ={"QUANT_AGENT_LOG_JSON": "sometimes"},
        )


def test_resolve_masks_environment_secrets_and_database_url() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/production.toml")
    resolved = settings.resolve(
        environ={"QUANT_AGENT_DATABASE_URL": "postgresql+psycopg://user:secret@db/quant"},
        vault_resolver=lambda path: f"resolved-{path}",
    )

    assert isinstance(resolved.storage.database_url, SecretStr)
    assert resolved.storage.database_url.get_secret_value().endswith("@db/quant")
    assert resolved.secrets.market_data_token.get_secret_value().startswith("resolved-")
    serialized = resolved.model_dump_json()
    assert "user:secret" not in serialized
    assert "resolved-" not in serialized


def test_resolve_fails_closed_for_missing_environment_value() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")

    with pytest.raises(ValueError, match="MARKET_DATA_TOKEN"):
        settings.resolve(environ={})


def test_resolve_fails_closed_without_vault_resolver() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/production.toml")

    with pytest.raises(ValueError, match="explicit resolver"):
        settings.resolve(
            environ={"QUANT_AGENT_DATABASE_URL": "sqlite:///:memory:"},
        )


def test_database_url_can_be_resolved_without_provider_credentials() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/production.toml")

    database_url = settings.resolve_database_url(
        environ={"QUANT_AGENT_DATABASE_URL": "sqlite:///runtime.db"},
    )

    assert database_url.get_secret_value() == "sqlite:///runtime.db"


def test_market_token_can_be_resolved_without_llm_key() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")

    token = settings.resolve_market_data_token(environ={"MARKET_DATA_TOKEN": "market-secret"})

    assert token.get_secret_value() == "market-secret"
    assert "market-secret" not in repr(token)
