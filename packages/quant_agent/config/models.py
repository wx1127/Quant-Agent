"""Versionable settings models with secure defaults."""

import os
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

VaultResolver = Callable[[str], str | None]

_ENVIRONMENT_OVERRIDES: dict[str, tuple[str, str]] = {
    "QUANT_AGENT_APP_ENV": ("runtime", "app_env"),
    "QUANT_AGENT_RUNTIME_MODE": ("runtime", "mode"),
    "QUANT_AGENT_TIMEZONE": ("runtime", "timezone"),
    "QUANT_AGENT_ALLOW_LIVE_AUTO": ("runtime", "allow_live_auto"),
    "QUANT_AGENT_LOG_LEVEL": ("logging", "level"),
    "QUANT_AGENT_LOG_JSON": ("logging", "json"),
    "QUANT_AGENT_DATABASE_URL": ("storage", "database_url"),
    "QUANT_AGENT_RESEARCH_STORAGE_PATH": ("storage", "research_storage_path"),
    "QUANT_AGENT_MARKET_DATA_TOKEN_REF": ("secrets", "market_data_token_ref"),
    "QUANT_AGENT_LLM_API_KEY_REF": ("secrets", "llm_api_key_ref"),
}
_BOOLEAN_OVERRIDES = {
    "QUANT_AGENT_ALLOW_LIVE_AUTO",
    "QUANT_AGENT_LOG_JSON",
}


class RuntimeMode(StrEnum):
    """Supported Harness modes."""

    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE_ASSISTED = "LIVE_ASSISTED"
    LIVE_AUTO = "LIVE_AUTO"


class AppEnvironment(StrEnum):
    """Deployment environments with distinct credentials and data."""

    LOCAL = "local"
    TEST = "test"
    REPLAY = "replay"
    PAPER = "paper"
    PRODUCTION = "production"


class SecretReference(str):
    """Reference to a secret without embedding its value in configuration."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[Any],
        handler: Any,
    ) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )

    @classmethod
    def validate(cls, value: str) -> "SecretReference":
        """Only permit supported secret reference schemes."""

        if not value.startswith(("env://", "vault://")):
            raise ValueError("secret must be an env:// or vault:// reference")
        if value in {"env://", "vault://"}:
            raise ValueError("secret reference target cannot be empty")
        return cls(value)


def _parse_boolean(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _resolve_reference(
    value: str,
    *,
    environ: Mapping[str, str],
    vault_resolver: VaultResolver | None,
) -> str:
    scheme, separator, target = value.partition("://")
    if not separator:
        return value
    if scheme not in {"env", "vault"}:
        return value
    if not target:
        raise ValueError("configuration reference target cannot be empty")
    if scheme == "env":
        resolved = environ.get(target)
        if not resolved:
            raise ValueError(f"required environment variable is not set: {target}")
        return resolved
    if scheme == "vault":
        if vault_resolver is None:
            raise ValueError("vault reference requires an explicit resolver")
        resolved = vault_resolver(target)
        if not resolved:
            raise ValueError(f"vault resolver returned no value for: {target}")
        return resolved
    raise AssertionError("unreachable reference scheme")


class RuntimeSettings(BaseModel):
    """Runtime safety boundary."""

    model_config = ConfigDict(frozen=True)

    app_env: AppEnvironment
    mode: RuntimeMode
    timezone: str = "Asia/Shanghai"
    allow_live_auto: bool = False

    @model_validator(mode="after")
    def reject_live_auto(self) -> "RuntimeSettings":
        """LIVE_AUTO is deliberately unavailable in the first release."""

        if self.mode is RuntimeMode.LIVE_AUTO or self.allow_live_auto:
            raise ValueError("LIVE_AUTO is disabled in the first release")
        if self.app_env is AppEnvironment.LOCAL and self.mode is not RuntimeMode.RESEARCH:
            raise ValueError("local environment must use RESEARCH mode")
        return self


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        """Normalize common logging level names."""

        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return normalized


class StorageSettings(BaseModel):
    """Storage connection references."""

    model_config = ConfigDict(frozen=True)

    database_url: str
    research_storage_path: str

    @field_validator("database_url", "research_storage_path")
    @classmethod
    def reject_empty_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("storage value cannot be empty")
        return value


class SecretSettings(BaseModel):
    """References to provider credentials."""

    model_config = ConfigDict(frozen=True)

    market_data_token_ref: SecretReference
    llm_api_key_ref: SecretReference


class ResolvedStorageSettings(BaseModel):
    """Runtime storage values with credentials hidden from representations."""

    model_config = ConfigDict(frozen=True)

    database_url: SecretStr
    research_storage_path: str


class ResolvedSecretSettings(BaseModel):
    """Resolved provider credentials hidden by Pydantic's secret type."""

    model_config = ConfigDict(frozen=True)

    market_data_token: SecretStr
    llm_api_key: SecretStr


class ResolvedQuantAgentSettings(BaseModel):
    """Values safe to pass to runtime components without leaking secrets in logs."""

    model_config = ConfigDict(frozen=True)

    runtime: RuntimeSettings
    logging: LoggingSettings
    storage: ResolvedStorageSettings
    secrets: ResolvedSecretSettings


class QuantAgentSettings(BaseModel):
    """Top-level validated settings."""

    model_config = ConfigDict(frozen=True)

    runtime: RuntimeSettings
    logging: LoggingSettings
    storage: StorageSettings
    secrets: SecretSettings

    @classmethod
    def from_toml(cls, path: str | Path) -> "QuantAgentSettings":
        """Load a TOML file through the validated settings model."""

        import tomllib

        config_path = Path(path)
        with config_path.open("rb") as config_file:
            raw = tomllib.load(config_file)
        return cls.model_validate(raw)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "QuantAgentSettings":
        """Load TOML and apply only the documented environment overrides."""

        source = os.environ if environ is None else environ
        payload = cls.from_toml(path).model_dump(by_alias=True)
        for name, (section, key) in _ENVIRONMENT_OVERRIDES.items():
            if name not in source:
                continue
            value: str | bool = source[name]
            if name == "QUANT_AGENT_DATABASE_URL":
                value = f"env://{name}"
            elif name in _BOOLEAN_OVERRIDES:
                value = _parse_boolean(name, source[name])
            section_payload = payload[section]
            if not isinstance(section_payload, dict):
                raise TypeError(f"configuration section is not an object: {section}")
            section_payload[key] = value
        return cls.model_validate(payload)

    def resolve_database_url(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        vault_resolver: VaultResolver | None = None,
    ) -> SecretStr:
        """Resolve and mask the database URL independently of provider secrets."""

        source = os.environ if environ is None else environ
        return SecretStr(
            _resolve_reference(
                self.storage.database_url,
                environ=source,
                vault_resolver=vault_resolver,
            )
        )

    def resolve_market_data_token(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        vault_resolver: VaultResolver | None = None,
    ) -> SecretStr:
        """Resolve only the market-data credential needed by ingestion workers."""

        source = os.environ if environ is None else environ
        return SecretStr(
            _resolve_reference(
                self.secrets.market_data_token_ref,
                environ=source,
                vault_resolver=vault_resolver,
            )
        )

    def resolve_llm_api_key(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        vault_resolver: VaultResolver | None = None,
    ) -> SecretStr:
        """Resolve only the LLM credential needed by a future Agent process."""

        source = os.environ if environ is None else environ
        return SecretStr(
            _resolve_reference(
                self.secrets.llm_api_key_ref,
                environ=source,
                vault_resolver=vault_resolver,
            )
        )

    def resolve(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        vault_resolver: VaultResolver | None = None,
    ) -> ResolvedQuantAgentSettings:
        """Resolve every external reference, failing closed when one is unavailable."""

        source = os.environ if environ is None else environ
        return ResolvedQuantAgentSettings(
            runtime=self.runtime,
            logging=self.logging,
            storage=ResolvedStorageSettings(
                database_url=self.resolve_database_url(
                    environ=source,
                    vault_resolver=vault_resolver,
                ),
                research_storage_path=self.storage.research_storage_path,
            ),
            secrets=ResolvedSecretSettings(
                market_data_token=self.resolve_market_data_token(
                    environ=source,
                    vault_resolver=vault_resolver,
                ),
                llm_api_key=self.resolve_llm_api_key(
                    environ=source,
                    vault_resolver=vault_resolver,
                ),
            ),
        )
