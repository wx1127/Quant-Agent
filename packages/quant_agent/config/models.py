"""Versionable settings models with secure defaults."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class SecretSettings(BaseModel):
    """References to provider credentials."""

    model_config = ConfigDict(frozen=True)

    market_data_token_ref: SecretReference
    llm_api_key_ref: SecretReference


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
