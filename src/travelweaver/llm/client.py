"""OpenAI-compatible chat-completion clients shared across components."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ApiRolloutError, ConfigurationError
from ..paths import project_root


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Provider-neutral configuration for an OpenAI-compatible endpoint."""

    api_key: str = field(repr=False)
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "model"
    timeout_seconds: float = 120.0
    max_tokens: int = 4096
    tool_choice: str = "auto"
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ConfigurationError("API key is empty.")
        if not self.base_url.strip():
            raise ConfigurationError("API base URL is empty.")
        if not self.model.strip():
            raise ConfigurationError("API model is empty.")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ConfigurationError("API timeout and max_tokens must be positive.")
        if self.tool_choice not in {"auto", "required", "none"}:
            raise ConfigurationError("tool_choice must be auto, required, or none.")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ConfigurationError("temperature must be between 0 and 2.")

    def completion_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "tool_choice": self.tool_choice,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            options["temperature"] = self.temperature
        return options

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        prefix: str = "OPENAI",
    ) -> OpenAICompatibleConfig:
        _load_dotenv(env_file)
        normalized_prefix = prefix.strip().upper()
        if not normalized_prefix:
            raise ConfigurationError("Environment variable prefix cannot be empty.")
        try:
            timeout_seconds = float(
                os.getenv(f"{normalized_prefix}_TIMEOUT_SECONDS", "120")
            )
            max_tokens = int(os.getenv(f"{normalized_prefix}_MAX_TOKENS", "4096"))
            temperature = _optional_float(
                os.getenv(f"{normalized_prefix}_TEMPERATURE"),
                f"{normalized_prefix}_TEMPERATURE",
            )
        except ValueError as error:
            raise ConfigurationError(
                f"{normalized_prefix}_TIMEOUT_SECONDS and "
                f"{normalized_prefix}_MAX_TOKENS must be numeric."
            ) from error
        return cls(
            api_key=os.getenv(f"{normalized_prefix}_API_KEY", ""),
            base_url=os.getenv(
                f"{normalized_prefix}_BASE_URL", "http://127.0.0.1:8000/v1"
            ),
            model=os.getenv(f"{normalized_prefix}_MODEL", "model"),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            tool_choice=os.getenv(f"{normalized_prefix}_TOOL_CHOICE", "auto").lower(),
            temperature=temperature,
        )


@dataclass(frozen=True)
class DeepSeekConfig(OpenAICompatibleConfig):
    """DeepSeek defaults and its optional provider-specific thinking switch."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    max_tokens: int = 16384
    thinking: str = "disabled"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.thinking not in {"enabled", "disabled"}:
            raise ConfigurationError("DEEPSEEK_THINKING must be enabled or disabled.")

    def completion_options(self) -> dict[str, Any]:
        options = super().completion_options()
        options["extra_body"] = {"thinking": {"type": self.thinking}}
        return options

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> DeepSeekConfig:
        _load_dotenv(env_file)
        try:
            timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))
            max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "16384"))
            temperature = _optional_float(
                os.getenv("DEEPSEEK_TEMPERATURE"), "DEEPSEEK_TEMPERATURE"
            )
        except ValueError as error:
            raise ConfigurationError(
                "DEEPSEEK_TIMEOUT_SECONDS and DEEPSEEK_MAX_TOKENS must be numeric."
            ) from error
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            thinking=os.getenv("DEEPSEEK_THINKING", "disabled").lower(),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            tool_choice=os.getenv("DEEPSEEK_TOOL_CHOICE", "auto").lower(),
            temperature=temperature,
        )


class OpenAICompatibleChatClient:
    """Thin transport wrapper around the OpenAI chat-completions interface."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ConfigurationError(
                    "API dependencies are missing. Run `uv sync --extra api --dev`."
                ) from error
            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=2,
            )
        self.client = client

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=deepcopy(messages),
                tools=deepcopy(tools),
                **self.config.completion_options(),
            )
        except Exception as error:
            raise ApiRolloutError(
                f"Model API request failed ({type(error).__name__}): {error}"
            ) from error


def _optional_float(value: str | None, name: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be numeric.") from error


def _load_dotenv(env_file: str | Path | None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise ConfigurationError(
            "API dependencies are missing. Run `uv sync --extra api --dev`."
        ) from error

    dotenv_path = Path(env_file) if env_file is not None else project_root() / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
