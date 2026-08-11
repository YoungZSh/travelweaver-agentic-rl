"""Provider-neutral model clients shared by synthesis, rollout, and evaluation."""

from .client import (
    DEFAULT_DEEPSEEK_CONCURRENCY,
    DeepSeekConfig,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)

__all__ = [
    "DEFAULT_DEEPSEEK_CONCURRENCY",
    "DeepSeekConfig",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
]
