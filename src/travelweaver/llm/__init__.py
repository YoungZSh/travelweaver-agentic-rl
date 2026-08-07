"""Provider-neutral model clients shared by synthesis, rollout, and evaluation."""

from .client import (
    DeepSeekConfig,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)

__all__ = [
    "DeepSeekConfig",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
]
