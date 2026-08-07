"""Backward-compatible model-client imports; new code uses :mod:`travelweaver.llm`."""

from ..llm import DeepSeekConfig, OpenAICompatibleChatClient, OpenAICompatibleConfig

__all__ = [
    "DeepSeekConfig",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
]
