"""physml.llm — Claude API backbone for the Mycelium local AI agent.

This sub-package provides:
- :class:`ClaudeClient` — a thin, caching-aware wrapper around the Anthropic SDK.
- :class:`PromptSystem` — natural-language router that maps user prompts to the
  right physml subsystem (agents, skills, tools, goals).

Both classes degrade gracefully when the ``anthropic`` SDK is absent or no API
key is configured.

Quick start::

    from physml.llm import ClaudeClient, PromptSystem

    client = ClaudeClient()            # reads ANTHROPIC_API_KEY from env
    result = client.chat("Hello!")
    print(result.text)

    ps = PromptSystem()
    action = ps.route("train a model on data.csv")
    print(action.intent)    # "train"
    print(action.payload)   # {"path": "data.csv"}
"""

from physml.llm.claude_client import ClaudeClient, ChatResult, ToolCallResult
from physml.llm.prompt_system import PromptSystem, PromptAction
from physml.llm.action_dispatcher import ActionDispatcher
from physml.llm.memory_store import UserMemory
from physml.llm.local_llm import LocalLLM, LocalChatResult

__all__ = [
    "ClaudeClient",
    "ChatResult",
    "ToolCallResult",
    "PromptSystem",
    "PromptAction",
    "ActionDispatcher",
    "UserMemory",
    "LocalLLM",
    "LocalChatResult",
]
