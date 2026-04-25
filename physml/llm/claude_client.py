"""physml.llm.claude_client — Claude API client with prompt caching and tool use.

This module wraps the ``anthropic`` SDK to provide:

* Prompt caching via ``cache_control`` on system prompts (reduces latency and
  token cost on repeated calls with the same system prompt).
* Streaming support via :meth:`ClaudeClient.stream`.
* Structured tool-use via :meth:`ClaudeClient.tool_call`.
* Graceful fallback: when the SDK is unavailable or no key is configured, all
  methods return a ``ChatResult(available=False)`` — no exceptions thrown.

The default model is ``claude-sonnet-4-6``.

Usage::

    from physml.llm.claude_client import ClaudeClient

    client = ClaudeClient()          # reads ANTHROPIC_API_KEY from environment
    result = client.chat("Hi!")
    print(result.text, result.available)

    # With a custom system prompt (cached on first call)
    result = client.chat(
        "Summarise the following table:",
        system="You are a concise data analyst. Respond in bullet points.",
    )

    # Structured tool call
    from physml.llm.claude_client import ToolCallResult
    tools = [
        {
            "name": "run_prediction",
            "description": "Run a prediction on feature values.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "features": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["features"],
            },
        }
    ]
    tc: ToolCallResult = client.tool_call("predict for values 1 2 3", tools=tools)
    if tc.tool_calls:
        print(tc.tool_calls[0]["name"], tc.tool_calls[0]["input"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence

from physml._log import get_logger

_logger = get_logger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ChatResult:
    """Result of a :meth:`ClaudeClient.chat` call.

    Attributes
    ----------
    text : str
        The generated response text (empty when ``available`` is False).
    available : bool
        ``True`` when the Claude API was used; ``False`` in fallback mode.
    model : str
        Model ID that produced the response.
    input_tokens : int
        Tokens consumed in the prompt.
    output_tokens : int
        Tokens generated in the response.
    cache_hit : bool
        ``True`` when the system prompt was served from the prompt cache.
    tool_calls : list of dict
        Any tool-use blocks returned by the model (empty for text-only responses).
    error : str or None
        Error description when the API call failed.
    """

    text: str = ""
    available: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ToolCallResult:
    """Result of a :meth:`ClaudeClient.tool_call` call.

    Attributes
    ----------
    tool_calls : list of dict
        List of ``{"id": ..., "name": ..., "input": {...}}`` dicts.
    text : str
        Any text the model returned alongside the tool calls.
    available : bool
        Whether the API was reachable.
    error : str or None
    """

    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    available: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ClaudeClient
# ---------------------------------------------------------------------------


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK with caching and tool-use support.

    Parameters
    ----------
    api_key : str or None
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
    model : str
        Model ID.  Defaults to ``"claude-sonnet-4-6"``.
    max_tokens : int
        Maximum tokens in each response.
    temperature : float
        Sampling temperature (0 = deterministic).
    use_caching : bool
        When ``True``, attaches ``cache_control: {type: ephemeral}`` to the
        system prompt so the API caches it across requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        use_caching: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_caching = use_caching
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None
        self._available: bool = False
        self._call_count: int = 0
        self._cache_hits: int = 0
        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        if not self._api_key:
            _logger.info(
                "ClaudeClient: no API key — running in fallback mode. "
                "Set ANTHROPIC_API_KEY to enable Claude."
            )
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._available = True
            _logger.info("ClaudeClient: SDK ready (model=%s)", self.model)
        except ImportError:
            _logger.info(
                "ClaudeClient: anthropic SDK not installed — "
                "pip install 'physml[llm]' to enable. Running in fallback mode."
            )

    @property
    def available(self) -> bool:
        """``True`` when the SDK is installed and an API key is set."""
        return self._available

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def _build_system_block(self, system: str) -> Any:
        """Return a system prompt in the correct format (cached or plain)."""
        if self.use_caching:
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system

    # ------------------------------------------------------------------
    # Core: chat
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResult:
        """Send a message and return a :class:`ChatResult`.

        Parameters
        ----------
        user_message : str
            The user's text.
        history : list of {"role": str, "content": str} dicts, optional
            Prior conversation turns.
        system : str, optional
            System prompt.
        tools : list of Anthropic tool definitions, optional
            When supplied, Claude may respond with tool-use blocks.

        Returns
        -------
        ChatResult
        """
        if not self._available or self._client is None:
            return ChatResult(available=False, error="Claude API not available")

        try:
            messages = _build_messages(history or [], user_message)
            default_system = (
                "You are Mycelium (Myco), a helpful local AI companion. "
                "Be concise and accurate."
            )
            sys_prompt = system or default_system

            req: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "system": self._build_system_block(sys_prompt),
            }
            if tools:
                req["tools"] = tools

            response = self._client.messages.create(**req)
            self._call_count += 1

            text = ""
            tool_calls: List[Dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {"id": block.id, "name": block.name, "input": block.input}
                    )

            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            if cache_read > 0:
                self._cache_hits += 1

            return ChatResult(
                text=text.strip(),
                available=True,
                model=self.model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_hit=cache_read > 0,
                tool_calls=tool_calls,
            )

        except Exception as exc:
            _logger.warning("ClaudeClient.chat failed: %s", exc)
            return ChatResult(available=False, error=str(exc))

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        user_message: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields text tokens as they arrive.

        Falls back to yielding the full response as one chunk when the API is
        unavailable.

        Yields
        ------
        str
            Raw text tokens.
        """
        if not self._available or self._client is None:
            yield f"[Claude unavailable] {user_message}"
            return

        import asyncio

        messages = _build_messages(history or [], user_message)
        default_system = "You are Myco, a helpful local AI companion."
        sys_prompt = system or default_system

        req: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system": sys_prompt,
        }

        chunks: List[str] = []

        def _run_stream() -> None:
            with self._client.messages.stream(**req) as s:
                for token in s.text_stream:
                    chunks.append(token)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_stream)
            self._call_count += 1
            for chunk in chunks:
                yield chunk
        except Exception as exc:
            _logger.warning("ClaudeClient.stream failed: %s", exc)
            result = self.chat(user_message, history=history, system=system)
            yield result.text or ""

    # ------------------------------------------------------------------
    # Structured tool call
    # ------------------------------------------------------------------

    def tool_call(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> ToolCallResult:
        """Send a message with tool definitions and parse the tool-use response.

        Parameters
        ----------
        user_message : str
        tools : list of Anthropic tool definition dicts
        system : str, optional
        history : list of prior message dicts, optional

        Returns
        -------
        ToolCallResult
        """
        result = self.chat(
            user_message=user_message,
            history=history,
            system=system,
            tools=tools,
        )
        return ToolCallResult(
            tool_calls=result.tool_calls,
            text=result.text,
            available=result.available,
            error=result.error,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: Optional[str] = None) -> ChatResult:
        """Single-turn completion (no history)."""
        return self.chat(user_message=prompt, system=system)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of API calls made."""
        return self._call_count

    @property
    def cache_hit_count(self) -> int:
        """Number of calls where the system prompt was served from cache."""
        return self._cache_hits

    def __repr__(self) -> str:
        return (
            f"ClaudeClient(model={self.model!r}, "
            f"available={self.available}, "
            f"calls={self._call_count})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages(
    history: Sequence[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Build the messages list in Anthropic API format.

    Trims to the last 20 turns and ensures the sequence ends with a user
    message.
    """
    msgs: List[Dict[str, str]] = []
    for msg in list(history)[-20:]:
        msgs.append({"role": msg["role"], "content": msg["content"]})

    # Remove trailing user message to avoid duplication
    if msgs and msgs[-1]["role"] == "user":
        msgs.pop()

    msgs.append({"role": "user", "content": user_message})
    return msgs
