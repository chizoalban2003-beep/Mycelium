"""Stage 121 — LLMIntegration: Claude API backbone for conversational AI.

Wraps the Anthropic SDK as an *optional* dependency.  When the SDK is
available and an API key is configured, all conversational responses route
through Claude.  When unavailable, the system falls back to the existing
keyword/rule-based NL router (Stage 106) transparently — no breakage.

Capabilities
------------
* Chat completion with full conversation history context.
* System-prompt construction from :class:`~physml.digital_soul.DigitalSoul`
  and :class:`~physml.user_profile.UserProfileLearner`.
* Prompt caching via ``cache_control`` headers for repeated system prompts
  (reduces latency and token cost on repeated calls).
* Tool-call bridge: Claude can invoke Mycelium physics engine, document
  processor, and local executor through structured tool definitions.
* Graceful fallback: if the SDK is missing or the API key is absent, returns
  ``LLMResult(available=False)`` — callers can degrade to rule-based logic.

Usage
-----
::

    from physml.llm_integration import LLMIntegration, LLMConfig

    llm = LLMIntegration(api_key="sk-ant-...")   # or set ANTHROPIC_API_KEY
    result = llm.chat("What is the weather like?", history=[], system="You are Myco.")
    print(result.text)        # Claude's response
    print(result.available)   # True
    print(result.model)       # "claude-sonnet-4-6"

    # Build system prompt from companion subsystems
    system = llm.build_system_prompt(soul=soul, profile=profile)
    result = llm.chat("predict my sales", history=history, system=system)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024


@dataclass
class LLMMessage:
    """A single message in a conversation.

    Attributes
    ----------
    role : str
        ``"user"`` or ``"assistant"``.
    content : str
        The message text.
    """

    role: str
    content: str


@dataclass
class LLMConfig:
    """Configuration for :class:`LLMIntegration`.

    Attributes
    ----------
    model : str
        Anthropic model ID.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature (0 = deterministic).
    use_caching : bool
        Enable prompt caching for the system prompt.
    """

    model: str = _DEFAULT_MODEL
    max_tokens: int = _MAX_TOKENS
    temperature: float = 0.7
    use_caching: bool = True


@dataclass
class LLMResult:
    """Result from an LLM chat call.

    Attributes
    ----------
    text : str
        The generated response text.
    available : bool
        ``True`` if the LLM was used; ``False`` if fallback mode.
    model : str
        Model that produced the response.
    input_tokens : int
        Tokens in the prompt.
    output_tokens : int
        Tokens in the response.
    cache_hit : bool
        ``True`` if the system prompt was served from cache.
    error : str or None
        Error message if the call failed.
    tool_calls : list
        Any tool calls Claude wants to make (for agentic use).
    """

    text: str = ""
    available: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core integration class
# ---------------------------------------------------------------------------


class LLMIntegration:
    """Claude API integration with graceful fallback.

    Parameters
    ----------
    api_key : str or None
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
    config : LLMConfig or None
        Model and generation settings.  Defaults to :class:`LLMConfig`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None
        self._sdk_available: bool = False
        self._call_count: int = 0
        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        if not self._api_key:
            _logger.info(
                "LLMIntegration: no API key found — running in fallback mode. "
                "Set ANTHROPIC_API_KEY to enable Claude."
            )
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._sdk_available = True
            _logger.info(
                "LLMIntegration: Anthropic SDK available; model=%s",
                self.config.model,
            )
        except ImportError:
            _logger.info(
                "LLMIntegration: anthropic SDK not installed — "
                "pip install anthropic to enable Claude. Running in fallback mode."
            )

    @property
    def available(self) -> bool:
        """``True`` if the Anthropic SDK is installed and an API key is set."""
        return self._sdk_available

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        soul: Any = None,
        profile: Any = None,
        extra: str = "",
    ) -> str:
        """Build a rich system prompt from agent identity and user profile.

        Parameters
        ----------
        soul : DigitalSoul or None
            Agent identity object (provides name, mood, stats, life story).
        profile : UserProfileLearner or None
            User preference object (provides top topics, feedback history).
        extra : str
            Additional instructions to append.

        Returns
        -------
        str
            A complete system prompt ready for the LLM.
        """
        parts: List[str] = []

        # Agent identity
        if soul is not None:
            name = getattr(soul, "name", "Mycelium")
            mood = getattr(soul, "mood", "curious")
            stats = getattr(soul, "stats", {})
            parts.append(
                f"You are {name}, a personal AI companion running entirely on the "
                f"user's local device. Your current mood is '{mood}'. "
                f"You have made {stats.get('total_predictions', 0)} predictions and "
                f"have been active for {stats.get('days_alive', 0)} day(s)."
            )
            # Life story excerpt
            try:
                story = soul.life_story()
                if story and len(story) > 10:
                    parts.append(f"Your history: {story[:500]}")
            except Exception:
                pass
        else:
            parts.append(
                "You are Mycelium (Myco), a personal AI companion running entirely "
                "on the user's local device. You are privacy-first, never send data "
                "to the cloud, and learn from the user over time."
            )

        # Core capabilities
        parts.append(
            "You can: run ML predictions on tabular data, train models on CSV files, "
            "process documents, execute safe local tasks, track user preferences, "
            "and answer questions. You are honest about what you can and cannot do."
        )

        # User profile
        if profile is not None:
            try:
                summary = profile.summary()
                topics = summary.get("top_topics", [])
                prefs = summary.get("preferences", {})
                count = summary.get("interaction_count", 0)
                if topics:
                    parts.append(
                        f"The user's top interests are: {', '.join(topics[:5])}."
                    )
                if prefs:
                    pref_str = "; ".join(f"{k}={v}" for k, v in list(prefs.items())[:5])
                    parts.append(f"User preferences: {pref_str}.")
                if count > 0:
                    parts.append(
                        f"You have had {count} previous interaction(s) with this user."
                    )
            except Exception:
                pass

        # Behaviour constraints
        parts.append(
            "Be concise and helpful. Never pretend to have capabilities you don't. "
            "If the user asks for a prediction and no model is loaded, say so and "
            "offer to train one. Respond in the same language the user writes in. "
            "All computation is local — reassure the user their data is private."
        )

        if extra:
            parts.append(extra)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        history: Optional[Sequence[LLMMessage]] = None,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResult:
        """Send a message and get a response.

        Parameters
        ----------
        user_message : str
            The user's latest message.
        history : list of LLMMessage, optional
            Prior conversation turns (alternating user/assistant).
        system : str, optional
            System prompt.  If ``None``, a minimal default is used.
        tools : list of dict, optional
            Anthropic-format tool definitions.  When provided, Claude may
            return tool_use blocks in the response.

        Returns
        -------
        LLMResult
        """
        if not self._sdk_available or self._client is None:
            return LLMResult(available=False, error="LLM not available")

        try:
            messages = _build_messages(history or [], user_message)
            system_prompt = system or (
                "You are Mycelium, a helpful local AI companion. "
                "Be concise and accurate."
            )

            # Build request kwargs
            req: Dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
            }

            if tools:
                req["tools"] = tools

            # Add system prompt with optional caching
            if self.config.use_caching:
                req["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                req["system"] = system_prompt

            response = self._client.messages.create(**req)
            self._call_count += 1

            # Parse response
            text = ""
            tool_calls: List[Dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            usage = response.usage
            cache_hit = (
                getattr(usage, "cache_read_input_tokens", 0) > 0
                if usage else False
            )

            return LLMResult(
                text=text.strip(),
                available=True,
                model=self.config.model,
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                cache_hit=cache_hit,
                tool_calls=tool_calls,
            )

        except Exception as exc:
            _logger.warning("LLMIntegration.chat failed: %s", exc)
            return LLMResult(available=False, error=str(exc))

    def chat_with_tool_results(
        self,
        tool_call_result_blocks: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        history: Optional[Sequence[LLMMessage]] = None,
        user_message: str = "",
        system: Optional[str] = None,
    ) -> LLMResult:
        """Continue a conversation after executing tool calls.

        Appends the assistant tool-use blocks and tool result blocks to the
        message history, then makes a second API call so Claude can produce
        a grounded final response.

        Parameters
        ----------
        tool_call_result_blocks : list of dict
            The raw tool call dicts from a previous :meth:`chat` result.
        tool_results : list of dict
            Results from :meth:`~physml.tool_bridge.ToolBridge.execute_all`.
        history, user_message, system
            Same as in :meth:`chat`.

        Returns
        -------
        LLMResult
        """
        if not self._sdk_available or self._client is None:
            return LLMResult(available=False, error="LLM not available")

        try:
            base_messages = _build_messages(history or [], user_message)

            # Reconstruct assistant message with tool_use content blocks
            assistant_content = []
            for call in tool_call_result_blocks:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["input"],
                    }
                )

            # Tool results go in a user turn
            tool_result_content = []
            for res in tool_results:
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": res["tool_use_id"],
                        "content": res["content"],
                    }
                )

            extended = list(base_messages)
            extended.append({"role": "assistant", "content": assistant_content})
            extended.append({"role": "user", "content": tool_result_content})

            system_prompt = system or (
                "You are Mycelium, a helpful local AI companion. Be concise."
            )
            req: Dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": extended,
            }
            if self.config.use_caching:
                req["system"] = [
                    {"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}}
                ]
            else:
                req["system"] = system_prompt

            response = self._client.messages.create(**req)
            self._call_count += 1
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            usage = response.usage
            return LLMResult(
                text=text.strip(),
                available=True,
                model=self.config.model,
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )

        except Exception as exc:
            _logger.warning("LLMIntegration.chat_with_tool_results failed: %s", exc)
            return LLMResult(available=False, error=str(exc))

    # ------------------------------------------------------------------
    # Convenience: single-shot completion
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResult:
        """Single-turn completion without history."""
        return self.chat(user_message=prompt, history=[], system=system)

    async def stream(
        self,
        user_message: str,
        history: Optional[Sequence] = None,
        system: Optional[str] = None,
    ):
        """Async generator that yields text tokens from a streaming completion.

        Falls back to yielding the full response as one chunk when the SDK is
        unavailable or streaming fails.

        Yields
        ------
        str
            Raw text tokens as they arrive from the API.
        """
        if not self._sdk_available or self._client is None:
            # No SDK — compute full response synchronously and yield it
            result = self.chat(user_message=user_message, history=history or [], system=system)
            yield result.text or ""
            return

        import asyncio
        try:
            messages = []
            for m in (history or []):
                messages.append({"role": m.role, "content": m.content})
            messages.append({"role": "user", "content": user_message})

            system_prompt = system or (
                "You are Myco, a helpful local AI companion. "
                "Be concise and friendly."
            )
            req = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
                "system": system_prompt,
            }

            # Run blocking stream() in a thread so the async generator doesn't block
            loop = asyncio.get_event_loop()
            chunks: List[str] = []

            def _run_stream():
                with self._client.messages.stream(**req) as stream:
                    for text in stream.text_stream:
                        chunks.append(text)

            await loop.run_in_executor(None, _run_stream)
            self._call_count += 1
            for chunk in chunks:
                yield chunk
        except Exception as exc:
            _logger.warning("LLMIntegration.stream error: %s", exc)
            result = self.chat(user_message=user_message, history=history or [], system=system)
            yield result.text or ""

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMIntegration("
            f"model={self.config.model!r}, "
            f"available={self.available})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages(
    history: Sequence[LLMMessage], user_message: str
) -> List[Dict[str, str]]:
    """Convert LLMMessage history + new user message into API format.

    Ensures the sequence always starts with a user turn (Anthropic requirement)
    and ends with the new user message.
    """
    msgs: List[Dict[str, str]] = []

    # Trim history to last 20 turns to avoid token overflow
    recent = list(history)[-20:]

    for msg in recent:
        msgs.append({"role": msg.role, "content": msg.content})

    # Ensure last message isn't already from user (would be duplicated)
    if msgs and msgs[-1]["role"] == "user":
        msgs.pop()

    msgs.append({"role": "user", "content": user_message})
    return msgs
