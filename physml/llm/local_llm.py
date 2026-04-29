"""physml.llm.local_llm — Local LLM backend for fully-offline operation.

Provides the same ``chat()`` / ``complete()`` interface as :class:`ClaudeClient`
but routes requests to a **local** model instead of the Anthropic API.

Two backends are supported (tried in order):

1. **Ollama** — zero extra dependencies, just ``urllib``.  Requires Ollama to be
   running locally (``ollama serve``) with at least one model pulled
   (e.g. ``ollama pull llama3.2`` or ``ollama pull llava`` for vision tasks).
   Detection: HTTP GET ``http://localhost:11434/api/tags`` succeeds.

2. **llama-cpp-python** — in-process GGUF inference; ``pip install llama-cpp-python``.
   Set ``MYCO_GGUF_PATH`` to your ``.gguf`` model file.

Both backends return a :class:`~physml.llm.claude_client.ChatResult`-compatible
object so callers can be backend-agnostic.

Quick start::

    from physml.llm.local_llm import LocalLLM

    llm = LocalLLM()          # auto-detects ollama or llama.cpp
    if llm.available:
        result = llm.chat("Explain Python decorators")
        print(result.text)
    else:
        print("No local LLM available — run 'ollama serve' to enable")
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from physml._log import get_logger

_logger = get_logger(__name__)

_OLLAMA_BASE = os.environ.get("MYCO_OLLAMA_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_MODEL = os.environ.get("MYCO_OLLAMA_MODEL", "llama3.2")
_GGUF_PATH = os.environ.get("MYCO_GGUF_PATH", "")
_DEFAULT_MAX_TOKENS = 1024
_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Shared result type (mirrors ChatResult from claude_client)
# ---------------------------------------------------------------------------
@dataclass
class LocalChatResult:
    """Result from a local LLM call."""

    text: str = ""
    available: bool = False
    model: str = ""
    backend: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    elapsed: float = 0.0
    cache_hit: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.available and not self.error


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
def _ollama_request(path: str, payload: Optional[dict] = None) -> dict:
    url = f"{_OLLAMA_BASE}{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _ollama_available() -> bool:
    try:
        _ollama_request("/api/tags")
        return True
    except Exception:
        return False


def _ollama_models() -> List[str]:
    try:
        data = _ollama_request("/api/tags")
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _ollama_chat(
    messages: List[Dict[str, str]],
    model: str,
    system: Optional[str],
    max_tokens: int,
) -> LocalChatResult:
    t0 = time.time()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if system:
        payload["system"] = system
    try:
        resp = _ollama_request("/api/chat", payload)
        text = resp.get("message", {}).get("content", "")
        eval_count = resp.get("eval_count", 0)
        prompt_eval_count = resp.get("prompt_eval_count", 0)
        return LocalChatResult(
            text=text.strip(),
            available=True,
            model=model,
            backend="ollama",
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            elapsed=time.time() - t0,
        )
    except Exception as exc:
        return LocalChatResult(
            available=False, backend="ollama", error=str(exc), elapsed=time.time() - t0
        )


def _ollama_vision(
    image_b64: str, prompt: str, model: str = "llava"
) -> LocalChatResult:
    """Vision query using an ollama vision model (e.g. llava, llava-phi3)."""
    t0 = time.time()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }
    try:
        resp = _ollama_request("/api/chat", payload)
        text = resp.get("message", {}).get("content", "")
        return LocalChatResult(
            text=text.strip(),
            available=True,
            model=model,
            backend="ollama-vision",
            elapsed=time.time() - t0,
        )
    except Exception as exc:
        return LocalChatResult(
            available=False, backend="ollama-vision", error=str(exc), elapsed=time.time() - t0
        )


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------
def _llama_cpp_chat(
    messages: List[Dict[str, str]],
    system: Optional[str],
    max_tokens: int,
    gguf_path: str,
) -> LocalChatResult:
    t0 = time.time()
    try:
        from llama_cpp import Llama  # type: ignore

        llm = Llama(model_path=gguf_path, n_ctx=4096, verbose=False)
        prompt_parts = []
        if system:
            prompt_parts.append(f"<system>{system}</system>")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"<{role}>{content}</{role}>")
        prompt_parts.append("<assistant>")
        full_prompt = "\n".join(prompt_parts)

        output = llm(full_prompt, max_tokens=max_tokens, stop=["</assistant>"])
        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        return LocalChatResult(
            text=text,
            available=True,
            model=gguf_path,
            backend="llama.cpp",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            elapsed=time.time() - t0,
        )
    except ImportError:
        return LocalChatResult(
            available=False,
            backend="llama.cpp",
            error="llama-cpp-python not installed: pip install llama-cpp-python",
            elapsed=time.time() - t0,
        )
    except Exception as exc:
        return LocalChatResult(
            available=False, backend="llama.cpp", error=str(exc), elapsed=time.time() - t0
        )


# ---------------------------------------------------------------------------
# LocalLLM — unified interface
# ---------------------------------------------------------------------------
class LocalLLM:
    """Local LLM backend — tries Ollama then llama.cpp.

    Parameters
    ----------
    model : str or None
        Ollama model name (e.g. ``"llama3.2"``, ``"mistral"``, ``"llava"``).
        Defaults to ``MYCO_OLLAMA_MODEL`` env var or ``"llama3.2"``.
    ollama_url : str
        Base URL of the Ollama server.
    gguf_path : str or None
        Path to a ``.gguf`` file for llama.cpp fallback.
        Defaults to ``MYCO_GGUF_PATH`` env var.
    max_tokens : int
        Max tokens to generate.
    vision_model : str
        Ollama model used for vision tasks (must support images, e.g. ``"llava"``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        ollama_url: str = _OLLAMA_BASE,
        gguf_path: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        vision_model: str = "llava",
    ) -> None:
        self.model = model or _DEFAULT_OLLAMA_MODEL
        self.ollama_url = ollama_url
        self.gguf_path = gguf_path or _GGUF_PATH
        self.max_tokens = max_tokens
        self.vision_model = vision_model

        self._ollama_ok: Optional[bool] = None
        self._llama_ok: Optional[bool] = None
        self._backend: str = "none"
        self._call_count: int = 0

        self._detect_backend()

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect_backend(self) -> None:
        self._ollama_ok = _ollama_available()
        if self._ollama_ok:
            models = _ollama_models()
            self._backend = "ollama"
            _logger.info(
                "LocalLLM: ollama detected at %s — %d model(s) available: %s",
                self.ollama_url,
                len(models),
                models[:5],
            )
            if models and self.model not in models:
                best = models[0]
                _logger.info(
                    "LocalLLM: requested model %r not found; using %r instead",
                    self.model,
                    best,
                )
                self.model = best
            return

        if self.gguf_path:
            try:
                import llama_cpp  # type: ignore  # noqa: F401
                self._llama_ok = True
                self._backend = "llama.cpp"
                _logger.info("LocalLLM: llama.cpp backend ready (%s)", self.gguf_path)
            except ImportError:
                self._llama_ok = False
                _logger.debug("LocalLLM: llama.cpp not installed")
        else:
            _logger.info(
                "LocalLLM: no backend available. "
                "Run 'ollama serve' (and 'ollama pull llama3.2') to enable local LLM."
            )

    @property
    def available(self) -> bool:
        return self._backend in ("ollama", "llama.cpp")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def call_count(self) -> int:
        return self._call_count

    # ------------------------------------------------------------------
    # Public interface (matches ClaudeClient)
    # ------------------------------------------------------------------
    def chat(
        self,
        user_message: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
        **_kwargs: Any,
    ) -> LocalChatResult:
        """Send a chat message and return a :class:`LocalChatResult`."""
        if not self.available:
            return LocalChatResult(
                available=False,
                error="No local LLM backend available. Run 'ollama serve' first.",
            )

        messages = list(history or []) + [{"role": "user", "content": user_message}]
        self._call_count += 1

        if self._backend == "ollama":
            return _ollama_chat(messages, self.model, system, self.max_tokens)
        if self._backend == "llama.cpp":
            return _llama_cpp_chat(messages, system, self.max_tokens, self.gguf_path)
        return LocalChatResult(available=False, error="Unknown backend")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **_kwargs: Any,
    ) -> LocalChatResult:
        """Single-turn completion (no history)."""
        return self.chat(user_message=prompt, system=system)

    def vision_chat(
        self,
        image_b64: str,
        prompt: str = "Describe what you see in this screenshot.",
    ) -> LocalChatResult:
        """Vision query — requires a vision-capable model (e.g. llava).

        Parameters
        ----------
        image_b64 : str
            Base-64 encoded PNG/JPEG image.
        prompt : str
            Text prompt to accompany the image.
        """
        if self._backend == "ollama":
            return _ollama_vision(image_b64, prompt, self.vision_model)
        return LocalChatResult(
            available=False,
            error="Vision requires ollama with a vision model (e.g. 'ollama pull llava')",
        )

    def list_models(self) -> List[str]:
        """Return available model names (ollama only)."""
        if self._backend == "ollama":
            return _ollama_models()
        return []

    def pull_model(self, model_name: str) -> bool:
        """Pull an ollama model (triggers download if not cached).

        Returns True on success.
        """
        if self._backend != "ollama":
            return False
        try:
            _ollama_request("/api/pull", {"name": model_name, "stream": False})
            _logger.info("LocalLLM: pulled model %r", model_name)
            return True
        except Exception as exc:
            _logger.warning("LocalLLM: pull failed: %s", exc)
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "available": self.available,
            "model": self.model,
            "vision_model": self.vision_model,
            "ollama_url": self.ollama_url,
            "gguf_path": self.gguf_path or None,
            "call_count": self._call_count,
            "models": self.list_models(),
        }

    def __repr__(self) -> str:
        return (
            f"LocalLLM(backend={self._backend!r}, model={self.model!r}, "
            f"available={self.available})"
        )
