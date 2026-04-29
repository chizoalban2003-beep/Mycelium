"""Stage 147 — VisionAgent: screenshot analysis and computer-use integration.

:class:`VisionAgent` converts raw screenshots into structured UI understanding
and actionable computer-use plans.  It bridges vision models with
:class:`~physml.screen_agent.ScreenAgent` to enable fully automated UI tasks.

Backends (in preference order)
-------------------------------
1. **Claude Vision** — Anthropic API with image input (requires API key).
2. **Ollama vision** — Local VLM (llava, llava-phi3, moondream) via
   :class:`~physml.llm.local_llm.LocalLLM`.  Start with ``ollama pull llava``.
3. **OCR fallback** — pytesseract text extraction when vision models unavailable.

Capabilities
------------
* ``analyse(screenshot_path)`` — returns :class:`VisionResult` with UI elements,
  description, suggested actions, and focussed app.
* ``find_element(description, screenshot_path)`` — locate a named UI element and
  return its approximate pixel coordinates.
* ``find_and_click(description)`` — take a screenshot, find element, click it.
* ``describe_goal_step(goal, step, screenshot_path)`` — ask the vision model how
  to perform a specific goal step given current screen state.
* ``watch_for(condition, timeout)`` — poll the screen until a condition is
  visually detected or timeout expires.

Usage
-----
::

    from physml import VisionAgent

    va = VisionAgent()
    result = va.analyse("/path/to/screenshot.png")
    print(result.description)
    print(result.elements)       # [UIElement(label='Save button', x=200, y=400), ...]
    print(result.suggested_actions)

    va.find_and_click("Save button")
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class UIElement:
    """A detected UI element from a screenshot."""

    label: str
    element_type: str = "unknown"   # button, text, input, menu, icon, link
    x: int = -1
    y: int = -1
    width: int = 0
    height: int = 0
    text: str = ""
    confidence: float = 0.0
    clickable: bool = True

    @property
    def center(self) -> Tuple[int, int]:
        if self.x >= 0 and self.y >= 0:
            return self.x + self.width // 2, self.y + self.height // 2
        return self.x, self.y


@dataclass
class VisionResult:
    """Result from a VisionAgent.analyse() call."""

    screenshot_path: str = ""
    description: str = ""
    elements: List[UIElement] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    active_app: str = ""
    active_window: str = ""
    text_content: str = ""
    backend: str = ""
    elapsed: float = 0.0
    success: bool = False
    error: str = ""

    def find_element(self, label: str) -> Optional[UIElement]:
        label_lower = label.lower()
        for el in self.elements:
            if label_lower in el.label.lower() or label_lower in el.text.lower():
                return el
        return None


# ---------------------------------------------------------------------------
# Vision analysis helpers
# ---------------------------------------------------------------------------
_ANALYSE_PROMPT = """Analyse this screenshot and respond with JSON only (no markdown, no explanation).

Return this exact structure:
{
  "description": "one sentence describing what is on screen",
  "active_app": "app name or empty string",
  "active_window": "window title or empty string",
  "text_content": "any important text visible (truncated to 200 chars)",
  "elements": [
    {"label": "element name", "type": "button|text|input|menu|icon|link|other",
     "x": pixel_x_or_-1, "y": pixel_y_or_-1, "text": "visible text", "clickable": true}
  ],
  "suggested_actions": ["action 1", "action 2", "action 3"]
}

List up to 8 interactive elements. Use -1 for unknown coordinates."""

_FIND_PROMPT = """Look at this screenshot. Where is the UI element described as: "{description}"?

Respond with JSON only:
{{"found": true/false, "x": pixel_x, "y": pixel_y, "label": "element name", "confidence": 0.0-1.0}}

If not found, set found=false and use -1 for coordinates."""

_GOAL_STEP_PROMPT = """You are helping execute a computer task.

Goal: {goal}
Current step: {step}

Look at the current screenshot and explain the single next action to take.
Respond with JSON only:
{{"action": "click|type|scroll|key|wait|none",
  "target_description": "what to interact with",
  "x": pixel_x_or_-1, "y": pixel_y_or_-1,
  "text": "text to type if action is type",
  "key": "key name if action is key (e.g. Return, Tab, Escape)",
  "explanation": "why this action"}}"""


def _load_image_b64(path: str) -> Optional[str]:
    try:
        return base64.standard_b64encode(Path(path).read_bytes()).decode()
    except Exception:
        return None


def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# VisionAgent
# ---------------------------------------------------------------------------
class VisionAgent:
    """Screenshot analysis and computer-use agent.

    Parameters
    ----------
    claude_client : ClaudeClient or None
        Anthropic Vision client.  Auto-created when None.
    local_llm : LocalLLM or None
        Local VLM backend (e.g. ollama with llava).  Auto-created when None.
    screen_agent : ScreenAgent or None
        Used for taking screenshots and executing actions.  Auto-created when None.
    vision_model : str
        Ollama vision model name (e.g. ``"llava"``, ``"llava-phi3"``).
    analyse_prompt : str or None
        Override the default analysis prompt.
    """

    def __init__(
        self,
        claude_client: Any = None,
        local_llm: Any = None,
        screen_agent: Any = None,
        vision_model: str = "llava",
        analyse_prompt: Optional[str] = None,
    ) -> None:
        self._claude = claude_client
        self._local = local_llm
        self._screen = screen_agent
        self._vision_model = vision_model
        self._analyse_prompt = analyse_prompt or _ANALYSE_PROMPT
        self._call_count = 0
        self._backend: str = "none"
        self._detect_backend()

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect_backend(self) -> None:
        if self._claude is None:
            try:
                from physml.llm.claude_client import ClaudeClient
                self._claude = ClaudeClient()
            except Exception:
                pass

        if self._claude is not None and getattr(self._claude, "available", False):
            # Check if it has vision capability (Claude SDK supports images)
            if not getattr(self._claude, "using_local_llm", False):
                self._backend = "claude-vision"
                _logger.info("VisionAgent: using Claude Vision backend")
                return
            # If using local LLM through ClaudeClient, check vision
            local = getattr(self._claude, "local_llm", None)
            if local is not None:
                self._local = local

        if self._local is None:
            try:
                from physml.llm.local_llm import LocalLLM
                self._local = LocalLLM(vision_model=self._vision_model)
            except Exception:
                pass

        if self._local is not None and getattr(self._local, "available", False):
            self._backend = "ollama-vision"
            _logger.info("VisionAgent: using ollama vision backend (%s)", self._vision_model)
            return

        self._backend = "ocr"
        _logger.info("VisionAgent: vision models unavailable — falling back to OCR")

    def _get_screen(self) -> Any:
        if self._screen is None:
            try:
                from physml.screen_agent import ScreenAgent
                self._screen = ScreenAgent()
            except Exception:
                pass
        return self._screen

    @property
    def available(self) -> bool:
        return self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def call_count(self) -> int:
        return self._call_count

    # ------------------------------------------------------------------
    # Vision query
    # ------------------------------------------------------------------
    def _vision_query(self, image_b64: str, prompt: str) -> str:
        """Send image + prompt to whichever vision backend is available."""
        self._call_count += 1

        if self._backend == "claude-vision" and self._claude is not None:
            try:
                client_obj = getattr(self._claude, "_client", None)
                if client_obj is None:
                    raise RuntimeError("Claude client not initialised")
                response = client_obj.messages.create(
                    model=self._claude.model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return response.content[0].text
            except Exception as exc:
                _logger.warning("VisionAgent: Claude Vision failed: %s", exc)

        if self._backend == "ollama-vision" and self._local is not None:
            result = self._local.vision_chat(image_b64, prompt)
            if result.text:
                return result.text

        return ""

    def _ocr_fallback(self, screenshot_path: str) -> str:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
            img = Image.open(screenshot_path)
            return pytesseract.image_to_string(img)[:1000]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyse(self, screenshot_path: str) -> VisionResult:
        """Analyse a screenshot and return structured UI understanding.

        Parameters
        ----------
        screenshot_path : str
            Path to a PNG/JPEG screenshot file.

        Returns
        -------
        VisionResult
        """
        t0 = time.time()
        result = VisionResult(screenshot_path=screenshot_path, backend=self._backend)

        image_b64 = _load_image_b64(screenshot_path)
        if image_b64 is None:
            result.error = f"Cannot read screenshot: {screenshot_path}"
            return result

        if self._backend in ("claude-vision", "ollama-vision"):
            raw = self._vision_query(image_b64, self._analyse_prompt)
            if raw:
                data = _parse_json_from_text(raw)
                result.description = data.get("description", "")
                result.active_app = data.get("active_app", "")
                result.active_window = data.get("active_window", "")
                result.text_content = data.get("text_content", "")
                result.suggested_actions = data.get("suggested_actions", [])
                for el_data in data.get("elements", []):
                    result.elements.append(UIElement(
                        label=el_data.get("label", ""),
                        element_type=el_data.get("type", "unknown"),
                        x=el_data.get("x", -1),
                        y=el_data.get("y", -1),
                        text=el_data.get("text", ""),
                        clickable=el_data.get("clickable", True),
                    ))
                result.success = True

        if not result.success:
            # OCR fallback
            text = self._ocr_fallback(screenshot_path)
            result.text_content = text
            result.description = f"Screen text (OCR): {text[:100]}" if text else "Screenshot captured"
            result.backend = "ocr"
            result.success = bool(text or screenshot_path)

        result.elapsed = time.time() - t0
        return result

    def analyse_current_screen(self) -> VisionResult:
        """Take a screenshot of the current screen and analyse it."""
        screen = self._get_screen()
        if screen is None:
            return VisionResult(error="ScreenAgent not available (install mss or pyautogui)")
        path = screen.screenshot()
        if path is None:
            return VisionResult(error="Screenshot capture failed")
        return self.analyse(path)

    def find_element(
        self, description: str, screenshot_path: Optional[str] = None
    ) -> Optional[UIElement]:
        """Locate a named UI element in a screenshot.

        Parameters
        ----------
        description : str
            Natural-language description of the element (e.g. "Save button").
        screenshot_path : str or None
            Screenshot to search.  Takes a new screenshot when None.

        Returns
        -------
        UIElement or None
        """
        if screenshot_path is None:
            screen = self._get_screen()
            if screen:
                screenshot_path = screen.screenshot()
        if not screenshot_path:
            return None

        image_b64 = _load_image_b64(screenshot_path)
        if image_b64 is None:
            return None

        prompt = _FIND_PROMPT.format(description=description)
        raw = self._vision_query(image_b64, prompt)
        if not raw:
            return None
        data = _parse_json_from_text(raw)
        if not data.get("found"):
            return None
        return UIElement(
            label=data.get("label", description),
            x=data.get("x", -1),
            y=data.get("y", -1),
            confidence=data.get("confidence", 0.5),
        )

    def find_and_click(self, description: str) -> bool:
        """Take a screenshot, find an element, and click it.

        Returns True on success.
        """
        element = self.find_element(description)
        if element is None:
            _logger.info("VisionAgent: element not found: %r", description)
            return False
        if element.x < 0 or element.y < 0:
            _logger.info("VisionAgent: element found but coords unknown: %r", description)
            return False
        screen = self._get_screen()
        if screen is None:
            return False
        cx, cy = element.center
        result = screen.click(cx, cy)
        _logger.info(
            "VisionAgent: clicked %r at (%d, %d) — %s", description, cx, cy,
            "ok" if result else "failed"
        )
        return result

    def describe_goal_step(
        self, goal: str, step: str, screenshot_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask the vision model how to perform a specific goal step.

        Returns a dict with keys: action, target_description, x, y, text, key, explanation.
        """
        if screenshot_path is None:
            screen = self._get_screen()
            if screen:
                screenshot_path = screen.screenshot()
        if not screenshot_path:
            return {"action": "none", "explanation": "No screenshot available"}

        image_b64 = _load_image_b64(screenshot_path)
        if image_b64 is None:
            return {"action": "none", "explanation": "Cannot read screenshot"}

        prompt = _GOAL_STEP_PROMPT.format(goal=goal, step=step)
        raw = self._vision_query(image_b64, prompt)
        if not raw:
            return {"action": "none", "explanation": "Vision backend unavailable"}
        return _parse_json_from_text(raw) or {"action": "none", "explanation": raw[:200]}

    def watch_for(
        self,
        condition: str,
        timeout: float = 30.0,
        interval: float = 2.0,
    ) -> bool:
        """Poll the screen until a visual condition is detected or timeout expires.

        Parameters
        ----------
        condition : str
            Natural-language description of what to look for (e.g. "dialog box appears").
        timeout : float
            Maximum seconds to wait.
        interval : float
            Seconds between screenshot checks.

        Returns
        -------
        bool
            True if condition was detected, False on timeout.
        """
        deadline = time.time() + timeout
        check_prompt = (
            f"Does the current screenshot show: {condition}? "
            "Reply with JSON: {{\"detected\": true/false, \"confidence\": 0.0-1.0}}"
        )
        screen = self._get_screen()
        if screen is None:
            return False

        while time.time() < deadline:
            path = screen.screenshot()
            if path:
                image_b64 = _load_image_b64(path)
                if image_b64:
                    raw = self._vision_query(image_b64, check_prompt)
                    data = _parse_json_from_text(raw)
                    if data.get("detected") and data.get("confidence", 0) >= 0.6:
                        _logger.info("VisionAgent: condition detected: %r", condition)
                        return True
            time.sleep(interval)

        _logger.info("VisionAgent: condition not detected within %.1fs: %r", timeout, condition)
        return False

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "available": self.available,
            "call_count": self._call_count,
            "vision_model": self._vision_model,
            "screen_agent_available": (
                getattr(self._get_screen(), "available", False)
            ),
        }

    def __repr__(self) -> str:
        return f"VisionAgent(backend={self._backend!r}, available={self.available})"
