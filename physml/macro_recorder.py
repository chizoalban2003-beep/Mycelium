"""physml.macro_recorder — Record user action sequences for imitation learning.

Captures mouse clicks, keyboard events, and active-window context into
:class:`MacroSequence` objects.  Sequences can be:

* **Saved** as named :class:`~physml.skill_library.Skill` entries
* **Replayed** via :class:`~physml.screen_agent.ScreenAgent`
* **Trained on** by :class:`~physml.imitation_learner.ImitationLearner`

Optional dependencies
---------------------
* ``pynput`` — keyboard + mouse listener (``pip install pynput``)
* Without it, ``MacroRecorder.available`` returns ``False`` and
  :meth:`record_text` can still be used for manual/test sequences.

Usage::

    from physml.macro_recorder import MacroRecorder

    recorder = MacroRecorder()
    recorder.start_recording("open_browser")
    # ... user performs actions ...
    seq = recorder.stop_recording()

    print(seq.name, len(seq.steps))   # "open_browser", 12
    recorder.save_to_skill_library(seq)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType:
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    TYPE_TEXT = "type_text"
    SCROLL = "scroll"
    DRAG = "drag"
    WINDOW_CHANGE = "window_change"
    PAUSE = "pause"


@dataclass
class ActionStep:
    """One recorded user action.

    Attributes
    ----------
    action_type : str
        One of the ActionType constants.
    timestamp : float
        Unix timestamp when the action occurred.
    x, y : int or None
        Screen coordinates (for mouse actions).
    key : str or None
        Key name (for keyboard actions).
    text : str or None
        Typed text (for TYPE_TEXT).
    app_name : str
        Active application when action occurred.
    window_title : str
        Active window title when action occurred.
    metadata : dict
        Extra context (button, scroll direction, etc.).
    """

    action_type: str
    timestamp: float = field(default_factory=time.time)
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None
    text: Optional[str] = None
    app_name: str = "unknown"
    window_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "timestamp": self.timestamp,
            "x": self.x, "y": self.y,
            "key": self.key, "text": self.text,
            "app_name": self.app_name, "window_title": self.window_title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionStep":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MacroSequence:
    """A recorded sequence of user actions forming a macro.

    Attributes
    ----------
    name : str
        Human-readable name for the macro.
    steps : list[ActionStep]
        Ordered list of recorded actions.
    description : str
        Auto-generated or user-provided description.
    created_at : float
        Unix timestamp.
    tags : list[str]
        Categorisation tags.
    """

    name: str
    steps: List[ActionStep] = field(default_factory=list)
    description: str = ""
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp

    @property
    def apps_used(self) -> List[str]:
        seen: List[str] = []
        for s in self.steps:
            if s.app_name not in seen:
                seen.append(s.app_name)
        return seen

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "tags": self.tags,
            "duration": self.duration,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MacroSequence":
        steps = [ActionStep.from_dict(s) for s in d.get("steps", [])]
        return cls(
            name=d.get("name", "unnamed"),
            steps=steps,
            description=d.get("description", ""),
            created_at=d.get("created_at", time.time()),
            tags=d.get("tags", []),
        )

    def summarise(self) -> str:
        """Return a short human-readable summary of the sequence."""
        if not self.steps:
            return f"Empty macro: {self.name!r}"
        app_str = ", ".join(self.apps_used[:3])
        return (
            f"Macro {self.name!r}: {len(self.steps)} steps over {self.duration:.1f}s "
            f"in {app_str}"
        )


class MacroRecorder:
    """Record user action sequences for imitation learning and skill saving.

    Parameters
    ----------
    skill_library : SkillLibrary or None
        Where to save recorded macros as reusable skills.
    save_dir : str
        Directory to persist raw macro JSON files.
    min_steps : int
        Minimum steps to consider a recording valid.
    merge_typing : bool
        Merge consecutive key-presses into TYPE_TEXT actions.
    """

    def __init__(
        self,
        skill_library: Any = None,
        save_dir: str = "~/.mycelium/macros",
        min_steps: int = 2,
        merge_typing: bool = True,
    ) -> None:
        self._skill_library = skill_library
        self.save_dir = Path(save_dir).expanduser()
        self.min_steps = min_steps
        self.merge_typing = merge_typing

        self._recording = False
        self._current_name: Optional[str] = None
        self._steps: List[ActionStep] = []
        self._lock = threading.Lock()
        self._listener_keyboard: Any = None
        self._listener_mouse: Any = None
        self._sequences: List[MacroSequence] = []

    @property
    def available(self) -> bool:
        """``True`` when ``pynput`` is installed."""
        try:
            import pynput  # noqa: F401  # type: ignore
            return True
        except ImportError:
            return False

    @property
    def recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self, name: str = "macro") -> None:
        """Begin recording user actions.

        Parameters
        ----------
        name : str
            Name to give the resulting :class:`MacroSequence`.
        """
        if self._recording:
            _logger.warning("MacroRecorder: already recording — stop first")
            return
        self._current_name = name
        self._steps = []
        self._recording = True
        _logger.info("MacroRecorder: started recording %r", name)

        if self.available:
            self._start_listeners()
        else:
            _logger.info("MacroRecorder: pynput not available — use record_step() manually")

    def stop_recording(self) -> Optional[MacroSequence]:
        """Stop recording and return the captured :class:`MacroSequence`.

        Returns ``None`` when fewer than ``min_steps`` were captured.
        """
        if not self._recording:
            return None
        self._recording = False
        self._stop_listeners()

        with self._lock:
            steps = list(self._steps)

        if len(steps) < self.min_steps:
            _logger.info("MacroRecorder: too few steps (%d < %d)", len(steps), self.min_steps)
            return None

        if self.merge_typing:
            steps = self._merge_typing(steps)

        seq = MacroSequence(
            name=self._current_name or "macro",
            steps=steps,
            description=self._auto_describe(steps),
        )
        self._sequences.append(seq)
        _logger.info("MacroRecorder: captured %r (%d steps, %.1fs)", seq.name, len(seq.steps), seq.duration)
        return seq

    def record_step(self, step: ActionStep) -> None:
        """Manually push a step (useful when pynput is unavailable)."""
        if not self._recording:
            return
        with self._lock:
            self._steps.append(step)

    def record_text_sequence(
        self,
        name: str,
        steps: List[Dict[str, Any]],
    ) -> MacroSequence:
        """Build a MacroSequence from a list of step dicts (for testing/scripting)."""
        seq = MacroSequence(
            name=name,
            steps=[ActionStep(**{k: v for k, v in s.items() if k in ActionStep.__dataclass_fields__}) for s in steps],
        )
        seq.description = self._auto_describe(seq.steps)
        self._sequences.append(seq)
        return seq

    # ------------------------------------------------------------------
    # Persistence + SkillLibrary
    # ------------------------------------------------------------------

    def save_sequence(self, seq: MacroSequence) -> str:
        """Save *seq* to disk as JSON. Returns file path."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in seq.name)
        path = self.save_dir / f"{safe_name}_{int(seq.created_at)}.json"
        path.write_text(json.dumps(seq.to_dict(), indent=2))
        _logger.info("MacroRecorder: saved sequence to %s", path)
        return str(path)

    def load_sequences(self) -> List[MacroSequence]:
        """Load all sequences from save_dir."""
        if not self.save_dir.is_dir():
            return []
        seqs = []
        for f in self.save_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                seqs.append(MacroSequence.from_dict(data))
            except Exception as exc:
                _logger.debug("MacroRecorder: failed to load %s: %s", f, exc)
        return seqs

    def save_to_skill_library(self, seq: MacroSequence) -> bool:
        """Register *seq* as a callable Skill in the SkillLibrary.

        The skill, when invoked, replays the action sequence using
        :class:`~physml.screen_agent.ScreenAgent`.

        Returns ``True`` on success.
        """
        lib = self._get_skill_library()
        if lib is None:
            return False
        try:
            # Build a replayable closure over the sequence steps
            steps_snapshot = [s.to_dict() for s in seq.steps]

            def _replay(**kwargs: Any) -> str:
                return _replay_sequence(steps_snapshot)

            lib.register(
                name=seq.name,
                fn=_replay,
                tags=["macro", "recorded"] + seq.tags,
                description=seq.description or seq.summarise(),
            )
            _logger.info("MacroRecorder: saved %r to SkillLibrary", seq.name)
            return True
        except Exception as exc:
            _logger.warning("MacroRecorder.save_to_skill_library error: %s", exc)
            return False

    @property
    def sequences(self) -> List[MacroSequence]:
        return list(self._sequences)

    def status(self) -> Dict[str, Any]:
        return {
            "recording": self._recording,
            "pynput_available": self.available,
            "sequences_captured": len(self._sequences),
            "current_steps": len(self._steps),
        }

    # ------------------------------------------------------------------
    # Listener wiring (pynput)
    # ------------------------------------------------------------------

    def _start_listeners(self) -> None:
        try:
            from pynput import mouse as _mouse, keyboard as _keyboard  # type: ignore

            def on_click(x, y, button, pressed):
                if not self._recording:
                    return False  # stop listener
                if pressed:
                    atype = ActionType.DOUBLE_CLICK if getattr(button, "name", "") == "middle" else ActionType.CLICK
                    if str(button) == "Button.right":
                        atype = ActionType.RIGHT_CLICK
                    app, title = _get_active_window_name()
                    with self._lock:
                        self._steps.append(ActionStep(
                            action_type=atype, x=x, y=y,
                            app_name=app, window_title=title,
                            metadata={"button": str(button)},
                        ))

            def on_scroll(x, y, dx, dy):
                if not self._recording:
                    return False
                app, title = _get_active_window_name()
                with self._lock:
                    self._steps.append(ActionStep(
                        action_type=ActionType.SCROLL, x=x, y=y,
                        app_name=app, window_title=title,
                        metadata={"dx": dx, "dy": dy},
                    ))

            def on_key_press(key):
                if not self._recording:
                    return False
                key_name = getattr(key, "char", None) or str(key)
                app, title = _get_active_window_name()
                with self._lock:
                    self._steps.append(ActionStep(
                        action_type=ActionType.KEY_PRESS, key=key_name,
                        app_name=app, window_title=title,
                    ))

            self._listener_mouse = _mouse.Listener(on_click=on_click, on_scroll=on_scroll)
            self._listener_keyboard = _keyboard.Listener(on_press=on_key_press)
            self._listener_mouse.start()
            self._listener_keyboard.start()
        except Exception as exc:
            _logger.debug("MacroRecorder: pynput listener start error: %s", exc)

    def _stop_listeners(self) -> None:
        for listener in (self._listener_mouse, self._listener_keyboard):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
        self._listener_mouse = None
        self._listener_keyboard = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_skill_library(self) -> Any:
        if self._skill_library is None:
            try:
                from physml.skill_library import SkillLibrary
                self._skill_library = SkillLibrary()
            except Exception:
                pass
        return self._skill_library

    @staticmethod
    def _merge_typing(steps: List[ActionStep]) -> List[ActionStep]:
        """Merge consecutive KEY_PRESS alphanumeric events into TYPE_TEXT."""
        merged: List[ActionStep] = []
        i = 0
        while i < len(steps):
            s = steps[i]
            if s.action_type == ActionType.KEY_PRESS and s.key and len(s.key) == 1:
                # Collect consecutive single-char keypresses
                chars = [s.key]
                j = i + 1
                while j < len(steps) and steps[j].action_type == ActionType.KEY_PRESS and steps[j].key and len(steps[j].key) == 1:
                    chars.append(steps[j].key)
                    j += 1
                merged.append(ActionStep(
                    action_type=ActionType.TYPE_TEXT,
                    text="".join(chars),
                    timestamp=s.timestamp,
                    app_name=s.app_name,
                    window_title=s.window_title,
                ))
                i = j
            else:
                merged.append(s)
                i += 1
        return merged

    @staticmethod
    def _auto_describe(steps: List[ActionStep]) -> str:
        if not steps:
            return "Empty macro"
        apps = list(dict.fromkeys(s.app_name for s in steps))
        action_counts = {}
        for s in steps:
            action_counts[s.action_type] = action_counts.get(s.action_type, 0) + 1
        top_actions = sorted(action_counts, key=action_counts.get, reverse=True)[:3]  # type: ignore
        return (
            f"{len(steps)} steps in {', '.join(apps[:2])}; "
            f"actions: {', '.join(top_actions)}"
        )


def _get_active_window_name() -> Tuple[str, str]:
    """Best-effort active window detection (cross-platform)."""
    import platform
    try:
        if platform.system() == "Linux":
            import subprocess
            r = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                               capture_output=True, text=True, timeout=1)
            title = r.stdout.strip()
            app = title.split(" — ")[-1].split(" - ")[-1][:30].strip() or "unknown"
            return app, title[:80]
        elif platform.system() == "Darwin":
            import subprocess
            script = 'tell app "System Events" to get name of first process whose frontmost is true'
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=1)
            return r.stdout.strip()[:30], ""
        elif platform.system() == "Windows":
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            return title[:30], title[:80]
    except Exception:
        pass
    return "unknown", ""


def _replay_sequence(steps_dicts: List[dict]) -> str:
    """Replay a recorded sequence using ScreenAgent."""
    try:
        from physml.screen_agent import ScreenAgent
        agent = ScreenAgent()
        replayed = 0
        for d in steps_dicts:
            atype = d.get("action_type", "")
            x, y = d.get("x"), d.get("y")
            if atype == ActionType.CLICK and x is not None:
                agent.click(x, y)
                replayed += 1
            elif atype == ActionType.TYPE_TEXT and d.get("text"):
                agent.type_text(d["text"])
                replayed += 1
            elif atype == ActionType.KEY_PRESS and d.get("key"):
                agent.hotkey(d["key"])
                replayed += 1
            time.sleep(0.05)
        return f"Replayed {replayed}/{len(steps_dicts)} steps"
    except Exception as exc:
        return f"Replay error: {exc}"
