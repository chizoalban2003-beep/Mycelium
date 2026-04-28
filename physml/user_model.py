"""physml.user_model — Unified representation of the user.

:class:`UserModel` is the central nervous system of Mycelium's understanding
of who the user is, what they care about, and how they work.  It aggregates:

* :class:`~physml.user_profile.UserProfileLearner` — explicit preferences + interaction history
* :class:`~physml.digital_soul.DigitalSoul` — personality, mood, life events
* :class:`~physml.personalisation.PersonalisationManager` — config preferences
* :class:`~physml.vector_memory.VectorMemory` — semantic memory of conversations/docs
* :class:`~physml.knowledge_graph.KnowledgeGraph` — factual knowledge about the user
* :class:`~physml.screen_observer.ScreenObserver` — current screen context
* :class:`~physml.macro_recorder.MacroRecorder` — recorded behavioral patterns

Everything flows through :meth:`update` and everything is queryable through
:meth:`current_context`, :meth:`behavioral_patterns`, and
:meth:`inject_into_prompt`.

Usage::

    from physml.user_model import UserModel

    model = UserModel()
    model.update({"type": "interaction", "intent": "train", "topic": "sales"})
    model.update({"type": "screen", "app": "VSCode", "description": "editing main.py"})

    print(model.current_context())
    # {"app": "VSCode", "mood": "focused", "top_topics": ["sales", "code"], ...}

    # Inject into LLM system prompt
    extra = model.inject_into_prompt()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from physml._log import get_logger

_logger = get_logger(__name__)


class UserModel:
    """Unified, continuously-updated model of the user.

    Parameters
    ----------
    user_profile : UserProfileLearner or None
    digital_soul : DigitalSoul or None
    personalisation : PersonalisationManager or None
    vector_memory : VectorMemory or None
    knowledge_graph : KnowledgeGraph or None
    screen_observer : ScreenObserver or None
    macro_recorder : MacroRecorder or None
    persist_dir : str
        Base directory for all sub-system persistence files.
    """

    def __init__(
        self,
        user_profile: Any = None,
        digital_soul: Any = None,
        personalisation: Any = None,
        vector_memory: Any = None,
        knowledge_graph: Any = None,
        screen_observer: Any = None,
        macro_recorder: Any = None,
        persist_dir: str = "~/.mycelium",
    ) -> None:
        self._up = user_profile
        self._soul = digital_soul
        self._pers = personalisation
        self._vm = vector_memory
        self._kg = knowledge_graph
        self._screen = screen_observer
        self._macro = macro_recorder
        self._persist_dir = persist_dir

        # Fast in-memory state
        self._current_app: str = "unknown"
        self._current_window: str = ""
        self._session_start: float = time.time()
        self._event_log: List[Dict[str, Any]] = []  # last 500 events

    # ------------------------------------------------------------------
    # Lazy subsystem init
    # ------------------------------------------------------------------

    def _get_up(self) -> Any:
        if self._up is None:
            try:
                from physml.user_profile import UserProfileLearner
                self._up = UserProfileLearner(
                    profile_path=f"{self._persist_dir}/profile.json",
                    auto_save=True,
                )
            except Exception as exc:
                _logger.debug("UserModel: UserProfileLearner unavailable: %s", exc)
        return self._up

    def _get_soul(self) -> Any:
        if self._soul is None:
            try:
                from physml.digital_soul import DigitalSoul
                self._soul = DigitalSoul(soul_path=f"{self._persist_dir}/soul.json")
            except Exception as exc:
                _logger.debug("UserModel: DigitalSoul unavailable: %s", exc)
        return self._soul

    def _get_pers(self) -> Any:
        if self._pers is None:
            try:
                from physml.personalisation import PersonalisationManager
                self._pers = PersonalisationManager(
                    config_path=f"{self._persist_dir}/config.json",
                )
            except Exception as exc:
                _logger.debug("UserModel: PersonalisationManager unavailable: %s", exc)
        return self._pers

    def _get_vm(self) -> Any:
        if self._vm is None:
            try:
                from physml.vector_memory import VectorMemory
                self._vm = VectorMemory(
                    persist_path=f"{self._persist_dir}/vector_memory.json"
                )
            except Exception as exc:
                _logger.debug("UserModel: VectorMemory unavailable: %s", exc)
        return self._vm

    def _get_kg(self) -> Any:
        if self._kg is None:
            try:
                from physml.knowledge_graph import KnowledgeGraph
                self._kg = KnowledgeGraph()
            except Exception as exc:
                _logger.debug("UserModel: KnowledgeGraph unavailable: %s", exc)
        return self._kg

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, event: Dict[str, Any]) -> None:
        """Process an event and update all relevant subsystems.

        Parameters
        ----------
        event : dict
            Must contain ``"type"`` key. Recognised types:

            * ``"interaction"`` — {intent, feedback, topic, metadata}
            * ``"screen"`` — {app, window, description}
            * ``"fact"`` — {subject, predicate, object}
            * ``"preference"`` — {key, value}
            * ``"goal_completed"`` — {goal_description, steps}
            * ``"text"`` — {text, source} — free-form text to memorize
        """
        event_type = event.get("type", "unknown")

        # Keep event log (capped at 500)
        self._event_log.append({**event, "_ts": time.time()})
        if len(self._event_log) > 500:
            self._event_log = self._event_log[-500:]

        try:
            if event_type == "interaction":
                self._handle_interaction(event)
            elif event_type == "screen":
                self._handle_screen(event)
            elif event_type == "fact":
                self._handle_fact(event)
            elif event_type == "preference":
                self._handle_preference(event)
            elif event_type == "goal_completed":
                self._handle_goal_completed(event)
            elif event_type == "text":
                self._handle_text(event)
        except Exception as exc:
            _logger.debug("UserModel.update error for type=%r: %s", event_type, exc)

    def current_context(self) -> Dict[str, Any]:
        """Return a snapshot of the user's current context."""
        ctx: Dict[str, Any] = {
            "app": self._current_app,
            "window": self._current_window,
            "session_seconds": int(time.time() - self._session_start),
        }

        # From UserProfileLearner
        up = self._get_up()
        if up is not None:
            try:
                ctx["top_topics"] = [t for t, _ in up.top_topics(5)]
                ctx["peak_hour"] = up.most_active_hour()
                ctx["feedback_score"] = round(up.feedback_score(), 2)
            except Exception:
                pass

        # From DigitalSoul
        soul = self._get_soul()
        if soul is not None:
            try:
                ctx["mood"] = soul.mood
                ctx["name"] = soul.name
            except Exception:
                pass

        # From PersonalisationManager
        pers = self._get_pers()
        if pers is not None:
            try:
                ctx["verbosity"] = pers.get("verbosity", "normal")
                ctx["language"] = pers.get("language", "en")
            except Exception:
                pass

        # Screen observer focus
        if self._screen is not None:
            try:
                ctx["top_apps"] = self._screen.top_apps(3)
            except Exception:
                pass

        return ctx

    def behavioral_patterns(self) -> List[Dict[str, Any]]:
        """Return a list of detected behavioral patterns.

        Each entry has: {description, frequency, app, action_type}
        """
        patterns: List[Dict[str, Any]] = []

        # From UserProfileLearner — topic frequency
        up = self._get_up()
        if up is not None:
            try:
                for topic, count in up.top_topics(5):
                    patterns.append({
                        "description": f"Frequently discusses {topic!r}",
                        "frequency": count,
                        "source": "interaction_history",
                    })
            except Exception:
                pass

        # From ScreenObserver — app usage
        if self._screen is not None:
            try:
                for app, seconds in self._screen.top_apps(5):
                    patterns.append({
                        "description": f"Spends time in {app!r}",
                        "seconds": round(seconds),
                        "source": "screen_activity",
                    })
            except Exception:
                pass

        # From MacroRecorder — recorded sequences
        if self._macro is not None:
            try:
                for seq in self._macro.sequences[-5:]:
                    patterns.append({
                        "description": seq.summarise(),
                        "source": "macro_recording",
                        "apps": seq.apps_used,
                    })
            except Exception:
                pass

        return patterns

    def remember_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Store a fact directly into KnowledgeGraph + VectorMemory."""
        self.update({"type": "fact", "subject": subject, "predicate": predicate, "object": obj})

    def recall(self, query: str, k: int = 5) -> List[str]:
        """Semantic search over the user's memory.

        Returns top-k relevant text snippets.
        """
        vm = self._get_vm()
        if vm is None:
            return []
        try:
            results = vm.search(query, k=k)
            return [r.text for r in results]
        except Exception as exc:
            _logger.debug("UserModel.recall error: %s", exc)
            return []

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference in PersonalisationManager + UserProfileLearner."""
        pers = self._get_pers()
        if pers is not None:
            try:
                pers.set(key, value)
            except Exception:
                pass
        up = self._get_up()
        if up is not None:
            try:
                up.set_preference(key, value)
            except Exception:
                pass

    def inject_into_prompt(self) -> str:
        """Return a concise system-prompt snippet describing the user.

        Designed to be appended to the LLM system prompt so every response
        is contextualised to the user's current state.
        """
        lines: List[str] = []
        ctx = self.current_context()

        name = ctx.get("name")
        if name and name != "Myco":
            lines.append(f"User name: {name}")

        mood = ctx.get("mood")
        if mood:
            lines.append(f"Current mood: {mood}")

        app = ctx.get("app", "unknown")
        if app and app != "unknown":
            lines.append(f"Currently using: {app}")

        topics = ctx.get("top_topics", [])
        if topics:
            lines.append(f"Frequent topics: {', '.join(topics[:4])}")

        verbosity = ctx.get("verbosity", "normal")
        if verbosity != "normal":
            lines.append(f"Preferred verbosity: {verbosity}")

        patterns = self.behavioral_patterns()
        if patterns:
            top = patterns[0]["description"]
            lines.append(f"Behavioral note: {top}")

        # Extra from PersonalisationManager
        pers = self._get_pers()
        if pers is not None:
            try:
                extra = pers.system_prompt_additions()
                if extra:
                    lines.append(extra)
            except Exception:
                pass

        if not lines:
            return ""
        return "User context:\n" + "\n".join(f"  • {line}" for line in lines)

    def summary(self) -> Dict[str, Any]:
        """Return full user model summary."""
        return {
            "context": self.current_context(),
            "behavioral_patterns": self.behavioral_patterns()[:5],
            "event_log_size": len(self._event_log),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "user_profile": self._get_up() is not None,
            "digital_soul": self._get_soul() is not None,
            "personalisation": self._get_pers() is not None,
            "vector_memory": self._get_vm() is not None,
            "knowledge_graph": self._get_kg() is not None,
            "screen_observer": self._screen is not None,
            "macro_recorder": self._macro is not None,
            "events_processed": len(self._event_log),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_interaction(self, event: Dict[str, Any]) -> None:
        up = self._get_up()
        if up is not None:
            up.record_interaction(
                intent=event.get("intent", "unknown"),
                feedback=event.get("feedback", "neutral"),
                topic=event.get("topic", "general"),
                metadata=event.get("metadata", {}),
            )
        soul = self._get_soul()
        if soul is not None:
            soul.record_event("interaction", description=event.get("topic", ""))

    def _handle_screen(self, event: Dict[str, Any]) -> None:
        self._current_app = event.get("app", self._current_app)
        self._current_window = event.get("window", self._current_window)
        desc = event.get("description", "")
        if desc:
            vm = self._get_vm()
            if vm is not None:
                vm.add(desc, metadata={"type": "screen", "app": self._current_app, "ts": time.time()})

    def _handle_fact(self, event: Dict[str, Any]) -> None:
        kg = self._get_kg()
        if kg is not None:
            subj = event.get("subject", "user")
            pred = event.get("predicate", "knows")
            obj = event.get("object", "")
            if obj:
                kg.add_node(subj, node_type="entity")
                kg.add_node(obj, node_type="entity")
                kg.add_edge(subj, obj, relation=pred)
        vm = self._get_vm()
        if vm is not None:
            text = f"{event.get('subject','user')} {event.get('predicate','')} {event.get('object','')}"
            vm.add(text.strip(), metadata={"type": "fact"})

    def _handle_preference(self, event: Dict[str, Any]) -> None:
        key = event.get("key")
        value = event.get("value")
        if key and value is not None:
            self.set_preference(key, value)

    def _handle_goal_completed(self, event: Dict[str, Any]) -> None:
        soul = self._get_soul()
        if soul is not None:
            soul.record_event("goal_completed", description=event.get("goal_description", ""))
            soul.increment_stat("goals_completed", 1)
        up = self._get_up()
        if up is not None:
            up.record_interaction(intent="goal_completed", feedback="positive",
                                  topic=event.get("goal_description", "goal")[:40])

    def _handle_text(self, event: Dict[str, Any]) -> None:
        text = event.get("text", "")
        if not text:
            return
        vm = self._get_vm()
        if vm is not None:
            vm.add(text[:2000], metadata={"type": "text", "source": event.get("source", ""), "ts": time.time()})

    def __repr__(self) -> str:
        ctx = self._current_app
        return f"UserModel(app={ctx!r}, events={len(self._event_log)})"
