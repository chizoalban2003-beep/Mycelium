"""Stage 120 — MyceliumCompanion: top-level digital companion integration.

THE top-level integration class.  Wraps every major subsystem into a single
``chat(text)`` interface — the product.

Subsystems wired together:
* :class:`~physml.mycelium_system.MyceliumSystem` — core ML engine
* :class:`~physml.conversation.ConversationManager` — dialogue history
* :class:`~physml.user_profile.UserProfileLearner` — preference learning
* :class:`~physml.response_formatter.ResponseFormatter` — NL responses
* :class:`~physml.local_executor.LocalTaskExecutor` — OS tasks
* :class:`~physml.nl_router.NaturalLanguageRouter` — intent routing
* :class:`~physml.plugin_registry.PluginRegistry` — user plugins
* :class:`~physml.device_monitor.DeviceMonitor` — device telemetry
* :class:`~physml.proactive_advisor.ProactiveAdvisor` — proactive alerts
* :class:`~physml.digital_soul.DigitalSoul` — agent identity
* :class:`~physml.secure_vault.SecureVault` — secrets store
* :class:`~physml.doc_processor.DocumentProcessor` — document ingestion
* :class:`~physml.knowledge_extractor.KnowledgeExtractor` — auto-fact extraction
* :class:`~physml.feedback_loop.FeedbackLoop` — live model corrections
* :class:`~physml.goal_engine.GoalEngine` — autonomous goal execution (Stage 137)
* :class:`~physml.personalisation.PersonalisationManager` — manual + auto config
* :class:`~physml.file_watcher.FileWatcher` — proactive file learning
* :class:`~physml.notifier.Notifier` — desktop notifications
* :class:`~physml.permission_manager.PermissionManager` — OS action gating
* :class:`~physml.screen_agent.ScreenAgent` — screen/UI automation
* :class:`~physml.browser_agent.BrowserAgent` — browser control
* :class:`~physml.task_decomposer.TaskDecomposer` — LLM-driven task planning

Usage
-----
::

    from physml.companion import MyceliumCompanion

    companion = MyceliumCompanion(name="Myco", data_dir="~/.mycelium")
    companion.start()

    response = companion.chat("predict my sales for tomorrow")
    response = companion.chat("read quarterly_report.csv and tell me the trends")
    response = companion.chat("what have you learned about me?")

    print(companion.status())
    companion.stop()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class MyceliumCompanion:
    """The Mycelium digital companion — the top-level product.

    Parameters
    ----------
    name : str, default "Myco"
        The agent's name (written into :class:`~physml.digital_soul.DigitalSoul`).
    data_dir : str, default "~/.mycelium"
        Root directory for all persisted state.
    verbosity : str, default "normal"
        Response verbosity: ``"concise"``, ``"normal"``, or ``"verbose"``.
    enable_device_monitor : bool, default False
        Start the background device monitor.
    enable_proactive_advisor : bool, default False
        Start the background proactive advisor.
    """

    def __init__(
        self,
        name: str = "Myco",
        data_dir: str = "~/.mycelium",
        verbosity: str = "normal",
        enable_device_monitor: bool = False,
        enable_proactive_advisor: bool = False,
        llm_api_key: Optional[str] = None,
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir).expanduser()
        self.verbosity = verbosity
        self._enable_device_monitor = enable_device_monitor
        self._enable_proactive_advisor = enable_proactive_advisor
        self._llm_api_key = llm_api_key
        self._started = False

        # Subsystems (initialised in start())
        self.soul: Any = None
        self.profile: Any = None
        self.formatter: Any = None
        self.router: Any = None
        self.llm: Any = None              # LLMIntegration (Stage 121)
        self.conversation: Any = None
        self.executor: Any = None
        self.doc_processor: Any = None
        self.plugin_registry: Any = None
        self.device_monitor: Any = None
        self.advisor: Any = None
        self.vault: Any = None
        self.model_manager: Any = None    # ModelManager (Stage 123)
        self.tool_bridge: Any = None      # ToolBridge (Stage 124)
        self.vector_memory: Any = None    # VectorMemory (Stage 126)
        self.knowledge_extractor: Any = None   # Stage 134
        self.feedback_loop: Any = None         # Stage 135
        self.personalisation: Any = None       # Stage 136
        self.file_watcher: Any = None          # Stage 132
        self.notifier: Any = None              # Stage 133
        self.permission_manager: Any = None    # Stage 131
        self.screen_agent: Any = None          # Stage 129
        self.browser_agent: Any = None         # Stage 130
        self.task_decomposer: Any = None       # Stage 92 + LLM
        self.goal_engine: Any = None           # Stage 137 — autonomous goal loop
        self.scheduler: Any = None             # Stage 138 — recurring scheduled goals
        self._last_features: Optional[List[float]] = None  # for feedback loop

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load all subsystems and start background threads."""
        if self._started:
            return

        # Auto-load .env if python-dotenv is installed — zero config for users
        try:
            from dotenv import load_dotenv as _ld
            _ld(override=False)   # don't override already-set env vars
        except ImportError:
            pass

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # DigitalSoul
        from physml.digital_soul import DigitalSoul
        self.soul = DigitalSoul(
            soul_path=str(self.data_dir / "soul.json"),
            name=self.name,
        )
        self.soul.name = self.name

        # UserProfileLearner
        from physml.user_profile import UserProfileLearner
        self.profile = UserProfileLearner(
            profile_path=str(self.data_dir / "profile.json"),
        )

        # ResponseFormatter (use verbosity from profile if set)
        verbosity = self.profile.get_preference("verbosity", self.verbosity)
        from physml.response_formatter import ResponseFormatter
        self.formatter = ResponseFormatter(verbosity=verbosity)

        # NaturalLanguageRouter
        from physml.nl_router import NaturalLanguageRouter, Intent
        self.router = _build_default_router()

        # ConversationManager
        from physml.conversation import ConversationManager
        self.conversation = ConversationManager(
            max_history=100,
            router=self.router,
        )

        # LocalTaskExecutor (read-only by default)
        from physml.local_executor import LocalTaskExecutor, ExecutionPolicy
        self.executor = LocalTaskExecutor(
            policy=ExecutionPolicy(read_only=True),
        )

        # DocumentProcessor
        from physml.doc_processor import DocumentProcessor
        self.doc_processor = DocumentProcessor()

        # PluginRegistry
        from physml.plugin_registry import PluginRegistry
        self.plugin_registry = PluginRegistry(
            plugin_dir=str(self.data_dir / "plugins"),
        )
        self.plugin_registry.load_all()

        # SecureVault (locked until user calls unlock)
        from physml.secure_vault import SecureVault
        self.vault = SecureVault(vault_path=str(self.data_dir / "vault.enc"))

        # DeviceMonitor
        from physml.device_monitor import DeviceMonitor
        self.device_monitor = DeviceMonitor(poll_interval=60)
        if self._enable_device_monitor:
            self.device_monitor.start_background()

        # ProactiveAdvisor
        from physml.proactive_advisor import ProactiveAdvisor
        self.advisor = ProactiveAdvisor()
        if self._enable_proactive_advisor:
            self.advisor.enable_background(interval=300)

        # Log a start event
        self.soul.record_event(
            "companion_started",
            details={"data_dir": str(self.data_dir)},
            description=f"{self.name} started",
        )
        self.soul.save()

        # LLM Integration (optional — graceful fallback if SDK absent / no key)
        from physml.llm_integration import LLMIntegration
        self.llm = LLMIntegration(api_key=self._llm_api_key)
        if self.llm.available:
            _logger.info("MyceliumCompanion: LLM backbone active (%s)", self.llm.config.model)
        else:
            _logger.info("MyceliumCompanion: LLM not available — using rule-based NL router")

        # ModelManager — persistent physics-ML model (Stage 123)
        from physml.model_manager import ModelManager
        self.model_manager = ModelManager(
            model_dir=str(self.data_dir / "model"),
        )
        loaded = self.model_manager.load()
        if loaded:
            ms = self.model_manager.status()
            _logger.info(
                "MyceliumCompanion: restored model (rows=%d, features=%d)",
                ms["n_training_rows"], ms["n_features"],
            )
            self.soul.record_event(
                "model_restored",
                description=f"Restored model trained on {ms['n_training_rows']} rows",
            )

        # ToolBridge — LLM → local executor bridge (Stage 124)
        from physml.tool_bridge import ToolBridge
        self.tool_bridge = ToolBridge(companion=self)

        # VectorMemory — semantic memory (Stage 126)
        from physml.vector_memory import VectorMemory
        self.vector_memory = VectorMemory(
            max_entries=2000,
            persist_path=str(self.data_dir / "vector_memory.json"),
        )
        _logger.info(
            "MyceliumCompanion: vector memory has %d entries (%s backend)",
            len(self.vector_memory), self.vector_memory.active_backend,
        )

        # PermissionManager — gate OS actions
        from physml.permission_manager import PermissionManager
        self.permission_manager = PermissionManager(
            config_path=str(self.data_dir / "permissions.json"),
        )

        # PersonalisationManager — manual + auto config
        from physml.personalisation import PersonalisationManager
        from physml.knowledge_graph import KnowledgeGraph
        _kg = KnowledgeGraph()
        self.personalisation = PersonalisationManager(
            config_path=str(self.data_dir / "config.json"),
            user_profile=self.profile,
            knowledge_graph=_kg,
        )

        # KnowledgeExtractor — auto-extract facts from chat
        from physml.knowledge_extractor import KnowledgeExtractor
        self.knowledge_extractor = KnowledgeExtractor(
            knowledge_graph=_kg,
            vector_memory=self.vector_memory,
            llm=self.llm if self.llm and self.llm.available else None,
        )

        # FeedbackLoop — live model corrections
        from physml.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop(
            model_manager=self.model_manager,
            vector_memory=self.vector_memory,
        )

        # Notifier — desktop notifications
        from physml.notifier import Notifier
        self.notifier = Notifier(app_name=self.name)

        # FileWatcher — proactive learning from new files
        from physml.file_watcher import FileWatcher
        watch_dirs = self.personalisation.get("watch_dirs", []) or []
        self.file_watcher = FileWatcher(
            watch_dirs=watch_dirs,
            callback=self._on_new_file,
        )
        if watch_dirs:
            self.file_watcher.start()

        # ScreenAgent — screenshot / UI automation (lazy, no auto-start)
        from physml.screen_agent import ScreenAgent
        self.screen_agent = ScreenAgent(
            screenshot_dir=str(self.data_dir / "screenshots"),
        )

        # BrowserAgent — browser automation (lazy, no auto-start)
        from physml.browser_agent import BrowserAgent
        self.browser_agent = BrowserAgent(headless=True)

        # TaskDecomposer — LLM-driven goal planning (Stage 92 + LLM)
        from physml.task_decomposer import TaskDecomposer
        self.task_decomposer = TaskDecomposer(
            llm=self.llm if self.llm and self.llm.available else None,
        )

        # Apply personalisation to formatter verbosity
        verbosity = self.personalisation.get("verbosity", self.verbosity)
        if hasattr(self.formatter, "verbosity"):
            self.formatter.verbosity = verbosity

        # GoalEngine — autonomous goal execution loop (Stage 137)
        from physml.goal_engine import GoalEngine
        self.goal_engine = GoalEngine(
            task_decomposer=self.task_decomposer,
            companion=self,
            notifier=self.notifier,
            permission_manager=self.permission_manager,
            llm=self.llm if self.llm and self.llm.available else None,
            state_dir=str(self.data_dir / "goals"),
        )

        # ScheduledGoals — recurring goal scheduler (Stage 138)
        from physml.scheduled_goals import ScheduledGoals
        self.scheduler = ScheduledGoals(
            goal_engine=self.goal_engine,
            notifier=self.notifier,
            state_dir=str(self.data_dir / "schedule"),
        )

        self._started = True
        _logger.info("MyceliumCompanion %r started (v0.31.0)", self.name)

    def personalise(self, key: str, value: Any) -> str:
        """Set a personalisation preference and return confirmation.

        Parameters
        ----------
        key : str
            One of: name, language, verbosity, agent_name, wake_word,
            timezone, notifications, watch_dirs, theme, …
        value : any
            The new value.
        """
        if not self._started:
            self.start()
        try:
            self.personalisation.set(key, value)
            # Apply immediately if we can
            if key == "verbosity" and hasattr(self.formatter, "verbosity"):
                self.formatter.verbosity = value
            elif key == "watch_dirs" and isinstance(value, list):
                if self.file_watcher is not None:
                    self.file_watcher.stop()
                    self.file_watcher.watch_dirs = value
                    if value:
                        self.file_watcher.start()
            return f"Got it — {key} set to {value!r}."
        except ValueError as exc:
            return f"Invalid value: {exc}"

    def plan(self, goal: str) -> str:
        """Decompose a goal into a step-by-step plan using Claude (or heuristics).

        Parameters
        ----------
        goal : str
            High-level goal description.
        """
        if not self._started:
            self.start()
        if self.task_decomposer is not None:
            return self.task_decomposer.decompose_and_summarise(goal)
        return f"Plan: {goal}"

    def add_goal(self, description: str, run_immediately: bool = False) -> str:
        """Add an autonomous goal for the GoalEngine to execute.

        Parameters
        ----------
        description : str
            Natural-language goal (e.g. "Read sales.csv and summarise trends").
        run_immediately : bool
            When True, execute the goal now (synchronous) instead of queuing.

        Returns
        -------
        str
            Confirmation message with the goal ID.
        """
        if not self._started:
            self.start()
        goal_id = self.goal_engine.add_goal(description, run_immediately=run_immediately)
        if run_immediately:
            goal = self.goal_engine.get(goal_id)
            status = goal.status if goal else "unknown"
            return f"Goal executed — status: {status} (id: {goal_id})"
        return (
            f"Goal queued (id: {goal_id}). "
            f"I'll work on it in the background. "
            f"Check progress with: companion.goals()"
        )

    def schedule_goal(self, description: str, schedule: str = "daily") -> str:
        """Register a recurring goal on a schedule.

        Parameters
        ----------
        description : str
            Natural-language goal.
        schedule : str
            When to run: ``"hourly"``, ``"daily"``, ``"weekly"``,
            ``"every 30 minutes"``, ``"every 2 hours"``, etc.

        Returns
        -------
        str
            Confirmation with the schedule ID.
        """
        if not self._started:
            self.start()
        try:
            sid = self.scheduler.add(description, schedule=schedule)
            return (
                f"Scheduled (id: {sid}): {description[:50]!r} — {schedule}.\n"
                f"Call companion.scheduler.start() to activate background scheduling."
            )
        except ValueError as exc:
            return f"Invalid schedule: {exc}"

    def goals(self, status: Optional[str] = None) -> str:
        """Return a formatted summary of all goals.

        Parameters
        ----------
        status : str, optional
            Filter by status: "pending", "active", "completed", "failed", "blocked".
        """
        if not self._started:
            self.start()
        from physml.goal_engine import GoalStatus
        filt = GoalStatus(status) if status else None
        records = self.goal_engine.goals(filt)
        if not records:
            label = f" with status={status!r}" if status else ""
            return f"No goals{label}."
        lines = [f"Goals ({len(records)}):"]
        for g in records:
            elapsed = f" ({g.elapsed:.0f}s)" if g.completed_at else ""
            n_steps = len(g.steps)
            done = sum(1 for s in g.steps if s.get("status") == "ok")
            progress = f" [{done}/{n_steps} steps]" if n_steps else ""
            lines.append(f"  [{g.status.value}] {g.id}: {g.description[:60]}{progress}{elapsed}")
        return "\n".join(lines)

    def take_screenshot(self) -> str:
        """Capture the screen and return the saved file path."""
        if not self._started:
            self.start()
        if self.screen_agent is None:
            return "Screen agent not initialised."
        if not self.permission_manager.check("screen.screenshot"):
            return "Permission denied for screen.screenshot."
        path = self.screen_agent.screenshot()
        if path:
            if self.notifier:
                self.notifier.send("Screenshot saved", path)
            return f"Screenshot saved: {path}"
        return "Screenshot failed (install mss or pyautogui)."

    def browse(self, url: str) -> str:
        """Fetch a URL and return its text content."""
        if not self._started:
            self.start()
        if not self.permission_manager.check("browser.navigate"):
            return "Permission denied for browser.navigate."
        result = self.doc_processor.process(url)
        if result.success:
            preview = result.text[:500]
            if self.vector_memory:
                self.vector_memory.add(f"Browsed {url}: {preview}", {"type": "url"})
            return f"Page: {url}\n\n{preview}{'...' if len(result.text) > 500 else ''}"
        return f"Could not fetch {url}: {result.error}"

    def _on_new_file(self, path: str) -> None:
        """Callback fired by FileWatcher when a new file appears."""
        _logger.info("MyceliumCompanion: new file detected — %s", path)
        try:
            result = self.doc_processor.process(path)
            if result.success:
                summary = result.text[:200]
                if self.vector_memory:
                    self.vector_memory.add(
                        f"Auto-ingested file {path}: {summary}",
                        {"type": "auto_file"},
                    )
                if result.df is not None and self.model_manager is not None:
                    # Auto-train on CSV files
                    tr = self.model_manager.train_from_csv(path)
                    if tr.success:
                        self.model_manager.save()
                        if self.notifier:
                            self.notifier.send_success(
                                f"Auto-trained on {path} ({tr.n_rows} rows)"
                            )
                        self.soul.record_event("auto_train", description=tr.message)
                        return
                if self.notifier:
                    self.notifier.send("New file", f"Ingested: {path}")
        except Exception as exc:
            _logger.warning("MyceliumCompanion: _on_new_file error: %s", exc)

    def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._started:
            return
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception:
                pass
        if self.goal_engine is not None:
            try:
                self.goal_engine.stop_loop()
            except Exception:
                pass
        if self.file_watcher is not None:
            try:
                self.file_watcher.stop()
            except Exception:
                pass
        if self.device_monitor is not None:
            try:
                self.device_monitor.stop()
            except Exception as e:
                _logger.warning("MyceliumCompanion: device_monitor stop failed: %s", e)
        if self.advisor is not None:
            try:
                self.advisor.disable_background()
            except Exception as e:
                _logger.warning("MyceliumCompanion: advisor stop failed: %s", e)
        try:
            self.soul.record_event("companion_stopped", description=f"{self.name} stopped")
            self.soul.save()
        except Exception as e:
            _logger.warning("MyceliumCompanion: soul save failed: %s", e)
        try:
            self.profile.save()
        except Exception as e:
            _logger.warning("MyceliumCompanion: profile save failed: %s", e)
        self._started = False
        _logger.info("MyceliumCompanion %r stopped", self.name)

    # ------------------------------------------------------------------
    # Main chat interface
    # ------------------------------------------------------------------

    def chat(self, text: str) -> str:
        """Process a natural-language message and return a response.

        Parameters
        ----------
        text : str
            User input.

        Returns
        -------
        str
            Formatted agent response.
        """
        if not self._started:
            self.start()

        text = str(text).strip()
        if not text:
            return ""

        # Record user turn (also routes intent)
        turn = self.conversation.add_turn("user", text)
        intent = turn.intent or "unknown"
        entities = turn.entities

        # Profile: record interaction
        self.profile.record_interaction(intent=intent, topic=intent)

        # Auto-extract facts from what the user says
        if self.knowledge_extractor is not None:
            try:
                facts = self.knowledge_extractor.extract_and_store(text)
                if facts:
                    # Persist newly learned name into personalisation
                    for f in facts:
                        if f["predicate"] == "name" and not self.personalisation.get("name"):
                            self.personalisation.set("name", f["object"])
            except Exception as _ke:
                _logger.debug("knowledge_extractor error: %s", _ke)

        # Check if user is giving a correction/feedback
        if self.feedback_loop is not None:
            try:
                feedback_reply = self.feedback_loop.parse_and_record(
                    text, last_features=self._last_features
                )
                if feedback_reply is not None:
                    self.conversation.add_turn("agent", feedback_reply)
                    self.soul.increment_stat("total_interactions")
                    return feedback_reply
            except Exception as _fe:
                _logger.debug("feedback_loop error: %s", _fe)

        # Dispatch by intent
        response = self._dispatch(text, intent, entities)

        # Record agent turn
        self.conversation.add_turn("agent", response)

        # Update soul
        self.soul.record_event(
            "interaction",
            details={"intent": intent},
            description=f"Handled: {intent}",
        )
        self.soul.increment_stat("total_interactions")

        # Check for proactive advice
        advice_str = self._maybe_advice()
        if advice_str:
            response = response + "\n\n" + advice_str

        return response

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, text: str, intent: str, entities: Dict[str, Any]) -> str:
        """Route the message to the appropriate handler.

        When LLM is available, structured actions (predict, train, report, save,
        status, help) still use their deterministic handlers for reliability.
        Unknown / conversational intents go to the LLM.
        """
        paths = entities.get("paths", [])

        # Training with a file path — route to model training, not document view
        if paths and intent == "train":
            return self._handle_train(text, entities)

        # Document read/analyze with a file path
        if paths and intent in ("process", "read", "load", "analyze", "analyse", "unknown"):
            return self._handle_document(paths[0], text)

        if intent == "predict":
            return self._handle_predict(text, entities)
        elif intent == "train":
            return self._handle_train(text, entities)
        elif intent == "report":
            return self._handle_report()
        elif intent == "save":
            return self._handle_save()
        elif intent == "status":
            return self.formatter.format_report(self.status())
        elif intent in ("profile", "learn_about", "about_me"):
            return self._handle_profile_query()
        elif intent == "plan":
            return self.plan(text)
        elif intent in ("goal", "run_goal", "do_goal", "execute_goal"):
            return self.add_goal(text)
        elif intent == "goals":
            return self.goals()
        elif intent in ("browse", "fetch", "open_url"):
            urls = [e for e in entities.get("paths", []) if e.startswith("http")]
            if urls:
                return self.browse(urls[0])
            return self._handle_generic(text)
        elif intent == "screenshot":
            return self.take_screenshot()
        elif intent == "help":
            return _HELP_TEXT
        else:
            # Conversational / open-ended — route to LLM when available
            return self._handle_generic(text)

    def _handle_predict(self, text: str, entities: Dict[str, Any]) -> str:
        numbers = entities.get("numbers", [])
        self.soul.increment_stat("total_predictions")
        self.soul.record_event("prediction", description="User requested prediction")

        # Use ModelManager for real prediction
        if self.model_manager is not None:
            if not self.model_manager.fitted:
                return (
                    f"I don't have a trained model yet. "
                    f"Try: 'train on <yourfile.csv>' so I can learn from your data first."
                )
            if numbers:
                self._last_features = numbers  # for feedback loop
                result = self.model_manager.predict(numbers)
                if result.error:
                    return f"Prediction error: {result.error}"
                # Store in vector memory
                mem_text = (
                    f"Predicted {result.value:.4g} (confidence {result.confidence:.0%}) "
                    f"for features {numbers}"
                )
                if self.vector_memory is not None:
                    self.vector_memory.add(mem_text, {"intent": "predict"})
                return self.formatter.format_prediction(
                    prediction=result.value,
                    confidence=result.confidence,
                    feature_names=result.feature_names or [f"x{i}" for i in range(len(numbers))],
                )
            else:
                return (
                    "To run a prediction, provide feature values. "
                    f"Example: 'predict 1.2 3.4 5.6' "
                    f"(model trained on: {', '.join(self.model_manager._feature_names[:5]) or 'unknown features'})"
                )

        # Fallback if model_manager not initialised
        return "Prediction system not yet initialised. Please restart the companion."

    def _handle_train(self, text: str, entities: Dict[str, Any]) -> str:
        paths = entities.get("paths", [])
        if paths:
            csv_path = paths[0]
            # Try to train the model manager on this file
            if self.model_manager is not None:
                result = self.model_manager.train_from_csv(csv_path)
                if result.success:
                    self.model_manager.save()
                    self.soul.record_event(
                        "model_trained",
                        description=result.message,
                        details={"path": csv_path, "rows": result.n_rows},
                    )
                    self.soul.increment_stat("total_training_rounds")
                    if self.vector_memory is not None:
                        self.vector_memory.add(result.message, {"intent": "train", "path": csv_path})
                    return result.message
                else:
                    return f"Training failed: {result.error or result.message}"
            return self._handle_document(csv_path, text)
        self.soul.record_event("training_requested", description="User requested training")
        return (
            f"To train {self.name}, provide a CSV file path. "
            "Example: 'train on data.csv'"
        )

    def _handle_document(self, path: str, original_text: str) -> str:
        result = self.doc_processor.process(path)
        if not result.success:
            return self.formatter.format_uncertainty(
                f"Could not read document: {result.error}"
            )
        meta = result.metadata
        doc_type = meta.get("type", "document")
        lines = [f"Processed {doc_type}: {path}"]
        if meta.get("rows"):
            lines.append(f"  Rows: {meta['rows']}, Columns: {meta.get('n_columns', '?')}")
        if meta.get("chars"):
            lines.append(f"  Characters: {meta['chars']}")
        if result.df is not None:
            try:
                desc = result.df.describe().to_string()
                lines.append(f"  Statistics:\n{desc}")
            except Exception:
                pass
        # Partial text preview
        if result.text:
            preview = result.text[:300].replace("\n", " ")
            lines.append(f"  Preview: {preview}…" if len(result.text) > 300 else f"  Content: {result.text[:300]}")
        return "\n".join(lines)

    def _handle_report(self) -> str:
        s = self.status()
        return self.formatter.format_report(s)

    def _handle_save(self) -> str:
        saved: List[str] = []
        try:
            self.soul.save()
            saved.append("soul")
        except Exception as e:
            _logger.warning("save soul failed: %s", e)
        try:
            self.profile.save()
            saved.append("profile")
        except Exception as e:
            _logger.warning("save profile failed: %s", e)
        return f"Saved: {', '.join(saved)}." if saved else "Nothing was saved."

    def _handle_profile_query(self) -> str:
        summary = self.profile.summary()
        soul_s = self.soul.summary()
        lines = [
            f"Here is what I know about you:",
            f"  You've had {summary['interaction_count']} interaction(s) with me.",
            f"  Your top topics: {', '.join(summary['top_topics']) or 'none yet'}.",
            f"  Your feedback score: {summary['feedback_score']:.0%}.",
        ]
        prefs = summary.get("preferences", {})
        if prefs:
            lines.append(f"  Preferences: {prefs}")
        lines.append("")
        lines.append(f"About me ({soul_s['name']}):")
        lines.append(f"  Mood: {soul_s['mood']}.")
        stats = soul_s.get("stats", {})
        lines.append(f"  Total predictions: {stats.get('total_predictions', 0)}.")
        lines.append(f"  Days alive: {stats.get('days_alive', 0)}.")
        return "\n".join(lines)

    def _handle_generic(self, text: str) -> str:
        # Use LLM when available
        if self.llm is not None and self.llm.available:
            return self._handle_llm(text)

        # Fallback: rule-based response
        mood = self.soul.mood if self.soul else "curious"
        mood_phrases = {
            "confident": "I'm feeling confident about this.",
            "curious": "I'm curious to learn more.",
            "learning": "I'm still learning.",
            "uncertain": "I'm not entirely sure, but I'll do my best.",
        }
        prefix = mood_phrases.get(mood, "")
        return (
            f"{prefix} You said: \"{text}\"\n"
            f"I understand this is a request. Could you clarify what you'd like me to do? "
            f"Try: 'predict', 'train on <file>', 'show report', or '/help'."
        )

    def _handle_llm(self, text: str) -> str:
        """Route through Claude API for open-ended conversational responses.

        Executes any tool calls Claude requests, then sends results back for
        a final grounded response.
        """
        from physml.llm_integration import LLMMessage
        from physml.tool_bridge import build_tool_definitions

        # Build conversation history — exclude the current user turn since
        # it is passed separately as user_message.
        history: List[LLMMessage] = []
        if self.conversation is not None:
            # Take the last 20 turns; skip the very last one (current user turn)
            prior_turns = list(self.conversation.turns)[:-1][-20:]
            for turn in prior_turns:
                role = "user" if turn.speaker == "user" else "assistant"
                history.append(LLMMessage(role=role, content=turn.text))

        # Inject semantic memory context into system prompt
        memory_context = ""
        if self.vector_memory is not None and len(self.vector_memory) > 0:
            results = self.vector_memory.search(text, k=3)
            if results:
                snippets = "\n".join(f"- {r.text}" for r in results if r.score > 0.1)
                if snippets:
                    memory_context = f"\nRelevant memory:\n{snippets}"

        # Personalisation additions
        personal_extra = ""
        if self.personalisation is not None:
            personal_extra = self.personalisation.system_prompt_additions()

        system = self.llm.build_system_prompt(
            soul=self.soul,
            profile=self.profile,
            extra=(personal_extra + "\n" + memory_context).strip(),
        )

        # First LLM call — may return tool calls
        result = self.llm.chat(
            user_message=text,
            history=history,
            system=system,
            tools=build_tool_definitions(),
        )

        if not result.available:
            _logger.warning("LLM call failed: %s", result.error)
            return (
                "I couldn't process that right now. "
                "Try: 'predict', 'train on <file>', 'show report', or '/help'."
            )

        # Execute tool calls if any
        if result.tool_calls and self.tool_bridge is not None:
            tool_results = self.tool_bridge.execute_all(result.tool_calls)

            # Second LLM call with tool results for grounded response
            result2 = self.llm.chat_with_tool_results(
                tool_call_result_blocks=result.tool_calls,
                tool_results=tool_results,
                history=history,
                user_message=text,
                system=system,
            )
            if result2.available and result2.text:
                # Store in vector memory
                if self.vector_memory is not None:
                    self.vector_memory.add(
                        f"User: {text}\nAssistant: {result2.text[:200]}",
                        {"intent": "llm_tool"},
                    )
                return result2.text

        if result.text:
            # Store exchange in vector memory
            if self.vector_memory is not None:
                self.vector_memory.add(
                    f"User: {text}\nAssistant: {result.text[:200]}",
                    {"intent": "llm_chat"},
                )
            return result.text

        return (
            "I couldn't generate a response. "
            "Try: 'predict', 'train on <file>', 'show report', or '/help'."
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return full system status.

        Returns
        -------
        dict
        """
        s: Dict[str, Any] = {
            "name": self.name,
            "started": self._started,
            "mood": self.soul.mood if self.soul else "unknown",
        }
        if self.soul:
            s.update(self.soul.stats)
        if self.profile:
            s["top_topics"] = self.profile.top_topics(3)
            s["feedback_score"] = round(self.profile.feedback_score(), 3)
        if self.conversation:
            s["conversation_turns"] = len(self.conversation.turns)
        if self.plugin_registry:
            s["plugins_loaded"] = len(self.plugin_registry.loaded)
        if self.personalisation:
            s["user_name"] = self.personalisation.get("name")
            s["verbosity"] = self.personalisation.get("verbosity")
        if self.feedback_loop:
            s["corrections_applied"] = self.feedback_loop.status()["total_applied"]
        if self.knowledge_extractor:
            s["facts_learned"] = self.knowledge_extractor.status()["facts_stored"]
        if self.screen_agent:
            s["screen_automation"] = self.screen_agent.available
        if self.browser_agent:
            s["browser_automation"] = self.browser_agent.available
        if self.file_watcher:
            s["file_watcher"] = self.file_watcher.status()
        if self.goal_engine:
            s["goals"] = self.goal_engine.status()
        if self.scheduler:
            s["scheduled_goals"] = self.scheduler.status()["total"]
        return s

    # ------------------------------------------------------------------
    # Proactive advice
    # ------------------------------------------------------------------

    def _maybe_advice(self) -> str:
        if self.advisor is None:
            return ""
        try:
            advices = self.advisor.check()
            if advices:
                return "\n".join(
                    self.formatter.format_advice(
                        a.message, action=a.action, severity=a.severity
                    )
                    for a in advices[:3]
                )
        except Exception as e:
            _logger.warning("MyceliumCompanion: advice check failed: %s", e)
        return ""

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MyceliumCompanion("
            f"name={self.name!r}, "
            f"started={self._started})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELP_TEXT = """
Mycelium (Myco) — your local AI companion

PREDICTIONS & LEARNING
  predict [values...]          — run a prediction with your model
  train on <file.csv>          — train on a CSV file
  that's wrong, it should be X — correct a prediction (live model update)

DOCUMENTS & WEB
  read / process <file>        — analyse a local file (CSV, PDF, TXT, image)
  open https://example.com     — fetch and read a webpage

AUTOMATION
  take a screenshot            — capture your screen
  plan <goal>                  — break a goal into steps (LLM-powered)

PERSONALISATION
  call companion.personalise("name", "Alex")   — set your name
  call companion.personalise("verbosity", "concise")
  call companion.personalise("watch_dirs", ["/home/me/Downloads"])

STATUS & SYSTEM
  show report / status         — full system status
  what have you learned about me? — profile + known facts
  save                         — save state to disk
  /help                        — show this message
""".strip()


def _build_default_router() -> Any:
    """Create a NaturalLanguageRouter with default intents."""
    from physml.nl_router import NaturalLanguageRouter, Intent

    router = NaturalLanguageRouter(min_confidence=0.1)
    router.register_many([
        Intent("predict", [
            "predict", "forecast", "estimate", "what will", "how many",
            "run inference", "classify", "run prediction",
        ]),
        Intent("train", [
            "train", "learn from", "fit", "teach", "update model",
            "train on", "learn on",
        ]),
        Intent("report", [
            "show report", "give report", "stats", "statistics",
            "how is the model", "model performance",
        ]),
        Intent("status", [
            "status", "system status", "what is running", "health check",
        ]),
        Intent("save", [
            "save", "persist", "store", "backup", "checkpoint",
        ]),
        Intent("read", [
            "read file", "open file", "load file", "read document",
        ]),
        Intent("analyze", [
            "analyze", "analyse", "process document", "look at file",
            "read and tell", "summarize", "summarise", "check the",
        ]),
        Intent("profile", [
            "what have you learned about me", "my preferences",
            "my profile", "about me",
        ]),
        Intent("help", [
            "help", "what can you do", "commands", "how do I",
        ]),
    ])
    return router
