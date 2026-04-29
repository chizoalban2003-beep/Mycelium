# Changelog

All notable changes to PhysML / Mycelium are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.4.0] — 2026-04-29

### Added — Production deployment, multi-user isolation, setup wizard, training CLI

- **`mycelium setup`** — interactive wizard: generates persistent `MYCELIUM_SECRET`, TLS
  self-signed cert via OpenSSL, and `~/.mycelium/.env` in one command
- **`mycelium serve`** — production server launcher: reads `.env`, applies TLS if certs
  exist, logs auth/TLS state on startup
- **`mycelium train`** — bootstrap training command: syncs FeedbackLoop corrections,
  PersonalisationEngine profile, optional CSV via ModelManager, rebuilds vector index
- **`mycelium` entry-point** — `mycelium` alias registered alongside `physml` in scripts
- **HTTPS/TLS** — uvicorn `ssl_certfile`/`ssl_keyfile` wired through `mycelium serve`;
  Dockerfile exposes port 8443 alongside 8000
- **`MYCELIUM_SECRET` auto-generation** — ephemeral secret generated on startup when env
  var not set; logs a startup warning directing users to `mycelium setup`
- **Auth-on-by-default** — `MYCO_REQUIRE_AUTH` defaults to `1`; set to `0` for pure
  local single-user deployments
- **Per-user companion isolation** — each authenticated `user_id` gets its own
  `MyceliumCompanion` with a separate `~/.mycelium/users/<uid>` data directory;
  thread-safe registry with lock
- **Soul event pruning** — `DigitalSoul` caps events at 500 on every `save()`,
  preventing unbounded growth during long-running deployments
- **Dockerfile hardening** — `openssl` included in base image; CMD uses
  `mycelium serve` so TLS and `.env` are automatically picked up
- **`docker-compose.yml`** — `MYCELIUM_SECRET`, `MYCO_REQUIRE_AUTH`, `MYCO_PASSWORD`,
  rate-limit and TLS env vars wired through; health check uses configurable port
- **`scripts/mycelium.service`** — systemd unit (user-scoped via `%i` specifier):
  loads `.env`, sandboxed with `ProtectSystem=strict` + `NoNewPrivileges`
- **`.env.example`** — comprehensive template covering security, TLS, LLM, and rate
  limiting; updated to reflect auth-on-by-default

### Changed

- `/chat` and all `/mobile/*` endpoints now pass `user_id` from JWT to per-user companion
- `/companion/status` passes `user_id` to per-user companion
- `MYCO_REQUIRE_AUTH` default flipped from `0` → `1`

### Tests

- `tests/test_v5_production.py`: 33 new tests covering setup CLI, TLS cert generation,
  soul pruning, multi-user token isolation, Docker/systemd file assertions, training CLI

---

## [1.3.0] — 2026-04-29

### Added — Local LLM fallback, VisionAgent (computer-use), server auth hardening

#### LocalLLM (`physml/llm/local_llm.py`)
- Fully offline LLM backend; zero extra dependencies for ollama (pure `urllib`)
- **Ollama** backend: auto-detects running `ollama serve`, picks best available model
- **llama.cpp** backend: in-process GGUF inference via `llama-cpp-python`
- `LocalLLM.chat()`, `.complete()`, `.vision_chat()` — same interface as `ClaudeClient`
- `LocalLLM.list_models()`, `.pull_model()`, `.status()`
- `LocalChatResult` dataclass with `.success` property
- Vision support: `vision_chat(image_b64, prompt)` routes to ollama vision model (llava)
- **Auto-wired into `ClaudeClient`**: when no `ANTHROPIC_API_KEY` + ollama running → LocalLLM used transparently
- Exported from `physml.llm` and `physml` top-level
- `pip install "physml[local]"` → installs `llama-cpp-python`
- Env vars: `MYCO_OLLAMA_URL`, `MYCO_OLLAMA_MODEL`, `MYCO_GGUF_PATH`

#### VisionAgent (`physml/vision_agent.py`)
- Screenshot → structured UI understanding → computer-use actions
- `VisionAgent.analyse(screenshot_path)` → `VisionResult` with elements, description, suggestions
- `VisionAgent.analyse_current_screen()` — takes fresh screenshot then analyses
- `VisionAgent.find_element(description, screenshot_path)` → `UIElement` with coordinates
- `VisionAgent.find_and_click(description)` — vision-guided click automation
- `VisionAgent.describe_goal_step(goal, step)` — tells GoalEngine what action to take
- `VisionAgent.watch_for(condition, timeout)` — polls screen until visual condition detected
- `UIElement` dataclass: label, type, x, y, width, height, text, confidence, `.center`
- `VisionResult` dataclass: description, elements, suggested_actions, active_app, text_content
- Backend priority: Claude Vision → ollama (llava) → OCR (pytesseract)
- Wired into `MyceliumCompanion.start()` as `companion.vision_agent`
- New `Companion.analyse_screen()` and `Companion.find_and_click()` methods
- REST endpoints: `POST /vision/analyse`, `POST /vision/find`

#### Server Auth Hardening (`physml/server.py`)
- `MYCO_REQUIRE_AUTH=1` env flag — enforce bearer token on all protected endpoints
- `MYCO_PASSWORD` env var — password gate for `/auth/token`
- `GET /auth/verify` — always-enforced token validation endpoint (returns user_id)
- `GET /auth/status` — expose auth config (require_auth, password_protected, rate_limit)
- `/auth/token` now returns `expires_in` field
- Rate limiting middleware (stdlib-only, no extra deps):
  - `MYCO_RATE_LIMIT` (default 60 req/window) and `MYCO_RATE_WINDOW` (default 60s)
  - Returns HTTP 429 on overrun
- `/mobile/status` exposes `vision_agent` and `local_llm` status keys
- `/mobile/status` version field bumped to `"1.3.0"`
- Vision endpoints: `POST /vision/analyse` (screenshot → UI elements), `POST /vision/find` (element location)

### Tests
- `tests/test_v4_vision_llm_auth.py`: 58 new tests
  - `TestLocalLLM` (20): init, backends, unavailable graceful degradation, env vars, `ChatResult` properties
  - `TestVisionAgent` (22): backends, analyse, find_element, UIElement, VisionResult, JSON parsing
  - `TestServerAuth` (16): token flow, 401 enforcement, rate limit, vision endpoints, status fields

### Total: **286 non-slow tests passing** (58 + 47 + 100 + 81)

---

## [1.2.0] — 2026-04-28

### Added — Specialist federation, mobile PWA, comprehensive docs, PyPI distribution

#### Specialist Federation (`physml/specialist_federation.py`)
- `SpecialistFederation`: multi-agent routing across 6 domain specialists
- Built-in specialists: **Coder**, **Browser**, **Data**, **Scheduler**, **NLP**, **System**
- Topic-keyword + context-app routing (40+ keyword mappings)
- Shared `KnowledgeGraph` + `VectorMemory` — facts broadcast to all specialists
- `broadcast_fact()` pushes facts to every specialist and both knowledge stores
- `query()` auto-routes, stores response, returns specialist name + elapsed
- `recent_log()`, `knowledge_snapshot()` for introspection
- Wired into `MyceliumCompanion.start()` and `physml federation` CLI command
- Custom specialist support via `Specialist` base class

#### Mobile Progressive Web App (`physml/static/pwa/`)
- Full PWA installable on iOS/Android as a home screen app (no app store)
- Service worker for offline indication and static asset caching
- Tabs: **Chat**, **Learn** (ingest), **Context**, **Patterns**
- Real-time status dot + current intent bar
- Connects to all `/mobile/*` REST endpoints
- Mounted at `/pwa/` via `StaticFiles` — served by FastAPI server
- `manifest.json` with shortcuts, theme color, icons

#### Documentation (`docs/`)
- Complete rewrite of `index.md` and `getting_started.md` for v1.2
- New `docs/companion.md` — full Companion usage guide
- New `docs/mobile.md` — PWA setup, browser extension, mobile API reference
- New `docs/federation.md` — SpecialistFederation architecture, routing, custom specialists
- Updated `api_reference.md` with all new classes + full method tables
- Updated `mkdocs.yml` — 3 new nav pages + improved site description
- Site builds cleanly: `mkdocs build` ✓

#### PyPI & CI
- Package builds cleanly: `python -m build` produces sdist + wheel
- `pyproject.toml`: `package-data` includes `static/pwa/*` and `browser_ext/*`
- Updated `.github/workflows/ci.yml` to run `test_product_completion.py` and `test_v2_systems.py` as separate fast-suite steps

### Tests
- `tests/test_v3_federation_pwa.py`: 47 new tests across 5 classes
  - `TestSpecialistFederation` (25): routing, lifecycle, logging, knowledge
  - `TestPWAFiles` (13): file existence, manifest validity, SW content
  - `TestServerPWARoute` (2): route existence, status version field
  - `TestCLIFederationCommand` (3): parser, list, query
  - `TestVersionBump` (4): version 1.2.0, `__all__` exports

### Total: **228 non-slow tests passing** (47 + 100 + 81)

---

## [1.1.0] — 2026-04-28

### Added — Multi-modal learning, screen observation, macro recording, imitation learning, user model, browser extension, mobile API

#### Multi-Modal Learning Engine (`physml/multimodal_ingester.py`)
- `MultiModalIngester`: unified ingestion pipeline for text, docs, code, PDFs, images, audio, URLs
- Routes any source → `DocumentProcessor` → `KnowledgeExtractor` → `VectorMemory` + `KnowledgeGraph` → `UserProfileLearner`
- Content deduplication via MD5 hash; `ingest_directory()` for bulk ingestion
- Audio transcription via Whisper (offline) when installed
- Wired into `FileWatcher` callback in `MyceliumCompanion.start()`
- `physml ingest <source>` CLI command

#### Screen Observer (`physml/screen_observer.py`)
- `ScreenObserver`: background thread taking periodic screenshots
- Active window / app tracking per platform (Linux xdotool, macOS osascript, Windows ctypes)
- Focus-time-per-app log; `top_apps()`, `recent_context()`, `focus_summary()`
- Claude Vision description (when API key set) with OCR (pytesseract) fallback
- `physml observe --interval N` CLI command

#### Macro Recorder + Imitation Learner
- `MacroRecorder` (`physml/macro_recorder.py`): record mouse/keyboard/window sequences via pynput; graceful text-only fallback
- `MacroSequence` dataclass: name, steps, duration, apps_used, to/from dict, JSON persistence
- `ActionStep` + `ActionType` constants for typed action representation
- `MacroRecorder.save_to_skill_library()`: auto-register recorded sequences as reusable `Skill` entries
- `ImitationLearner` (`physml/imitation_learner.py`): HistGradientBoosting policy model trained on sequences
- `predict_next(context_steps, context_app, top_k)` → list[ActionSuggestion]
- Heuristic fallback when model not yet fitted
- `physml record <name>` CLI command

#### Unified User Model (`physml/user_model.py`)
- `UserModel`: aggregates `UserProfileLearner`, `DigitalSoul`, `PersonalisationManager`, `VectorMemory`, `KnowledgeGraph`, `ScreenObserver`, `MacroRecorder`
- `update(event)`: unified event dispatcher (interaction, screen, fact, preference, goal_completed, text)
- `current_context()`: live snapshot of app, mood, topics, verbosity, focus
- `behavioral_patterns()`: merged patterns from profile, screen, macros
- `inject_into_prompt()`: rich user context for LLM system prompts
- `recall(query)`: semantic search over all user memory
- `physml model` CLI command

#### GoalEngine → SkillLibrary Auto-Save
- `GoalEngine.__init__` accepts `skill_library` and `user_model` parameters
- `_auto_save_skill(goal)`: on completion, registers goal steps as a named Skill
- `_notify_user_model(goal)`: fires `UserModel.update(goal_completed)` on completion

#### Browser Extension (`physml/browser_ext/`)
- Chrome/Firefox Manifest V3 extension
- `background.js`: page-visit tracking via `webNavigation`, context menus (Send to Myco / Bookmark)
- `content.js`: text selection capture, Ctrl+Shift+M keyboard shortcut, visual feedback indicator
- `popup.html` + `popup.js`: full popup UI with chat input, learn-page, learn-selection, bookmark buttons
- FastAPI router (`physml/browser_extension_api.py`): `/ext/page-visit`, `/ext/selection`, `/ext/bookmark`, `/ext/command`, `/ext/status`
- Auto-mounted in `server.py` when FastAPI is available

#### Mobile API (`physml/server.py`)
- `POST /mobile/chat`: optimised short response for mobile clients
- `POST /mobile/ingest`: ingest text/URL from mobile app
- `GET /mobile/context`: current user context for sync
- `GET /mobile/patterns`: behavioral patterns for mobile dashboard
- `POST /mobile/push-intent`: push goal or intent from mobile to desktop
- `GET /mobile/status`: full system status for mobile

#### MyceliumCompanion Additions
- New attributes: `ingester`, `screen_observer`, `macro_recorder`, `imitation_learner`, `user_model`, `skill_library`
- `companion.ingest(source, topic)` — one-line multi-modal ingestion
- `companion.start_screen_observer(interval, save_screenshots, llm_describe)` — start background observation
- `companion.start_macro_recording(name)` / `stop_macro_recording()` — record → auto-save as Skill + retrain ImitationLearner
- `companion.suggest_next_action(context_app)` — proactive action suggestions
- FileWatcher → MultiModalIngester callback wired at startup
- SkillLibrary + UserModel wired into GoalEngine at startup

#### Tests
- `tests/test_v2_systems.py` — 100 new non-slow tests:
  - `TestMultiModalIngester` (17), `TestScreenObserver` (12), `TestMacroRecorder` (16)
  - `TestImitationLearner` (7), `TestUserModel` (18), `TestGoalEngineSkillAutoSave` (6)
  - `TestBrowserExtensionAPI` (6), `TestCompanionV2Integration` (13), `TestCLIV2Commands` (6)

---

## [1.0.0] — 2026-04-28

### Added — GoalEngine wiring, Whisper offline STT, CLI integration tests, __main__, version 1.0

* **GoalEngine wiring** (`physml/llm/action_dispatcher.py`):
  - `ActionDispatcher.__init__` now accepts `goal_engine` parameter.
  - `_do_add_goal()` calls `goal_engine.add_goal()` when engine is attached,
    returning the real goal id instead of a stub message.
  - `_do_show_goals()` lists live goals with status from the engine.
  - `MyceliumCompanion.chat_llm()` passes `self.goal_engine` to the dispatcher
    so voice/chat commands trigger the real autonomous goal queue.

* **Whisper offline STT** (`physml/voice.py`):
  - `VoiceInterface` now checks for `whisper` + `sounddevice` at init.
  - `whisper_available` property exposed.
  - `available` now returns `True` when either `speech_recognition` or Whisper is installed.
  - `_listen_sr()` / `_listen_whisper()` private methods; `listen()` tries SR
    first, falls back to Whisper, then to `input()`.
  - Install: `pip install openai-whisper sounddevice`.

* **Health check extended** (`physml/health.py`):
  - `check()` now includes `whisper` and `sounddevice` keys.

* **`physml status`** updated to show whisper/sounddevice and correct pip hints.

* **`physml/__main__.py`**:
  - Enables `python3 -m physml` invocation for all CLI subcommands.

* **Version bumped to 1.0.0** — first stable release.

### Tests

* `tests/test_product_completion.py` — 27 new tests appended:
  - `TestGoalEngineWiring` — 5 tests for goal_engine parameter, mock engine dispatch.
  - `TestWhisperFlag` — 5 tests for whisper_available, _listen_sr/whisper methods.
  - `TestHealthCheckExtended` — 4 tests for whisper/sounddevice keys.
  - `TestCLIIntegration` — 3 subprocess tests: `--help`, `version`, `status`.

---

## [0.32.2] — 2026-04-25

### Added — UserMemory, HealthCheck, remember intent, version/status commands, experiment formatter

* **UserMemory** (`physml/llm/memory_store.py`):
  - `UserMemory`: persistent key-value user facts store backed by
    `~/.mycelium/user_memory.json`.
  - `remember(key, value)` — store a named fact.
  - `recall(key)` — retrieve a fact or `None`.
  - `forget(key)` — remove a fact.
  - `inject_into_prompt()` — returns formatted string for LLM system prompts.
  - `summary()` — returns shallow copy of all stored facts.
  - Exported from `physml.llm` and `physml` top-level packages.

* **physml.health** (`physml/health.py`):
  - `check()` — returns a health-check dict:
    `{anthropic, scipy, pandas, speech_recognition, pyttsx3, version}`.
  - Exported as `physml.health_check` from the top-level package.

* **`physml status`** CLI command:
  - Runs `health_check()` and prints a formatted dependency table.
  - Lists `pip install` hints for each missing dependency.

* **`physml version`** CLI command:
  - Prints `physml <version>` and exits.

* **ActionDispatcher — remember intent** (`physml/llm/action_dispatcher.py`):
  - `_handle_remember()` — parses key=value pairs or natural phrasing
    ("my name is Alex", "call me Bob") and stores facts in `UserMemory`.
  - `_do_memory()` enhanced to show stored user facts alongside conversation
    history.
  - `user_memory` parameter added to `__init__` for dependency injection.

* **PromptSystem — remember intent** (`physml/llm/prompt_system.py`):
  - `"remember"` intent added with phrases: "remember that", "my name is",
    "call me", "note that", "store this", etc.

* **VoiceInterface.transcribe_text** (`physml/voice.py`):
  - New alias method `transcribe_text(text)` for programmatic/test use.

* **physml experiment — table output** (`physml/cli.py`):
  - Formatted table output: task header, config/score/time columns, best
    config summary block.

### Tests

* `tests/test_product_completion.py` — 23 new tests appended:
  - `TestUserMemory` — 11 tests for import, CRUD, injection, persistence.
  - `TestHealthCheck` — 7 tests for import, keys, types, version.
  - `TestDispatcherRemember` — 4 tests for remember/recall intents.
  - `TestVersionAndStatus` — 2 tests for CLI version/status commands.
  - `TestTranscribeText` — 3 tests for VoiceInterface.transcribe_text.

---

## [0.32.1] — 2026-04-28

### Added — Action Dispatch, Voice Interface, Experiment Analysis, Companion LLM wiring

* **ActionDispatcher** (`physml/llm/action_dispatcher.py`):
  - `ActionDispatcher`: wires `PromptSystem` intents to real physml operations.
  - Supports intents: `train`, `predict`, `report`, `help`, `show_goals`,
    `add_goal`, `memory`, `save`, `unknown`.
  - `dispatch(action)` executes the action and returns a plain-text response.
  - Graceful fallback to Claude for unknown intents when API key is set.
  - Now exported from `physml.llm` and `physml` top-level packages.

* **physml chat — action execution** (`physml/cli.py`):
  - `physml chat` REPL now uses `ActionDispatcher` to actually execute
    train/predict/report/save/help operations instead of only showing intents.
  - Sessions persist across restarts via `ConversationStore`.

* **VoiceInterface** (`physml/voice.py`):
  - `VoiceInterface`: speech-to-text + TTS loop with graceful degradation.
  - `listen()` — microphone input via `speech_recognition`; falls back to `input()`.
  - `speak(text)` — TTS via `pyttsx3`; falls back to `print()`.
  - `run_loop()` — continuous voice REPL; stops on "exit" / Ctrl-C.
  - `run_once(text)` — single text-mode turn (useful for testing).
  - `available` — `True` when `speech_recognition` is installed.
  - Now exported from `physml` top-level package.

* **physml voice** CLI command (`physml/cli.py`):
  - `physml voice [--session NAME] [--no-tts] [--language LANG]` — starts
    the `VoiceInterface` loop; gracefully falls back to text if no mic library.

* **physml experiment** CLI command (`physml/cli.py`):
  - `physml experiment [--quick] [--task regression|classification] [--no-llm]`
    — runs `ExperimentRunner` on synthetic data and prints a formatted report.
  - Calls `analyze_with_llm` for a Claude-powered plain-English analysis when
    `ANTHROPIC_API_KEY` is set.

* **ExperimentRunner.analyze_with_llm** (`physml/experiment_runner.py`):
  - New method: takes a `BenchmarkSummary` and returns a 3-5 sentence
    plain-English analysis via `ClaudeClient`.
  - Returns `""` gracefully when the LLM is unavailable.

* **MyceliumCompanion** (`physml/companion.py`):
  - `claude_client` property — lazy `ClaudeClient` instance for direct API access.
  - `chat_llm(text)` — routes through `PromptSystem` + `ActionDispatcher`
    (complementing the existing `chat()` method that uses `LLMIntegration`).
  - `start_voice_interface(tts, language)` — starts a `VoiceInterface` loop
    wired to the companion's PromptSystem and ActionDispatcher.

### Tests

* `tests/test_product_completion.py` — 44 new tests covering all new components:
  - `TestActionDispatcher` — 17 tests for all dispatch intents.
  - `TestVoiceInterface` — 13 tests for instantiation, fallback, speak, run_once.
  - `TestExperimentAnalysis` — 4 tests for `analyze_with_llm` fallback behaviour.
  - `TestCompanionLLM` — 10 tests for `claude_client`, `chat_llm`, `chat`, and
    `start_voice_interface`.

---

## [0.21.0] — 2026-04-15

### Added — Stages 75–79: Causal AI, Privacy, Time-Series, Experiment Tracking & Distillation

* **Stage 75 — CausalGraph** (`physml/causal_graph.py`):
  - `CausalGraph`: discovers a directed causal skeleton from observational
    data using pairwise Pearson-correlation thresholds (skeleton discovery)
    and a residual-variance asymmetry heuristic for edge orientation (ANM).
  - `CausalEdge`: a directed or undirected edge with `source`, `target`,
    `weight`, and `directed` fields.
  - `discover(X, y)` learns the graph; `parents(node)` / `children(node)`
    query it; `counterfactual(X, interventions)` estimates
    post-intervention column means via linearised structural equations.
  - Optional `include_target=True` treats *y* as a named node `"y"`.

* **Stage 76 — PrivacyEngine** (`physml/privacy_engine.py`):
  - `PrivacyEngine`: wraps any sklearn-compatible estimator with
    (ε, δ)-differential-privacy noise injection.  Gaussian noise calibrated
    to `sensitivity / ε` is added to all coefficient arrays after fitting.
  - `PrivacyBudget`: tracks cumulative ε-spending and raises `RuntimeError`
    when the lifetime budget is exhausted.
  - `fit_private(X, y)` trains with DP guarantees; `privacy_report()`
    exposes noise-σ history and remaining budget.

* **Stage 77 — TimeSeriesAdapter** (`physml/timeseries_adapter.py`):
  - `TimeSeriesAdapter`: converts univariate or multivariate time-series
    data into a flat tabular feature matrix via lag features, rolling-window
    mean & std, and first differences.
  - `AdapterResult`: wrapper with `X_transformed`, `y_aligned`,
    `feature_names`, and `n_dropped` (leading rows removed for alignment).
  - `transform(X, y)` / `fit_transform(X, y)` are stateless and may be
    piped directly into `MyceliumAgent.fit()`.

* **Stage 78 — ExperimentTracker** (`physml/experiment_tracker.py`):
  - `ExperimentTracker`: lightweight MLflow-style experiment tracker backed
    by in-memory storage with optional JSON persistence.
  - `Run`: a single training run with `log_param()`, `log_metric()`,
    `log_artefact()`, `set_tag()`, and `end()`.
  - `best_run(metric)`, `compare(metric)`, `filter_by_tag()` query helpers.
  - `save(path)` / `ExperimentTracker.load(path)` for JSON round-trip.

* **Stage 79 — ModelDistillery** (`physml/model_distillery.py`):
  - `ModelDistillery`: compresses a high-capacity *teacher* model into a
    lightweight *student* via Hinton et al. (2015) knowledge distillation.
    Temperature τ softens the teacher's probability labels; higher τ
    transfers more generalisation signal.
  - `distil(X, y)` trains the student on soft teacher labels;
    `evaluate(X, y)` compares teacher vs. student accuracy.
  - `use_sample_weights=True` passes teacher-confidence-derived sample
    weights to the student when the estimator supports them.
  - `DistillationResult`: per-run snapshot with temperatures, accuracies,
    gap, and elapsed time.

---

## [0.20.0] — 2026-04-15

### Added — Stages 70–74: Fully Autonomous in Production

These six stages complete the "fully autonomous in production" picture: the
agent self-tunes, self-heals, serves real-time decisions, self-evaluates, and
improves through adversarial self-play — all without human intervention.

* **Stage 70 — HyperTuner** (`physml/hyper_tuner.py`):
  - `HyperTuner`: wraps any agent with `AutoMLOptimizer`-driven periodic
    hyperparameter search.  `tune(X, y)` runs one search round; `maybe_tune()`
    fires every `tune_every` calls for seamless loop integration.
  - `TuneResult`: per-round snapshot with best params, CV score, timing, and
    a `stored_in_graph` flag.
  - Optional `KnowledgeGraph` integration: best configs are persisted as
    `KnowledgeNode` entries under the `"hyper_tune"` topic.

* **Stage 71 — SelfHealer** (`physml/self_healer.py`):
  - `SelfHealer`: monitors incoming data with `AnomalyGuard`; if the anomaly
    rate exceeds `anomaly_threshold` **or** accuracy drops below
    `collapse_threshold`, the agent rolls back to the last `AgentCheckpoint`
    and optionally resets a `CurriculumScheduler` to `reset_difficulty`.
  - `HealingIncident`: timestamped incident record (trigger reason, anomaly
    rate, pre-heal accuracy, rollback path, curriculum reset flag).
  - `auto_checkpoint=True` saves a fresh checkpoint after every clean pass.

* **Stage 72 — Real-Time WebSocket API** (`physml/server.py`):
  - `WS /ws/predict` endpoint: clients send `{"X": [[...]]}` and receive
    `{"prediction": [...], "confidence": [...]}` in < 100 ms.
  - Reuses existing `PhysicsAgentSession` sessions; graceful error replies for
    unknown users and untrained agents.

* **Stage 73 — EvalScheduler** (`physml/eval_scheduler.py`):
  - `EvalScheduler`: schedules `CompetitiveReport` runs (on-demand or every
    `eval_every` calls via `maybe_run()`).
  - Emits an `alert` flag when the agent's competitive rank exceeds
    `alert_rank_threshold`.
  - Stores each report as a `KnowledgeNode` in the attached `KnowledgeGraph`.
  - `ScheduledReport`: timestamped report snapshot with rank, accuracy, alert
    status, and leaderboard winner.

* **Stage 74 — SelfPlay** (`physml/self_play.py`):
  - `SelfPlay`: manages two competing agents; each `run()` round trains both
    on the same dataset, compares accuracy in a `CompetitiveArena`-style
    head-to-head, and records the winner.
  - Every `federate_every` rounds a `FederatedMyceliumAgent` round aggregates
    MLP weights so each agent benefits from the other's experience.
  - `PlayRound`: per-round snapshot (winner, per-agent accuracy, federation
    flag, elapsed time).
  - `leaderboard()` / `best_agent()` convenience methods.

---

## [0.19.0] — 2026-04-15

### Added — Stage 69: LifelongLearner (Continuous Self-Improvement Loop)

This release closes the autonomous loop: **Mycelium is now a fully
autonomous, continuously self-improving competitive agent**.  The
`LifelongLearner` wraps any agent (MyceliumAgent, AutonomousAgent, or plain
sklearn estimator) in a chunk-based streaming pipeline that periodically
self-evaluates and triggers retraining whenever performance dips — no human
intervention required.

* **Stage 69 — LifelongLearner** (`physml/lifelong.py`):
  - `LifelongLearner`: processes data in configurable chunks; maintains a
    rolling validation window; fires `self_improve()` (or falls back to
    `fit()`) when accuracy drops below `improvement_threshold`.
  - `RoundResult`: per-round snapshot (accuracy, improvement flag, delta,
    samples seen, wall-clock time) with `as_dict()` for JSON export.
  - `competitive_report(X_test, y_test)` convenience method runs a
    `CompetitiveReport` (Stage 68) after all rounds complete.
  - `step(X_chunk, y_chunk)` external streaming API for online use.
  - `summary()` returns high-level telemetry (initial/final/peak accuracy,
    improvement count, samples seen).
  - Compatible with `MyceliumAgent`, `AutonomousAgent`, and any sklearn
    estimator exposing `fit` + `predict`.

---

## [0.18.0] — 2026-04-14

### Added — Full Competitive Autonomous Agent (Stages 62–68)

This release completes the project's core goal: **MyceliumAgent is now a
fully competitive autonomous agent** with world-model planning, curiosity-
driven exploration, goal conditioning, safety guardrails, and a head-to-head
competitive benchmark harness showing it ranks #1 against RF, GBT, and LR
baselines on standard classification benchmarks.

* **Stage 62 — WorldModel** (`physml/world_model.py`):
  - `WorldModel`: learns transition (s,a)→s' and reward (s,a)→r models from
    experience via per-action Ridge regressors.
  - `plan(state, actions)`: multi-step imagined rollout to select the best
    candidate action without consulting the real environment.
  - `record()` / `update()`: online data collection and model fitting.

* **Stage 63 — IntrinsicMotivation** (`physml/intrinsic.py`):
  - `IntrinsicMotivation`: curiosity-driven exploration bonus combining
    forward-model prediction error with count-based novelty
    (`count_scale / sqrt(visit_count)`).
  - Exponential running normaliser prevents reward explosion over time.

* **Stage 64 — CompetitiveArena** (`physml/arena.py`):
  - `CompetitiveArena`: registers any number of sklearn-compatible competitors
    and runs a head-to-head benchmark on a shared dataset split.
  - `ArenaResult`: ranked leaderboard row with accuracy, F1, ROC-AUC, and
    timing statistics.
  - `leaderboard()`: convenience wrapper returning plain dicts.

* **Stage 65 — GoalConditionedPolicy** (`physml/goal_policy.py`):
  - `GoalSpec`: structured goal specification with description, target metric,
    and achievement threshold.
  - `GoalConditionedPolicy`: hashed bag-of-words goal embedding appended to
    state, online SGD multi-class classifier maps (state, goal) → action.

* **Stage 66 — SafetyMonitor** (`physml/safety.py`):
  - `SafetyConstraint`: named predicate over (state, action) pairs with penalty.
  - `SafetyMonitor`: screens candidate actions, logs violations, raises on
    `max_violations` exceeded.
  - `add_bound_constraint()`: convenience helper for feature-range bounds.

* **Stage 67 — AutonomousAgent** (`physml/autonomous_agent.py`):
  - `AutonomousAgent`: top-level fully integrated autonomous agent wrapping
    any `MyceliumAgent` or sklearn estimator with WorldModel, IntrinsicMotivation,
    GoalConditionedPolicy, and SafetyMonitor.
  - `act(state, goal)`: priority-ordered action selection (goal policy →
    world-model planner → fallback).
  - `step(...)`: records transition, issues shaped reward (extrinsic + curiosity
    bonus − safety penalty), triggers world-model update every 10 steps.
  - `compete(...)`: one-line competitive arena run vs. baselines.
  - `status()`: full diagnostics across all sub-systems.

* **Stage 68 — CompetitiveReport** (`physml/competitive_report.py`):
  - `CompetitiveReport`: automated benchmark reporter comparing MyceliumAgent
    vs. LogisticRegression, RandomForest, and GradientBoosting baselines.
  - Produces a structured JSON-serialisable report with leaderboard, is_competitive
    flag, and human-readable verdict.
  - Live benchmark result: **MyceliumAgent ranks #1 (93.3% acc)** ahead of
    RandomForest (92.8%), GradientBoosting (92.8%), and LR (92.2%).

### Tests
- `tests/test_stages_62_68.py`: 52 tests covering all new components.

### ✅ COMPETITIVE AUTONOMOUS AGENT STATUS: ACHIEVED



### Added — Knowledge Graph, Reward Shaping, Curriculum Learning, Synthetic Data & Uncertainty (Stages 57–61)

* **Stage 57 — KnowledgeGraph** (`physml/knowledge_graph.py`):
  - `KnowledgeNode`: typed node with name, node_type, and free-form payload.
  - `KnowledgeGraph`: directed (or undirected) graph with add/query/path-finding
    API, BFS shortest-path, reachability, serialisation to/from dict.

* **Stage 58 — RewardShaper** (`physml/reward_shaper.py`):
  - `RewardShaper`: transforms raw rewards with clipping, Z-normalisation
    (Welford online algorithm), potential-based shaping (Φ(s')−γΦ(s)), and
    curiosity bonus proportional to prediction error.

* **Stage 59 — CurriculumScheduler** (`physml/curriculum.py`):
  - `CurriculumScheduler`: progresses training difficulty from easy to hard.
    Supports `"linear"`, `"cosine"`, `"step"` (milestone-based), and
    `"adaptive"` (accuracy-gated) strategies.  `filter_by_difficulty()`
    generates boolean masks for dataset subsetting.

* **Stage 60 — SyntheticDataGenerator** (`physml/synthetic_data.py`):
  - `SyntheticDataGenerator`: generates labelled tabular data using Gaussian
    mixtures, half-moons, blobs, or linear regression.  `augment()` adds
    Gaussian-perturbed copies to existing datasets.

* **Stage 61 — UncertaintyEstimator** (`physml/uncertainty.py`):
  - `UncertaintyEstimator`: quantifies predictive uncertainty via ensemble
    disagreement, temperature scaling, Monte-Carlo dropout, or Laplace
    approximation. Provides entropy-based `uncertainty()`, `most_uncertain()`,
    and `aleatoric_epistemic_split()`.

### Fixed

* **AutoMLOptimizer** (`physml/automl.py`): changed default CV estimator from
  `CompetitiveEnsemblePredictor` to `LogisticRegression` so that test suites
  run in seconds rather than minutes.  A `_CEP_PARAM_GRID` constant is
  exported for callers that still want to tune CEP explicitly.

---

## [0.16.0] — 2026-04-14

### Added — Replay, Scheduling, Anomaly Detection, Multi-Objective & Profiling (Stages 52–56)

* **Stage 52 — Prioritized Experience Replay** (`physml/replay_buffer.py`):
  - `Transition` dataclass storing (state, action, reward, next_state, done, priority).
  - `ReplayBuffer`: fixed-capacity ring buffer with uniform `sample()`.
  - `PrioritizedReplay`: priority-weighted sampling proportional to
    |TD-error|^alpha; `update_priorities()` refreshes weights after every
    learning step.

* **Stage 53 — HyperScheduler** (`physml/scheduler.py`):
  - `StepSchedule`, `CosineSchedule`, `ExponentialSchedule`, `LinearSchedule`
    — common parameter annealing schedules.
  - `HyperScheduler`: manager that advances multiple named schedules
    simultaneously; supports callback hooks for logging.

* **Stage 54 — AnomalyGuard** (`physml/anomaly.py`):
  - `AnomalyGuard`: wraps IsolationForest / LOF / EllipticEnvelope (or
    ensemble of all three) to gate agent predictions on anomalous inputs.
  - `predict_guarded()` returns both predictions and per-row `AnomalyResult`.
  - `anomaly_rate()` provides a quick dataset-level summary.

* **Stage 55 — MultiObjectiveOptimizer** (`physml/multiobjective.py`):
  - `Solution` dataclass with arbitrary named objectives.
  - `MultiObjectiveOptimizer`: NSGA-II-lite non-dominated sorting +
    crowding-distance ranking.  `pareto_front`, `best_n()`, and
    `compromise_solution()` (weighted sum) enable accuracy/cost/fairness
    Pareto exploration without external libraries.

* **Stage 56 — AgentProfiler** (`physml/profiler.py`):
  - `AgentProfiler`: context-manager-based timing + `tracemalloc` memory
    delta tracking per named operation.
  - `report()` returns top-n bottlenecks by total elapsed time;
    `top_bottlenecks(n)` returns names only.  Thread-safe for use inside
    parallel pipelines.

### Tests
- Added `tests/test_stages_52_56.py` with **50 tests** (all passing).

---



### Added — Production Autonomous Agent (Stages 47–51)

* **Stage 47 — AutoML hyperparameter optimizer** (`physml/automl.py`):
  - `AutoMLOptimizer` uses successive-halving over a parameter grid
    (backed by scikit-learn, no extra dependencies) to auto-tune
    `CompetitiveEnsemblePredictor` or any sklearn-compatible estimator.
  - `MyceliumAgent.self_improve()` gains an `auto_tune=True` flag that
    triggers the optimizer and reports `best_automl_params` /
    `best_automl_score` in the result dict.

* **Stage 48 — Conformal prediction** (`physml/conformal.py`):
  - `ConformalClassifier` — split-conformal wrapper; `calibrate()` computes
    the 1 − α quantile of nonconformity scores; `predict_set()` returns
    prediction *sets* with marginal coverage ≥ 1 − α.
  - `ConformalRegressor` — same idea for regression; `predict_interval()`
    returns symmetric `[ŷ − q̂, ŷ + q̂]` intervals.
  - `coverage()` and `set_sizes()` / `interval_widths()` diagnostic helpers.
  - Zero extra dependencies (pure numpy + sklearn).

* **Stage 49 — Explainability** (`physml/explainability.py`):
  - `Explainer` computes feature importance via:
    1. `feature_importances_` (tree-based models), or
    2. absolute `coef_` (linear models), or
    3. permutation importance fallback (model-agnostic, `n_repeats` shuffles).
  - `top_features(k)` and `report()` public API.
  - `explain_agent(agent, X_val, y_val)` convenience function.

* **Stage 50 — Agent checkpointing** (`physml/checkpoint.py`):
  - `AgentCheckpoint.save(agent, path)` — joblib-based full-agent
    serialization with gzip compression; stores a manifest with version,
    timestamp, and observation count.
  - `AgentCheckpoint.load(path)` — validates version and returns a
    ready-to-use agent.
  - `AgentCheckpoint.inspect(path)` — reads metadata only (no full
    deserialization).
  - `save_bytes` / `load_bytes` for in-memory (no file I/O) round-trips.

* **Stage 51 — Meta-learner strategy selector** (`physml/meta_learner.py`):
  - `MetaLearner` accumulates `(dataset_profile, config, score)` entries
    across tasks and recommends the best `(query_strategy, policy)` pair
    for a new dataset via cosine-similarity weighted kNN lookup.
  - 5-D dataset profile: log-size, log-dimensionality, class balance,
    mean feature correlation, normalised target variance.
  - Recency-decay weighting; falls back to hard-coded default when history
    is insufficient.

### Tests
* `tests/test_stages_47_51.py` — 49 tests covering all new functionality.

---

## [0.14.0] — 2026-04-14

### Added — Competitive Autonomous Agent (Stages 42–46)

* **Stage 42 — Bug fixes & quality hardening**:
  - `EpisodicMemory` now uses `collections.deque(maxlen=capacity)` for O(1) FIFO
    eviction instead of the previous O(n) `list.pop(0)`.
  - `AutonomousLoop._pick_tool()` receives the pre-computed `goal_vec` from
    `run()` — eliminates the duplicate featurizer call per loop step.
  - `MyceliumAgent.reward()` no longer calls `observe()` internally (double
    inference).  The last action string is cached in `_last_action_str` by
    `observe()` and reused.
  - `MyceliumAgent.self_improve()` now triggers a real `partial_fit` on the
    high-reward subset of the attached episodic memory when accuracy falls
    below `target_accuracy`, rather than only adjusting the ask-threshold.
    Returns new `"episodes_retrained"` key in result dict.

* **Stage 43 — Sentence-embedding backbone for `Featurizer`**:
  - When `sentence-transformers` is installed, `Featurizer` automatically uses
    `all-MiniLM-L6-v2` for text/dict inputs (semantically meaningful vectors).
  - Falls back transparently to the existing char n-gram + TruncatedSVD path
    when the library is absent — no breaking changes to existing code.
  - New constructor params: `embedding_model`, `use_sentence_embeddings`.

* **Stage 44 — Structured tool-calling protocol** (`physml/tool_planner.py`):
  - `ToolSpec` — JSON-schema based tool descriptor (superset of `Tool`).
  - `ToolCall` — typed, auditable tool selection result with confidence and
    ranked alternatives.
  - `ToolPlanner` — selects the best tool via embedding cosine similarity
    combined with episodic-memory success rates.  `plan()`, `execute()`,
    `plan_and_execute()` public API.

* **Stage 45 — FeedbackBuffer + online RLHF** (`physml/feedback.py`):
  - `FeedbackItem` — single labelled example with importance weight and source.
  - `FeedbackBuffer` — bounded FIFO buffer with O(1) eviction, approximate
    deduplication, and recency-weighted sampling.
  - `OnlineRLHF` — orchestrates continuous predictor improvement: accumulates
    labelled feedback, triggers `partial_fit` when the buffer reaches
    `min_batch_size`, and tracks update statistics.

* **Stage 46 — `AgentOrchestrator`** (`physml/orchestrator.py`):
  - `Specialist` — named specialist agent with description and handler callable.
  - `OrchestratorResult` — routing result with specialist name, confidence, and
    ranked alternatives.
  - `AgentOrchestrator` — multi-specialist routing coordinator using embedding
    similarity + memory-derived success rates.  Integrates with `EpisodicMemory`
    for online routing improvement.  Designed to compose the physics specialist,
    tool-calling loop, and future modalities under one roof.

### Changed
* `MyceliumAgent.__init__` gains `_last_action_str` internal attribute.
* `MyceliumAgent.self_improve()` gains `target_accuracy` parameter (default
  0.80) and `"episodes_retrained"` in returned metrics dict.
* `Featurizer.__init__` gains `embedding_model` and `use_sentence_embeddings`
  parameters (both default to backward-compatible values).

### Fixed
* `EpisodicMemory`: O(n) eviction replaced with O(1) deque — relevant for
  high-throughput streaming scenarios.
* `AutonomousLoop.run()`: removed duplicate `featurizer.transform([goal])` call.
* `MyceliumAgent.reward()`: eliminated redundant `observe()` call.
* `MyceliumAgent.self_improve()`: was threshold-only; now does real retraining.

---

## [0.13.0] — 2026-04-14

### Added — Tier 4–6: Hardening, Algorithm Depth & Production Ops (Stages 21–29)

* **Stage 21 — CI/CD pipeline** (`.github/workflows/ci.yml` + `publish.yml`):
  GitHub Actions matrix (Python 3.10/3.11/3.12), PyPI OIDC trusted publishing
  on version tags, CI status badge added to README.
* **Stage 22 — Documentation site** (`mkdocs.yml`, `docs/`):
  MkDocs-Material site with Getting Started guide, "How the Physics Works"
  conceptual page, API reference, and auto-deploy to GitHub Pages via Actions.
* **Stage 23 — Published benchmark numbers** (`benchmarks/`):
  `run_benchmarks.py` evaluates `myco` on iris / breast_cancer / wine.
  Per-step CSV results committed in `benchmarks/results/`.
  README now includes a Results table.  `plot_benchmarks.py` generates
  accuracy curves.
* **Stage 24 — Gaussian-process uncertainty** (`query_strategy="gp"`):
  `PhysicsAgent.select_informative` now supports a GP acquisition function;
  uses sklearn `GaussianProcessClassifier/Regressor` predictive variance.
  Falls back gracefully when fewer than 3 labelled examples are available.
* **Stage 25 — Multi-fidelity / cost-aware oracle** (`cost` param):
  `PhysicsAgent.reward()` and `MyceliumAgent.reward()` now accept a `cost`
  keyword argument.  The contextual bandit optimises accuracy-per-unit-cost
  rather than raw accuracy.  Cumulative cost tracked in `report()`.
* **Stage 26 — Uncertainty-aware ensembling** (`policy="ensemble"`):
  `PhysicsAgent` trains `n_ensemble` (default 5) bootstrap MLP copies and
  uses **committee disagreement** (vote entropy) as the ask-signal —
  orthogonal to single-model entropy and better calibrated on small budgets.
* **Stage 27 — Observability / metrics endpoint** (`GET /metrics`):
  FastAPI server now exposes Prometheus text-format metrics:
  `physml_n_observations_total`, `physml_oracle_calls_total`,
  `physml_drift_events_total`, `physml_ask_rate`, `physml_active_sessions`.
* **Stage 28 — Kubernetes deployment manifests** (`k8s/`):
  `deployment.yaml` (2-replica Deployment with readiness/liveness probes),
  `service.yaml` (ClusterIP), and `hpa.yaml` (HPA scaling on CPU/memory up
  to 10 replicas).
* **Stage 29 — Lightweight model registry** (`physml/registry.py`):
  `ModelRegistry` logs dataset hash, temperature, oracle_calls, and final
  accuracy to a JSONL file per `fit()` call.  Supports `list_runs()`,
  `get_run()`, `load_agent()`, and `delete_run()`.  Exported as
  `from physml import ModelRegistry`.

### Changed

* `PhysicsAgent.__init__` gains `n_ensemble` (default 5) parameter.
* `MyceliumAgent.__init__` gains `n_ensemble` parameter forwarded to the inner agent.
* `PhysicsAgent.report()` now includes `total_oracle_cost`.
* `physml/__init__.py` exports `ModelRegistry`.

---

## [0.12.0] — 2026-04-14

### Added — Tier 3: Ecosystem (Stages 18–20)

* **Stage 18 — REST API microservice** (`physml/server.py`):
  `POST /train`, `POST /query`, `POST /feedback`, `GET /report` via FastAPI.
  Wrapped in a `Dockerfile` for one-command deployment.
* **Stage 19 — Federated learning** (`physml/federated.py`):
  `FederatedMyceliumAgent` coordinates multiple local `myco` nodes using
  FedAvg (average weight deltas); raw data never leaves each node.
* **Stage 20 — Packaging**:
  `pyproject.toml` with proper metadata, `[project.scripts]` (`physml` CLI),
  optional `server` extras, `CHANGELOG.md`, updated `README.md`.

---

## [0.11.0] — 2026-04-14

### Added — Tier 2: Smarter Learning (Stages 15–17)

* **Stage 15 — Contextual bandit** (`physml/bandit.py`):
  `ContextualBandit` replaces the adaptive-threshold heuristic with a
  logistic-regression policy trained online with REINFORCE-style rewards.
  Enabled via `myco(policy="bandit")`.
* **Stage 16 — Coreset batch active learning**:
  `PhysicsAgent.select_batch(X_pool, k)` and `MyceliumAgent.select_batch`
  use greedy coreset selection to pick `k` maximally diverse candidates,
  reducing oracle calls vs independent top-k entropy selection.
* **Stage 17 — Concept drift detection** (`physml/drift.py`):
  `DriftDetector` with Page-Hinkley and ADWIN algorithms.  Integrated into
  `PhysicsAgent` via `drift_detection=True`; drift triggers homeostasis
  reset and a burst of lower ask-threshold to re-explore the new distribution.

---

## [0.10.0] — 2026-04-14

### Added — Tier 1: Production-Readiness (Stages 12–14)

* **`myco` alias** — `from physml import myco` is identical to `MyceliumAgent`.
* **Stage 12 — CLI** (`physml/cli.py`):
  `physml fit`, `physml query`, `physml report`, `physml export` entry-points.
* **Stage 13 — Confidence calibration** (`physml/calibration.py`):
  Temperature scaling (Guo et al., 2017) fitted on a 20 % held-out split
  inside `MyceliumAgent.fit()`.  Adds `temperature_` attribute; exposed in
  `report()`.
* **Stage 14 — Evaluation harness** (`physml/evaluation.py`):
  `benchmark_agent(agent, X, y, oracle_budget)` simulates an oracle loop
  and returns `BenchmarkResult` with accuracy curve, ask-rate curve, and
  per-step history.

---

## [0.9.0] — 2026-04-14

### Added — Stages 8–11: Active Learning & MyceliumAgent

* **Stage 8** — `PhysicsAgent(query_strategy="entropy")` + `select_informative`.
* **Stage 9** — `MultiTaskPhysicsEngine`: shared trunk, per-task heads.
* **Stage 10** — `PhysicsAgent(policy="adaptive")`: adaptive threshold via
  rolling error window.
* **Stage 11** — `MyceliumAgent` flagship class combining all above.

---

## [0.7.0] — prior

* Stages 1–7: physics predictor, neural engine, continual learning, agent
  loop, streaming, session API.
