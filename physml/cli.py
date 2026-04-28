"""Stage 12 — Command-line interface for PhysML / myco.

Provides the ``physml`` entry-point with four sub-commands:

* ``physml fit``     — train an agent from a CSV file and save it.
* ``physml query``   — load a saved agent and predict for new samples.
* ``physml report``  — print a summary of a saved agent.
* ``physml export``  — export predictions to a CSV file.

Examples
--------
::

    physml fit train.csv --target y --out agent.pkl
    physml query agent.pkl test.csv --out predictions.csv
    physml report agent.pkl
    physml export agent.pkl test.csv --out predictions.csv

All sub-commands accept ``--help`` for detailed usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> "tuple[Any, Any]":
    """Load a CSV into (X_array, column_names). Raises SystemExit on error."""
    try:
        import pandas as pd
    except ImportError:
        _die("pandas is required for CSV I/O: pip install pandas")
    p = Path(path)
    if not p.exists():
        _die(f"File not found: {path}")
    df = pd.read_csv(p)
    return df, list(df.columns)


def _die(msg: str, code: int = 1) -> None:
    print(f"physml: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _load_agent(path: str) -> Any:
    from physml.mycelium_agent import MyceliumAgent
    try:
        return MyceliumAgent.load(path)
    except Exception as exc:
        _die(f"Could not load agent from {path!r}: {exc}")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def _cmd_fit(args: argparse.Namespace) -> None:
    """Train a MyceliumAgent on a CSV and save it to disk."""
    from physml.mycelium_agent import MyceliumAgent

    df, cols = _load_csv(args.csv)

    if args.target not in cols:
        _die(
            f"Target column {args.target!r} not found in {args.csv}. "
            f"Available columns: {cols}"
        )

    y = df[args.target].to_numpy()
    X = df.drop(columns=[args.target]).to_numpy(dtype=float)

    agent = MyceliumAgent(
        uncertainty_threshold=args.threshold,
        query_strategy=args.query_strategy,
        policy=args.policy,
    )
    print(f"Fitting agent on {X.shape[0]} samples, {X.shape[1]} features …")
    agent.fit(X, y)

    out_path = Path(args.out)
    agent.save(out_path)
    print(f"Agent saved to {out_path}")


def _cmd_query(args: argparse.Namespace) -> None:
    """Load a saved agent and predict on a CSV file."""

    agent = _load_agent(args.agent)

    df, cols = _load_csv(args.csv)
    X = df.to_numpy(dtype=float)

    predictions = []
    confidences = []
    actions = []

    for i in range(X.shape[0]):
        action = agent.observe(X[i : i + 1])
        predictions.append(action.prediction)
        confidences.append(action.confidence)
        actions.append(action.action)

    result_df = df.copy()
    result_df["prediction"] = predictions
    result_df["confidence"] = confidences
    result_df["action"] = actions

    out_path = Path(args.out)
    result_df.to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")
    n_asks = sum(1 for a in actions if a == "ask")
    print(f"  {X.shape[0]} samples, {n_asks} uncertain (action='ask')")


def _cmd_report(args: argparse.Namespace) -> None:
    """Print a human-readable summary of a saved agent."""
    agent = _load_agent(args.agent)
    report = agent.report()

    if args.json:
        # Serialise to JSON (convert non-serialisable values to strings)
        def _default(o: Any) -> str:
            return str(o)
        print(json.dumps(report, indent=2, default=_default))
    else:
        _print_report(report)


def _print_report(report: dict, indent: int = 0) -> None:
    prefix = "  " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_report(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def _cmd_export(args: argparse.Namespace) -> None:
    """Export predictions for every row in a CSV (alias for query with different defaults)."""
    # Delegate to _cmd_query
    _cmd_query(args)


def _cmd_chat(args: argparse.Namespace) -> None:
    """Run a natural-language REPL using the LLM layer (falls back to rule-based routing)."""
    from physml.llm import PromptSystem, ClaudeClient, ActionDispatcher
    from physml.conversation_store import ConversationStore

    session_id = getattr(args, "session", "default")
    store_path = f"~/.mycelium/conversations/{session_id}.json"
    store = ConversationStore(path=store_path)

    # Try to load a previously saved agent
    agent = None
    agent_path = getattr(args, "agent", "agent.pkl")
    try:
        agent = _load_agent(agent_path)
    except SystemExit:
        pass  # no agent on disk yet — that's fine

    print("Mycelium REPL — type your request in plain English (Ctrl-C or 'exit' to quit).")
    print(f"Session: {session_id!r} | history: {len(store)} turns loaded")
    print("LLM backend: Claude claude-sonnet-4-6 (falls back to rule-based if no API key)")
    print("Type 'help' to see available commands.\n")

    try:
        client = ClaudeClient()
        ps = PromptSystem(client=client)
        dispatcher = ActionDispatcher(agent=agent, store=store, client=client)

        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                print("Bye!")
                break
            if user_input.lower() == "/history":
                for turn in store:
                    role = turn.get("role", "?")
                    text = turn.get("content", "")[:120]
                    print(f"  {role}: {text}")
                print()
                continue
            if user_input.lower() == "/clear":
                store.clear()
                print("myco> History cleared.\n")
                continue

            # Route the intent
            action = ps.route(user_input)
            store.add(role="user", content=user_input, metadata={"intent": action.intent})

            # Dispatch to real action; keep agent in sync after train/save
            reply = dispatcher.dispatch(action)

            # Sync agent reference (dispatcher may have created one during train)
            agent = dispatcher.agent

            store.add(role="assistant", content=reply)
            tokens_str = ""
            if client.available:
                tokens_str = f" [intent={action.intent}]"
            print(f"myco> {reply}{tokens_str}")
            print()

    except Exception as exc:
        _die(f"REPL error: {exc}")


def _cmd_voice(args: argparse.Namespace) -> None:
    """Start a voice interaction loop (falls back to text if no mic library)."""
    from physml.llm import PromptSystem, ClaudeClient, ActionDispatcher
    from physml.voice import VoiceInterface
    from physml.conversation_store import ConversationStore

    session_id = getattr(args, "session", "default")
    store_path = f"~/.mycelium/conversations/{session_id}.json"
    store = ConversationStore(path=store_path)

    client = ClaudeClient()
    ps = PromptSystem(client=client)
    dispatcher = ActionDispatcher(store=store, client=client)

    voice = VoiceInterface(
        prompt_system=ps,
        dispatcher=dispatcher,
        tts=not getattr(args, "no_tts", False),
        language=getattr(args, "language", "en-US"),
    )

    if not voice.available:
        print(
            "speech_recognition not installed — running in text mode.\n"
            "Install it with: pip install SpeechRecognition pyaudio\n"
        )

    voice.run_loop()


def _cmd_experiment(args: argparse.Namespace) -> None:
    """Run benchmark experiments on synthetic data, optionally with Claude analysis."""
    from physml.experiment_runner import ExperimentRunner

    task = getattr(args, "task", "regression")
    quick = getattr(args, "quick", False)

    configs = [{"plane": "liquid", "n_cycles": 3}] if quick else None
    n_samples = 100 if quick else 200
    n_features = 5

    print(f"Task: {task} | Samples: {n_samples} | Features: {n_features}")
    print(f"Mode: {'quick' if quick else 'full'}\n")
    runner = ExperimentRunner(configs=configs)
    summary = runner.run(task=task, n_samples=n_samples, n_features=n_features)

    # Print formatted table
    metric_label = "R²" if task == "regression" else "Accuracy"
    col_w = max(30, max(len(str(r.config)) for r in summary.results) + 2)
    header = f"{'Config':<{col_w}}  {metric_label:>8}  {'Time(s)':>8}"
    sep = f"{'-' * col_w}  {'-' * 8}  {'-' * 8}"
    print(header)
    print(sep)
    for r in summary.results:
        cfg_str = str(r.config)
        score_str = f"{r.score:.4f}" if r.error is None else "FAILED"
        print(f"{cfg_str:<{col_w}}  {score_str:>8}  {r.fit_time_s:>8.3f}")
    print(sep)
    print(f"\nBest config : {summary.best_config}")
    print(f"Best {metric_label:<9}: {summary.best_score:.4f}")
    print(f"Mean ± std  : {summary.mean_score:.4f} ± {summary.std_score:.4f}")
    print(f"Total time  : {summary.total_time_s:.1f}s")

    # Optional Claude analysis
    from physml.llm import ClaudeClient
    client = ClaudeClient()
    if client.available and not getattr(args, "no_llm", False):
        analysis = runner.analyze_with_llm(summary, client)
        if analysis:
            print(f"\nClaude analysis:\n{analysis}")
    else:
        print("\n(Set ANTHROPIC_API_KEY for Claude-powered analysis.)")


def _cmd_version(_args: argparse.Namespace) -> None:
    """Print the physml version string."""
    from physml import __version__
    print(f"physml {__version__}")


def _cmd_status(_args: argparse.Namespace) -> None:
    """Run a health-check and print dependency status table."""
    from physml.health import check

    status = check()
    print(f"physml {status.pop('version')}\n")
    print(f"{'Dependency':<25}  {'Status':<10}")
    print(f"{'-' * 25}  {'-' * 10}")
    for dep, ok in status.items():
        mark = "OK" if ok else "missing"
        print(f"{dep:<25}  {mark}")
    print()
    missing = [dep for dep, ok in status.items() if not ok]
    if missing:
        print("To install missing dependencies:")
        install_hints = {
            "anthropic": "pip install anthropic",
            "scipy": "pip install scipy",
            "pandas": "pip install pandas",
            "speech_recognition": "pip install SpeechRecognition pyaudio",
            "pyttsx3": "pip install pyttsx3",
            "whisper": "pip install openai-whisper",
            "sounddevice": "pip install sounddevice",
        }
        for dep in missing:
            hint = install_hints.get(dep, f"pip install {dep}")
            print(f"  {hint}")
    else:
        print("All dependencies present.")


def _cmd_explain(args: argparse.Namespace) -> None:
    """Ask Claude to explain what a saved agent has learned."""
    import pickle
    from physml.llm import ClaudeClient

    agent_path = args.agent
    try:
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
    except Exception as exc:
        _die(f"Could not load agent from {agent_path!r}: {exc}")

    # Gather stats from the agent
    lines = [f"Agent loaded from: {agent_path}"]
    for attr in ("n_samples_seen_", "classes_", "feature_importances_", "runtime_state_"):
        val = getattr(agent, attr, None)
        if val is not None:
            lines.append(f"  {attr}: {val}")
    summary = "\n".join(lines)

    client = ClaudeClient()
    if client.available:
        prompt = (
            f"The user has a trained Mycelium physics-ML agent. Here are its stats:\n\n"
            f"{summary}\n\n"
            "Explain in 3-5 plain English sentences what this model has learned, "
            "how confident it is, and one practical suggestion for improving it."
        )
        result = client.chat(prompt)
        print(result.text)
    else:
        print(summary)
        print("\n(Set ANTHROPIC_API_KEY for a plain-English explanation.)")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physml",
        description="PhysML / myco command-line interface.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── fit ──────────────────────────────────────────────────────────────
    p_fit = sub.add_parser("fit", help="Train a myco agent on a CSV file.")
    p_fit.add_argument("csv", metavar="CSV", help="Path to the training CSV file.")
    p_fit.add_argument(
        "--target", "-t", required=True,
        help="Name of the target column in the CSV.",
    )
    p_fit.add_argument(
        "--out", "-o", default="agent.pkl",
        help="Output path for the saved agent (default: agent.pkl).",
    )
    p_fit.add_argument(
        "--threshold", type=float, default=0.35,
        help="Uncertainty threshold (default: 0.35).",
    )
    p_fit.add_argument(
        "--query-strategy", dest="query_strategy",
        choices=["entropy", "threshold"], default="entropy",
        help="Active-learning query strategy (default: entropy).",
    )
    p_fit.add_argument(
        "--policy", choices=["adaptive", "fixed"], default="adaptive",
        help="Threshold policy (default: adaptive).",
    )
    p_fit.set_defaults(func=_cmd_fit)

    # ── query ─────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Predict on new samples using a saved agent.")
    p_query.add_argument("agent", metavar="AGENT", help="Path to a saved agent (.pkl).")
    p_query.add_argument("csv", metavar="CSV", help="Path to the input CSV file.")
    p_query.add_argument(
        "--out", "-o", default="predictions.csv",
        help="Output CSV path (default: predictions.csv).",
    )
    p_query.set_defaults(func=_cmd_query)

    # ── report ────────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Print a summary of a saved agent.")
    p_report.add_argument("agent", metavar="AGENT", help="Path to a saved agent (.pkl).")
    p_report.add_argument(
        "--json", action="store_true",
        help="Output in JSON format.",
    )
    p_report.set_defaults(func=_cmd_report)

    # ── export ────────────────────────────────────────────────────────────
    p_export = sub.add_parser(
        "export",
        help="Export predictions to CSV (alias for query).",
    )
    p_export.add_argument("agent", metavar="AGENT", help="Path to a saved agent (.pkl).")
    p_export.add_argument("csv", metavar="CSV", help="Path to the input CSV file.")
    p_export.add_argument(
        "--out", "-o", default="predictions.csv",
        help="Output CSV path (default: predictions.csv).",
    )
    p_export.set_defaults(func=_cmd_export)

    # ── chat (REPL) ───────────────────────────────────────────────────────
    p_chat = sub.add_parser(
        "chat",
        help="Start a natural-language REPL (uses Claude if ANTHROPIC_API_KEY is set).",
    )
    p_chat.add_argument(
        "--session", "-s", default="default",
        help="Session name for persistent conversation history (default: 'default').",
    )
    p_chat.set_defaults(func=_cmd_chat)

    # ── explain ───────────────────────────────────────────────────────────
    p_explain = sub.add_parser(
        "explain",
        help="Ask Claude to explain what a saved agent has learned (requires ANTHROPIC_API_KEY).",
    )
    p_explain.add_argument("agent", metavar="AGENT", help="Path to a saved agent (.pkl).")
    p_explain.set_defaults(func=_cmd_explain)

    # ── voice ─────────────────────────────────────────────────────────────
    p_voice = sub.add_parser(
        "voice",
        help="Start a voice interaction loop (falls back to text if speech_recognition missing).",
    )
    p_voice.add_argument(
        "--session", "-s", default="default",
        help="Session name for persistent conversation history (default: 'default').",
    )
    p_voice.add_argument(
        "--no-tts", dest="no_tts", action="store_true",
        help="Disable text-to-speech output.",
    )
    p_voice.add_argument(
        "--language", default="en-US",
        help="BCP-47 language tag for speech recognition (default: en-US).",
    )
    p_voice.set_defaults(func=_cmd_voice)

    # ── experiment ────────────────────────────────────────────────────────
    p_exp = sub.add_parser(
        "experiment",
        help="Run benchmark experiments on synthetic data with optional Claude analysis.",
    )
    p_exp.add_argument(
        "--quick", action="store_true",
        help="Run a quick single-config benchmark (100 samples).",
    )
    p_exp.add_argument(
        "--task", choices=["regression", "classification"], default="regression",
        help="Prediction task type (default: regression).",
    )
    p_exp.add_argument(
        "--no-llm", dest="no_llm", action="store_true",
        help="Skip Claude analysis even if ANTHROPIC_API_KEY is set.",
    )
    p_exp.set_defaults(func=_cmd_experiment)

    # ── version ───────────────────────────────────────────────────────────
    p_ver = sub.add_parser("version", help="Print the physml version and exit.")
    p_ver.set_defaults(func=_cmd_version)

    # ── status ────────────────────────────────────────────────────────────
    p_status = sub.add_parser(
        "status",
        help="Run a dependency health-check and print results.",
    )
    p_status.set_defaults(func=_cmd_status)

    # ── ingest ────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest a file, URL, or text into Mycelium's learning stack.",
    )
    p_ingest.add_argument("source", help="File path, URL, or quoted text string.")
    p_ingest.add_argument("--topic", default="document", help="Topic tag (default: document).")
    p_ingest.set_defaults(func=_cmd_ingest)

    # ── observe ───────────────────────────────────────────────────────────
    p_obs = sub.add_parser(
        "observe",
        help="Start the background screen observer.",
    )
    p_obs.add_argument("--interval", type=float, default=60.0, help="Seconds between snapshots.")
    p_obs.add_argument("--save-screenshots", action="store_true", help="Save PNG screenshots to disk.")
    p_obs.add_argument("--no-llm", action="store_true", help="Disable Claude vision descriptions.")
    p_obs.set_defaults(func=_cmd_observe)

    # ── record ────────────────────────────────────────────────────────────
    p_rec = sub.add_parser(
        "record",
        help="Record a macro (user action sequence) and save as a Skill.",
    )
    p_rec.add_argument("name", help="Name for the recorded macro.")
    p_rec.set_defaults(func=_cmd_record)

    # ── model ─────────────────────────────────────────────────────────────
    p_model = sub.add_parser(
        "model",
        help="Show the current user model (context, preferences, patterns).",
    )
    p_model.set_defaults(func=_cmd_model)

    return parser


# ---------------------------------------------------------------------------
# v1.1 commands
# ---------------------------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a file, URL, or raw text into the learning stack."""
    from physml.multimodal_ingester import MultiModalIngester
    ing = MultiModalIngester()
    print(f"Ingesting: {args.source!r} (topic={args.topic!r})")
    result = ing.ingest(args.source, topic=args.topic)
    if result.success:
        print(f"  text chars : {len(result.text)}")
        print(f"  facts found: {len(result.facts)}")
        print(f"  elapsed    : {result.elapsed:.2f}s")
        if result.facts:
            print("  top facts  :")
            for f in result.facts[:5]:
                print(f"    {f.get('subject')} → {f.get('predicate')} → {f.get('object')}")
    else:
        print(f"  error: {result.error}")


def _cmd_observe(args: argparse.Namespace) -> None:
    """Start background screen observer (runs until Ctrl-C)."""
    from physml.screen_observer import ScreenObserver
    obs = ScreenObserver(
        interval=args.interval,
        save_screenshots=args.save_screenshots,
        llm_describe=not args.no_llm,
    )
    print(f"Screen observer started (interval={args.interval:.0f}s). Press Ctrl-C to stop.")
    obs.start()
    try:
        import time
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        print("\nFocus summary:")
        for app, secs in obs.top_apps(10):
            print(f"  {app:<30} {secs:.0f}s")


def _cmd_record(args: argparse.Namespace) -> None:
    """Record a macro and save as a Skill."""
    from physml.macro_recorder import MacroRecorder
    recorder = MacroRecorder()
    if not recorder.available:
        print("pynput not installed — manual recording only. Install: pip install pynput")
        print("Recording text-only demo sequence...")
        seq = recorder.record_text_sequence(args.name, [
            {"action_type": "click", "x": 100, "y": 100, "app_name": "demo"},
        ])
        saved = recorder.save_to_skill_library(seq)
        print(f"Demo sequence saved to skill library: {saved}")
        return

    print(f"Recording macro {args.name!r} — perform your actions. Press Ctrl-C when done.")
    recorder.start_recording(args.name)
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    seq = recorder.stop_recording()
    if seq is None:
        print("Too few steps recorded.")
        return
    path = recorder.save_sequence(seq)
    saved = recorder.save_to_skill_library(seq)
    print(f"Macro {seq.name!r}: {len(seq.steps)} steps, {seq.duration:.1f}s")
    print(f"Saved to: {path}")
    print(f"Registered as Skill: {saved}")


def _cmd_model(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Show the current user model."""
    from physml.user_model import UserModel
    model = UserModel()
    ctx = model.current_context()
    print("User Model — current context:")
    for k, v in ctx.items():
        print(f"  {k:<20} {v}")
    patterns = model.behavioral_patterns()
    if patterns:
        print("\nBehavioral patterns:")
        for p in patterns[:5]:
            print(f"  • {p.get('description', '')}")
    else:
        print("\nNo patterns detected yet. Interact more with Myco to build your model.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Main entry-point for the ``physml`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
