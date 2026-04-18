"""Stage 115 — LocalREPL: interactive readline-based agent REPL.

Starts a readline-based prompt loop that:
1. Accepts natural-language commands.
2. Routes them via :class:`~physml.nl_router.NaturalLanguageRouter`.
3. Executes via :class:`~physml.local_executor.LocalTaskExecutor` or
   :class:`~physml.mycelium_agent.MyceliumAgent`.
4. Formats output via :class:`~physml.response_formatter.ResponseFormatter`.
5. Prints the response.

History is saved to ``~/.mycelium/history.txt``.
Special commands: ``/help``, ``/status``, ``/save``, ``/quit``.

Usage
-----
::

    from physml.local_repl import LocalREPL

    repl = LocalREPL(agent=mycelium_agent, profile=user_profile)
    repl.run()   # blocking readline loop

    # Single command execution (non-interactive)
    result = repl.execute("predict for values 1.2 3.4")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

_HELP_TEXT = """
Mycelium Companion REPL
-----------------------
Type natural-language commands, or use:

  /help    — show this message
  /status  — show system status
  /save    — save current session
  /quit    — exit the REPL

Examples:
  predict for values 1.2 3.4 5.6
  train on data.csv
  show report
  what have you learned about me?
"""


class LocalREPL:
    """Interactive command-line REPL for the Mycelium agent.

    Parameters
    ----------
    agent : any
        The agent object (e.g. :class:`~physml.companion.MyceliumCompanion`
        or :class:`~physml.mycelium_agent.MyceliumAgent`).  Must expose a
        ``chat(text)`` or ``step(text)`` method.
    profile : UserProfileLearner or None
        Optional user profile for personalisation.
    formatter : ResponseFormatter or None
        If ``None``, a default formatter is created.
    history_path : str, default "~/.mycelium/history.txt"
        Path to the readline history file.
    prompt : str, default "myco> "
        REPL prompt string.
    """

    def __init__(
        self,
        agent: Any = None,
        profile: Any = None,
        formatter: Any = None,
        history_path: str = "~/.mycelium/history.txt",
        prompt: str = "myco> ",
    ) -> None:
        self.agent = agent
        self.profile = profile
        self.history_path = Path(history_path).expanduser()
        self.prompt = prompt
        self._running = False

        # Formatter — import lazily to avoid circular deps
        if formatter is not None:
            self.formatter = formatter
        else:
            from physml.response_formatter import ResponseFormatter
            verbosity = "normal"
            if profile is not None:
                verbosity = profile.get_preference("verbosity", "normal")
            self.formatter = ResponseFormatter(verbosity=verbosity)

        # Try to set up readline
        self._readline_available = False
        try:
            import readline  # noqa: F401 # type: ignore

            self._readline_available = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the blocking REPL loop."""
        self._setup_readline()
        self._running = True
        print(_HELP_TEXT.strip())
        print()
        try:
            while self._running:
                try:
                    line = input(self.prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break
                if not line:
                    continue
                self._save_history_entry(line)
                response = self.execute(line)
                print(response)
                print()
        finally:
            self._save_readline_history()
            self._running = False

    def execute(self, text: str) -> str:
        """Execute a single command and return the formatted response.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        text = text.strip()
        if not text:
            return ""

        # Special slash commands
        if text.startswith("/"):
            return self._handle_slash(text)

        # Route through agent
        return self._dispatch(text)

    # ------------------------------------------------------------------
    # Special commands
    # ------------------------------------------------------------------

    def _handle_slash(self, text: str) -> str:
        cmd = text.split()[0].lower()
        if cmd == "/help":
            return _HELP_TEXT.strip()
        elif cmd == "/quit":
            self._running = False
            return "Goodbye!"
        elif cmd == "/status":
            return self._cmd_status()
        elif cmd == "/save":
            return self._cmd_save()
        else:
            return f"Unknown command: {cmd}. Type /help for options."

    def _cmd_status(self) -> str:
        if self.agent is None:
            return "No agent loaded."
        if hasattr(self.agent, "status"):
            try:
                s = self.agent.status()
                return self.formatter.format_report(s)
            except Exception as e:
                return f"Status error: {e}"
        return "Agent has no status() method."

    def _cmd_save(self) -> str:
        saved = []
        if self.profile is not None:
            try:
                self.profile.save()
                saved.append("profile")
            except Exception as e:
                _logger.warning("LocalREPL: profile save failed: %s", e)
        if self.agent is not None and hasattr(self.agent, "save"):
            try:
                self.agent.save()
                saved.append("agent")
            except Exception as e:
                _logger.warning("LocalREPL: agent save failed: %s", e)
        if saved:
            return f"Saved: {', '.join(saved)}."
        return "Nothing to save."

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, text: str) -> str:
        if self.agent is None:
            return "(No agent configured)"

        # MyceliumCompanion-style
        if hasattr(self.agent, "chat"):
            try:
                result = self.agent.chat(text)
                return str(result)
            except Exception as e:
                _logger.warning("LocalREPL: chat() failed: %s", e)
                return self.formatter.format_action_result(
                    type("R", (), {"success": False, "operation": "chat", "error": str(e), "output": None})()
                )

        # Generic step / predict
        if hasattr(self.agent, "step"):
            try:
                result = self.agent.step(text)
                return self.formatter.format_action_result(result)
            except Exception as e:
                return f"Error: {e}"

        return "Agent does not expose chat() or step()."

    # ------------------------------------------------------------------
    # Readline helpers
    # ------------------------------------------------------------------

    def _setup_readline(self) -> None:
        if not self._readline_available:
            return
        try:
            import readline

            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            if self.history_path.exists():
                readline.read_history_file(str(self.history_path))
            readline.set_history_length(1000)
        except Exception as e:
            _logger.warning("LocalREPL: readline setup failed: %s", e)

    def _save_readline_history(self) -> None:
        if not self._readline_available:
            return
        try:
            import readline

            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(self.history_path))
        except Exception as e:
            _logger.warning("LocalREPL: could not write readline history: %s", e)

    def _save_history_entry(self, line: str) -> None:
        """Append a raw line to the history file."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            _logger.warning("LocalREPL: could not append history: %s", e)

    def __repr__(self) -> str:
        return f"LocalREPL(agent={type(self.agent).__name__ if self.agent else None!r})"
