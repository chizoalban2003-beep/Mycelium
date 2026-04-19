"""Stage 124 — ToolBridge: execute LLM tool calls on the local device.

When :class:`~physml.llm_integration.LLMIntegration` returns structured
tool calls from Claude, this module routes them to the appropriate local
subsystem and returns results that can be fed back to the LLM for a
complete agentic loop.

Tools available to Claude
--------------------------
``run_prediction``
    Predict a target value from a feature vector using the fitted model.
``train_on_file``
    Load a local CSV file and train / update the model.
``read_document``
    Read and summarise a local file (CSV, JSON, TXT, PDF).
``show_report``
    Return the companion's current status and model report.
``execute_task``
    Run a safe shell command or file operation via
    :class:`~physml.local_executor.LocalTaskExecutor`.

Usage
-----
::

    from physml.tool_bridge import ToolBridge, build_tool_definitions

    bridge = ToolBridge(companion=companion)

    # Get tool specs to pass to Claude
    tools = build_tool_definitions()

    # Execute a tool call returned by LLM
    result_text = bridge.execute(tool_name="run_prediction", tool_input={"features": [1.2, 3.4]})
    print(result_text)   # "Prediction: 42.3  (confidence 87%)"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions for Claude API
# ---------------------------------------------------------------------------


def build_tool_definitions() -> List[Dict[str, Any]]:
    """Return Anthropic-format tool definitions for the companion's capabilities.

    Returns
    -------
    list of dict
        Tool specs ready to pass as ``tools=`` in
        :meth:`~physml.llm_integration.LLMIntegration.chat`.
    """
    return [
        {
            "name": "run_prediction",
            "description": (
                "Run a prediction using the locally trained physics ML model. "
                "Provide feature values as a list of numbers. Returns the predicted "
                "value and confidence score."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric feature values in order.",
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional names for each feature value.",
                    },
                },
                "required": ["features"],
            },
        },
        {
            "name": "train_on_file",
            "description": (
                "Train or update the local ML model from a CSV file. "
                "The model learns to predict the target column. "
                "Optionally specify which column is the target."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to a CSV file.",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column (auto-detected if omitted).",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "read_document",
            "description": (
                "Read and process a local file (CSV, JSON, TXT, PDF). "
                "Returns a summary and relevant content from the file."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "show_report",
            "description": (
                "Show the current model status, training history, accuracy metrics, "
                "and companion system state."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "execute_task",
            "description": (
                "Execute a safe local task: list files in a directory, read a file, "
                "check if a path exists, get system info. "
                "Only safe, non-destructive operations are permitted."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_dir", "read_file", "exists", "system_info"],
                        "description": "The action to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (required for list_dir, read_file, exists).",
                    },
                },
                "required": ["action"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# ToolBridge
# ---------------------------------------------------------------------------


class ToolBridge:
    """Execute LLM tool calls using companion subsystems.

    Parameters
    ----------
    companion : MyceliumCompanion or None
        The companion instance whose subsystems handle the calls.
        When ``None``, a lightweight standalone mode is used.
    """

    def __init__(self, companion: Any = None) -> None:
        self._companion = companion

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Dispatch a tool call and return the result as a string.

        Parameters
        ----------
        tool_name : str
        tool_input : dict

        Returns
        -------
        str
            Human-readable result, or an error message.
        """
        try:
            if tool_name == "run_prediction":
                return self._run_prediction(tool_input)
            elif tool_name == "train_on_file":
                return self._train_on_file(tool_input)
            elif tool_name == "read_document":
                return self._read_document(tool_input)
            elif tool_name == "show_report":
                return self._show_report()
            elif tool_name == "execute_task":
                return self._execute_task(tool_input)
            else:
                return f"Unknown tool: {tool_name!r}"
        except Exception as exc:
            _logger.warning("ToolBridge.execute(%r): %s", tool_name, exc)
            return f"Tool execution error: {exc}"

    def execute_all(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return results in API format.

        Parameters
        ----------
        tool_calls : list of dict
            Each dict has ``id``, ``name``, ``input`` keys (from LLMResult).

        Returns
        -------
        list of dict
            Each dict has ``type``, ``tool_use_id``, ``content`` keys ready
            to append to a messages list for the next LLM call.
        """
        results = []
        for call in tool_calls:
            result_text = self.execute(
                tool_name=call.get("name", ""),
                tool_input=call.get("input", {}),
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.get("id", ""),
                    "content": result_text,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _run_prediction(self, inp: Dict[str, Any]) -> str:
        features = inp.get("features", [])
        feature_names = inp.get("feature_names")

        if not features:
            return "Please provide feature values as a list of numbers."

        if self._companion is not None and hasattr(self._companion, "model_manager"):
            mgr = self._companion.model_manager
            result = mgr.predict(features, feature_names=feature_names)
            if not result.model_fitted:
                return result.error or "No model trained yet."
            if result.error:
                return f"Prediction error: {result.error}"
            fnames = result.feature_names or [f"x{i}" for i in range(len(features))]
            feature_str = ", ".join(
                f"{n}={v:.3g}" for n, v in zip(fnames[: len(features)], features)
            )
            return (
                f"Prediction: **{result.value:.4g}**  (confidence {result.confidence:.0%})\n"
                f"Target: {result.target_column}\n"
                f"Features: {feature_str}"
            )
        return "No model available. Train one first with 'train on <file.csv>'."

    def _train_on_file(self, inp: Dict[str, Any]) -> str:
        path = inp.get("path", "")
        target = inp.get("target_column")

        if not path:
            return "Please provide a file path."

        if self._companion is not None and hasattr(self._companion, "model_manager"):
            mgr = self._companion.model_manager
            result = mgr.train_from_csv(path, target_column=target)
            if result.success:
                mgr.save()  # persist immediately
            return result.message
        return "Model manager not available."

    def _read_document(self, inp: Dict[str, Any]) -> str:
        path = inp.get("path", "")
        if not path:
            return "Please provide a file path."

        if self._companion is not None and hasattr(self._companion, "doc_processor"):
            result = self._companion.doc_processor.process(path)
            if not result.success:
                return f"Could not read file: {result.error}"
            lines = [f"File: {path}"]
            meta = result.metadata or {}
            if meta.get("rows"):
                lines.append(f"Rows: {meta['rows']}, Columns: {meta.get('n_columns', '?')}")
            if meta.get("chars"):
                lines.append(f"Characters: {meta['chars']}")
            if result.text:
                preview = result.text[:600]
                lines.append(f"Content preview:\n{preview}")
            return "\n".join(lines)
        return "Document processor not available."

    def _show_report(self) -> str:
        if self._companion is None:
            return "Companion not connected."
        parts = []
        s = self._companion.status()
        parts.append(f"System status: {s.get('name', 'Mycelium')}")
        parts.append(f"  Mood: {s.get('mood', 'unknown')}")
        parts.append(f"  Total interactions: {s.get('total_interactions', 0)}")
        parts.append(f"  Total predictions: {s.get('total_predictions', 0)}")
        parts.append(f"  Days alive: {s.get('days_alive', 0)}")
        if hasattr(self._companion, "model_manager"):
            ms = self._companion.model_manager.status()
            if ms["fitted"]:
                parts.append(f"  Model: trained on {ms['n_training_rows']} rows")
                if ms["last_accuracy"] is not None:
                    parts.append(f"  Accuracy: {ms['last_accuracy']:.2%}")
                parts.append(f"  Predictions made: {ms['n_predictions']}")
            else:
                parts.append("  Model: not trained yet")
        return "\n".join(parts)

    def _execute_task(self, inp: Dict[str, Any]) -> str:
        action = inp.get("action", "")
        path = inp.get("path", "")

        if self._companion is None or not hasattr(self._companion, "executor"):
            # Standalone safe fallback
            import os

            if action == "system_info":
                return f"Platform: {os.uname().sysname}, CWD: {os.getcwd()}"
            if action == "exists":
                return f"Path {path!r}: {'exists' if os.path.exists(path) else 'does not exist'}"
            return "Executor not available."

        executor = self._companion.executor
        try:
            if action == "list_dir":
                result = executor.list_dir(path or ".")
            elif action == "read_file":
                result = executor.read_file(path)
            elif action == "exists":
                result = executor.exists(path)
            elif action == "system_info":
                result = executor.system_info()
            else:
                return f"Unknown action: {action!r}"

            if hasattr(result, "output"):
                return str(result.output)
            return str(result)

        except Exception as exc:
            return f"Task error: {exc}"
