"""Stage 44 — Structured tool-calling protocol.

Extends the Stage 31 tool infrastructure with:

* :class:`ToolSpec` — JSON-schema based tool specification (name, description,
  ``input_schema`` dict). Backward compatible with bare :class:`~physml.tools.Tool`.
* :class:`ToolCall` — typed return from :class:`ToolPlanner` carrying the
  chosen tool name, parsed arguments, and confidence score.
* :class:`ToolPlanner` — selects and structures a tool call from a goal
  string using embedding cosine similarity + prior success rate from
  :class:`~physml.memory.EpisodicMemory`.  Falls back gracefully when memory
  or embeddings are unavailable.

Design rationale
----------------
The Stage 31 ``_pick_tool`` was a private helper using raw cosine similarity
over character-ngram vectors.  Stage 44 makes tool selection a first-class,
auditable operation:

1. Each tool carries a JSON schema so calling code can validate inputs.
2. ``ToolPlanner.plan()`` combines embedding similarity with an empirical
   success rate from episodic memory, producing a ranked list of candidates.
3. The structured :class:`ToolCall` makes the selection traceable and
   allows downstream components (e.g. ``AutonomousLoop``) to log structured
   audit trails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from physml.featurizer import Featurizer
    from physml.memory import EpisodicMemory


@dataclass
class ToolSpec:
    """A tool with a JSON-schema input specification.

    Parameters
    ----------
    name : str
        Unique tool identifier.
    description : str
        Human-readable description used for semantic matching.
    fn : Callable[[str], str]
        The tool function — takes a string, returns a string.
    input_schema : dict, optional
        JSON Schema dict describing the expected input.  Example::

            {"type": "object", "properties": {"query": {"type": "string"}}}

        When omitted the tool accepts any string without validation.
    """

    name: str
    description: str
    fn: Callable[[str], str]
    input_schema: dict = field(default_factory=dict)

    def validate_input(self, input_str: str) -> bool:
        """Return ``True`` if *input_str* satisfies ``input_schema``.

        Currently validates ``minLength`` and ``maxLength`` for plain-string
        schemas.  Richer validation requires ``jsonschema`` (optional dep).
        """
        if not self.input_schema:
            return True
        schema_type = self.input_schema.get("type", "string")
        if schema_type == "string":
            min_len = self.input_schema.get("minLength", 0)
            max_len = self.input_schema.get("maxLength", 2**31)
            return min_len <= len(input_str) <= max_len
        # For object schemas, try jsonschema if available
        try:
            import jsonschema  # type: ignore[import-untyped]
            try:
                parsed = json.loads(input_str)
            except json.JSONDecodeError:
                return False
            jsonschema.validate(parsed, self.input_schema)
            return True
        except Exception:
            return True  # permissive fallback


@dataclass
class ToolCall:
    """The structured result of :class:`ToolPlanner` tool selection.

    Attributes
    ----------
    tool_name : str
        Name of the selected tool.
    input_str : str
        The input that should be passed to the tool function.
    confidence : float
        Similarity score in [0, 1] — higher means better match.
    schema_valid : bool
        Whether *input_str* satisfies the tool's ``input_schema``.
    ranked_alternatives : list[str]
        Names of other candidate tools in descending score order.
    """

    tool_name: str
    input_str: str
    confidence: float
    schema_valid: bool = True
    ranked_alternatives: list[str] = field(default_factory=list)


class ToolPlanner:
    """Select tools via embedding similarity + memory-based success rates.

    Parameters
    ----------
    featurizer : Featurizer
        Fitted featurizer used to embed goal and tool descriptions.
    memory : EpisodicMemory or None, default None
        Episodic memory whose stored actions (tool names) are used to compute
        empirical success rates.  When ``None``, only similarity is used.
    memory_weight : float, default 0.3
        Weight of the memory-derived success rate in the final score.
        ``final_score = (1 - memory_weight) * similarity + memory_weight * success_rate``
    """

    def __init__(
        self,
        featurizer: "Featurizer",
        memory: "EpisodicMemory | None" = None,
        memory_weight: float = 0.3,
    ) -> None:
        self.featurizer = featurizer
        self.memory = memory
        self.memory_weight = float(np.clip(memory_weight, 0.0, 1.0))

        self._tool_specs: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> "ToolPlanner":
        """Register a :class:`ToolSpec`.

        Parameters
        ----------
        spec : ToolSpec

        Returns
        -------
        self
        """
        self._tool_specs[spec.name] = spec
        return self

    def plan(self, goal: str) -> ToolCall:
        """Select the best tool for *goal* and return a :class:`ToolCall`.

        Parameters
        ----------
        goal : str
            Natural-language goal or query string.

        Returns
        -------
        ToolCall

        Raises
        ------
        RuntimeError
            If no tools are registered.
        """
        if not self._tool_specs:
            raise RuntimeError("No tools registered. Call register() first.")

        goal_vec = self.featurizer.transform([goal])[0]
        success_rates = self._compute_success_rates()

        scores: dict[str, float] = {}
        for name, spec in self._tool_specs.items():
            sim = self._cosine_similarity(goal_vec, spec)
            sr = success_rates.get(name, 0.5)  # default 0.5 (unknown)
            scores[name] = (1.0 - self.memory_weight) * sim + self.memory_weight * sr

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        best_name = ranked[0]
        best_score = float(scores[best_name])
        alternatives = ranked[1:]

        spec = self._tool_specs[best_name]
        schema_valid = spec.validate_input(goal)

        return ToolCall(
            tool_name=best_name,
            input_str=goal,
            confidence=best_score,
            schema_valid=schema_valid,
            ranked_alternatives=alternatives,
        )

    def execute(self, tool_call: ToolCall) -> str:
        """Execute the tool described by *tool_call*.

        Parameters
        ----------
        tool_call : ToolCall

        Returns
        -------
        str — raw tool output.

        Raises
        ------
        KeyError
            If ``tool_call.tool_name`` is not registered.
        """
        spec = self._tool_specs[tool_call.tool_name]
        return spec.fn(tool_call.input_str)

    def plan_and_execute(self, goal: str) -> tuple[ToolCall, str]:
        """Plan and immediately execute the best tool for *goal*.

        Returns
        -------
        (ToolCall, str) — the selection metadata and the tool's output.
        """
        call = self.plan(goal)
        output = self.execute(call)
        return call, output

    def list_specs(self) -> list[dict]:
        """Return all registered tool specs as plain dicts."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            }
            for s in self._tool_specs.values()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, goal_vec: np.ndarray, spec: ToolSpec) -> float:
        """Compute cosine similarity between *goal_vec* and *spec* description."""
        try:
            desc_text = spec.name + " " + spec.description
            desc_vec = self.featurizer.transform([desc_text])[0]
            denom = (np.linalg.norm(goal_vec) * np.linalg.norm(desc_vec)) + 1e-8
            return float(np.dot(goal_vec, desc_vec) / denom)
        except Exception:
            return 0.0

    def _compute_success_rates(self) -> dict[str, float]:
        """Derive per-tool success rates from episodic memory.

        Returns a dict mapping tool name → mean outcome in [0, 1].
        Missing tools get a prior of 0.5.
        """
        if self.memory is None or len(self.memory) == 0:
            return {}

        actions = list(self.memory._actions)
        outcomes = list(self.memory._outcomes)

        totals: dict[str, list[float]] = {}
        for act, out in zip(actions, outcomes):
            totals.setdefault(act, []).append(float(out))

        return {
            name: float(np.mean(vals))
            for name, vals in totals.items()
        }
