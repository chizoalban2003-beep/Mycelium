"""Stage 31 — Tool-calling support and autonomous agent loop.

Provides:
* :class:`Tool` — a named, callable function with a description.
* :class:`ToolRegistry` — register and invoke named tools.
* :class:`AutonomousLoop` — wraps :class:`~physml.mycelium_agent.MyceliumAgent`
  with a :class:`ToolRegistry` and :class:`~physml.featurizer.Featurizer` for
  agentic tool-use loops driven by a text *goal*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from physml.featurizer import Featurizer
    from physml.mycelium_agent import MyceliumAgent


@dataclass
class Tool:
    """A named callable with a human-readable description.

    Parameters
    ----------
    name : str
        Unique identifier used to look up the tool.
    description : str
        Short description of what the tool does (used for relevance scoring).
    fn : Callable[[str], str]
        The actual function: takes a string input, returns a string output.
    """

    name: str
    description: str
    fn: Callable[[str], str]


class ToolRegistry:
    """Registry for :class:`Tool` objects.

    Tools can be registered by name and invoked via :meth:`call`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register *tool*, overwriting any previous tool with the same name.

        Parameters
        ----------
        tool : Tool
        """
        self._tools[tool.name] = tool

    def call(self, name: str, input_str: str) -> str:
        """Invoke the tool named *name* with *input_str*.

        Parameters
        ----------
        name : str
        input_str : str

        Returns
        -------
        str

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name].fn(input_str)

    def list_tools(self) -> list[dict]:
        """Return a list of ``{name, description}`` dicts for all registered tools.

        Returns
        -------
        list[dict]
        """
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]


class AutonomousLoop:
    """Wraps :class:`~physml.mycelium_agent.MyceliumAgent` + :class:`ToolRegistry`
    for agentic tool-use loops.

    Given a text *goal*, the loop featurizes it, asks the agent whether to
    predict or call a tool, and iterates up to *max_steps* times.

    Parameters
    ----------
    agent : MyceliumAgent
        A fitted MyceliumAgent used for action selection.
    registry : ToolRegistry
        Collection of available tools.
    featurizer : Featurizer
        A fitted Featurizer used to embed text strings.
    max_steps : int, default 10
        Maximum number of iterations.
    """

    def __init__(
        self,
        agent: "MyceliumAgent",
        registry: ToolRegistry,
        featurizer: "Featurizer",
        max_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.registry = registry
        self.featurizer = featurizer
        self.max_steps = int(max_steps)

    def run(self, goal: str) -> dict:
        """Execute the agentic loop for *goal*.

        Returns
        -------
        dict with keys:
            ``steps`` (list of step dicts), ``result`` (str), ``n_tool_calls`` (int).
        """
        steps: list[dict] = []
        n_tool_calls = 0
        result: str = goal
        tools = self.registry.list_tools()

        # Featurize the goal
        goal_vec: np.ndarray = self.featurizer.transform([goal])[0]

        for step_idx in range(self.max_steps):
            step_info: dict[str, Any] = {"step": step_idx}

            # Get action from agent
            try:
                action = self.agent.observe(goal_vec.reshape(1, -1))
            except Exception as exc:
                step_info["error"] = str(exc)
                steps.append(step_info)
                break

            action_label = getattr(action, "action", str(action))
            confidence = getattr(action, "confidence", 1.0)
            step_info["action"] = str(action_label)

            # Decide whether to call a tool
            should_use_tool = (action_label == "ask") or (confidence is not None and float(confidence) < 0.5)

            if should_use_tool and tools:
                tool_name = self._pick_tool(goal, tools)
                tool_output = self.registry.call(tool_name, goal)
                n_tool_calls += 1
                step_info["tool"] = tool_name
                step_info["tool_output"] = tool_output
                result = tool_output

                # Update agent with tool-output feedback
                try:
                    tool_vec = self.featurizer.transform([tool_output])[0]
                    self.agent.reward(tool_vec.reshape(1, -1), np.array([1.0]))
                except Exception:
                    pass
            else:
                # Agent made a confident prediction — stop
                prediction = getattr(action, "prediction", None)
                step_info["prediction"] = str(prediction) if prediction is not None else ""
                if prediction is not None:
                    result = str(prediction)
                steps.append(step_info)
                break

            steps.append(step_info)

        return {"steps": steps, "result": result, "n_tool_calls": n_tool_calls}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_tool(self, goal: str, tools: list[dict]) -> str:
        """Return the name of the most relevant tool via cosine similarity."""
        if len(tools) == 1:
            return tools[0]["name"]

        goal_vec = self.featurizer.transform([goal])[0]
        best_name = tools[0]["name"]
        best_sim = -2.0

        for t in tools:
            try:
                desc_vec = self.featurizer.transform([t["name"] + " " + t["description"]])[0]
                denom = (np.linalg.norm(goal_vec) * np.linalg.norm(desc_vec)) + 1e-8
                sim = float(np.dot(goal_vec, desc_vec) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_name = t["name"]
            except Exception:
                pass

        return best_name
