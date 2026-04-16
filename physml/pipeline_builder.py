"""Stage 89 — PipelineBuilder: sklearn Pipeline factory.

Provides a fluent builder API for assembling scikit-learn ``Pipeline``
objects from named transformation steps and a final estimator.

Classes
-------
PipelineStep
    Metadata about a single step in the pipeline.
PipelineBuilder
    Accumulates steps and produces a sklearn Pipeline on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineStep:
    """Metadata for a single pipeline step.

    Attributes
    ----------
    name : str
        Unique identifier for this step in the pipeline.
    component : Any
        The transformer or estimator instance.
    is_estimator : bool
        ``True`` when this is the final (estimator) step.
    """

    name: str
    component: Any
    is_estimator: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": repr(self.component),
            "is_estimator": self.is_estimator,
        }

    def __repr__(self) -> str:
        return (
            f"PipelineStep(name={self.name!r}, "
            f"component={self.component!r}, "
            f"is_estimator={self.is_estimator})"
        )


class PipelineBuilder:
    """Fluent builder for sklearn ``Pipeline`` objects.

    Steps are accumulated in insertion order.  Calling :meth:`build`
    assembles them into a ``Pipeline``.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = (
    ...     PipelineBuilder()
    ...     .add_step("scaler", StandardScaler())
    ...     .add_estimator("clf", LogisticRegression())
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._steps: list[PipelineStep] = []

    # ------------------------------------------------------------------
    # Step registration
    # ------------------------------------------------------------------

    def add_step(self, name: str, transformer: Any) -> "PipelineBuilder":
        """Append a transformer step.

        Parameters
        ----------
        name : str
            Unique name for this step.
        transformer : sklearn transformer
            Must implement ``fit`` / ``transform``.

        Returns
        -------
        self
        """
        self._validate_name(name)
        self._steps.append(PipelineStep(name=name, component=transformer, is_estimator=False))
        return self

    def add_estimator(self, name: str, estimator: Any) -> "PipelineBuilder":
        """Set the final estimator step (replaces any previously set estimator).

        Parameters
        ----------
        name : str
            Unique name for the estimator step.
        estimator : sklearn estimator
            Must implement ``fit`` / ``predict``.

        Returns
        -------
        self
        """
        self._validate_name(name)
        # Remove any existing estimator step before appending the new one
        self._steps = [s for s in self._steps if not s.is_estimator]
        self._steps.append(PipelineStep(name=name, component=estimator, is_estimator=True))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Any:
        """Assemble and return a scikit-learn ``Pipeline``.

        The steps are used in insertion order.  Transformer steps come
        first; the estimator step (if any) is placed last.

        Returns
        -------
        sklearn.pipeline.Pipeline

        Raises
        ------
        ValueError
            If no steps have been added.
        """
        from sklearn.pipeline import Pipeline

        if not self._steps:
            raise ValueError("PipelineBuilder has no steps. Add at least one step.")

        # Separate transformers from estimator to enforce correct ordering
        transformers = [s for s in self._steps if not s.is_estimator]
        estimators = [s for s in self._steps if s.is_estimator]
        ordered = transformers + estimators
        sklearn_steps = [(s.name, s.component) for s in ordered]
        return Pipeline(sklearn_steps)

    # ------------------------------------------------------------------
    # Properties / introspection
    # ------------------------------------------------------------------

    @property
    def step_names(self) -> list[str]:
        """Names of all registered steps (in insertion order)."""
        return [s.name for s in self._steps]

    @property
    def steps(self) -> list[PipelineStep]:
        """All registered :class:`PipelineStep` objects."""
        return list(self._steps)

    def get_step(self, name: str) -> PipelineStep:
        """Return the step registered under *name*.

        Raises
        ------
        KeyError
            If no step with that name exists.
        """
        for step in self._steps:
            if step.name == name:
                return step
        raise KeyError(f"No step named {name!r}.")

    def clear(self) -> "PipelineBuilder":
        """Remove all registered steps.

        Returns
        -------
        self
        """
        self._steps = []
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Step name must be a non-empty string.")
        existing = [s.name for s in self._steps if not s.is_estimator]
        if name in existing:
            raise ValueError(f"A transformer step named {name!r} already exists.")

    def __repr__(self) -> str:
        names = ", ".join(self.step_names)
        return f"PipelineBuilder(steps=[{names}])"
