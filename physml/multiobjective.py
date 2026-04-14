"""Stage 55 — MultiObjectiveOptimizer: Pareto front for accuracy vs cost.

Implements a lightweight NSGA-II-inspired non-dominated sorting and
crowding-distance selection so users can explore accuracy/cost/fairness
tradeoffs without a full evolutionary framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Solution:
    """A candidate solution with named objectives and optional config."""

    objectives: Dict[str, float]  # name -> value (all treated as "minimise")
    config: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0

    def __repr__(self) -> str:
        obj_str = ", ".join(f"{k}={v:.4f}" for k, v in self.objectives.items())
        return f"Solution({obj_str}, rank={self.rank})"


class MultiObjectiveOptimizer:
    """Store and rank a population of :class:`Solution` objects by Pareto dominance.

    Objectives are always *minimised* (negate for maximisation).

    Parameters
    ----------
    objective_names : list[str]
        Names of objectives in the same order used in each Solution.
    """

    def __init__(self, objective_names: List[str]) -> None:
        if not objective_names:
            raise ValueError("At least one objective is required")
        self.objective_names = list(objective_names)
        self._population: List[Solution] = []

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _validate(self, solution: Solution) -> None:
        """Ensure solution has all required objectives."""
        for k in self.objective_names:
            if k not in solution.objectives:
                raise ValueError(
                    f"Solution missing objective '{k}'. "
                    f"Expected: {self.objective_names}"
                )

    def add(self, solution: Solution) -> None:
        """Add a solution and re-rank the population."""
        self._validate(solution)
        self._population.append(solution)
        self._rank_population()

    def add_many(self, solutions: Sequence[Solution]) -> None:
        for s in solutions:
            self._validate(s)
            self._population.append(s)
        self._rank_population()

    def clear(self) -> None:
        self._population = []

    # ------------------------------------------------------------------
    # Pareto ranking
    # ------------------------------------------------------------------

    def _dominates(self, a: Solution, b: Solution) -> bool:
        """Return True if *a* dominates *b* (a ≤ b on all, < on at least one)."""
        a_vals = [a.objectives[k] for k in self.objective_names]
        b_vals = [b.objectives[k] for k in self.objective_names]
        return all(av <= bv for av, bv in zip(a_vals, b_vals)) and any(
            av < bv for av, bv in zip(a_vals, b_vals)
        )

    def _rank_population(self) -> None:
        pop = self._population
        n = len(pop)
        dominated_by: List[int] = [0] * n
        dominates: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(pop[i], pop[j]):
                    dominates[i].append(j)
                    dominated_by[j] += 1
                elif self._dominates(pop[j], pop[i]):
                    dominates[j].append(i)
                    dominated_by[i] += 1

        for i in range(n):
            if dominated_by[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)

        f = 0
        while fronts[f]:
            next_front: List[int] = []
            for i in fronts[f]:
                for j in dominates[i]:
                    dominated_by[j] -= 1
                    if dominated_by[j] == 0:
                        pop[j].rank = f + 1
                        next_front.append(j)
            fronts.append(next_front)
            f += 1

        self._compute_crowding_distances(fronts[:-1])

    def _compute_crowding_distances(self, fronts: List[List[int]]) -> None:
        pop = self._population
        for front in fronts:
            m = len(front)
            for idx in front:
                pop[idx].crowding_distance = 0.0
            for k_name in self.objective_names:
                sorted_front = sorted(front, key=lambda i: pop[i].objectives[k_name])
                pop[sorted_front[0]].crowding_distance = float("inf")
                pop[sorted_front[-1]].crowding_distance = float("inf")
                obj_vals = [pop[i].objectives[k_name] for i in sorted_front]
                obj_range = obj_vals[-1] - obj_vals[0]
                if obj_range == 0:
                    continue
                for idx in range(1, m - 1):
                    pop[sorted_front[idx]].crowding_distance += (
                        obj_vals[idx + 1] - obj_vals[idx - 1]
                    ) / obj_range

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def pareto_front(self) -> List[Solution]:
        """Return all rank-0 solutions."""
        return [s for s in self._population if s.rank == 0]

    def best_n(self, n: int) -> List[Solution]:
        """Top-n solutions by (rank ASC, crowding_distance DESC)."""
        return sorted(
            self._population,
            key=lambda s: (s.rank, -s.crowding_distance),
        )[:n]

    def compromise_solution(self, weights: Optional[Dict[str, float]] = None) -> Optional[Solution]:
        """Weighted-sum compromise: minimises sum(w_i * obj_i)."""
        if not self._population:
            return None
        if weights is None:
            weights = {k: 1.0 for k in self.objective_names}
        # Normalise objectives across current population
        obj_arrays = {
            k: np.array([s.objectives[k] for s in self._population])
            for k in self.objective_names
        }
        ranges = {
            k: (obj_arrays[k].max() - obj_arrays[k].min()) or 1.0
            for k in self.objective_names
        }
        scores = []
        for s in self._population:
            score = sum(
                weights.get(k, 1.0) * (s.objectives[k] - obj_arrays[k].min()) / ranges[k]
                for k in self.objective_names
            )
            scores.append(score)
        return self._population[int(np.argmin(scores))]

    def summary(self) -> Dict[str, Any]:
        front = self.pareto_front
        return {
            "population_size": len(self._population),
            "pareto_front_size": len(front),
            "objective_names": self.objective_names,
        }
