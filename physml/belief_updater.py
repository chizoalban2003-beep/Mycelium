"""Stage 98 — BeliefUpdater: Bayesian belief state maintenance.

Maintains a probability distribution over a discrete set of hypotheses
(world states) and updates it with Bayes' rule when new evidence arrives.

Classes
-------
Belief
    A snapshot of the current probability distribution.
BeliefUpdater
    Manages hypothesis probabilities and applies Bayesian updates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Belief:
    """One snapshot of the belief distribution.

    Attributes
    ----------
    timestamp : float
        Unix time when the belief was recorded.
    distribution : dict[str, float]
        Hypothesis → probability mapping.
    evidence : str or None
        The evidence that triggered this update.
    most_likely : str
        Hypothesis with the highest probability.
    """

    timestamp: float
    distribution: Dict[str, float]
    evidence: Optional[str]
    most_likely: str


class BeliefUpdater:
    """Maintains and updates a discrete Bayesian belief state.

    Parameters
    ----------
    hypotheses : list[str]
        Names of the world-state hypotheses.
    prior : dict[str, float], optional
        Initial probability for each hypothesis.  If *None*, a uniform
        prior is used.  Probabilities are normalised automatically.

    Attributes
    ----------
    distribution_ : dict[str, float]
        Current probability distribution over hypotheses.
    history_ : list[Belief]
        All belief snapshots recorded via :meth:`update`.
    likelihoods_ : dict[str, dict[str, float]]
        ``likelihoods_[evidence][hypothesis]`` = P(evidence | hypothesis).
        Populated via :meth:`set_likelihood`.
    """

    def __init__(
        self,
        hypotheses: List[str],
        prior: Optional[Dict[str, float]] = None,
    ) -> None:
        if not hypotheses:
            raise ValueError("hypotheses must be non-empty.")
        self._hypotheses = list(hypotheses)
        if prior is not None:
            dist = {h: float(prior.get(h, 0.0)) for h in hypotheses}
        else:
            dist = {h: 1.0 for h in hypotheses}
        self.distribution_ = self._normalise(dist)
        self.history_: List[Belief] = []
        self.likelihoods_: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def set_likelihood(
        self, evidence: str, likelihoods: Dict[str, float]
    ) -> None:
        """Register P(evidence | hypothesis) for every hypothesis.

        Parameters
        ----------
        evidence : str
            The observation/evidence label.
        likelihoods : dict[str, float]
            Mapping from hypothesis name to likelihood value.
        """
        self.likelihoods_[evidence] = {h: float(likelihoods.get(h, 1e-9)) for h in self._hypotheses}

    # ------------------------------------------------------------------
    def update(self, evidence: str) -> Belief:
        """Apply Bayes' rule with *evidence* and record the new belief.

        If no likelihood is registered for *evidence*, the distribution
        is left unchanged.

        Parameters
        ----------
        evidence : str

        Returns
        -------
        Belief
        """
        if evidence in self.likelihoods_:
            liks = self.likelihoods_[evidence]
            unnorm = {h: self.distribution_[h] * liks.get(h, 1e-9) for h in self._hypotheses}
            self.distribution_ = self._normalise(unnorm)

        belief = Belief(
            timestamp=time.time(),
            distribution=dict(self.distribution_),
            evidence=evidence,
            most_likely=max(self.distribution_, key=lambda h: self.distribution_[h]),
        )
        self.history_.append(belief)
        return belief

    # ------------------------------------------------------------------
    def most_likely(self) -> str:
        """Return the hypothesis with the highest current probability."""
        return max(self.distribution_, key=lambda h: self.distribution_[h])

    def entropy(self) -> float:
        """Return the Shannon entropy of the current distribution (nats)."""
        import math

        total = 0.0
        for p in self.distribution_.values():
            if p > 0:
                total -= p * math.log(p)
        return total

    def reset(self, prior: Optional[Dict[str, float]] = None) -> None:
        """Reset the distribution to *prior* (or uniform if *None*)."""
        if prior is not None:
            dist = {h: float(prior.get(h, 0.0)) for h in self._hypotheses}
        else:
            dist = {h: 1.0 for h in self._hypotheses}
        self.distribution_ = self._normalise(dist)
        self.history_.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(dist: Dict[str, float]) -> Dict[str, float]:
        total = sum(dist.values())
        if total == 0:
            n = len(dist)
            return {k: 1.0 / n for k in dist}
        return {k: v / total for k, v in dist.items()}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BeliefUpdater(hypotheses={self._hypotheses}, "
            f"most_likely={self.most_likely()})"
        )
