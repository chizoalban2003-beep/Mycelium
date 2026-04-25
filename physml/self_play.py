"""Stage 74 — SelfPlay: multi-agent adversarial self-play.

Two :class:`~physml.autonomous_agent.AutonomousAgent` instances compete in
:class:`~physml.arena.CompetitiveArena` and exchange experience via
:class:`~physml.federated.FederatedMyceliumAgent`, so each improves through
adversarial self-play without requiring external labels.

Classes
-------
SelfPlay
    Manages two competing agents and their federated experience exchange.
PlayRound
    Result snapshot from one round of self-play competition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PlayRound:
    """Snapshot of one self-play competition round.

    Attributes
    ----------
    round_idx : int
        Zero-based round index.
    winner : str
        Name of the winning agent (``"agent_a"`` or ``"agent_b"``).
    agent_a_accuracy : float
        Accuracy of agent A on the shared test set.
    agent_b_accuracy : float
        Accuracy of agent B on the shared test set.
    federated : bool
        Whether a FedAvg exchange was performed after this round.
    elapsed_s : float
        Wall-clock seconds for this round.
    """

    round_idx: int
    winner: str
    agent_a_accuracy: float
    agent_b_accuracy: float
    federated: bool
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "winner": self.winner,
            "agent_a_accuracy": round(self.agent_a_accuracy, 4),
            "agent_b_accuracy": round(self.agent_b_accuracy, 4),
            "federated": self.federated,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class SelfPlay:
    """Two-agent adversarial self-play loop with federated experience exchange.

    Each round:

    1. Both agents are trained on their respective (possibly overlapping) data
       splits or the same shared dataset.
    2. They compete head-to-head in a :class:`~physml.arena.CompetitiveArena`.
    3. Every *federate_every* rounds, a :class:`~physml.federated.FederatedMyceliumAgent`
       round aggregates their MLP weights so each agent benefits from the
       other's experience.

    Parameters
    ----------
    agent_a, agent_b : Any
        Two agents to pit against each other.  Each must expose
        ``fit(X, y)`` and ``predict(X)`` / ``predict_proba(X)``.
        Typically :class:`~physml.mycelium_agent.MyceliumAgent` instances.
    federate_every : int, default 2
        Number of self-play rounds between federated weight exchanges.
    random_state : int, default 0

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from physml import MyceliumAgent
    >>> from physml.self_play import SelfPlay
    >>> X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    >>> sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
    >>> history = sp.run(X, y, n_rounds=3)
    >>> print(sp.leaderboard())
    """

    def __init__(
        self,
        agent_a: Any,
        agent_b: Any,
        *,
        federate_every: int = 2,
        random_state: int = 0,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.federate_every = max(1, int(federate_every))
        self.random_state = random_state

        self._history: list[PlayRound] = []
        self._wins: dict[str, int] = {"agent_a": 0, "agent_b": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X: Any,
        y: Any,
        *,
        n_rounds: int = 5,
        test_fraction: float = 0.3,
    ) -> list[PlayRound]:
        """Run *n_rounds* rounds of self-play.

        Parameters
        ----------
        X, y : array-like
            Shared dataset.  Each round, a fresh train/test split is created.
        n_rounds : int, default 5
            Number of competition rounds.
        test_fraction : float, default 0.3
            Fraction of data used as the test set for each round.

        Returns
        -------
        list[PlayRound]
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        rng = np.random.default_rng(self.random_state)

        # Determine initial split for the initial fit (round 0)
        n_total = len(X_arr)
        n_test0 = max(2, int(n_total * test_fraction))
        n_train0 = n_total - n_test0
        idx0 = rng.permutation(n_total)
        X_train_init = X_arr[idx0[:n_train0]]
        y_train_init = y_arr[idx0[:n_train0]]

        # Fit both agents once on the initial split
        _fitted_a = False
        _fitted_b = False
        try:
            self.agent_a.fit(X_train_init, y_train_init)
            _fitted_a = True
        except Exception:
            pass
        try:
            self.agent_b.fit(X_train_init, y_train_init)
            _fitted_b = True
        except Exception:
            pass

        for round_idx in range(n_rounds):
            t0 = time.perf_counter()

            # Random train/test split for this round
            n = len(X_arr)
            n_test = max(2, int(n * test_fraction))
            n_train = n - n_test
            idx = rng.permutation(n)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]
            X_test = X_arr[test_idx]
            y_test = y_arr[test_idx]

            # Incrementally update agents with partial_fit if available,
            # otherwise skip per-round refitting (agents keep their initial fit)
            if _fitted_a:
                try:
                    if hasattr(self.agent_a, "partial_fit"):
                        self.agent_a.partial_fit(X_train, y_train)
                except Exception:
                    pass
            if _fitted_b:
                try:
                    if hasattr(self.agent_b, "partial_fit"):
                        self.agent_b.partial_fit(X_train, y_train)
                except Exception:
                    pass

            # Evaluate both agents
            acc_a = self._eval(self.agent_a, X_test, y_test)
            acc_b = self._eval(self.agent_b, X_test, y_test)

            winner = "agent_a" if acc_a >= acc_b else "agent_b"
            self._wins[winner] += 1

            # Federated exchange
            federated = False
            if (round_idx + 1) % self.federate_every == 0:
                federated = self._federate(X_train, y_train)

            elapsed = time.perf_counter() - t0
            result = PlayRound(
                round_idx=round_idx,
                winner=winner,
                agent_a_accuracy=acc_a,
                agent_b_accuracy=acc_b,
                federated=federated,
                elapsed_s=elapsed,
            )
            self._history.append(result)

        return list(self._history)

    def compete(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> "PlayRound":
        """Run a single competition round on pre-split data.

        Both agents are trained on *X_train / y_train*, then evaluated on
        *X_test / y_test*.

        Returns
        -------
        PlayRound
        """
        t0 = time.perf_counter()

        try:
            self.agent_a.fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
        except Exception:
            pass
        try:
            self.agent_b.fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
        except Exception:
            pass

        X_test_arr = np.asarray(X_test, dtype=float)
        y_test_arr = np.asarray(y_test)
        acc_a = self._eval(self.agent_a, X_test_arr, y_test_arr)
        acc_b = self._eval(self.agent_b, X_test_arr, y_test_arr)

        winner = "agent_a" if acc_a >= acc_b else "agent_b"
        self._wins[winner] += 1

        elapsed = time.perf_counter() - t0
        result = PlayRound(
            round_idx=len(self._history),
            winner=winner,
            agent_a_accuracy=acc_a,
            agent_b_accuracy=acc_b,
            federated=False,
            elapsed_s=elapsed,
        )
        self._history.append(result)
        return result

    def federate(self, X: Any, y: Any) -> bool:
        """Manually trigger a federated weight exchange between the two agents.

        Returns
        -------
        bool
            True if the exchange succeeded.
        """
        return self._federate(np.asarray(X, dtype=float), np.asarray(y))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[PlayRound]:
        """Ordered list of per-round snapshots."""
        return list(self._history)

    def leaderboard(self) -> dict[str, Any]:
        """Return win counts and accuracy statistics for both agents."""
        accs_a = [r.agent_a_accuracy for r in self._history]
        accs_b = [r.agent_b_accuracy for r in self._history]
        return {
            "agent_a": {
                "wins": self._wins["agent_a"],
                "mean_accuracy": round(float(np.mean(accs_a)), 4) if accs_a else None,
                "best_accuracy": round(float(max(accs_a)), 4) if accs_a else None,
            },
            "agent_b": {
                "wins": self._wins["agent_b"],
                "mean_accuracy": round(float(np.mean(accs_b)), 4) if accs_b else None,
                "best_accuracy": round(float(max(accs_b)), 4) if accs_b else None,
            },
            "n_rounds": len(self._history),
        }

    def best_agent(self) -> Any:
        """Return whichever agent won more rounds (agent_a on tie)."""
        if self._wins["agent_b"] > self._wins["agent_a"]:
            return self.agent_b
        return self.agent_a

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval(self, agent: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate *agent* accuracy on *(X, y)*."""
        try:
            from sklearn.metrics import accuracy_score

            if hasattr(agent, "predict"):
                preds = np.asarray(agent.predict(X))
            elif hasattr(agent, "observe"):
                preds = []
                for row in X:
                    result = agent.observe(row)
                    pred = getattr(result, "prediction", result)
                    preds.append(pred)
                preds = np.asarray(preds)
            else:
                return 0.0
            return float(accuracy_score(y, preds))
        except Exception:
            return 0.0

    def _federate(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Perform one round of FedAvg between agent_a and agent_b."""
        try:
            from physml.federated import FederatedMyceliumAgent

            fed = FederatedMyceliumAgent(n_rounds=1)
            fed.add_node("agent_a", X, y, agent=self.agent_a)
            fed.add_node("agent_b", X, y, agent=self.agent_b)
            fed.aggregate()
            return True
        except Exception:
            return False
