"""Stage 54 — AnomalyGuard: anomaly detection gate for agent predictions.

Uses sklearn IsolationForest and/or LocalOutlierFactor to flag anomalous
inputs before they reach the predictor, preventing silent mispredictions on
out-of-distribution data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class AnomalyResult:
    """Per-sample anomaly verdict."""

    is_anomaly: bool
    score: float  # lower = more anomalous for IF; negative = anomalous for LOF
    detector: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyGuard:
    """Wraps one or more detectors and gates agent predictions.

    Parameters
    ----------
    method : {"isolation_forest", "lof", "elliptic", "ensemble"}
        Detection algorithm.  ``"ensemble"`` runs all three and flags a
        sample if *any* detector marks it anomalous.
    contamination : float
        Expected fraction of outliers (0, 0.5].
    n_estimators : int
        Trees for IsolationForest.
    n_neighbors : int
        Neighbours for LOF.
    random_state : int, optional
    """

    def __init__(
        self,
        method: Literal["isolation_forest", "lof", "elliptic", "ensemble"] = "isolation_forest",
        contamination: float = 0.05,
        n_estimators: int = 100,
        n_neighbors: int = 20,
        random_state: Optional[int] = 42,
    ) -> None:
        self.method = method
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.n_neighbors = int(n_neighbors)
        self.random_state = random_state
        self._detectors: Dict[str, Any] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AnomalyGuard":
        """Fit the guard on clean training data *X*."""
        X = np.asarray(X, dtype=float)
        if self.method in ("isolation_forest", "ensemble"):
            self._detectors["isolation_forest"] = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            ).fit(X)
        if self.method in ("lof", "ensemble"):
            self._detectors["lof"] = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X) - 1),
                contamination=self.contamination,
                novelty=True,
            ).fit(X)
        if self.method in ("elliptic", "ensemble"):
            try:
                self._detectors["elliptic"] = EllipticEnvelope(
                    contamination=self.contamination,
                    random_state=self.random_state,
                ).fit(X)
            except Exception:
                pass  # may fail if n_features > n_samples; skip gracefully
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> List[AnomalyResult]:
        """Return anomaly verdicts for each row in *X*."""
        if not self._is_fitted:
            raise RuntimeError("AnomalyGuard must be fitted before predict()")
        X = np.asarray(X, dtype=float)
        results: List[AnomalyResult] = []
        for row in X:
            row2d = row.reshape(1, -1)
            votes: Dict[str, bool] = {}
            scores: Dict[str, float] = {}
            for name, det in self._detectors.items():
                pred = det.predict(row2d)[0]  # 1 = inlier, -1 = outlier
                score = det.score_samples(row2d)[0]
                votes[name] = pred == -1
                scores[name] = float(score)

            is_anomaly = any(votes.values()) if votes else False
            avg_score = float(np.mean(list(scores.values()))) if scores else 0.0
            results.append(
                AnomalyResult(
                    is_anomaly=is_anomaly,
                    score=avg_score,
                    detector=self.method,
                    metadata={"votes": votes, "scores": scores},
                )
            )
        return results

    def predict_guarded(
        self, X: np.ndarray, predictor: Any
    ) -> Tuple[np.ndarray, List[AnomalyResult]]:
        """Run *predictor.predict(X)* and attach anomaly flags.

        Returns
        -------
        predictions : np.ndarray
            Output of the wrapped predictor (unchanged).
        anomaly_results : list[AnomalyResult]
        """
        anomaly_results = self.predict(X)
        predictions = predictor.predict(X)
        return predictions, anomaly_results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def anomaly_rate(self, X: np.ndarray) -> float:
        """Fraction of rows in *X* flagged as anomalies."""
        results = self.predict(X)
        return sum(r.is_anomaly for r in results) / len(results)

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "contamination": self.contamination,
            "detectors": list(self._detectors.keys()),
            "fitted": self._is_fitted,
        }
