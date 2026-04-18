"""Stage 114 — ResponseFormatter: template-based natural language responses.

Translates raw agent outputs (predictions, reports, action results) into
natural-language responses adapted to the user's verbosity level and domain
vocabulary.  No LLM required — template-based with dynamic data injection.

Also formats uncertainty, confidence, and recommendations.

Usage
-----
::

    from physml.response_formatter import ResponseFormatter

    fmt = ResponseFormatter(verbosity="concise")
    text = fmt.format_prediction(prediction=1, confidence=0.87,
                                  feature_names=["age", "income"])
    # → "Predicted: 1 (87% confidence). Top signal: income."

    text = fmt.format_report({"n_queries": 10, "accuracy": 0.91})
    text = fmt.format_action_result(task_result)
    text = fmt.format_uncertainty("Model is uncertain about this sample")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class ResponseFormatter:
    """Format agent outputs as natural language.

    Parameters
    ----------
    verbosity : str, default "normal"
        ``"concise"``, ``"normal"``, or ``"verbose"``.
    domain : str, optional
        Domain label (e.g. ``"sales"``) for context hints.
    """

    _VERBOSITY_LEVELS = {"concise", "normal", "verbose"}

    def __init__(
        self,
        verbosity: str = "normal",
        domain: Optional[str] = None,
    ) -> None:
        if verbosity not in self._VERBOSITY_LEVELS:
            _logger.warning(
                "ResponseFormatter: unknown verbosity %r; defaulting to 'normal'", verbosity
            )
            verbosity = "normal"
        self.verbosity = verbosity
        self.domain = domain

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def format_prediction(
        self,
        prediction: Any,
        confidence: float = 1.0,
        feature_names: Optional[List[str]] = None,
        importances: Optional[List[float]] = None,
    ) -> str:
        """Format a model prediction.

        Parameters
        ----------
        prediction : any
            The predicted value.
        confidence : float, default 1.0
            Confidence / probability in [0, 1].
        feature_names : list of str, optional
        importances : list of float, optional
            Feature importances aligned with *feature_names*.

        Returns
        -------
        str
        """
        conf_pct = f"{confidence * 100:.0f}%"
        base = f"Predicted: {prediction} ({conf_pct} confidence)"

        top_feature = None
        if feature_names and importances:
            try:
                top_idx = max(range(len(importances)), key=lambda i: importances[i])
                top_feature = feature_names[top_idx]
            except Exception:
                pass
        elif feature_names:
            top_feature = feature_names[-1]  # heuristic: last feature

        if self.verbosity == "concise":
            if top_feature:
                return f"{base}. Top signal: {top_feature}."
            return f"{base}."

        if self.verbosity == "normal":
            parts = [base + "."]
            if top_feature:
                parts.append(f"The most influential factor was **{top_feature}**.")
            return " ".join(parts)

        # verbose
        parts = [base + "."]
        if feature_names and importances:
            pairs = sorted(
                zip(feature_names, importances), key=lambda x: -x[1]
            )[:5]
            parts.append(
                "Feature importances: "
                + ", ".join(f"{n} ({v:.3f})" for n, v in pairs)
                + "."
            )
        elif feature_names:
            parts.append(f"Features used: {', '.join(feature_names[:5])}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def format_report(self, metrics: Dict[str, Any]) -> str:
        """Format a metrics/report dictionary.

        Parameters
        ----------
        metrics : dict

        Returns
        -------
        str
        """
        if self.verbosity == "concise":
            key_metrics = {k: v for k, v in metrics.items()
                           if k in ("accuracy", "f1", "mse", "rmse", "n_queries", "queries")}
            if not key_metrics:
                key_metrics = dict(list(metrics.items())[:3])
            parts = [f"{k}: {self._fmt_val(v)}" for k, v in key_metrics.items()]
            return "Report — " + ", ".join(parts) + "."

        if self.verbosity == "normal":
            parts = [f"{k}: {self._fmt_val(v)}" for k, v in metrics.items()]
            return "System Report:\n  " + "\n  ".join(parts)

        # verbose
        lines = ["=== System Report ==="]
        for k, v in metrics.items():
            lines.append(f"  {k}: {self._fmt_val(v)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action result
    # ------------------------------------------------------------------

    def format_action_result(self, result: Any) -> str:
        """Format the result of a task/action execution.

        Parameters
        ----------
        result : any
            Can be a :class:`~physml.local_executor.TaskResult`, a dict,
            a string, or any object.

        Returns
        -------
        str
        """
        # Handle TaskResult-like objects
        if hasattr(result, "success") and hasattr(result, "operation"):
            op = result.operation
            if result.success:
                out = result.output
                if self.verbosity == "concise":
                    return f"Done ({op})."
                if self.verbosity == "normal":
                    return f"Task '{op}' completed successfully. Output: {_truncate(str(out), 200)}"
                return (
                    f"Task '{op}' completed successfully.\n"
                    f"  Output: {_truncate(str(out), 500)}\n"
                    f"  Elapsed: {getattr(result, 'elapsed', 'N/A'):.3f}s"
                )
            else:
                err = getattr(result, "error", "unknown error")
                if self.verbosity == "concise":
                    return f"Failed ({op}): {err}."
                return f"Task '{op}' failed: {err}"

        # Dict
        if isinstance(result, dict):
            return self.format_report(result)

        # Fallback
        text = str(result)
        if self.verbosity == "concise":
            return _truncate(text, 100)
        return _truncate(text, 500)

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def format_uncertainty(self, message: str) -> str:
        """Format an uncertainty / low-confidence message.

        Parameters
        ----------
        message : str

        Returns
        -------
        str
        """
        if self.verbosity == "concise":
            return f"⚠ {message}"
        if self.verbosity == "normal":
            return f"Uncertainty notice: {message}"
        return (
            f"[Uncertainty Warning]\n"
            f"  {message}\n"
            f"  Consider providing more data or retraining the model."
        )

    # ------------------------------------------------------------------
    # Advice
    # ------------------------------------------------------------------

    def format_advice(self, message: str, action: Optional[str] = None, severity: str = "info") -> str:
        """Format a proactive advice message.

        Parameters
        ----------
        message : str
        action : str, optional
            Recommended action.
        severity : str, default "info"
            ``"info"``, ``"warning"``, or ``"critical"``.

        Returns
        -------
        str
        """
        prefix = {"info": "ℹ", "warning": "⚠", "critical": "🚨"}.get(severity, "ℹ")
        if self.verbosity == "concise":
            if action:
                return f"{prefix} {message} → {action}."
            return f"{prefix} {message}."
        if self.verbosity == "normal":
            parts = [f"{prefix} {message}"]
            if action:
                parts.append(f"Recommended action: {action}.")
            return " ".join(parts)
        # verbose
        lines = [f"[{severity.upper()}] {message}"]
        if action:
            lines.append(f"  Recommended action: {action}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def __repr__(self) -> str:
        return f"ResponseFormatter(verbosity={self.verbosity!r})"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"
