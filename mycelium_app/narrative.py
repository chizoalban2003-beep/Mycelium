"""Narrative layer — gives the digital organism a voice.

Generates natural-language summaries of the ecosystem state, sedimentation
results, and prediction outcomes. The narrative adapts to the growth stage:

    - **Infant** — observational, descriptive ("I'm still watching…")
    - **Toddler** — exploratory, questioning ("I noticed a pattern…")
    - **Adolescent** — confident, actionable ("Based on your patterns…")

This module uses deterministic template-based generation by default.
When a local LLM endpoint is configured (NARRATIVE_LLM_ENDPOINT in settings),
it can optionally enhance summaries with natural language.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import urllib.request
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


def _stage_voice(stage: str) -> dict[str, str]:
    """Return tone parameters for each growth stage."""
    voices = {
        "infant": {
            "pronoun": "I",
            "tone": "observational",
            "prefix": "I'm still learning to see your world.",
            "confidence": "low",
        },
        "toddler": {
            "pronoun": "I",
            "tone": "curious",
            "prefix": "I'm starting to understand your patterns.",
            "confidence": "growing",
        },
        "adolescent": {
            "pronoun": "I",
            "tone": "confident",
            "prefix": "I can see meaningful patterns in your digital life.",
            "confidence": "high",
        },
    }
    return voices.get(stage, voices["infant"])


def generate_ecosystem_narrative(
    *,
    stage: str,
    summary: dict[str, Any] | None = None,
    sedimentation: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate a narrative summary of the ecosystem state.

    Returns a dict with keys: headline, body, insight, next_step.
    """
    voice = _stage_voice(stage)
    summary = summary or {}
    sedimentation = sedimentation or {}
    prediction = prediction or {}

    n_signals = int(summary.get("n_signals", 0))
    hours_active = float(summary.get("hours_active", 0))
    top_apps = summary.get("top_apps", {})
    cpu_mean = summary.get("cpu_mean")
    battery_mean = summary.get("battery_mean")

    layers = sedimentation.get("layers", {})
    bedrock_features = sedimentation.get("top_bedrock", [])
    n_complexes = int(sedimentation.get("n_complexes", 0))

    pred_target = prediction.get("target")
    pred_r2 = prediction.get("r2")
    top_weights = prediction.get("top_weights", [])

    # --- Headline ---
    if stage == "infant":
        if n_signals == 0:
            headline = "Waiting for first signals"
        elif n_signals < 10:
            headline = f"First {n_signals} signals received"
        else:
            headline = f"Observing — {n_signals} signals in {hours_active:.0f}h"
    elif stage == "toddler":
        headline = f"Learning — {n_signals} signals, {n_complexes} pattern groups found"
    else:
        if pred_r2 is not None and pred_r2 > 0.3:
            headline = f"Growing — R²={pred_r2:.2f} on {pred_target}"
        else:
            headline = f"Analyzing — {n_signals} signals across {hours_active:.0f}h"

    # --- Body ---
    body_parts: list[str] = [voice["prefix"]]

    if top_apps:
        app_list = ", ".join(list(top_apps.keys())[:5])
        body_parts.append(f"Your most active apps: {app_list}.")

    if cpu_mean is not None:
        body_parts.append(f"Average CPU: {cpu_mean:.0f}%.")
    if battery_mean is not None:
        body_parts.append(f"Battery averaged {battery_mean:.0f}%.")

    if layers:
        bedrock_count = layers.get("bedrock", 0)
        suspension_count = layers.get("suspension", 0)
        turbulent_count = layers.get("turbulent", 0)
        body_parts.append(
            f"Ecosystem layers: {bedrock_count} bedrock, "
            f"{suspension_count} suspension, {turbulent_count} turbulent features."
        )

    if bedrock_features:
        body_parts.append(
            f"Your foundation signals: {', '.join(bedrock_features[:3])}."
        )

    body = " ".join(body_parts)

    # --- Insight ---
    if stage == "infant":
        insight = "Still observing. The ecosystem structure will emerge as more signals arrive."
    elif pred_r2 is not None and pred_r2 > 0.3:
        top_feat = top_weights[0]["feature"] if top_weights else "signal patterns"
        insight = (
            f"Your {pred_target} is most influenced by {top_feat}. "
            f"The model explains {pred_r2 * 100:.0f}% of the variance."
        )
    elif n_complexes > 0:
        insight = (
            f"Found {n_complexes} correlated signal groups. "
            "These complexes represent stable behavioral patterns."
        )
    else:
        insight = "Signals are diverse but no strong patterns yet. More data will help."

    # --- Next step ---
    if stage == "infant":
        next_step = "Keep using your device naturally. I learn from every signal."
    elif stage == "toddler":
        next_step = "I'm starting to predict patterns. Give me feedback to help me grow."
    else:
        next_step = "Review my predictions and accept or reject them to refine my understanding."

    result = {
        "headline": headline,
        "body": body,
        "insight": insight,
        "next_step": next_step,
        "stage": stage,
        "voice_tone": voice["tone"],
    }

    # Optional LLM enhancement
    llm_endpoint = str(os.environ.get("NARRATIVE_LLM_ENDPOINT", "") or "").strip()
    if llm_endpoint and _HAS_URLLIB:
        try:
            result = _enhance_with_llm(result, llm_endpoint, summary, sedimentation, prediction)
        except Exception:
            pass

    return result


def _enhance_with_llm(
    base_narrative: dict[str, str],
    endpoint: str,
    summary: dict[str, Any],
    sedimentation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, str]:
    """Optionally enhance the narrative using a local LLM endpoint (Ollama-compatible)."""
    stage = base_narrative.get("stage", "infant")
    voice = _stage_voice(stage)

    prompt = (
        f"You are a {voice['tone']} AI companion at the {stage} growth stage. "
        f"Your motto is 'Grow with Data.' "
        f"Summarize the user's digital ecosystem in 2-3 sentences:\n\n"
        f"Signals: {json.dumps(summary, default=str)}\n"
        f"Ecosystem: {json.dumps(sedimentation, default=str)}\n"
        f"Prediction: {json.dumps(prediction, default=str)}\n\n"
        f"Be warm, transparent, and brief. Mention specific apps or patterns. "
        f"End with one actionable suggestion."
    )

    payload = json.dumps({
        "model": os.environ.get("NARRATIVE_LLM_MODEL", "llama3"),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200},
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            llm_text = str(data.get("response", "")).strip()
            if llm_text and len(llm_text) > 20:
                base_narrative["body"] = llm_text
                base_narrative["llm_enhanced"] = "true"
    except Exception:
        pass

    return base_narrative
