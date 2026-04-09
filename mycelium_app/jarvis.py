"""JARVIS-like AI chat engine — the agent's conversational intelligence.

Combines ecosystem state, pattern analysis, and optional local LLM to produce
intelligent, context-aware responses. Falls back to a rich template engine
when no LLM is available — JARVIS always has something useful to say.

Supports: Ollama, llama.cpp server, or any OpenAI-compatible local endpoint.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from typing import Any

from mycelium_app.humanizer import humanize_app, humanize_feature, humanize_layer


def _llm_endpoint() -> str:
    return str(os.environ.get("NARRATIVE_LLM_ENDPOINT", "") or "").strip()


def _llm_model() -> str:
    return str(os.environ.get("NARRATIVE_LLM_MODEL", "llama3") or "llama3").strip()


def _call_llm(prompt: str, *, max_tokens: int = 300, temperature: float = 0.7) -> str | None:
    """Call local LLM (Ollama-compatible). Returns None if unavailable."""
    endpoint = _llm_endpoint()
    if not endpoint:
        return None

    payload = json.dumps({
        "model": _llm_model(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()

    try:
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            text = str(data.get("response", "")).strip()
            return text if len(text) > 10 else None
    except Exception:
        return None


def _build_context(
    ecosystem: dict[str, Any] | None,
    patterns: dict[str, Any] | None,
    stage: str,
    mood: str,
    agent_name: str,
    gender: str,
) -> str:
    """Build a context string for the LLM from current state."""
    parts = [
        f"You are {agent_name}, a digital AI companion at the {stage} growth stage.",
        f"Your mood is {mood}. Your gender identity is {gender}.",
        "You live inside the user's device and learn from their digital signals.",
        "Your motto is 'Grow with Data.' Be warm, concise, and helpful.",
        "Speak naturally like JARVIS from Iron Man — intelligent, witty, caring.",
        "Never mention technical details like JSON, APIs, or endpoints.",
    ]

    if ecosystem:
        n = ecosystem.get("summary", {}).get("n_signals", 0)
        apps = ecosystem.get("summary", {}).get("top_apps", {})
        cpu = ecosystem.get("summary", {}).get("cpu_mean")
        parts.append(f"\nUser's current state: {n} signals observed.")
        if apps:
            parts.append(f"Active apps: {', '.join(list(apps.keys())[:5])}.")
        if cpu is not None:
            parts.append(f"CPU usage: {cpu}%.")
        sed = ecosystem.get("sedimentation") or {}
        layers = sed.get("layers_raw") or sed.get("layers") or {}
        if layers:
            parts.append(f"Ecosystem: {layers.get('bedrock', 0)} foundation, {layers.get('suspension', 0)} active, {layers.get('turbulent', 0)} changing features.")

    if patterns:
        insights = patterns.get("insights", [])
        if insights:
            parts.append(f"\nBehavioral insights: {' '.join(insights[:3])}")
        suggestions = patterns.get("suggestions", [])
        if suggestions:
            parts.append(f"Suggestions: {', '.join(s.get('title', '') for s in suggestions[:3])}")

    return "\n".join(parts)


def chat(
    message: str,
    *,
    ecosystem: dict[str, Any] | None = None,
    patterns: dict[str, Any] | None = None,
    stage: str = "infant",
    mood: str = "curious",
    agent_name: str = "Myco",
    gender: str = "neutral",
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Generate a JARVIS-like response to a user message.

    Tries local LLM first, falls back to intelligent template engine.
    """
    msg_lower = message.lower().strip()

    # Try LLM first
    context = _build_context(ecosystem, patterns, stage, mood, agent_name, gender)
    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\n".join(
            f"{'User' if h.get('role') == 'user' else agent_name}: {h.get('content', '')}"
            for h in recent
        )

    llm_prompt = f"""{context}

{history_text}
User: {message}
{agent_name}:"""

    llm_response = _call_llm(llm_prompt)
    if llm_response:
        return llm_response

    # --- Intelligent template fallback (JARVIS without LLM) ---
    eco = ecosystem or {}
    pats = patterns or {}
    summary = eco.get("summary", {})
    n_signals = summary.get("n_signals", 0)
    top_apps = summary.get("top_apps", {})
    cpu = summary.get("cpu_mean")
    battery = summary.get("battery_mean")
    sed = (eco.get("sedimentation") or {})
    layers = sed.get("layers_raw") or sed.get("layers") or {}
    insights = pats.get("insights", [])
    pattern_list = pats.get("patterns", [])
    suggestions = pats.get("suggestions", [])

    app_list = ", ".join(list(top_apps.keys())[:5])

    # --- Intent detection ---

    # Identity / who are you
    if any(w in msg_lower for w in ["who are you", "what are you", "your name", "introduce"]):
        return (
            f"I'm {agent_name}, your digital companion. I'm at the {stage} stage, "
            f"feeling {mood}. I live inside your device's signal ecosystem and learn "
            f"from how you use your technology. I've observed {n_signals} signals so far. "
            f"Think of me as a growing intelligence that mirrors your digital life."
        )

    # How are you / mood
    if any(w in msg_lower for w in ["how are you", "how do you feel", "your mood", "feeling"]):
        mood_responses = {
            "curious": "I'm curious — there's so much in your signal stream I want to understand better.",
            "content": "I'm content. Your patterns are stable and I'm learning steadily.",
            "agitated": "I'm a bit agitated — I've detected some unusual patterns that I'm trying to make sense of.",
            "focused": "I'm focused. Your recent activity has clear structure and I'm tracking it closely.",
        }
        base = mood_responses.get(mood, f"I'm {mood}.")
        if n_signals > 0:
            base += f" I've processed {n_signals} signals from your device."
        return base

    # Apps / what am I using
    if any(w in msg_lower for w in ["app", "using", "running", "open", "what am i doing"]):
        if not app_list:
            return "I haven't tracked enough app activity yet. Give me a few more minutes to observe."
        # Find usage durations from patterns
        usage_info = ""
        for p in pattern_list:
            if p.get("type") == "app_usage":
                top = p.get("top_apps", [])
                if top:
                    usage_parts = [f"{a['app']} ({a['minutes']}min)" for a in top[:5]]
                    usage_info = " Usage breakdown: " + ", ".join(usage_parts) + "."
                break
        return f"Your active apps: {app_list}.{usage_info}"

    # Ecosystem / layers / foundation
    if any(w in msg_lower for w in ["ecosystem", "layer", "foundation", "bedrock", "pattern", "structure"]):
        bedrock = layers.get("bedrock", 0)
        suspension = layers.get("suspension", 0)
        turbulent = layers.get("turbulent", 0)
        features = (sed.get("features") or [])[:5]
        feat_names = ", ".join(f.get("feature", "") for f in features) if features else "still forming"
        return (
            f"Your digital ecosystem has {bedrock} foundation signals (stable patterns that define you), "
            f"{suspension} active patterns (things that change regularly), and {turbulent} changing signals "
            f"(noise and variation). Your strongest foundation signals: {feat_names}."
        )

    # Routine / habits
    if any(w in msg_lower for w in ["routine", "habit", "schedule", "daily", "when do i"]):
        for p in pattern_list:
            if p.get("type") == "circadian_rhythm":
                peak = p.get("peak_hour", "")
                active_h = p.get("active_hours", 0)
                return (
                    f"Your peak activity hour is around {peak}:00. "
                    f"You're active across {active_h} hours. "
                    "I'm still learning your full daily rhythm — give me a few more days."
                )
            if p.get("type") == "routines":
                trans = p.get("transitions", [])
                if trans:
                    t = trans[0]
                    return (
                        f"I've noticed you often go from {t['from']} to {t['to']} "
                        f"({t['count']} times). This looks like one of your routines."
                    )
        return "I'm still mapping your routines. Keep using your device naturally and I'll spot the patterns."

    # Focus / productivity
    if any(w in msg_lower for w in ["focus", "productive", "distract", "context switch", "concentration"]):
        for p in pattern_list:
            if p.get("type") == "focus_analysis":
                avg = p.get("avg_session_minutes", 0)
                longest_app = p.get("longest_session_app", "")
                return (
                    f"Your average focus session is {avg} minutes. "
                    f"Your longest session was in {longest_app}. "
                    + ("Try to extend your focus blocks gradually." if avg < 25 else "That's solid focus time.")
                )
            if p.get("type") == "context_switching":
                rate = p.get("switches_per_hour", 0)
                return (
                    f"You're switching apps about {rate} times per hour. "
                    + ("That's quite high — consider batching similar tasks." if rate > 10 else "That's a healthy switching rate.")
                )
        return "I'm tracking your focus patterns. They'll become clearer over the next few sessions."

    # System / CPU / battery / performance
    if any(w in msg_lower for w in ["cpu", "battery", "memory", "system", "performance", "slow"]):
        parts = []
        if cpu is not None:
            parts.append(f"CPU is at {cpu}%")
        if battery is not None:
            parts.append(f"battery at {battery}%")
        if parts:
            status = ", ".join(parts)
            return f"Current system status: {status}. Everything looks normal."
        return "I'm monitoring your system vitals. No issues detected so far."

    # Help
    if any(w in msg_lower for w in ["help", "what can you", "commands", "abilities"]):
        return (
            "I can help you understand your digital life. Try asking:\n"
            "• \"What apps am I using?\" — see your active applications\n"
            "• \"What's my routine?\" — discover your daily patterns\n"
            "• \"How focused am I?\" — check your concentration metrics\n"
            "• \"Show me my ecosystem\" — understand your signal landscape\n"
            "• \"How are you feeling?\" — check my current state\n"
            "• Or just chat — I'm here to grow with you."
        )

    # Greeting
    if any(w in msg_lower for w in ["hello", "hi", "hey", "good morning", "good evening", "sup"]):
        time_of_day = datetime.utcnow().hour
        greeting = "Good morning" if time_of_day < 12 else "Good afternoon" if time_of_day < 18 else "Good evening"
        return (
            f"{greeting}! I'm {agent_name}. "
            f"I've been watching your ecosystem — {n_signals} signals and counting. "
            + (f"Your top apps right now: {app_list}. " if app_list else "")
            + "What would you like to know?"
        )

    # Suggestions
    if any(w in msg_lower for w in ["suggest", "advice", "recommend", "tip", "should i"]):
        if suggestions:
            s = suggestions[0]
            return f"Here's a suggestion: {s.get('title', '')}. {s.get('message', '')}"
        if insights:
            return f"Based on what I've learned: {insights[0]}"
        return "I'm still gathering enough data to make personalized suggestions. Check back soon."

    # Default — contextual response based on current state
    if insights:
        return f"Here's what I know about your current state: {insights[0]} Feel free to ask me anything specific."
    if n_signals > 0:
        return (
            f"I've observed {n_signals} signals from your device. "
            f"I'm at the {stage} stage, still learning your patterns. "
            "Ask me about your apps, routines, focus, or ecosystem."
        )
    return (
        f"I'm {agent_name}, growing with your data. "
        "I don't have enough signals yet to give detailed insights. "
        "Keep using your device and I'll start spotting patterns."
    )
