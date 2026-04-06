from __future__ import annotations

import hashlib


def _pick(items: list[str], h: int) -> str:
    if not items:
        return ""
    return items[h % len(items)]


def present_identity(*, identity_hash: str, mood: str) -> dict[str, str]:
    """Deterministic 'name and face' from identity_hash + mood.

    This is intentionally lightweight and non-LLM.
    """

    ih = (identity_hash or "").strip()
    m = (mood or "curious").strip().lower()

    seed = hashlib.sha256((ih + "|" + m).encode("utf-8")).digest()
    h = int.from_bytes(seed[:8], "big", signed=False)

    names = [
        "Sprout",
        "Mycel",
        "Spore",
        "Lumen",
        "Weave",
        "Knot",
        "Pulse",
        "Glyph",
    ]
    tones = ["gentle", "direct", "curious", "calm", "focused", "playful", "precise"]
    roles = ["Guide", "Analyst", "Archivist", "Navigator", "Synthesist", "Caretaker"]

    base = _pick(names, h)
    role = _pick(roles, h >> 7)
    tone = _pick(tones, h >> 13)

    # Simple palette (Tailwind-ish). Keep stable.
    colors = [
        {"bg": "bg-indigo-900", "fg": "text-indigo-200", "accent": "indigo"},
        {"bg": "bg-emerald-900", "fg": "text-emerald-200", "accent": "emerald"},
        {"bg": "bg-sky-900", "fg": "text-sky-200", "accent": "sky"},
        {"bg": "bg-rose-900", "fg": "text-rose-200", "accent": "rose"},
        {"bg": "bg-amber-900", "fg": "text-amber-200", "accent": "amber"},
    ]
    palette = colors[h % len(colors)]

    display_name = f"{base} ({role})"

    mood_phrase = {
        "agitated": "I’m running hot; I’ll be conservative.",
        "curious": "I’m curious; I’ll explore carefully.",
        "focused": "I’m focused; I’ll optimize for signal.",
        "calm": "I’m calm; I’ll keep things stable.",
    }.get(m, "I’m here; I’ll stay transparent.")

    tagline = f"Tone: {tone}. Mood: {m}. {mood_phrase}"

    return {
        "display_name": display_name,
        "tagline": tagline,
        "bg": palette["bg"],
        "fg": palette["fg"],
        "accent": palette["accent"],
    }
