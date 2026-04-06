from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IonizedEvent:
    kind: str
    payload: dict[str, Any]
    confidence: float


_CURRENCY_RE = re.compile(
    r"(?P<prefix>\$|usd\s*)?(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?P<suffix>usd|dollars?)?",
    flags=re.IGNORECASE,
)


def ionize_finance(text: str) -> list[IonizedEvent]:
    """Extract simple expense/income facts from plain text.

    This is deterministic and intentionally conservative (no LLM required).

    Recognizes patterns like:
    - "spent $50 on coffee"
    - "paid 120 usd for rent"
    - "I earned $500"
    """

    t = (text or "").strip()
    if not t:
        return []

    events: list[IonizedEvent] = []
    lowered = t.lower()

    is_spend = any(w in lowered for w in ["spent", "paid", "bought", "purchase", "purchased"])
    is_income = any(w in lowered for w in ["earned", "income", "salary", "received", "got paid"])

    matches = list(_CURRENCY_RE.finditer(t))
    if not matches:
        return []

    for m in matches[:5]:
        amt_s = (m.group("amount") or "").replace(",", "")
        try:
            amount = float(amt_s)
        except Exception:
            continue

        # Category: try a crude "on <thing>" extraction.
        category = None
        on_idx = lowered.find(" on ")
        if on_idx != -1:
            category = t[on_idx + 4 :].strip()[:80]
        else:
            # fallback: last 3 words
            words = [w for w in re.split(r"\s+", t) if w]
            if len(words) >= 2:
                category = " ".join(words[-3:])

        if is_income and not is_spend:
            kind = "income"
            conf = 0.75
        elif is_spend and not is_income:
            kind = "expense"
            conf = 0.80
        else:
            kind = "money"
            conf = 0.55

        events.append(
            IonizedEvent(
                kind=kind,
                payload={"amount": amount, "currency": "USD", "category": category},
                confidence=conf,
            )
        )

    return events


def style_profile(text: str) -> dict[str, Any]:
    """Compute deterministic writing-style stats usable as a personal style 'fingerprint'."""

    t = (text or "").strip()
    if not t:
        return {
            "chars": 0,
            "words": 0,
            "sentences": 0,
            "avg_words_per_sentence": None,
            "avg_word_len": None,
            "punct": {"!": 0, "?": 0, ",": 0, ".": 0},
            "caps_words": 0,
        }

    words = [w for w in re.findall(r"[A-Za-z']+", t)]
    n_words = len(words)
    n_chars = len(t)

    # Sentence heuristic: ., !, ? as terminators.
    sentences = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    n_sent = max(1, len(sentences))

    avg_wps = (float(n_words) / float(n_sent)) if n_sent > 0 else None
    avg_wlen = (sum(len(w) for w in words) / float(n_words)) if n_words > 0 else None

    punct = {"!": t.count("!"), "?": t.count("?"), ",": t.count(","), ".": t.count(".")}
    caps_words = sum(1 for w in words if w.isupper() and len(w) >= 2)

    return {
        "chars": n_chars,
        "words": n_words,
        "sentences": n_sent,
        "avg_words_per_sentence": None if avg_wps is None else round(avg_wps, 3),
        "avg_word_len": None if avg_wlen is None else round(float(avg_wlen), 3),
        "punct": punct,
        "caps_words": int(caps_words),
    }


def grammar_suggest(text: str) -> dict[str, Any]:
    """A tiny deterministic 'Grammarly-like' pass.

    Not an LLM: it fixes a few common mechanical issues and returns a diff-friendly response.
    """

    original = text or ""
    t = original

    # Normalize whitespace.
    t = re.sub(r"[\t ]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Fix lowercase standalone "i".
    t = re.sub(r"\bi\b", "I", t)

    # Ensure space after punctuation where missing.
    t = re.sub(r"([,;:.!?])(\S)", r"\1 \2", t)

    # Capitalize first character of the text if it's a letter.
    if t and t[0].isalpha():
        t = t[0].upper() + t[1:]

    changed = t != original
    return {
        "changed": bool(changed),
        "suggested": t,
        "profile": style_profile(t),
    }
