from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")

# Broad match; we run a Luhn check to reduce false positives.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(email|e-mail|phone|ssn|social|password|passwd|token|secret|api[_-]?key|address|full[_-]?name|name)$"
)


def _luhn_ok(digits: str) -> bool:
    xs = [int(c) for c in digits if c.isdigit()]
    if len(xs) < 13 or len(xs) > 19:
        return False
    s = 0
    alt = False
    for d in reversed(xs):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return (s % 10) == 0


def _iter_strings(obj: Any, *, path: str = "$") -> Iterable[tuple[str, str]]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield path, obj
        return
    if isinstance(obj, (int, float, bool)):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            yield from _iter_strings(v, path=f"{path}.{ks}")
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_strings(v, path=f"{path}[{i}]")
        return
    # fallback: stringified
    yield path, str(obj)


def _iter_key_values(obj: Any, *, path: str = "$") -> Iterable[tuple[str, str, Any]]:
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        ks = str(k)
        yield path, ks, v
        if isinstance(v, dict):
            yield from _iter_key_values(v, path=f"{path}.{ks}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    yield from _iter_key_values(item, path=f"{path}.{ks}[{i}]")


@dataclass(frozen=True)
class MembraneResult:
    ok: bool
    reasons: list[str]


def _redact_text(t: str) -> tuple[str, bool]:
    s = str(t)
    changed = False

    if _EMAIL_RE.search(s):
        s2 = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
        changed = changed or (s2 != s)
        s = s2
    if _SSN_RE.search(s):
        s2 = _SSN_RE.sub("[REDACTED_SSN]", s)
        changed = changed or (s2 != s)
        s = s2
    if _PHONE_RE.search(s):
        s2 = _PHONE_RE.sub("[REDACTED_PHONE]", s)
        changed = changed or (s2 != s)
        s = s2

    def _card_repl(m: re.Match[str]) -> str:
        cand = m.group(0)
        digits = "".join(ch for ch in cand if ch.isdigit())
        if _luhn_ok(digits):
            return "[REDACTED_CARD]"
        return cand

    if _CARD_CANDIDATE_RE.search(s):
        s2 = _CARD_CANDIDATE_RE.sub(_card_repl, s)
        changed = changed or (s2 != s)
        s = s2

    return s, changed


def redact_hive_payload(payload: Any) -> tuple[Any, bool]:
    """Return a redacted copy of payload and whether changes were made."""

    if payload is None:
        return None, False
    if isinstance(payload, str):
        return _redact_text(payload)
    if isinstance(payload, (int, float, bool)):
        return payload, False
    if isinstance(payload, list):
        changed_any = False
        out_list: list[Any] = []
        for item in payload:
            red, ch = redact_hive_payload(item)
            changed_any = changed_any or ch
            out_list.append(red)
        return out_list, changed_any
    if isinstance(payload, dict):
        changed_any = False
        out_dict: dict[str, Any] = {}
        for k, v in payload.items():
            ks = str(k)
            if _SENSITIVE_KEY_RE.match(ks.strip()):
                # If key itself is sensitive, blank out its value.
                if v not in (None, "", 0, False, [], {}):
                    changed_any = True
                    out_dict[ks] = None
                else:
                    out_dict[ks] = v
                continue

            red, ch = redact_hive_payload(v)
            changed_any = changed_any or ch
            out_dict[ks] = red
        return out_dict, changed_any

    # fallback: redact stringified
    s, ch = _redact_text(str(payload))
    return s, ch


def check_hive_payload(payload: dict[str, Any]) -> MembraneResult:
    """Best-effort scan to prevent obvious PII/sensitive leaks.

    This is intentionally conservative and heuristic-based.
    """

    reasons: list[str] = []

    # Sensitive keys.
    for path, key, value in _iter_key_values(payload):
        if _SENSITIVE_KEY_RE.match(key.strip()):
            # Allow empty/default values, but flag if non-trivial.
            if isinstance(value, str) and value.strip():
                reasons.append(f"sensitive_key:{path}.{key}")
            elif value not in (None, "", 0, False):
                reasons.append(f"sensitive_key:{path}.{key}")

    # Sensitive patterns in string leaves.
    for path, text in _iter_strings(payload):
        if not text or len(text) < 3:
            continue
        t = str(text)
        if _EMAIL_RE.search(t):
            reasons.append(f"email:{path}")
        if _SSN_RE.search(t):
            reasons.append(f"ssn:{path}")
        if _PHONE_RE.search(t):
            reasons.append(f"phone:{path}")

        m = _CARD_CANDIDATE_RE.search(t)
        if m:
            cand = m.group(0)
            digits = "".join(ch for ch in cand if ch.isdigit())
            if _luhn_ok(digits):
                reasons.append(f"card:{path}")

        if len(reasons) >= 8:
            break

    return MembraneResult(ok=(len(reasons) == 0), reasons=reasons)
