from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import GrowthLedgerEntry


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _metric_direction(metric: str) -> int:
    """Return +1 if higher-is-better, -1 if lower-is-better."""
    m = (metric or "").strip().lower()
    if m in {"mae", "rmse", "mse", "logloss", "cross_entropy"}:
        return -1
    return +1


def _stable_identity_hash(rows: list[GrowthLedgerEntry]) -> str:
    """Hash a stable summary of best accepted sweeps.

    The intent is *identity as memory-of-success*: a fingerprint of what has
    worked so far.
    """

    best_by_key: dict[tuple[str, str], float] = {}
    for r in rows:
        if not bool(r.accepted):
            continue
        key = (str(r.domain or ""), str(r.metric or ""))
        score = float(r.score)
        cur = best_by_key.get(key)
        if cur is None:
            best_by_key[key] = score
            continue
        direction = _metric_direction(key[1])
        if direction > 0 and score > cur:
            best_by_key[key] = score
        if direction < 0 and score < cur:
            best_by_key[key] = score

    payload = {
        "best": [{"domain": k[0], "metric": k[1], "score": float(v)} for k, v in sorted(best_by_key.items())],
        "n_best": len(best_by_key),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest


@dataclass(frozen=True)
class Reflection:
    mood: str
    mood_signal: dict[str, float]
    identity_hash: str
    top_preferences: list[dict[str, object]]
    causal_hints: list[str]
    stats: dict[str, object]


def _compute_mood(rows: list[GrowthLedgerEntry]) -> tuple[str, dict[str, float]]:
    """Map recent performance stability into a simple "mood".

    This is deliberately transparent: it’s not claiming real sentience, it’s a
    UX layer that translates numeric tension into a readable internal state.
    """

    accepted = [r for r in rows if bool(r.accepted)]
    n_total = len(rows)
    n_acc = len(accepted)

    # Compute a simple volatility over direction-normalized scores.
    vals: list[float] = []
    for r in accepted:
        direction = _metric_direction(str(r.metric))
        vals.append(float(direction) * float(r.score))

    if not vals:
        return "curious", {"accepted_rate": 0.0, "stability": 0.0, "tension": 1.0}

    mean = sum(vals) / float(len(vals))
    var = sum((x - mean) ** 2 for x in vals) / float(max(1, len(vals) - 1))
    std = math.sqrt(var)

    accepted_rate = float(n_acc / float(max(1, n_total)))

    # Turn std into a stability score in [0,1].
    stability = float(1.0 / (1.0 + std))

    # Tension rises when acceptance is low OR stability is low.
    tension = float(min(1.0, (1.0 - accepted_rate) * 0.6 + (1.0 - stability) * 0.4))

    if accepted_rate >= 0.65 and stability >= 0.7:
        mood = "content"
    elif tension >= 0.75:
        mood = "agitated"
    elif accepted_rate >= 0.35:
        mood = "focused"
    else:
        mood = "curious"

    return mood, {
        "accepted_rate": float(round(accepted_rate, 6)),
        "stability": float(round(stability, 6)),
        "tension": float(round(tension, 6)),
    }


def _top_sweeps(rows: list[GrowthLedgerEntry], *, limit: int = 5) -> list[GrowthLedgerEntry]:
    accepted = [r for r in rows if bool(r.accepted)]

    # Keep one best sweep per (domain, metric).
    best_by_key: dict[tuple[str, str], GrowthLedgerEntry] = {}

    for r in accepted:
        key = (str(r.domain or ""), str(r.metric or ""))
        cur = best_by_key.get(key)
        if cur is None:
            best_by_key[key] = r
            continue

        direction = _metric_direction(key[1])
        if direction > 0 and float(r.score) > float(cur.score):
            best_by_key[key] = r
        if direction < 0 and float(r.score) < float(cur.score):
            best_by_key[key] = r

    best = list(best_by_key.values())

    # Sort by direction-aware score.
    def _sort_key(x: GrowthLedgerEntry) -> float:
        direction = _metric_direction(str(x.metric))
        return float(direction) * float(x.score)

    best.sort(key=_sort_key, reverse=True)
    return best[: max(1, min(int(limit), 25))]


def _causal_hints(rows: list[GrowthLedgerEntry]) -> list[str]:
    """Extract simple, human-readable 'why' hints from accepted vs rejected sweeps.

    This does not attempt causal inference; it surfaces recurring knobs and
    outcomes as introspection material.
    """

    accepted = [r for r in rows if bool(r.accepted)]
    rejected = [r for r in rows if not bool(r.accepted)]

    acc_knobs: Counter[str] = Counter()
    rej_knobs: Counter[str] = Counter()

    def _ingest(row: GrowthLedgerEntry, bucket: Counter[str]) -> None:
        proposal = _loads_dict(row.proposal_json)
        outcome = _loads_dict(row.outcome_json)

        # Pull a few stable, safe keys.
        for k in ("model", "window_hours", "min_pairs", "field_effect_enabled", "multibuffer_enabled"):
            if k in proposal:
                bucket[f"proposal:{k}={proposal.get(k)}"] += 1
        for k in ("accuracy", "n_pairs", "accepted"):
            if k in outcome:
                bucket[f"outcome:{k}={outcome.get(k)}"] += 1

    for r in accepted:
        _ingest(r, acc_knobs)
    for r in rejected:
        _ingest(r, rej_knobs)

    hints: list[str] = []

    # Compare which knobs appear more in accepted than rejected.
    for k, v in acc_knobs.most_common(25):
        delta = v - rej_knobs.get(k, 0)
        if delta <= 1:
            continue
        hints.append(f"Preference signal: {k} (Δ={delta})")

    return hints[:10]


def compute_self_reflection(
    session: Session,
    *,
    user_id: int,
    project_id: int | None = None,
    window_days: int = 30,
    top_limit: int = 5,
) -> Reflection:
    """Compute a reflective snapshot from the GrowthLedger.

    Returns:
    - mood: qualitative internal state derived from stability
    - identity_hash: fingerprint of best accepted sweeps
    - top_preferences: best sweeps + notable proposal/outcome fields
    - causal_hints: recurring knobs correlated with acceptance
    """

    window_days = max(1, min(int(window_days), 365))
    since = datetime.utcnow() - timedelta(days=window_days)

    q = select(GrowthLedgerEntry).where(
        GrowthLedgerEntry.created_by_user_id == user_id,
        GrowthLedgerEntry.created_at >= since,
    )
    if project_id is not None:
        q = q.where(GrowthLedgerEntry.project_id == project_id)

    q = q.order_by(GrowthLedgerEntry.created_at.desc()).limit(2500)
    rows = list(session.exec(q).all())

    mood, mood_signal = _compute_mood(rows)
    identity_hash = _stable_identity_hash(rows)

    top = _top_sweeps(rows, limit=top_limit)
    top_preferences: list[dict[str, object]] = []
    for r in top:
        top_preferences.append(
            {
                "created_at": r.created_at,
                "domain": r.domain,
                "metric": r.metric,
                "score": float(r.score),
                "accepted": bool(r.accepted),
                "notes": r.notes,
                "proposal": _loads_dict(r.proposal_json),
                "outcome": _loads_dict(r.outcome_json),
            }
        )

    hints = _causal_hints(rows)

    accepted = [r for r in rows if bool(r.accepted)]
    by_domain: dict[str, int] = dict(Counter(str(r.domain) for r in accepted).most_common(20))

    stats: dict[str, object] = {
        "window_days": window_days,
        "n_total": len(rows),
        "n_accepted": len(accepted),
        "accepted_domains": by_domain,
    }

    return Reflection(
        mood=mood,
        mood_signal=mood_signal,
        identity_hash=identity_hash,
        top_preferences=top_preferences,
        causal_hints=hints,
        stats=stats,
    )
