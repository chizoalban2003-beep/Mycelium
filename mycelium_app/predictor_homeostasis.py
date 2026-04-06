from __future__ import annotations

"""Homeostasis → Predictor bridge.

Goal:
- Make the "Global Workspace" effect reusable across the codebase.

Why a separate module (instead of embedding in physics_predictor.py):
- `physics_predictor.py` is intentionally data/compute focused.
- Reading HomeostasisState requires DB access (Session) and user context.
- Putting the bridge here lets API routes, CLI scripts, and background jobs all
  share the exact same policy.

Safety:
- Only a small allowlist of stable knobs is adjusted.
- Changes are monotonic, bounded, and reported back for transparency.
"""

import json
from datetime import datetime, timedelta

from sqlmodel import Session, select

from mycelium_app.models import GrowthLedgerEntry, HomeostasisState
from mycelium_app.settings import settings


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def get_homeostasis_state(session: Session, *, user_id: int) -> HomeostasisState | None:
    """Fetch latest global (project_id=None) HomeostasisState for a user."""

    q = (
        select(HomeostasisState)
        .where(HomeostasisState.user_id == user_id)
        .where(HomeostasisState.project_id.is_(None))
        .order_by(HomeostasisState.updated_at.desc())
        .limit(1)
    )
    return session.exec(q).first()


def recent_deep_breath(session: Session, *, user_id: int, cooldown_minutes: int) -> bool:
    """True if a Deep Breath was recorded recently for this user."""

    since = datetime.utcnow() - timedelta(minutes=max(1, int(cooldown_minutes)))
    q = (
        select(GrowthLedgerEntry)
        .where(GrowthLedgerEntry.created_by_user_id == user_id)
        .where(GrowthLedgerEntry.created_at >= since)
        .where(GrowthLedgerEntry.domain == "homeostasis")
        .where(GrowthLedgerEntry.metric == "deep_breath")
        .order_by(GrowthLedgerEntry.created_at.desc())
        .limit(1)
    )
    return bool(session.exec(q).first())


def apply_homeostasis_to_predictor_kwargs(
    base_kwargs: dict[str, object],
    *,
    mood: str | None,
    recent_deep_breath_flag: bool,
) -> tuple[dict[str, object], list[str]]:
    """Apply a small allowlisted adjustment to predictor knobs.

    - If recent deep breath: reset LR to baseline.
    - If mood is agitated: reduce LR slightly and encourage gentle decay.

    Returns the modified kwargs and a list of applied change descriptions.
    """

    applied: list[str] = []
    out = dict(base_kwargs)

    # Baseline from run_physics_prediction signature.
    lr_default = 0.18

    if recent_deep_breath_flag:
        out["cycle_learning_rate"] = float(lr_default)
        out["cycle_learning_rate_schedule"] = "constant"
        out["cycle_learning_rate_exp_decay"] = 1.0
        applied.append("deep_breath_reset_lr")
        return out, applied

    if (mood or "").strip().lower() != "agitated":
        return out, applied

    # 15% reduction, clamped.
    try:
        cur_lr = float(out.get("cycle_learning_rate", lr_default))
    except Exception:
        cur_lr = lr_default

    new_lr = max(0.02, min(0.50, cur_lr * 0.85))
    if abs(new_lr - cur_lr) > 1e-12:
        out["cycle_learning_rate"] = float(new_lr)
        applied.append(f"cycle_learning_rate:{cur_lr:.4g}->{new_lr:.4g}")

    # Nudge exp_decay down (more decay) when applicable.
    schedule = str(out.get("cycle_learning_rate_schedule", "constant")).strip().lower()
    if schedule in {"exp_decay", "constant"}:
        try:
            cur_exp = float(out.get("cycle_learning_rate_exp_decay", 1.0))
        except Exception:
            cur_exp = 1.0
        new_exp = min(cur_exp, 0.997)
        if abs(new_exp - cur_exp) > 1e-12:
            out["cycle_learning_rate_exp_decay"] = float(new_exp)
            applied.append(f"cycle_learning_rate_exp_decay:{cur_exp:.4g}->{new_exp:.4g}")

    # Optional: tighten multibuffer transition width (reduce oscillations).
    if bool(out.get("multibuffer_enabled")) and "multibuffer_transition_frac" in out:
        try:
            cur_t = float(out.get("multibuffer_transition_frac", 0.0))
        except Exception:
            cur_t = 0.0
        if cur_t > 0.0:
            new_t = max(0.0, min(0.50, cur_t * 0.85))
            if abs(new_t - cur_t) > 1e-12:
                out["multibuffer_transition_frac"] = float(new_t)
                applied.append(f"multibuffer_transition_frac:{cur_t:.4g}->{new_t:.4g}")

    return out, applied


def apply_homeostasis_from_db(
    session: Session,
    *,
    user_id: int,
    base_kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Convenience wrapper that reads DB state and returns (kwargs, info).

    `info` is suitable to return to clients for transparency.
    """

    if not bool(getattr(settings, "nexus_homeostasis_enabled", False)):
        return dict(base_kwargs), None

    hs = get_homeostasis_state(session, user_id=user_id)
    cooldown = int(getattr(settings, "nexus_homeostasis_deep_breath_cooldown_minutes", 30))
    recent = recent_deep_breath(session, user_id=user_id, cooldown_minutes=cooldown)

    mood = None if hs is None else str(hs.mood)
    identity_hash = None if hs is None else str(hs.identity_hash)
    mood_signal = {} if hs is None else _loads_dict(hs.mood_signal_json)

    patched, applied = apply_homeostasis_to_predictor_kwargs(
        base_kwargs,
        mood=mood,
        recent_deep_breath_flag=bool(recent),
    )

    info: dict[str, object] = {
        "enabled": True,
        "mood": mood,
        "identity_hash": identity_hash,
        "mood_signal": mood_signal,
        "recent_deep_breath": bool(recent),
        "applied": applied,
    }

    return patched, info
