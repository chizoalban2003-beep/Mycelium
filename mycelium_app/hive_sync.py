from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from mycelium_app.models import ExperienceBufferEntry


def _loads_dict(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def build_anonymized_report(
    entries: list[ExperienceBufferEntry],
    *,
    device_id: str,
    project_id: int | None,
) -> dict[str, object]:
    """Build an anonymized, aggregation-only report.

    Intentional limitations:
    - Never includes raw text
    - Only uses extracted_json payloads (already 'ionized')
    - Produces counts and coarse summary stats suitable for federated learning

    This is an MVP scaffold; DP noise and poisoning defense live above this layer.
    """

    now = datetime.utcnow().isoformat() + "Z"
    modality_counts: Counter[str] = Counter()

    # Finance aggregates
    n_fin_events = 0
    sum_expense = 0.0
    sum_income = 0.0
    cat_counts: Counter[str] = Counter()

    # Style aggregates
    style_n = 0
    style_avg_word_len_sum = 0.0
    style_avg_wps_sum = 0.0

    # Grammar aggregates
    grammar_n = 0
    grammar_changed = 0

    for e in entries:
        modality_counts[str(e.modality or "").lower()] += 1
        extracted = _loads_dict(e.extracted_json)

        kind = str(extracted.get("kind", "")).lower()

        # Finance
        finance_payload = None
        if kind == "finance":
            finance_payload = extracted
        elif kind == "auto":
            finance_payload = extracted.get("finance") if isinstance(extracted.get("finance"), dict) else None

        if isinstance(finance_payload, dict):
            events = finance_payload.get("events")
            if isinstance(events, list):
                for ev in events[:200]:
                    if not isinstance(ev, dict):
                        continue
                    ek = str(ev.get("kind", "money")).lower()
                    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                    amount = payload.get("amount")
                    try:
                        amount_f = float(amount)
                    except Exception:
                        amount_f = None
                    category = payload.get("category")
                    if isinstance(category, str) and category.strip():
                        cat_counts[category.strip().lower()[:32]] += 1

                    n_fin_events += 1
                    if amount_f is not None:
                        if ek == "expense":
                            sum_expense += amount_f
                        elif ek == "income":
                            sum_income += amount_f

        # Style
        style_payload = None
        if kind == "style":
            style_payload = extracted
        elif kind == "auto":
            style_payload = extracted.get("style") if isinstance(extracted.get("style"), dict) else None
        if isinstance(style_payload, dict):
            prof = style_payload.get("profile") if isinstance(style_payload.get("profile"), dict) else None
            if isinstance(prof, dict):
                awl = prof.get("avg_word_len")
                wps = prof.get("avg_words_per_sentence")
                try:
                    awl_f = float(awl)
                    wps_f = float(wps)
                except Exception:
                    awl_f = None
                    wps_f = None
                if awl_f is not None and wps_f is not None:
                    style_n += 1
                    style_avg_word_len_sum += awl_f
                    style_avg_wps_sum += wps_f

        # Grammar
        grammar_payload = None
        if kind == "grammar":
            grammar_payload = extracted
        elif kind == "auto":
            grammar_payload = extracted.get("grammar") if isinstance(extracted.get("grammar"), dict) else None
        if isinstance(grammar_payload, dict):
            grammar_n += 1
            if bool(grammar_payload.get("changed")):
                grammar_changed += 1

    style_avg_word_len = None if style_n == 0 else round(style_avg_word_len_sum / float(style_n), 4)
    style_avg_wps = None if style_n == 0 else round(style_avg_wps_sum / float(style_n), 4)

    grammar_changed_rate = None if grammar_n == 0 else round(grammar_changed / float(grammar_n), 4)

    top_categories = [{"category": k, "count": int(v)} for k, v in cat_counts.most_common(20)]

    return {
        "meta": {
            "created_at": now,
            "device_id": device_id,
            "project_id": project_id,
            "n_entries": int(len(entries)),
        },
        "counts": {
            "modalities": dict(modality_counts),
            "finance_events": int(n_fin_events),
        },
        "finance": {
            "sum_expense_usd": round(float(sum_expense), 4),
            "sum_income_usd": round(float(sum_income), 4),
            "top_categories": top_categories,
        },
        "style": {
            "n": int(style_n),
            "avg_word_len_mean": style_avg_word_len,
            "avg_words_per_sentence_mean": style_avg_wps,
        },
        "grammar": {
            "n": int(grammar_n),
            "changed_rate": grammar_changed_rate,
        },
    }
