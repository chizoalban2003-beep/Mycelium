#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlmodel import Session, create_engine, select

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mycelium_app.models import SignalLedgerEvent, User  # noqa: E402
from mycelium_app.settings import settings  # noqa: E402


def _normalize_database_url(url: str) -> str:
    u = str(url or "").strip()
    if u.startswith("postgres://"):
        return "postgresql+psycopg://" + u[len("postgres://") :]
    if u.startswith("postgresql://") and "+" not in u.split("://", 1)[0]:
        return "postgresql+psycopg://" + u[len("postgresql://") :]
    return u


def _parse_json_dict(text: str) -> dict[str, object]:
    if not text:
        return {}
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _iso(dt: datetime) -> str:
    # Keep UTC-ish. The DB stores naive UTC timestamps by convention.
    return dt.isoformat() + "Z"


def main() -> int:
    p = argparse.ArgumentParser(description="Export Nexus telemetry (SignalLedgerEvent) rows to CSV")
    p.add_argument("--db-url", default=str(getattr(settings, "database_url", "sqlite:///storage/mycelium.db")))
    p.add_argument("--out", default="storage/telemetry_export.csv")

    filt = p.add_argument_group("filters")
    filt.add_argument("--since-hours", type=int, default=24)
    filt.add_argument("--user-id", type=int, default=None)
    filt.add_argument("--email", default=None, help="Resolve user_id by email")
    filt.add_argument("--project-id", type=int, default=None)
    filt.add_argument("--device-id", default=None)
    filt.add_argument("--signal-type", default=None)
    filt.add_argument("--limit", type=int, default=50_000)

    args = p.parse_args()

    db_url = _normalize_database_url(args.db_url)
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)

    since_hours = max(1, min(int(args.since_hours), 24 * 365))
    since = datetime.utcnow() - timedelta(hours=since_hours)

    user_id: int | None = None
    if args.user_id is not None:
        user_id = int(args.user_id)
    elif args.email:
        with Session(engine) as session:
            u = session.exec(select(User).where(User.email == str(args.email).strip().lower())).first()
            if not u or not u.id:
                raise SystemExit(f"No user found for email={args.email!r}")
            user_id = int(u.id)

    limit = max(1, min(int(args.limit), 500_000))

    q = select(SignalLedgerEvent).where(SignalLedgerEvent.created_at >= since)
    if user_id is not None:
        q = q.where(SignalLedgerEvent.created_by_user_id == int(user_id))
    if args.project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == int(args.project_id))
    if args.device_id:
        q = q.where(SignalLedgerEvent.device_id == str(args.device_id).strip()[:64])
    if args.signal_type:
        q = q.where(SignalLedgerEvent.signal_type == str(args.signal_type).strip().lower()[:64])

    q = q.order_by(SignalLedgerEvent.created_at.asc()).limit(limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with Session(engine) as session, out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "created_at",
                "user_id",
                "project_id",
                "device_id",
                "signal_type",
                "payload_json",
                "payload_app",
                "payload_backend",
            ],
        )
        w.writeheader()

        for r in session.exec(q).all():
            payload = _parse_json_dict(str(r.payload_json or "{}"))
            w.writerow(
                {
                    "id": int(r.id or 0),
                    "created_at": _iso(r.created_at),
                    "user_id": int(r.created_by_user_id),
                    "project_id": ("" if r.project_id is None else int(r.project_id)),
                    "device_id": str(r.device_id or ""),
                    "signal_type": str(r.signal_type or ""),
                    "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "payload_app": str(payload.get("app") or payload.get("app_name") or payload.get("bundle_id") or ""),
                    "payload_backend": str(payload.get("backend") or ""),
                }
            )
            n += 1

    print(f"wrote {n} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
