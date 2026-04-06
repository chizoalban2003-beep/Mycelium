#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mycelium_app.db import engine  # noqa: E402
from mycelium_app.models import NexusNudge, User  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Mark old unseen nudges as seen, keeping newest N unseen")
    p.add_argument("--email", required=True)
    p.add_argument("--keep", type=int, default=1, help="How many newest unseen nudges to keep")
    args = p.parse_args()

    keep = max(0, int(args.keep))
    email = str(args.email).strip().lower()

    with Session(engine) as s:
        u = s.exec(select(User).where(User.email == email)).first()
        if not u or not u.id:
            raise SystemExit(f"User not found: {email}")

        rows = s.exec(
            select(NexusNudge)
            .where(NexusNudge.created_by_user_id == int(u.id), NexusNudge.seen_at.is_(None))
            .order_by(NexusNudge.created_at.desc())
        ).all()

        if len(rows) <= keep:
            print(f"No changes. unseen={len(rows)} keep={keep}")
            return 0

        to_ack = rows[keep:]
        now = datetime.utcnow()
        for r in to_ack:
            r.seen_at = now
            s.add(r)
        s.commit()

        print(f"Acked {len(to_ack)} old nudges for {email}; kept newest {keep} unseen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
