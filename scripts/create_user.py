from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlmodel import Session, select

from mycelium_app.db import create_db_and_tables, engine
from mycelium_app.models import User
from mycelium_app.security import hash_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Mycelium user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--full-name", default="")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="If the user already exists, reset their password (and optionally full name).",
    )
    args = parser.parse_args()

    create_db_and_tables()

    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == args.email)).first()
        if existing:
            if not bool(getattr(args, "reset", False)):
                raise SystemExit("User already exists (use --reset to change password)")
            existing.hashed_password = hash_password(args.password)
            if str(args.full_name or "").strip():
                existing.full_name = str(args.full_name)
            session.add(existing)
            session.commit()
            session.refresh(existing)
            print(f"Reset password for user id={existing.id} email={existing.email}")
            return
        user = User(email=args.email, full_name=args.full_name, hashed_password=hash_password(args.password))
        session.add(user)
        session.commit()
        session.refresh(user)
        print(f"Created user id={user.id} email={user.email}")


if __name__ == "__main__":
    main()
