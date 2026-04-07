#!/usr/bin/env python3
from __future__ import annotations

import os


def main() -> int:
    mode = str(os.getenv("DB_MIGRATION_MODE", "create_all") or "create_all").strip().lower()
    auto_create = str(os.getenv("DB_AUTO_CREATE_TABLES", "true") or "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    ok = True
    reasons: list[str] = []
    if mode == "migrate" and auto_create:
        ok = False
        reasons.append("DB_MIGRATION_MODE=migrate requires DB_AUTO_CREATE_TABLES=false")

    print(
        {
            "ok": ok,
            "db_migration_mode": mode,
            "db_auto_create_tables": auto_create,
            "reasons": reasons,
        }
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
