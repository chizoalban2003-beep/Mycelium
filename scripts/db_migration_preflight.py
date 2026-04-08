#!/usr/bin/env python3
from __future__ import annotations

import os


def main() -> int:
    strict = str(os.getenv("STRICT_PRODUCTION", "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mode = str(os.getenv("DB_MIGRATION_MODE", "create_all") or "create_all").strip().lower()
    auto_create = str(os.getenv("DB_AUTO_CREATE_TABLES", "true") or "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    secret_key = str(os.getenv("SECRET_KEY", "") or "").strip()
    cookie_secure = str(os.getenv("COOKIE_SECURE", "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    database_url = str(os.getenv("DATABASE_URL", "") or "").strip()

    ok = True
    reasons: list[str] = []
    if mode == "migrate" and auto_create:
        ok = False
        reasons.append("DB_MIGRATION_MODE=migrate requires DB_AUTO_CREATE_TABLES=false")
    if strict:
        if not secret_key or secret_key == "dev-secret-change-me" or secret_key == "change-me-please":
            ok = False
            reasons.append("SECRET_KEY must be a long random production value")
        if not cookie_secure:
            ok = False
            reasons.append("COOKIE_SECURE should be true in production")
        if not database_url or database_url.startswith("sqlite:"):
            ok = False
            reasons.append("DATABASE_URL should point to Postgres in production mode")

    print(
        {
            "ok": ok,
            "strict": strict,
            "db_migration_mode": mode,
            "db_auto_create_tables": auto_create,
            "cookie_secure": cookie_secure,
            "database_url_scheme": database_url.split(":", 1)[0] if database_url else "",
            "reasons": reasons,
        }
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
