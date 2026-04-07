#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

import mycelium_app.db as db_module
import mycelium_app.main as main_module
from mycelium_app.models import ConversationMessage, HiveDevice, MissionLogLedgerEntry, SignalLedgerEvent, User
from mycelium_app.parental_policy import set_policy


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _seed_user(session: Session) -> User:
    user = User(
        email="webhook-smoke@example.com",
        full_name="Webhook Smoke",
        hashed_password="hash",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _seed_data(session: Session, user: User) -> None:
    session.add_all(
        [
            MissionLogLedgerEntry(
                created_by_user_id=int(user.id),
                project_id=None,
                device_id="webhook-smoke-device",
                source_kind="diagnostic",
                source_ref="webhook-smoke-1",
                mode="calm",
                tier="baseline",
                title="Seed trace one",
                detail="smoke",
                delta=1.0,
                delta_text="+1.00",
            ),
            MissionLogLedgerEntry(
                created_by_user_id=int(user.id),
                project_id=None,
                device_id="webhook-smoke-device",
                source_kind="diagnostic",
                source_ref="webhook-smoke-2",
                mode="calm",
                tier="baseline",
                title="Seed trace two",
                detail="smoke",
                delta=1.0,
                delta_text="+1.00",
            ),
            SignalLedgerEvent(
                created_by_user_id=int(user.id),
                project_id=None,
                device_id="phone",
                signal_type="app",
                payload_json='{"app":"Telegram"}',
            ),
            HiveDevice(
                device_id="phone",
                last_source="telegram_smoke",
            ),
        ]
    )
    session.commit()


def _post(client: TestClient, payload: dict[str, object], secret: str | None = None):
    headers = {}
    if secret is not None:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret
    return client.post("/api/nexus/chat/telegram/webhook", json=payload, headers=headers)


def _latest_assistant_message(session: Session) -> str:
    row = session.exec(
        select(ConversationMessage)
        .where(ConversationMessage.channel == "telegram")
        .where(ConversationMessage.role == "assistant")
        .order_by(ConversationMessage.created_at.desc())
    ).first()
    return str(getattr(row, "content", "") or "") if row else ""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mycelium-webhook-http-smoke-") as tmp_dir:
        db_path = Path(tmp_dir) / "webhook-smoke.db"
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

        original_db_engine = db_module.engine
        original_main_engine = main_module.engine
        original_secret = str(getattr(main_module.settings, "notifications_telegram_webhook_secret", "") or "")

        db_module.engine = engine
        main_module.engine = engine
        main_module.settings.notifications_telegram_webhook_secret = "webhook-secret"
        main_module.settings.notifications_bridge_enabled = False
        main_module.settings.notifications_telegram_bot_token = ""

        try:
            SQLModel.metadata.create_all(engine)

            with Session(engine) as session:
                user = _seed_user(session)
                set_policy(
                    session,
                    int(user.id),
                    {
                        "notifications": {
                            "enabled": True,
                            "telegram_enabled": True,
                            "telegram_chat_id": "12345",
                        }
                    },
                )
                _seed_data(session, user)

            with TestClient(main_module.app) as client:
                unauthorized = _post(
                    client,
                    {
                        "message": {
                            "chat": {"id": 12345},
                            "text": "/status",
                        }
                    },
                    secret=None,
                )
                _assert(unauthorized.status_code == 401, f"expected 401 for missing webhook secret, got {unauthorized.status_code}")

                status_resp = _post(
                    client,
                    {
                        "message": {
                            "chat": {"id": 12345},
                            "text": "/status",
                        }
                    },
                    secret="webhook-secret",
                )
                _assert(status_resp.status_code == 200, f"status webhook failed: {status_resp.status_code} {status_resp.text}")
                _assert(status_resp.json().get("ok") is True, f"unexpected status payload: {status_resp.text}")

                with Session(engine) as session:
                    status_text = _latest_assistant_message(session)
                    _assert("Nodes:" in status_text, f"status reply missing node count: {status_text!r}")
                    _assert("Tier:" in status_text, f"status reply missing permission tier: {status_text!r}")
                    _assert("Mode:" in status_text, f"status reply missing mode summary: {status_text!r}")
                    _assert("η:" in status_text, f"status reply missing viscosity summary: {status_text!r}")

                prune_invalid = _post(
                    client,
                    {
                        "message": {
                            "chat": {"id": 12345},
                            "text": "/prune abc",
                        }
                    },
                    secret="webhook-secret",
                )
                _assert(prune_invalid.status_code == 200, f"invalid prune webhook failed: {prune_invalid.status_code} {prune_invalid.text}")

                with Session(engine) as session:
                    invalid_text = _latest_assistant_message(session)
                    _assert("couldn’t parse" in invalid_text.lower() or "could not parse" in invalid_text.lower(), f"invalid prune did not return a helpful parse error: {invalid_text!r}")

                prune_clear = _post(
                    client,
                    {
                        "message": {
                            "chat": {"id": 12345},
                            "text": "/clear_logs",
                        }
                    },
                    secret="webhook-secret",
                )
                _assert(prune_clear.status_code == 200, f"clear webhook failed: {prune_clear.status_code} {prune_clear.text}")

                with Session(engine) as session:
                    remaining = session.exec(select(MissionLogLedgerEntry)).all()
                    _assert(len(remaining) == 0, f"expected clear_logs to empty mission log, found {len(remaining)} rows")

            print("telegram_webhook_http_smoketest=ok")
            return 0
        finally:
            db_module.engine = original_db_engine
            main_module.engine = original_main_engine
            main_module.settings.notifications_telegram_webhook_secret = original_secret


if __name__ == "__main__":
    raise SystemExit(main())
