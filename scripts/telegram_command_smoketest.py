#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mycelium_app.models import MissionLogLedgerEntry, Project, ProjectMember, ProjectRole, SignalLedgerEvent, User
from mycelium_app.parental_policy import set_policy
from mycelium_app.routes.chat import _build_status_message, _parse_mission_log_command, _resolve_user_id_from_telegram_chat_id
from mycelium_app.routes.live import MissionLogPruneRequest, prune_mission_log


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mycelium-telegram-smoke-") as tmp_dir:
        db_path = Path(tmp_dir) / "smoke.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            user = User(
                email="telegram-smoke@example.com",
                full_name="Telegram Smoke",
                hashed_password="hash",
            )
            session.add(user)
            session.commit()
            session.refresh(user)

            project = Project(name="Telegram Smoke Project", description="smoke", created_by_user_id=int(user.id))
            session.add(project)
            session.commit()
            session.refresh(project)
            session.add(ProjectMember(project_id=int(project.id), user_id=int(user.id), role=ProjectRole.owner))

            session.add_all(
                [
                    MissionLogLedgerEntry(
                        created_by_user_id=int(user.id),
                        project_id=int(project.id),
                        device_id="smoke-device",
                        source_kind="diagnostic",
                        source_ref="telegram-smoke-1",
                        mode="calm",
                        tier="baseline",
                        title="Seed trace one",
                        detail="smoke",
                        delta=1.0,
                        delta_text="+1.00",
                    ),
                    MissionLogLedgerEntry(
                        created_by_user_id=int(user.id),
                        project_id=int(project.id),
                        device_id="smoke-device",
                        source_kind="diagnostic",
                        source_ref="telegram-smoke-2",
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
                        device_id="smoke-device",
                        signal_type="app",
                        payload_json='{"app":"Telegram"}',
                    ),
                ]
            )
            session.commit()

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

            resolved = _resolve_user_id_from_telegram_chat_id(session, "12345")
            _assert(resolved == int(user.id), f"expected chat mapping to resolve to {int(user.id)}, got {resolved!r}")

            _assert(_parse_mission_log_command("/clear_logs") == ("clear", None, None), "clear command parsing failed")
            _assert(_parse_mission_log_command("/prune 24") == ("prune", None, 24), "prune hours parsing failed")
            _assert(
                _parse_mission_log_command("/prune project 7 12") == ("prune", 7, 12),
                "project-scoped prune parsing failed",
            )
            _assert(_parse_mission_log_command("/prune abc") is None, "invalid prune syntax should be rejected")

            status_text = _build_status_message(session, user_id=int(user.id), assistant_name="Synapse")
            _assert("η:" in status_text, "status text did not include viscosity summary")
            _assert("Nodes:" in status_text, "status text did not include node count")

            prune_result = prune_mission_log(
                payload=MissionLogPruneRequest(project_id=int(project.id), older_than_hours=None, clear_all=True),
                current_user=user,
                session=session,
            )
            _assert(int(prune_result.pruned_count) == 2, f"expected 2 pruned rows, got {prune_result.pruned_count!r}")
            _assert(int(prune_result.remaining_count) == 0, f"expected 0 remaining rows, got {prune_result.remaining_count!r}")

        print("telegram_command_smoketest=ok")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
