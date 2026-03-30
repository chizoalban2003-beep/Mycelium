from __future__ import annotations

import math
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ProjectMember, ProjectRole, User


router = APIRouter(prefix="/api", tags=["game"])


def _require_member(session: Session, project_id: int, user_id: int, min_role: ProjectRole) -> None:
    role_order = {ProjectRole.viewer: 1, ProjectRole.editor: 2, ProjectRole.owner: 3}
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    if role_order[member.role] < role_order[min_role]:
        raise HTTPException(status_code=403, detail="Insufficient role")


@router.get("/projects/{project_id}/game/state")
def get_game_state(
    project_id: int,
    left_name: str = "Home",
    right_name: str = "Away",
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Return a live-updating sports-like state.

    MVP behavior: deterministic simulation based on time + project_id.
    Next step: replace with dataset/model-driven state (predictions, SHAP, drift, etc).
    """

    _require_member(session, project_id, current_user.id, ProjectRole.viewer)

    t = time.time()
    tick = int(t)

    # Pitch coordinates normalized to [0,1] x [0,1]
    ball_x = 0.5 + 0.35 * math.sin(t * 0.7 + project_id)
    ball_y = 0.5 + 0.25 * math.sin(t * 1.1 + project_id * 0.3)

    players = []
    for i in range(11):
        # Home team
        x = 0.25 + 0.12 * math.sin(t * 0.9 + i) + 0.05 * math.sin(t * 1.7 + i * 0.5)
        y = (i + 1) / 12 + 0.06 * math.sin(t * 0.8 + i * 0.9)
        players.append({"id": f"L{i}", "team": "left", "x": float(max(0.05, min(0.95, x))), "y": float(max(0.05, min(0.95, y)))})

        # Away team
        x2 = 0.75 + 0.12 * math.sin(t * 0.85 + i + 1.4) + 0.05 * math.sin(t * 1.65 + i * 0.55)
        y2 = (i + 1) / 12 + 0.06 * math.sin(t * 0.78 + i * 0.88 + 0.4)
        players.append({"id": f"R{i}", "team": "right", "x": float(max(0.05, min(0.95, x2))), "y": float(max(0.05, min(0.95, y2)))})

    # Fake score that changes slowly
    left_score = int((math.sin(t / 25 + project_id) + 1) * 1.2)
    right_score = int((math.sin(t / 27 + project_id * 0.7 + 1.1) + 1) * 1.2)

    return {
        "tick": tick,
        "teams": {"left": {"name": left_name, "score": left_score}, "right": {"name": right_name, "score": right_score}},
        "ball": {"x": float(max(0.02, min(0.98, ball_x))), "y": float(max(0.02, min(0.98, ball_y)))},
        "players": players,
    }
