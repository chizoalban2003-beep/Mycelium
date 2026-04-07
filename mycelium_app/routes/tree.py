from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import NodeRun, NodeRunStatus, ProjectMember, ProjectRole, TreeNode, User
from mycelium_app.schemas import NodeRunPublic, TreeNodeCreate, TreeNodePublic
from mycelium_app.stimulus import record_stimulus_event


router = APIRouter(prefix="/api", tags=["tree"])


def _require_member(session: Session, project_id: int, user_id: int, min_role: ProjectRole) -> None:
    role_order = {ProjectRole.viewer: 1, ProjectRole.editor: 2, ProjectRole.owner: 3}
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user_id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    if role_order[member.role] < role_order[min_role]:
        raise HTTPException(status_code=403, detail="Insufficient role")


@router.get("/projects/{project_id}/nodes", response_model=list[TreeNodePublic])
def list_nodes(project_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    _require_member(session, project_id, current_user.id, ProjectRole.viewer)
    nodes = session.exec(select(TreeNode).where(TreeNode.project_id == project_id)).all()
    return [
        TreeNodePublic(
            id=n.id,
            project_id=n.project_id,
            parent_id=n.parent_id,
            name=n.name,
            node_type=n.node_type,
            config_json=n.config_json,
            created_by_user_id=n.created_by_user_id,
            created_at=n.created_at,
        )
        for n in nodes
    ]


@router.post("/projects/{project_id}/nodes", response_model=TreeNodePublic)
def create_node(
    project_id: int,
    payload: TreeNodeCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _require_member(session, project_id, current_user.id, ProjectRole.editor)
    if payload.parent_id is not None:
        parent = session.get(TreeNode, payload.parent_id)
        if not parent or parent.project_id != project_id:
            raise HTTPException(status_code=400, detail="Invalid parent_id")
    node = TreeNode(
        project_id=project_id,
        parent_id=payload.parent_id,
        name=payload.name,
        node_type=payload.node_type,
        config_json=payload.config_json,
        created_by_user_id=current_user.id,
    )
    session.add(node)
    session.commit()
    session.refresh(node)

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=project_id,
            device_id="local",
            source="tree_api",
            modality="workspace",
            signal_type="node_create",
            stimulus={
                "node_type": str(payload.node_type),
                "name_len": len(str(payload.name or "")),
                "has_parent": payload.parent_id is not None,
                "config_len": len(str(payload.config_json or "")),
            },
            occurred_at=node.created_at,
        )
    except Exception:
        pass

    return TreeNodePublic(
        id=node.id,
        project_id=node.project_id,
        parent_id=node.parent_id,
        name=node.name,
        node_type=node.node_type,
        config_json=node.config_json,
        created_by_user_id=node.created_by_user_id,
        created_at=node.created_at,
    )


@router.post("/nodes/{node_id}/run", response_model=NodeRunPublic)
def run_node(node_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    node = session.get(TreeNode, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    _require_member(session, node.project_id, current_user.id, ProjectRole.editor)

    # MVP: create a run record and mark as failed with a placeholder log.
    run = NodeRun(node_id=node_id, status=NodeRunStatus.running, started_at=datetime.utcnow())
    session.add(run)
    session.commit()
    session.refresh(run)

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=node.project_id,
            device_id="local",
            source="tree_api",
            modality="execution",
            signal_type="node_run",
            stimulus={"node_type": str(node.node_type), "status": str(run.status), "name_len": len(str(node.name or ""))},
            occurred_at=run.started_at or datetime.utcnow(),
        )
    except Exception:
        pass

    run.status = NodeRunStatus.failed
    run.finished_at = datetime.utcnow()
    run.logs = "Node execution engine not implemented yet."
    session.add(run)
    session.commit()

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=node.project_id,
            device_id="local",
            source="tree_api",
            modality="execution",
            signal_type="node_run_result",
            stimulus={"node_type": str(node.node_type), "status": str(run.status), "result": "not_implemented"},
            occurred_at=run.finished_at or datetime.utcnow(),
        )
    except Exception:
        pass

    return NodeRunPublic(
        id=run.id,
        node_id=run.node_id,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        logs=run.logs,
    )
