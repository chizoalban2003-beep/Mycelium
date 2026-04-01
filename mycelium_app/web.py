from __future__ import annotations

import io

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

import pandas as pd

from mycelium_app.db import get_session
from mycelium_app.models import Project, ProjectMember, ProjectRole, TreeNode, User
from mycelium_app.physics_predictor import PhysicsPlane, PredictorError, run_physics_prediction
from mycelium_app.security import create_access_token
from mycelium_app.security import decode_token
from mycelium_app.settings import settings


templates = Jinja2Templates(directory="templates")
router = APIRouter(include_in_schema=False)


def _get_web_user(request: Request, session: Session) -> User | None:
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    try:
        payload = decode_token(token)
        subject = payload.get("sub")
        if not subject:
            return None
        user_id = int(subject)
    except Exception:
        return None
    user = session.exec(select(User).where(User.id == user_id)).first()
    if not user or not user.is_active:
        return None
    return user


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return RedirectResponse(url="/projects", status_code=302)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "app_name": settings.app_name})


@router.post("/login")
def login_action(
    request: Request,
    session: Session = Depends(get_session),
    email: str = Form(...),
    password: str = Form(...),
):
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    # Reuse the same verifier as API via passlib context
    from mycelium_app.security import verify_password

    if not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(subject=str(user.id))
    response = RedirectResponse(url="/projects", status_code=302)
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.access_token_expire_minutes * 60,
    )
    return response


@router.post("/logout")
def logout_action():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(settings.cookie_name)
    return response


@router.get("/projects", response_class=HTMLResponse)
def projects_page(
    request: Request,
    session: Session = Depends(get_session),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    memberships = session.exec(select(ProjectMember).where(ProjectMember.user_id == current_user.id)).all()
    project_ids = [m.project_id for m in memberships]
    projects = session.exec(select(Project).where(Project.id.in_(project_ids))).all() if project_ids else []
    return templates.TemplateResponse(
        "projects.html",
        {"request": request, "user": current_user, "projects": projects, "app_name": settings.app_name},
    )


@router.post("/projects")
def create_project_action(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    session: Session = Depends(get_session),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    project = Project(name=name, description=description, created_by_user_id=current_user.id)
    session.add(project)
    session.commit()
    session.refresh(project)
    session.add(ProjectMember(project_id=project.id, user_id=current_user.id, role=ProjectRole.owner))
    session.commit()
    return RedirectResponse(url=f"/projects/{project.id}", status_code=302)


def _build_tree(nodes: list[TreeNode]):
    by_parent: dict[int | None, list[TreeNode]] = {}
    for n in nodes:
        by_parent.setdefault(n.parent_id, []).append(n)
    for children in by_parent.values():
        children.sort(key=lambda x: (x.created_at, x.id or 0))
    return by_parent


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail_page(
    project_id: int,
    request: Request,
    session: Session = Depends(get_session),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == current_user.id)
    ).first()
    if not member:
        return RedirectResponse(url="/projects", status_code=302)
    project = session.get(Project, project_id)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    nodes = session.exec(select(TreeNode).where(TreeNode.project_id == project_id)).all()
    tree = _build_tree(nodes)
    return templates.TemplateResponse(
        "project_tree.html",
        {
            "request": request,
            "user": current_user,
            "project": project,
            "member": member,
            "tree": tree,
            "app_name": settings.app_name,
        },
    )


@router.get("/projects/{project_id}/game", response_class=HTMLResponse)
def project_game_page(
    project_id: int,
    request: Request,
    session: Session = Depends(get_session),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == current_user.id)
    ).first()
    if not member:
        return RedirectResponse(url="/projects", status_code=302)
    project = session.get(Project, project_id)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)

    return templates.TemplateResponse(
        "game.html",
        {
            "request": request,
            "user": current_user,
            "project": project,
            "member": member,
            "app_name": settings.app_name,
        },
    )


@router.get("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    session: Session = Depends(get_session),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "user": current_user,
            "app_name": settings.app_name,
            "result": None,
            "error": None,
            "columns": None,
            "target_col": "",
            "plane": PhysicsPlane.solid.value,
            "top_k": 30,
        },
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict_action(
    request: Request,
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    target_col: str = Form(""),
    plane: str = Form(PhysicsPlane.solid.value),
    top_k: int = Form(30),
    max_rows: int = Form(5000),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    error: str | None = None
    result = None
    columns = None

    try:
        plane_enum = PhysicsPlane(plane)
    except Exception:
        plane_enum = PhysicsPlane.solid

    try:
        raw = await file.read()
        if not raw:
            raise PredictorError("Empty upload")
        df = pd.read_csv(io.BytesIO(raw), nrows=max(1, min(int(max_rows), 200_000)))
        columns = list(df.columns)
        if not target_col:
            raise PredictorError(
                "Please enter a target column name and submit again. "
                f"Detected columns: {columns}"
            )
        top_k = max(1, min(int(top_k), 200))

        pred = run_physics_prediction(
            df,
            target_col=target_col,
            plane=plane_enum,
            top_k_weights=top_k,
        )

        result = {
            "target": pred.target,
            "target_kind": pred.target_kind,
            "plane": pred.plane.value,
            "weights": [
                {
                    "feature": w.feature,
                    "weight": round(float(w.weight), 6),
                    "method": w.method,
                    "kind": w.feature_kind,
                    "signed": w.signed,
                }
                for w in pred.weights
            ],
            "metrics": {
                "target_kind": pred.metrics.target_kind,
                "n_rows": pred.metrics.n_rows,
                "n_features_used": pred.metrics.n_features_used,
                "mae": None if pred.metrics.mae is None else round(float(pred.metrics.mae), 6),
                "rmse": None if pred.metrics.rmse is None else round(float(pred.metrics.rmse), 6),
                "accuracy": None if pred.metrics.accuracy is None else round(float(pred.metrics.accuracy), 6),
            },
            "preview": pred.preview_rows,
        }

    except PredictorError as e:
        error = str(e)
    except Exception as e:
        # Keep the UI helpful without leaking stack traces into HTML.
        error = f"Failed to run predictor: {type(e).__name__}: {e}"

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "user": current_user,
            "app_name": settings.app_name,
            "result": result,
            "error": error,
            "columns": columns,
            "target_col": target_col,
            "plane": plane_enum.value,
            "top_k": top_k,
        },
    )
