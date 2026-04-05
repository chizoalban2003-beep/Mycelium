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
from mycelium_app.physics_predictor import PhysicsPlane, PredictorError, infer_target_kind, run_physics_prediction
from mycelium_app.presets import (
    PRODUCTION_CLASSIFICATION_BALANCED_KWARGS,
    PRODUCTION_CLASSIFICATION_BALANCED_PRESET_NAME,
    PRODUCTION_CLASSIFICATION_MAX_ACCURACY_KWARGS,
    PRODUCTION_CLASSIFICATION_MAX_ACCURACY_PRESET_NAME,
    PRODUCTION_CLASSIFICATION_MAX_COVERAGE_KWARGS,
    PRODUCTION_CLASSIFICATION_MAX_COVERAGE_PRESET_NAME,
    PRODUCTION_REGRESSION_KWARGS,
    PRODUCTION_REGRESSION_PRESET_NAME,
)
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
            "train_ratio": 0.8,
            "random_seed": 42,
            "no_split": False,
            "n_cycles": 30,
            "cycle_learning_rate": 0.18,
            "cascade_enabled": True,
            "competitive_inhibition": True,
            "thermal_noise": False,
            "stage2_cycles": 2,
            "stage2_trigger_cycle": 50,
            "inhibition_strength": 0.7,
            "scavenger_cycles": 1,
            "stage2_shatter_complexes": True,
            "low_confidence_mode": "none",
            "low_confidence_threshold": 0.0,
            "low_confidence_entropy_threshold": 0.0,
            "low_confidence_smear_metric": "entropy",
            "low_confidence_combine_rule": "or",
            "low_confidence_auto_conf_quantile": 0.20,
            "low_confidence_auto_smear_quantile": 0.80,
            "low_confidence_require_ionized": False,
            "low_confidence_ionization_pvalue": 0.05,
            "low_confidence_ionization_z_min": 0.25,
            "low_confidence_confirmatory_enabled": False,
            "low_confidence_confirmatory_conf_min": 0.50,
            "low_confidence_confirmatory_conf_max": 0.90,
            "low_confidence_confirmatory_consensus_threshold": 0.60,
            "low_confidence_confirmatory_min_ion_hits": 0,
            "low_confidence_secondary_enabled": False,
            "low_confidence_secondary_cycles": 0,
            "low_confidence_secondary_viscosity_multiplier": 0.75,
            "low_confidence_secondary_viscosity_anneal": False,
            "low_confidence_secondary_viscosity_multiplier_start": 0.95,
            "low_confidence_secondary_inhibition_multiplier": 0.85,
            "low_confidence_secondary_shear_multiplier": 1.10,
            "low_confidence_secondary_relax_ionization_gate": True,
            "low_confidence_secondary_ionization_z_min": 0.10,
            "low_confidence_secondary_relaxed_ion_conf_min": 0.55,
            "low_confidence_secondary_use_spearman": True,
            "low_confidence_secondary_spearman_min_abs": 0.015,
            "low_confidence_secondary_spearman_margin": 0.010,
            "low_confidence_secondary_promote_min_zone_votes": 3,
            "low_confidence_secondary_promote_z_min": 0.50,
            "low_confidence_secondary_promote_conf_min": 0.42,
            "low_confidence_primary_sieve_enabled": False,
            "low_confidence_primary_sieve_cycle_a": 30,
            "low_confidence_primary_sieve_cycle_b": 45,
            "low_confidence_primary_sieve_shake_cycles": 2,
            "low_confidence_primary_sieve_reverse_multiplier": 1.0,
            "low_confidence_primary_sieve_noise_std": 0.08,
            "low_confidence_primary_sieve_instability_min": 0.50,
            "low_confidence_primary_sieve_conf_delta_max": 0.003,
            "low_confidence_secondary_sieve_enabled": False,
            "low_confidence_secondary_sieve_cycles": 2,
            "low_confidence_secondary_sieve_reverse_multiplier": 0.75,
            "low_confidence_secondary_sieve_noise_std": 0.04,
            "low_confidence_secondary_sieve_instability_min": 0.65,
            "low_confidence_secondary_sieve_conf_delta_max": 0.002,
            "low_confidence_secondary_sieve_update_norm_max": 0.003,
            "classification_goal": "balanced",
            "cleaning_enabled": True,
            "cleaning_outlier_strategy": "winsorize",
            "cleaning_outlier_fold": 1.5,
            "cleaning_outlier_q_low": 0.005,
            "cleaning_outlier_q_high": 0.995,
            "cleaning_arbitrary_min": None,
            "cleaning_arbitrary_max": None,
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
    train_ratio: float = Form(0.8),
    random_seed: int = Form(42),
    no_split: str | None = Form(None),
    max_rows: int = Form(5000),
    n_cycles: int = Form(30),
    cycle_learning_rate: float = Form(0.18),
    cascade_enabled: str | None = Form(None),
    competitive_inhibition: str | None = Form(None),
    thermal_noise: str | None = Form(None),
    stage2_cycles: int = Form(2),
    stage2_trigger_cycle: int = Form(50),
    inhibition_strength: float = Form(0.7),
    scavenger_cycles: int = Form(1),
    stage2_shatter_complexes: str | None = Form(None),
    low_confidence_mode: str = Form("none"),
    low_confidence_threshold: float = Form(0.0),
    low_confidence_entropy_threshold: float = Form(0.0),
    low_confidence_smear_metric: str = Form("entropy"),
    low_confidence_combine_rule: str = Form("or"),
    low_confidence_auto_conf_quantile: float = Form(0.20),
    low_confidence_auto_smear_quantile: float = Form(0.80),
    low_confidence_require_ionized: str | None = Form(None),
    low_confidence_ionization_pvalue: float = Form(0.05),
    low_confidence_ionization_z_min: float = Form(0.25),
    low_confidence_confirmatory_enabled: str | None = Form(None),
    low_confidence_confirmatory_conf_min: float = Form(0.50),
    low_confidence_confirmatory_conf_max: float = Form(0.90),
    low_confidence_confirmatory_consensus_threshold: float = Form(0.60),
    low_confidence_confirmatory_min_ion_hits: int = Form(0),
    low_confidence_secondary_enabled: str | None = Form(None),
    low_confidence_secondary_cycles: int = Form(0),
    low_confidence_secondary_viscosity_multiplier: float = Form(0.75),
    low_confidence_secondary_viscosity_anneal: str | None = Form(None),
    low_confidence_secondary_viscosity_multiplier_start: float = Form(0.95),
    low_confidence_secondary_inhibition_multiplier: float = Form(0.85),
    low_confidence_secondary_shear_multiplier: float = Form(1.10),
    low_confidence_secondary_relax_ionization_gate: str | None = Form(None),
    low_confidence_secondary_ionization_z_min: float = Form(0.10),
    low_confidence_secondary_relaxed_ion_conf_min: float = Form(0.55),
    low_confidence_secondary_use_spearman: str | None = Form(None),
    low_confidence_secondary_spearman_min_abs: float = Form(0.015),
    low_confidence_secondary_spearman_margin: float = Form(0.010),
    low_confidence_secondary_promote_min_zone_votes: int = Form(3),
    low_confidence_secondary_promote_z_min: float = Form(0.50),
    low_confidence_secondary_promote_conf_min: float = Form(0.42),
    low_confidence_primary_sieve_enabled: str | None = Form(None),
    low_confidence_primary_sieve_cycle_a: int = Form(30),
    low_confidence_primary_sieve_cycle_b: int = Form(45),
    low_confidence_primary_sieve_shake_cycles: int = Form(2),
    low_confidence_primary_sieve_reverse_multiplier: float = Form(1.0),
    low_confidence_primary_sieve_noise_std: float = Form(0.08),
    low_confidence_primary_sieve_instability_min: float = Form(0.50),
    low_confidence_primary_sieve_conf_delta_max: float = Form(0.003),
    low_confidence_secondary_sieve_enabled: str | None = Form(None),
    low_confidence_secondary_sieve_cycles: int = Form(2),
    low_confidence_secondary_sieve_reverse_multiplier: float = Form(0.75),
    low_confidence_secondary_sieve_noise_std: float = Form(0.04),
    low_confidence_secondary_sieve_instability_min: float = Form(0.65),
    low_confidence_secondary_sieve_conf_delta_max: float = Form(0.002),
    low_confidence_secondary_sieve_update_norm_max: float = Form(0.003),
    classification_goal: str = Form("balanced"),
    cleaning_enabled: str | None = Form(None),
    cleaning_outlier_strategy: str = Form("winsorize"),
    cleaning_outlier_fold: float = Form(1.5),
    cleaning_outlier_q_low: float = Form(0.005),
    cleaning_outlier_q_high: float = Form(0.995),
    cleaning_arbitrary_min: float | None = Form(None),
    cleaning_arbitrary_max: float | None = Form(None),
):
    current_user = _get_web_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    error: str | None = None
    result = None
    columns = None

    # Initialize derived booleans early so the error path can still render the form.
    cascade_enabled_bool = bool(cascade_enabled)
    competitive_inhibition_bool = bool(competitive_inhibition)
    thermal_noise_bool = bool(thermal_noise)
    low_confidence_require_ionized_bool = bool(low_confidence_require_ionized)
    low_confidence_confirmatory_enabled_bool = bool(low_confidence_confirmatory_enabled)
    low_confidence_secondary_enabled_bool = bool(low_confidence_secondary_enabled)
    low_confidence_secondary_viscosity_anneal_bool = bool(low_confidence_secondary_viscosity_anneal)
    low_confidence_secondary_relax_ionization_gate_bool = bool(low_confidence_secondary_relax_ionization_gate)
    low_confidence_secondary_use_spearman_bool = bool(low_confidence_secondary_use_spearman)
    low_confidence_secondary_sieve_enabled_bool = bool(low_confidence_secondary_sieve_enabled)
    low_confidence_primary_sieve_enabled_bool = bool(low_confidence_primary_sieve_enabled)
    cleaning_enabled_bool = bool(cleaning_enabled)

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

        # Split controls
        no_split_enabled = bool(no_split)
        if no_split_enabled:
            train_ratio = 1.0
        else:
            # Friendly clamping before predictor validation
            try:
                train_ratio = float(train_ratio)
            except Exception:
                train_ratio = 0.8
            train_ratio = max(0.05, min(0.95, train_ratio))

        # Advanced electrophoresis controls (safe parsing + clamping)
        try:
            n_cycles = int(n_cycles)
        except Exception:
            n_cycles = 30
        n_cycles = max(1, min(200, n_cycles))

        try:
            cycle_learning_rate = float(cycle_learning_rate)
        except Exception:
            cycle_learning_rate = 0.18
        cycle_learning_rate = max(0.01, min(1.0, cycle_learning_rate))

        try:
            stage2_cycles = int(stage2_cycles)
        except Exception:
            stage2_cycles = 2
        stage2_cycles = max(0, min(50, stage2_cycles))

        try:
            stage2_trigger_cycle = int(stage2_trigger_cycle)
        except Exception:
            stage2_trigger_cycle = 50
        stage2_trigger_cycle = max(0, min(200, stage2_trigger_cycle))

        try:
            inhibition_strength = float(inhibition_strength)
        except Exception:
            inhibition_strength = 0.7
        inhibition_strength = max(0.0, min(2.0, inhibition_strength))

        try:
            scavenger_cycles = int(scavenger_cycles)
        except Exception:
            scavenger_cycles = 1
        scavenger_cycles = max(0, min(10, scavenger_cycles))

        # Low-confidence readout controls
        low_confidence_mode = str(low_confidence_mode or "none")
        try:
            low_confidence_threshold = float(low_confidence_threshold)
        except Exception:
            low_confidence_threshold = 0.0
        try:
            low_confidence_entropy_threshold = float(low_confidence_entropy_threshold)
        except Exception:
            low_confidence_entropy_threshold = 0.0
        low_confidence_smear_metric = str(low_confidence_smear_metric or "entropy")
        low_confidence_combine_rule = str(low_confidence_combine_rule or "or")

        try:
            low_confidence_auto_conf_quantile = float(low_confidence_auto_conf_quantile)
        except Exception:
            low_confidence_auto_conf_quantile = 0.20
        low_confidence_auto_conf_quantile = max(0.0, min(1.0, low_confidence_auto_conf_quantile))

        try:
            low_confidence_auto_smear_quantile = float(low_confidence_auto_smear_quantile)
        except Exception:
            low_confidence_auto_smear_quantile = 0.80
        low_confidence_auto_smear_quantile = max(0.0, min(1.0, low_confidence_auto_smear_quantile))

        low_confidence_require_ionized_bool = bool(low_confidence_require_ionized)
        try:
            low_confidence_ionization_pvalue = float(low_confidence_ionization_pvalue)
        except Exception:
            low_confidence_ionization_pvalue = 0.05
        low_confidence_ionization_pvalue = max(0.0, min(1.0, low_confidence_ionization_pvalue))
        try:
            low_confidence_ionization_z_min = float(low_confidence_ionization_z_min)
        except Exception:
            low_confidence_ionization_z_min = 0.25
        low_confidence_ionization_z_min = max(0.0, low_confidence_ionization_z_min)

        low_confidence_confirmatory_enabled_bool = bool(low_confidence_confirmatory_enabled)
        try:
            low_confidence_confirmatory_conf_min = float(low_confidence_confirmatory_conf_min)
        except Exception:
            low_confidence_confirmatory_conf_min = 0.50
        try:
            low_confidence_confirmatory_conf_max = float(low_confidence_confirmatory_conf_max)
        except Exception:
            low_confidence_confirmatory_conf_max = 0.90
        try:
            low_confidence_confirmatory_consensus_threshold = float(low_confidence_confirmatory_consensus_threshold)
        except Exception:
            low_confidence_confirmatory_consensus_threshold = 0.60
        low_confidence_confirmatory_consensus_threshold = max(0.0, min(1.0, low_confidence_confirmatory_consensus_threshold))
        try:
            low_confidence_confirmatory_min_ion_hits = int(low_confidence_confirmatory_min_ion_hits)
        except Exception:
            low_confidence_confirmatory_min_ion_hits = 0
        low_confidence_confirmatory_min_ion_hits = max(0, min(50, low_confidence_confirmatory_min_ion_hits))

        low_confidence_secondary_enabled_bool = bool(low_confidence_secondary_enabled)
        try:
            low_confidence_secondary_cycles = int(low_confidence_secondary_cycles)
        except Exception:
            low_confidence_secondary_cycles = 0
        low_confidence_secondary_cycles = max(0, min(50, low_confidence_secondary_cycles))
        try:
            low_confidence_secondary_viscosity_multiplier = float(low_confidence_secondary_viscosity_multiplier)
        except Exception:
            low_confidence_secondary_viscosity_multiplier = 0.75
        low_confidence_secondary_viscosity_multiplier = max(0.10, min(2.50, low_confidence_secondary_viscosity_multiplier))

        low_confidence_secondary_viscosity_anneal_bool = bool(low_confidence_secondary_viscosity_anneal)
        try:
            low_confidence_secondary_viscosity_multiplier_start = float(low_confidence_secondary_viscosity_multiplier_start)
        except Exception:
            low_confidence_secondary_viscosity_multiplier_start = 0.95
        low_confidence_secondary_viscosity_multiplier_start = max(
            0.10, min(2.50, low_confidence_secondary_viscosity_multiplier_start)
        )
        try:
            low_confidence_secondary_inhibition_multiplier = float(low_confidence_secondary_inhibition_multiplier)
        except Exception:
            low_confidence_secondary_inhibition_multiplier = 0.85
        low_confidence_secondary_inhibition_multiplier = max(0.0, min(2.0, low_confidence_secondary_inhibition_multiplier))
        try:
            low_confidence_secondary_shear_multiplier = float(low_confidence_secondary_shear_multiplier)
        except Exception:
            low_confidence_secondary_shear_multiplier = 1.10
        low_confidence_secondary_shear_multiplier = max(0.0, min(4.0, low_confidence_secondary_shear_multiplier))
        low_confidence_secondary_relax_ionization_gate_bool = bool(low_confidence_secondary_relax_ionization_gate)
        try:
            low_confidence_secondary_ionization_z_min = float(low_confidence_secondary_ionization_z_min)
        except Exception:
            low_confidence_secondary_ionization_z_min = 0.10
        low_confidence_secondary_ionization_z_min = max(0.0, low_confidence_secondary_ionization_z_min)
        try:
            low_confidence_secondary_relaxed_ion_conf_min = float(low_confidence_secondary_relaxed_ion_conf_min)
        except Exception:
            low_confidence_secondary_relaxed_ion_conf_min = 0.55
        low_confidence_secondary_relaxed_ion_conf_min = max(0.0, min(1.0, low_confidence_secondary_relaxed_ion_conf_min))
        low_confidence_secondary_use_spearman_bool = bool(low_confidence_secondary_use_spearman)
        try:
            low_confidence_secondary_spearman_min_abs = float(low_confidence_secondary_spearman_min_abs)
        except Exception:
            low_confidence_secondary_spearman_min_abs = 0.015
        low_confidence_secondary_spearman_min_abs = max(0.0, min(1.0, low_confidence_secondary_spearman_min_abs))
        try:
            low_confidence_secondary_spearman_margin = float(low_confidence_secondary_spearman_margin)
        except Exception:
            low_confidence_secondary_spearman_margin = 0.010
        low_confidence_secondary_spearman_margin = max(0.0, min(1.0, low_confidence_secondary_spearman_margin))
        try:
            low_confidence_secondary_promote_min_zone_votes = int(low_confidence_secondary_promote_min_zone_votes)
        except Exception:
            low_confidence_secondary_promote_min_zone_votes = 3
        low_confidence_secondary_promote_min_zone_votes = max(0, min(50, low_confidence_secondary_promote_min_zone_votes))
        try:
            low_confidence_secondary_promote_z_min = float(low_confidence_secondary_promote_z_min)
        except Exception:
            low_confidence_secondary_promote_z_min = 0.50
        low_confidence_secondary_promote_z_min = max(0.0, low_confidence_secondary_promote_z_min)
        try:
            low_confidence_secondary_promote_conf_min = float(low_confidence_secondary_promote_conf_min)
        except Exception:
            low_confidence_secondary_promote_conf_min = 0.42
        low_confidence_secondary_promote_conf_min = max(0.0, min(1.0, low_confidence_secondary_promote_conf_min))

        try:
            low_confidence_primary_sieve_cycle_a = int(low_confidence_primary_sieve_cycle_a)
        except Exception:
            low_confidence_primary_sieve_cycle_a = 30
        low_confidence_primary_sieve_cycle_a = max(1, min(200, low_confidence_primary_sieve_cycle_a))
        try:
            low_confidence_primary_sieve_cycle_b = int(low_confidence_primary_sieve_cycle_b)
        except Exception:
            low_confidence_primary_sieve_cycle_b = 45
        low_confidence_primary_sieve_cycle_b = max(1, min(200, low_confidence_primary_sieve_cycle_b))
        try:
            low_confidence_primary_sieve_shake_cycles = int(low_confidence_primary_sieve_shake_cycles)
        except Exception:
            low_confidence_primary_sieve_shake_cycles = 2
        low_confidence_primary_sieve_shake_cycles = max(0, min(25, low_confidence_primary_sieve_shake_cycles))
        try:
            low_confidence_primary_sieve_reverse_multiplier = float(low_confidence_primary_sieve_reverse_multiplier)
        except Exception:
            low_confidence_primary_sieve_reverse_multiplier = 1.0
        low_confidence_primary_sieve_reverse_multiplier = max(0.0, min(2.0, low_confidence_primary_sieve_reverse_multiplier))
        try:
            low_confidence_primary_sieve_noise_std = float(low_confidence_primary_sieve_noise_std)
        except Exception:
            low_confidence_primary_sieve_noise_std = 0.08
        low_confidence_primary_sieve_noise_std = max(0.0, min(0.50, low_confidence_primary_sieve_noise_std))
        try:
            low_confidence_primary_sieve_instability_min = float(low_confidence_primary_sieve_instability_min)
        except Exception:
            low_confidence_primary_sieve_instability_min = 0.50
        low_confidence_primary_sieve_instability_min = max(0.0, min(1.0, low_confidence_primary_sieve_instability_min))
        try:
            low_confidence_primary_sieve_conf_delta_max = float(low_confidence_primary_sieve_conf_delta_max)
        except Exception:
            low_confidence_primary_sieve_conf_delta_max = 0.003
        low_confidence_primary_sieve_conf_delta_max = max(0.0, min(0.25, low_confidence_primary_sieve_conf_delta_max))

        low_confidence_secondary_sieve_enabled_bool = bool(low_confidence_secondary_sieve_enabled)
        try:
            low_confidence_secondary_sieve_cycles = int(low_confidence_secondary_sieve_cycles)
        except Exception:
            low_confidence_secondary_sieve_cycles = 2
        low_confidence_secondary_sieve_cycles = max(0, min(25, low_confidence_secondary_sieve_cycles))
        try:
            low_confidence_secondary_sieve_reverse_multiplier = float(low_confidence_secondary_sieve_reverse_multiplier)
        except Exception:
            low_confidence_secondary_sieve_reverse_multiplier = 0.75
        low_confidence_secondary_sieve_reverse_multiplier = max(0.0, min(2.0, low_confidence_secondary_sieve_reverse_multiplier))
        try:
            low_confidence_secondary_sieve_noise_std = float(low_confidence_secondary_sieve_noise_std)
        except Exception:
            low_confidence_secondary_sieve_noise_std = 0.04
        low_confidence_secondary_sieve_noise_std = max(0.0, min(0.50, low_confidence_secondary_sieve_noise_std))
        try:
            low_confidence_secondary_sieve_instability_min = float(low_confidence_secondary_sieve_instability_min)
        except Exception:
            low_confidence_secondary_sieve_instability_min = 0.65
        low_confidence_secondary_sieve_instability_min = max(0.0, min(1.0, low_confidence_secondary_sieve_instability_min))
        try:
            low_confidence_secondary_sieve_conf_delta_max = float(low_confidence_secondary_sieve_conf_delta_max)
        except Exception:
            low_confidence_secondary_sieve_conf_delta_max = 0.002
        low_confidence_secondary_sieve_conf_delta_max = max(0.0, min(0.25, low_confidence_secondary_sieve_conf_delta_max))
        try:
            low_confidence_secondary_sieve_update_norm_max = float(low_confidence_secondary_sieve_update_norm_max)
        except Exception:
            low_confidence_secondary_sieve_update_norm_max = 0.003
        low_confidence_secondary_sieve_update_norm_max = max(0.0, min(10.0, low_confidence_secondary_sieve_update_norm_max))

        # Cleaning/outlier controls
        cleaning_outlier_strategy = str(cleaning_outlier_strategy or "winsorize").strip().lower()
        if cleaning_outlier_strategy not in ("winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "none"):
            cleaning_outlier_strategy = "winsorize"
        try:
            cleaning_outlier_fold = float(cleaning_outlier_fold)
        except Exception:
            cleaning_outlier_fold = 1.5
        cleaning_outlier_fold = max(0.01, min(50.0, cleaning_outlier_fold))
        try:
            cleaning_outlier_q_low = float(cleaning_outlier_q_low)
        except Exception:
            cleaning_outlier_q_low = 0.005
        try:
            cleaning_outlier_q_high = float(cleaning_outlier_q_high)
        except Exception:
            cleaning_outlier_q_high = 0.995
        cleaning_outlier_q_low = max(0.0, min(0.49, cleaning_outlier_q_low))
        cleaning_outlier_q_high = max(0.51, min(1.0, cleaning_outlier_q_high))
        if cleaning_arbitrary_min is not None:
            try:
                cleaning_arbitrary_min = float(cleaning_arbitrary_min)
            except Exception:
                cleaning_arbitrary_min = None
        if cleaning_arbitrary_max is not None:
            try:
                cleaning_arbitrary_max = float(cleaning_arbitrary_max)
            except Exception:
                cleaning_arbitrary_max = None

        cascade_enabled_bool = bool(cascade_enabled)
        competitive_inhibition_bool = bool(competitive_inhibition)
        thermal_noise_bool = bool(thermal_noise)

        base_kwargs: dict[str, object] = (lambda: {
            "target_col": target_col,
            "plane": plane_enum,
            "train_fraction": float(train_ratio),
            "random_seed": int(random_seed),
            "top_k_weights": top_k,
            "n_cycles": n_cycles,
            "cycle_learning_rate": cycle_learning_rate,
            "cascade_enabled": cascade_enabled_bool,
            "competitive_inhibition": competitive_inhibition_bool,
            "thermal_noise": thermal_noise_bool,
            "stage2_cycles": stage2_cycles,
            "stage2_trigger_cycle": stage2_trigger_cycle,
            "inhibition_strength": inhibition_strength,
            "scavenger_cycles": scavenger_cycles,
            "stage2_shatter_complexes": bool(stage2_shatter_complexes),
            "low_confidence_mode": low_confidence_mode,
            "low_confidence_threshold": low_confidence_threshold,
            "low_confidence_entropy_threshold": low_confidence_entropy_threshold,
            "low_confidence_smear_metric": low_confidence_smear_metric,
            "low_confidence_combine_rule": low_confidence_combine_rule,
            "low_confidence_auto_conf_quantile": low_confidence_auto_conf_quantile,
            "low_confidence_auto_smear_quantile": low_confidence_auto_smear_quantile,
            "low_confidence_require_ionized": low_confidence_require_ionized_bool,
            "low_confidence_ionization_pvalue": low_confidence_ionization_pvalue,
            "low_confidence_ionization_z_min": low_confidence_ionization_z_min,
            "low_confidence_confirmatory_enabled": low_confidence_confirmatory_enabled_bool,
            "low_confidence_confirmatory_conf_min": low_confidence_confirmatory_conf_min,
            "low_confidence_confirmatory_conf_max": low_confidence_confirmatory_conf_max,
            "low_confidence_confirmatory_consensus_threshold": low_confidence_confirmatory_consensus_threshold,
            "low_confidence_confirmatory_min_ion_hits": low_confidence_confirmatory_min_ion_hits,
            "low_confidence_secondary_enabled": low_confidence_secondary_enabled_bool,
            "low_confidence_secondary_cycles": low_confidence_secondary_cycles,
            "low_confidence_secondary_viscosity_multiplier": low_confidence_secondary_viscosity_multiplier,
            "low_confidence_secondary_viscosity_anneal": low_confidence_secondary_viscosity_anneal_bool,
            "low_confidence_secondary_viscosity_multiplier_start": low_confidence_secondary_viscosity_multiplier_start,
            "low_confidence_secondary_inhibition_multiplier": low_confidence_secondary_inhibition_multiplier,
            "low_confidence_secondary_shear_multiplier": low_confidence_secondary_shear_multiplier,
            "low_confidence_secondary_relax_ionization_gate": low_confidence_secondary_relax_ionization_gate_bool,
            "low_confidence_secondary_ionization_z_min": low_confidence_secondary_ionization_z_min,
            "low_confidence_secondary_relaxed_ion_conf_min": low_confidence_secondary_relaxed_ion_conf_min,
            "low_confidence_secondary_use_spearman": low_confidence_secondary_use_spearman_bool,
            "low_confidence_secondary_spearman_min_abs": low_confidence_secondary_spearman_min_abs,
            "low_confidence_secondary_spearman_margin": low_confidence_secondary_spearman_margin,
            "low_confidence_secondary_promote_min_zone_votes": low_confidence_secondary_promote_min_zone_votes,
            "low_confidence_secondary_promote_z_min": low_confidence_secondary_promote_z_min,
            "low_confidence_secondary_promote_conf_min": low_confidence_secondary_promote_conf_min,
            "low_confidence_primary_sieve_enabled": low_confidence_primary_sieve_enabled_bool,
            "low_confidence_primary_sieve_cycle_a": low_confidence_primary_sieve_cycle_a,
            "low_confidence_primary_sieve_cycle_b": low_confidence_primary_sieve_cycle_b,
            "low_confidence_primary_sieve_shake_cycles": low_confidence_primary_sieve_shake_cycles,
            "low_confidence_primary_sieve_reverse_multiplier": low_confidence_primary_sieve_reverse_multiplier,
            "low_confidence_primary_sieve_noise_std": low_confidence_primary_sieve_noise_std,
            "low_confidence_primary_sieve_instability_min": low_confidence_primary_sieve_instability_min,
            "low_confidence_primary_sieve_conf_delta_max": low_confidence_primary_sieve_conf_delta_max,
            "low_confidence_secondary_sieve_enabled": low_confidence_secondary_sieve_enabled_bool,
            "low_confidence_secondary_sieve_cycles": low_confidence_secondary_sieve_cycles,
            "low_confidence_secondary_sieve_reverse_multiplier": low_confidence_secondary_sieve_reverse_multiplier,
            "low_confidence_secondary_sieve_noise_std": low_confidence_secondary_sieve_noise_std,
            "low_confidence_secondary_sieve_instability_min": low_confidence_secondary_sieve_instability_min,
            "low_confidence_secondary_sieve_conf_delta_max": low_confidence_secondary_sieve_conf_delta_max,
            "low_confidence_secondary_sieve_update_norm_max": low_confidence_secondary_sieve_update_norm_max,
            "cleaning_enabled": cleaning_enabled_bool,
            "cleaning_outlier_strategy": cleaning_outlier_strategy,
            "cleaning_outlier_fold": cleaning_outlier_fold,
            "cleaning_outlier_q_low": cleaning_outlier_q_low,
            "cleaning_outlier_q_high": cleaning_outlier_q_high,
            "cleaning_arbitrary_min": cleaning_arbitrary_min,
            "cleaning_arbitrary_max": cleaning_arbitrary_max,
        })()

        preset_applied: str | None = None
        try:
            tk = infer_target_kind(df[target_col])
        except Exception:
            tk = "numeric"
        if tk == "categorical" and settings.predictor_lock_production_classification_preset:
            goal = str(classification_goal or "balanced").strip().lower()
            if goal in ("max_accuracy", "accuracy", "precise"):
                base_kwargs.update(PRODUCTION_CLASSIFICATION_MAX_ACCURACY_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_MAX_ACCURACY_PRESET_NAME
            elif goal in ("max_coverage", "coverage", "broad"):
                base_kwargs.update(PRODUCTION_CLASSIFICATION_MAX_COVERAGE_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_MAX_COVERAGE_PRESET_NAME
            else:
                base_kwargs.update(PRODUCTION_CLASSIFICATION_BALANCED_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_BALANCED_PRESET_NAME
        elif settings.predictor_lock_production_regression_preset and tk in ("numeric", "datetime"):
            # Override base kwargs with locked production regression settings.
            base_kwargs.update(PRODUCTION_REGRESSION_KWARGS)
            base_kwargs["plane"] = PhysicsPlane(str(PRODUCTION_REGRESSION_KWARGS.get("plane", "gas")))
            preset_applied = PRODUCTION_REGRESSION_PRESET_NAME

        pred = run_physics_prediction(df, **base_kwargs)

        result = {
            "production_preset": preset_applied,
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
            "migration_map": [
                {
                    "feature": m.feature,
                    "kind": m.feature_kind,
                    "method": m.method,
                    "charge": round(float(m.charge), 6),
                    "ionization": m.ionization,
                    "normality_p": None if m.normality_p is None else round(float(m.normality_p), 8),
                    "p_value": None if m.p_value is None else round(float(m.p_value), 8),
                    "mass": round(float(m.mass), 6),
                    "stable": bool(m.stable),
                    "complex_id": m.complex_id,
                    "complex_size": m.complex_size,
                    "entropy": round(float(m.entropy), 6),
                    "variance": round(float(m.variance), 6),
                    "standard_error": round(float(m.standard_error), 6),
                    "kl_divergence": round(float(m.kl_divergence), 6),
                    "density": round(float(m.density), 6),
                    "viscosity": round(float(m.viscosity), 6),
                    "terminal_velocity": round(float(m.terminal_velocity), 6),
                    "arrival_speed": round(float(m.arrival_speed), 6),
                    "direction": m.direction,
                    "state": m.state,
                }
                for m in pred.migration_map
            ],
            "bonding_map": [
                {
                    "feature_a": b.feature_a,
                    "feature_b": b.feature_b,
                    "affinity": round(float(b.affinity), 6),
                    "bonding_factor": round(float(b.bonding_factor), 6),
                    "bond_type": getattr(b, "bond_type", "affinity"),
                }
                for b in pred.bonding_map
            ],
            "iteration_gains": [
                {
                    "cycle": int(it.cycle),
                    "test_accuracy": None
                    if it.test_accuracy is None
                    else round(float(it.test_accuracy), 6),
                    "test_mae": None if it.test_mae is None else round(float(it.test_mae), 6),
                    "test_rmse": None if it.test_rmse is None else round(float(it.test_rmse), 6),
                    "lift_over_baseline": None
                    if it.lift_over_baseline is None
                    else round(float(it.lift_over_baseline), 6),
                }
                for it in pred.iteration_gains
            ],
            "equilibrium_zones": [
                {
                    "zone_id": int(ez.zone_id),
                    "features": ez.features,
                    "avg_pI": round(float(ez.avg_pI), 6),
                    "avg_momentum": round(float(ez.avg_momentum), 6),
                    "strength": round(float(ez.strength), 6),
                }
                for ez in pred.equilibrium_zones
            ],
            "metrics": {
                "target_kind": pred.metrics.target_kind,
                "n_rows": pred.metrics.n_rows,
                "n_train": pred.metrics.n_train,
                "n_test": pred.metrics.n_test,
                "train_fraction": round(float(pred.metrics.train_fraction), 4),
                "random_seed": int(pred.metrics.random_seed),
                "n_features_used": pred.metrics.n_features_used,
                "buffer_ionization": pred.metrics.buffer_ionization,
                "buffer_normality_p": None
                if pred.metrics.buffer_normality_p is None
                else round(float(pred.metrics.buffer_normality_p), 8),
                "mae": None if pred.metrics.mae is None else round(float(pred.metrics.mae), 6),
                "rmse": None if pred.metrics.rmse is None else round(float(pred.metrics.rmse), 6),
                "baseline_mae": None
                if pred.metrics.baseline_mae is None
                else round(float(pred.metrics.baseline_mae), 6),
                "baseline_rmse": None
                if pred.metrics.baseline_rmse is None
                else round(float(pred.metrics.baseline_rmse), 6),
                "accuracy": None if pred.metrics.accuracy is None else round(float(pred.metrics.accuracy), 6),
                "baseline_accuracy": None
                if pred.metrics.baseline_accuracy is None
                else round(float(pred.metrics.baseline_accuracy), 6),
                "best_cycle": pred.metrics.best_cycle,
                "best_lift": None if pred.metrics.best_lift is None else round(float(pred.metrics.best_lift), 6),
                "gel_band_sharpness": None
                if pred.metrics.gel_band_sharpness is None
                else round(float(pred.metrics.gel_band_sharpness), 6),
                "gel_smearing": None
                if pred.metrics.gel_smearing is None
                else round(float(pred.metrics.gel_smearing), 6),
                "gel_ghost_band_rate": None
                if pred.metrics.gel_ghost_band_rate is None
                else round(float(pred.metrics.gel_ghost_band_rate), 6),
                "gel_confidence_mean": None
                if pred.metrics.gel_confidence_mean is None
                else round(float(pred.metrics.gel_confidence_mean), 6),
                "gel_confidence_std": None
                if pred.metrics.gel_confidence_std is None
                else round(float(pred.metrics.gel_confidence_std), 6),
                "abstain_rate": None
                if getattr(pred.metrics, "abstain_rate", None) is None
                else round(float(pred.metrics.abstain_rate), 6),
                "coverage": None
                if getattr(pred.metrics, "coverage", None) is None
                else round(float(pred.metrics.coverage), 6),
                "selective_accuracy": None
                if getattr(pred.metrics, "selective_accuracy", None) is None
                else round(float(pred.metrics.selective_accuracy), 6),
            },
            "diagnostics": {
                "trapped_features": sorted(
                    [
                        {
                            "feature": m.feature,
                            "kind": m.feature_kind,
                            "ionization": m.ionization,
                            "method": m.method,
                            "charge": round(float(m.charge), 6),
                            "viscosity": round(float(m.viscosity), 6),
                            "terminal_velocity": round(float(m.terminal_velocity), 6),
                            "p_value": None if m.p_value is None else round(float(m.p_value), 8),
                        }
                        for m in pred.migration_map
                        if str(getattr(m, "state", "")) == "trapped"
                    ],
                    key=lambda d: (float(d.get("viscosity", 0.0)), abs(float(d.get("terminal_velocity", 0.0)))),
                    reverse=True,
                )[:12]
            },
            "preview": pred.preview_rows,
        }

        # Predictor-level diagnostics (e.g. abstain reason breakdown).
        try:
            if getattr(pred, "diagnostics", None):
                cleaning = (pred.diagnostics or {}).get("cleaning")
                if cleaning:
                    result.setdefault("diagnostics", {})["cleaning"] = cleaning
                sel = (pred.diagnostics or {}).get("selective")
                if sel:
                    result.setdefault("diagnostics", {})["abstain_breakdown"] = sel
        except Exception:
            pass

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
            "train_ratio": train_ratio,
            "random_seed": random_seed,
            "no_split": bool(no_split),
            "n_cycles": n_cycles,
            "cycle_learning_rate": cycle_learning_rate,
            "cascade_enabled": cascade_enabled_bool,
            "competitive_inhibition": competitive_inhibition_bool,
            "thermal_noise": thermal_noise_bool,
            "stage2_cycles": stage2_cycles,
            "stage2_trigger_cycle": stage2_trigger_cycle,
            "inhibition_strength": inhibition_strength,
            "scavenger_cycles": scavenger_cycles,
            "stage2_shatter_complexes": bool(stage2_shatter_complexes),
            "low_confidence_mode": low_confidence_mode,
            "low_confidence_threshold": low_confidence_threshold,
            "low_confidence_entropy_threshold": low_confidence_entropy_threshold,
            "low_confidence_smear_metric": low_confidence_smear_metric,
            "low_confidence_combine_rule": low_confidence_combine_rule,
            "low_confidence_auto_conf_quantile": low_confidence_auto_conf_quantile,
            "low_confidence_auto_smear_quantile": low_confidence_auto_smear_quantile,
            "low_confidence_require_ionized": low_confidence_require_ionized_bool,
            "low_confidence_ionization_pvalue": low_confidence_ionization_pvalue,
            "low_confidence_ionization_z_min": low_confidence_ionization_z_min,
            "low_confidence_confirmatory_enabled": low_confidence_confirmatory_enabled_bool,
            "low_confidence_confirmatory_conf_min": low_confidence_confirmatory_conf_min,
            "low_confidence_confirmatory_conf_max": low_confidence_confirmatory_conf_max,
            "low_confidence_confirmatory_consensus_threshold": low_confidence_confirmatory_consensus_threshold,
            "low_confidence_confirmatory_min_ion_hits": low_confidence_confirmatory_min_ion_hits,
            "low_confidence_secondary_enabled": low_confidence_secondary_enabled_bool,
            "low_confidence_secondary_cycles": low_confidence_secondary_cycles,
            "low_confidence_secondary_viscosity_multiplier": low_confidence_secondary_viscosity_multiplier,
            "low_confidence_secondary_viscosity_anneal": low_confidence_secondary_viscosity_anneal_bool,
            "low_confidence_secondary_viscosity_multiplier_start": low_confidence_secondary_viscosity_multiplier_start,
            "low_confidence_secondary_inhibition_multiplier": low_confidence_secondary_inhibition_multiplier,
            "low_confidence_secondary_shear_multiplier": low_confidence_secondary_shear_multiplier,
            "low_confidence_secondary_relax_ionization_gate": low_confidence_secondary_relax_ionization_gate_bool,
            "low_confidence_secondary_ionization_z_min": low_confidence_secondary_ionization_z_min,
            "low_confidence_secondary_relaxed_ion_conf_min": low_confidence_secondary_relaxed_ion_conf_min,
            "low_confidence_secondary_use_spearman": low_confidence_secondary_use_spearman_bool,
            "low_confidence_secondary_spearman_min_abs": low_confidence_secondary_spearman_min_abs,
            "low_confidence_secondary_spearman_margin": low_confidence_secondary_spearman_margin,
            "low_confidence_secondary_promote_min_zone_votes": low_confidence_secondary_promote_min_zone_votes,
            "low_confidence_secondary_promote_z_min": low_confidence_secondary_promote_z_min,
            "low_confidence_secondary_promote_conf_min": low_confidence_secondary_promote_conf_min,
            "low_confidence_primary_sieve_enabled": low_confidence_primary_sieve_enabled_bool,
            "low_confidence_primary_sieve_cycle_a": low_confidence_primary_sieve_cycle_a,
            "low_confidence_primary_sieve_cycle_b": low_confidence_primary_sieve_cycle_b,
            "low_confidence_primary_sieve_shake_cycles": low_confidence_primary_sieve_shake_cycles,
            "low_confidence_primary_sieve_reverse_multiplier": low_confidence_primary_sieve_reverse_multiplier,
            "low_confidence_primary_sieve_noise_std": low_confidence_primary_sieve_noise_std,
            "low_confidence_primary_sieve_instability_min": low_confidence_primary_sieve_instability_min,
            "low_confidence_primary_sieve_conf_delta_max": low_confidence_primary_sieve_conf_delta_max,
            "low_confidence_secondary_sieve_enabled": low_confidence_secondary_sieve_enabled_bool,
            "low_confidence_secondary_sieve_cycles": low_confidence_secondary_sieve_cycles,
            "low_confidence_secondary_sieve_reverse_multiplier": low_confidence_secondary_sieve_reverse_multiplier,
            "low_confidence_secondary_sieve_noise_std": low_confidence_secondary_sieve_noise_std,
            "low_confidence_secondary_sieve_instability_min": low_confidence_secondary_sieve_instability_min,
            "low_confidence_secondary_sieve_conf_delta_max": low_confidence_secondary_sieve_conf_delta_max,
            "low_confidence_secondary_sieve_update_norm_max": low_confidence_secondary_sieve_update_norm_max,
            "classification_goal": classification_goal,
            "cleaning_enabled": cleaning_enabled_bool,
            "cleaning_outlier_strategy": cleaning_outlier_strategy,
            "cleaning_outlier_fold": cleaning_outlier_fold,
            "cleaning_outlier_q_low": cleaning_outlier_q_low,
            "cleaning_outlier_q_high": cleaning_outlier_q_high,
            "cleaning_arbitrary_min": cleaning_arbitrary_min,
            "cleaning_arbitrary_max": cleaning_arbitrary_max,
        },
    )
