from __future__ import annotations

from datetime import datetime
import hashlib
import math
import random
from typing import Any


WORLD_VERSION = 1
WORLD_BOUNDS = 96.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:16], 16)


def _layer_anchor(layer: str) -> float:
    if layer == "bedrock":
        return -16.0
    if layer == "gaseous":
        return 16.0
    return 0.0


def _empty_world(seed: int) -> dict[str, Any]:
    return {
        "version": WORLD_VERSION,
        "seed": int(seed),
        "tick": 0,
        "entities": [],
        "infrastructure": {"nodes": [], "links": []},
        "events": [],
        "metrics": {
            "entity_count": 0,
            "mean_energy": 0.0,
            "mean_cohesion": 0.0,
            "node_count": 0,
            "link_count": 0,
            "life_index": 0.0,
            "stability": 0.0,
        },
        "as_of": _now_iso(),
    }


def _normalize_layer(raw: str) -> str:
    text = str(raw or "").lower()
    if text in {"bedrock", "solid", "immutable"}:
        return "bedrock"
    if text in {"gaseous", "gas", "turbulent"}:
        return "gaseous"
    return "liquid"


def _bootstrap_entities(
    *,
    dwellers: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    seed: int,
    max_entities: int = 220,
) -> list[dict[str, Any]]:
    rng = random.Random(seed ^ 0xA5A5A5A5)
    entities: list[dict[str, Any]] = []

    for idx, row in enumerate(dwellers):
        if len(entities) >= max_entities:
            break
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "active")
        if status == "retired":
            continue
        vol = float(row.get("volatility_score", 0.45) or 0.45)
        util = float(row.get("utility_signal", 0.55) or 0.55)
        cycles = int(row.get("survival_cycles", 0) or 0)
        layer = "bedrock" if cycles >= 3 else "gaseous" if vol >= 0.7 else "liquid"
        anchor = _layer_anchor(layer)
        base = _stable_seed(str(row.get("id") or row.get("name") or f"dweller-{idx}"))
        row_rng = random.Random(base ^ seed)
        entities.append(
            {
                "id": f"dweller::{str(row.get('id') or idx)}",
                "label": str(row.get("name") or f"Dweller {idx + 1}"),
                "kind": "dweller",
                "layer": layer,
                "x": row_rng.uniform(-42.0, 42.0),
                "y": anchor + row_rng.uniform(-5.0, 5.0),
                "z": row_rng.uniform(-42.0, 42.0),
                "vx": row_rng.uniform(-0.16, 0.16),
                "vy": row_rng.uniform(-0.12, 0.12),
                "vz": row_rng.uniform(-0.16, 0.16),
                "mass": _clip(0.9 + util * 1.8, 0.8, 3.8),
                "energy": _clip(0.45 + util * 0.5, 0.2, 1.0),
                "cohesion": _clip(0.25 + (cycles * 0.08) + (util * 0.3), 0.0, 1.0),
                "volatility": _clip(vol, 0.0, 1.0),
                "age_ticks": 0,
                "dna": hashlib.sha1(f"{base}:{layer}:{idx}".encode("utf-8")).hexdigest()[:12],
            }
        )

    # Signals become ambient particles that can seed new infrastructure.
    for idx, row in enumerate(signals):
        if len(entities) >= max_entities:
            break
        if not isinstance(row, dict):
            continue
        sig_type = str(row.get("signal_type") or row.get("type") or "signal")
        app_name = str(row.get("app_name") or row.get("device") or sig_type)
        score = _clip(float(row.get("session_seconds", 0.0) or 0.0) / 3600.0, 0.0, 1.0)
        layer = "gaseous" if score < 0.15 else "liquid"
        anchor = _layer_anchor(layer)
        base = _stable_seed(f"{app_name}:{sig_type}:{idx}")
        row_rng = random.Random(base ^ (seed << 1))
        entities.append(
            {
                "id": f"signal::{idx}",
                "label": app_name[:42],
                "kind": "signal",
                "layer": layer,
                "x": row_rng.uniform(-58.0, 58.0),
                "y": anchor + row_rng.uniform(-7.0, 7.0),
                "z": row_rng.uniform(-58.0, 58.0),
                "vx": row_rng.uniform(-0.25, 0.25),
                "vy": row_rng.uniform(-0.14, 0.14),
                "vz": row_rng.uniform(-0.25, 0.25),
                "mass": _clip(0.35 + score * 1.2, 0.2, 2.0),
                "energy": _clip(0.25 + score * 0.55, 0.1, 0.95),
                "cohesion": _clip(0.12 + score * 0.35, 0.0, 1.0),
                "volatility": _clip(0.65 - (score * 0.4), 0.05, 0.95),
                "age_ticks": 0,
                "dna": hashlib.sha1(f"{app_name}:{base}:{idx}".encode("utf-8")).hexdigest()[:12],
            }
        )

    if not entities:
        for idx in range(18):
            layer = ("gaseous", "liquid", "bedrock")[idx % 3]
            entities.append(
                {
                    "id": f"seed::{idx}",
                    "label": f"Seed {idx + 1}",
                    "kind": "seed",
                    "layer": layer,
                    "x": rng.uniform(-28.0, 28.0),
                    "y": _layer_anchor(layer) + rng.uniform(-4.0, 4.0),
                    "z": rng.uniform(-28.0, 28.0),
                    "vx": rng.uniform(-0.08, 0.08),
                    "vy": rng.uniform(-0.08, 0.08),
                    "vz": rng.uniform(-0.08, 0.08),
                    "mass": 0.7,
                    "energy": 0.5,
                    "cohesion": 0.4,
                    "volatility": 0.4,
                    "age_ticks": 0,
                    "dna": hashlib.sha1(f"seed::{idx}".encode("utf-8")).hexdigest()[:12],
                }
            )

    return entities


def _infrastructure_from_entities(
    *,
    entities: list[dict[str, Any]],
    previous: dict[str, Any],
    tick: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prev_nodes = {str(row.get("id")): row for row in list((previous or {}).get("nodes") or []) if isinstance(row, dict)}
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cell_span = 22.0
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("kind")) == "signal" and float(entity.get("cohesion", 0.0) or 0.0) < 0.18:
            continue
        x = float(entity.get("x", 0.0) or 0.0)
        z = float(entity.get("z", 0.0) or 0.0)
        key = (int((x + WORLD_BOUNDS) // cell_span), int((z + WORLD_BOUNDS) // cell_span))
        buckets.setdefault(key, []).append(entity)

    for (gx, gz), members in buckets.items():
        if len(members) < 3:
            continue
        xs = [float(row.get("x", 0.0) or 0.0) for row in members]
        ys = [float(row.get("y", 0.0) or 0.0) for row in members]
        zs = [float(row.get("z", 0.0) or 0.0) for row in members]
        cohesions = [float(row.get("cohesion", 0.0) or 0.0) for row in members]
        energies = [float(row.get("energy", 0.0) or 0.0) for row in members]
        avg_y = sum(ys) / max(1, len(ys))
        layer = "bedrock" if avg_y <= -6 else "liquid" if avg_y <= 6 else "gaseous"
        node_id = f"hub::{gx}:{gz}"
        prev = prev_nodes.get(node_id, {})
        age = int(prev.get("age_ticks", 0) or 0) + 1
        integrity = _clip((sum(cohesions) / len(cohesions)) * 0.65 + (sum(energies) / len(energies)) * 0.35, 0.0, 1.0)
        throughput = int(len(members) + round(sum(energies)))
        nodes.append(
            {
                "id": node_id,
                "kind": "hub",
                "layer": layer,
                "x": round(sum(xs) / len(xs), 3),
                "y": round(avg_y, 3),
                "z": round(sum(zs) / len(zs), 3),
                "integrity": round(integrity, 4),
                "throughput": throughput,
                "age_ticks": age,
                "members": [str(row.get("id")) for row in members[:12]],
            }
        )
        if age == 1:
            events.append(
                {
                    "at_tick": tick,
                    "event_type": "infrastructure_birth",
                    "node_id": node_id,
                    "layer": layer,
                    "throughput": throughput,
                }
            )

    # Connect nearby hubs with flow links.
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            dx = float(a["x"]) - float(b["x"])
            dz = float(a["z"]) - float(b["z"])
            distance = math.sqrt((dx * dx) + (dz * dz))
            if distance > 34.0:
                continue
            flow = int((int(a["throughput"]) + int(b["throughput"])) / 2)
            stability = _clip((float(a["integrity"]) + float(b["integrity"])) / 2, 0.0, 1.0)
            link_id = f"link::{min(a['id'], b['id'])}::{max(a['id'], b['id'])}"
            links.append(
                {
                    "id": link_id,
                    "source": a["id"],
                    "target": b["id"],
                    "flow": flow,
                    "stability": round(stability, 4),
                    "distance": round(distance, 3),
                }
            )

    # Keep deterministic but bounded topology.
    links.sort(key=lambda row: (-(int(row.get("flow", 0) or 0)), str(row.get("id"))))
    links = links[:48]
    nodes.sort(key=lambda row: (str(row.get("layer")), -int(row.get("throughput", 0) or 0), str(row.get("id"))))

    return {"nodes": nodes, "links": links}, events


def _tick_entities(
    *,
    entities: list[dict[str, Any]],
    seed: int,
    tick: int,
    secondary_force: float,
    adaptive_heat: float,
) -> None:
    rng = random.Random(seed ^ (tick * 7919))
    n = len(entities)
    if n == 0:
        return
    sample_n = min(16, max(6, n // 8))
    indices = list(range(n))

    for idx, entity in enumerate(entities):
        x = float(entity.get("x", 0.0) or 0.0)
        y = float(entity.get("y", 0.0) or 0.0)
        z = float(entity.get("z", 0.0) or 0.0)
        vx = float(entity.get("vx", 0.0) or 0.0)
        vy = float(entity.get("vy", 0.0) or 0.0)
        vz = float(entity.get("vz", 0.0) or 0.0)
        mass = _clip(float(entity.get("mass", 1.0) or 1.0), 0.2, 5.0)
        energy = _clip(float(entity.get("energy", 0.5) or 0.5), 0.01, 1.0)
        cohesion = _clip(float(entity.get("cohesion", 0.4) or 0.4), 0.0, 1.0)
        volatility = _clip(float(entity.get("volatility", 0.4) or 0.4), 0.0, 1.0)
        layer = _normalize_layer(str(entity.get("layer") or "liquid"))
        anchor_y = _layer_anchor(layer)

        fx = 0.0
        fy = (anchor_y - y) * 0.035
        fz = 0.0

        # Deterministic local interaction sample per entity.
        base_rng = random.Random(seed ^ (idx * 31337) ^ (tick * 104729))
        neighbors = base_rng.sample(indices, k=sample_n) if n > sample_n else indices
        for n_idx in neighbors:
            if n_idx == idx:
                continue
            other = entities[n_idx]
            ox = float(other.get("x", 0.0) or 0.0)
            oy = float(other.get("y", 0.0) or 0.0)
            oz = float(other.get("z", 0.0) or 0.0)
            om = _clip(float(other.get("mass", 1.0) or 1.0), 0.2, 5.0)
            oc = _clip(float(other.get("cohesion", 0.4) or 0.4), 0.0, 1.0)
            dx = ox - x
            dy = oy - y
            dz = oz - z
            dist_sq = (dx * dx) + (dy * dy) + (dz * dz) + 1.2
            inv_dist = 1.0 / math.sqrt(dist_sq)
            attraction = ((cohesion + oc) * 0.5) * 0.07 * om / dist_sq
            repulsion = 0.02 / dist_sq
            force = attraction - repulsion
            fx += dx * inv_dist * force
            fy += dy * inv_dist * force * 0.75
            fz += dz * inv_dist * force

        stress_noise = (secondary_force * 0.75) + (adaptive_heat * 0.25)
        jitter = (rng.random() - 0.5) * (0.04 + (stress_noise * 0.06))
        fx += jitter
        fz -= jitter * 0.7
        if layer == "liquid":
            fy += (stress_noise - 0.3) * 0.05
        if layer == "gaseous":
            fy += 0.015 + (volatility * 0.02)
        if layer == "bedrock":
            fy -= 0.018 + (cohesion * 0.015)

        damping = 0.85 - (volatility * 0.22) + (cohesion * 0.09)
        damping = _clip(damping, 0.45, 0.96)
        vx = (vx * damping) + (fx / max(0.3, mass))
        vy = (vy * damping) + (fy / max(0.3, mass))
        vz = (vz * damping) + (fz / max(0.3, mass))

        x = _clip(x + vx, -WORLD_BOUNDS, WORLD_BOUNDS)
        y = _clip(y + vy, -WORLD_BOUNDS * 0.6, WORLD_BOUNDS * 0.6)
        z = _clip(z + vz, -WORLD_BOUNDS, WORLD_BOUNDS)

        speed = math.sqrt((vx * vx) + (vy * vy) + (vz * vz))
        energy = _clip(energy + (0.014 * cohesion) - (0.022 * speed) - (0.02 * stress_noise * volatility), 0.01, 1.0)
        cohesion = _clip(cohesion + (0.012 * (1.0 - volatility)) - (0.01 * speed) + (0.006 * (1.0 - secondary_force)), 0.0, 1.0)

        if y >= 8.0:
            layer = "gaseous"
        elif y <= -8.0:
            layer = "bedrock"
        else:
            layer = "liquid"

        entity["x"] = round(x, 4)
        entity["y"] = round(y, 4)
        entity["z"] = round(z, 4)
        entity["vx"] = round(vx, 5)
        entity["vy"] = round(vy, 5)
        entity["vz"] = round(vz, 5)
        entity["energy"] = round(energy, 5)
        entity["cohesion"] = round(cohesion, 5)
        entity["layer"] = layer
        entity["age_ticks"] = int(entity.get("age_ticks", 0) or 0) + 1


def _world_metrics(entities: list[dict[str, Any]], infrastructure: dict[str, Any]) -> dict[str, float | int]:
    if not entities:
        return {
            "entity_count": 0,
            "mean_energy": 0.0,
            "mean_cohesion": 0.0,
            "node_count": 0,
            "link_count": 0,
            "life_index": 0.0,
            "stability": 0.0,
        }
    energies = [float(row.get("energy", 0.0) or 0.0) for row in entities]
    cohesions = [float(row.get("cohesion", 0.0) or 0.0) for row in entities]
    speeds = [
        math.sqrt(
            (float(row.get("vx", 0.0) or 0.0) ** 2)
            + (float(row.get("vy", 0.0) or 0.0) ** 2)
            + (float(row.get("vz", 0.0) or 0.0) ** 2)
        )
        for row in entities
    ]
    node_count = len(list((infrastructure or {}).get("nodes") or []))
    link_count = len(list((infrastructure or {}).get("links") or []))
    mean_energy = sum(energies) / len(energies)
    mean_cohesion = sum(cohesions) / len(cohesions)
    infra_density = _clip((node_count + (link_count * 0.5)) / max(8.0, len(entities) * 0.22), 0.0, 1.0)
    turbulence = _clip((sum(speeds) / max(1.0, len(speeds))) / 2.8, 0.0, 1.0)
    life_index = _clip((mean_energy * 0.34) + (mean_cohesion * 0.41) + (infra_density * 0.25), 0.0, 1.0)
    stability = _clip((mean_cohesion * 0.62) + ((1.0 - turbulence) * 0.38), 0.0, 1.0)
    return {
        "entity_count": len(entities),
        "mean_energy": round(mean_energy, 5),
        "mean_cohesion": round(mean_cohesion, 5),
        "node_count": node_count,
        "link_count": link_count,
        "life_index": round(life_index, 5),
        "stability": round(stability, 5),
    }


def evolve_world_state(
    *,
    existing_state: dict[str, Any] | None,
    dwellers: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    seed_key: str,
    secondary_force: float,
    adaptive_heat: float,
    ticks: int = 1,
) -> dict[str, Any]:
    seed = _stable_seed(seed_key)
    state = existing_state if isinstance(existing_state, dict) else _empty_world(seed)
    if int(state.get("version", 0) or 0) != WORLD_VERSION:
        state = _empty_world(seed)
    state["seed"] = int(state.get("seed", seed) or seed)
    state.setdefault("entities", [])
    state.setdefault("infrastructure", {"nodes": [], "links": []})
    state.setdefault("events", [])

    entities = list(state.get("entities") or [])
    if not entities:
        entities = _bootstrap_entities(dwellers=dwellers, signals=signals, seed=state["seed"])

    tick_count = max(1, min(int(ticks or 1), 24))
    new_events: list[dict[str, Any]] = []
    for _ in range(tick_count):
        current_tick = int(state.get("tick", 0) or 0) + 1
        _tick_entities(
            entities=entities,
            seed=int(state["seed"]),
            tick=current_tick,
            secondary_force=_clip(float(secondary_force or 0.0), 0.0, 1.0),
            adaptive_heat=_clip(float(adaptive_heat or 0.0), 0.0, 1.0),
        )
        infrastructure, infra_events = _infrastructure_from_entities(
            entities=entities,
            previous=state.get("infrastructure") if isinstance(state.get("infrastructure"), dict) else {},
            tick=current_tick,
        )
        state["infrastructure"] = infrastructure
        new_events.extend(infra_events)
        state["tick"] = current_tick

    state["entities"] = entities[:260]
    state["metrics"] = _world_metrics(state["entities"], state.get("infrastructure", {}))
    state["events"] = (list(state.get("events") or []) + new_events)[-40:]
    state["as_of"] = _now_iso()

    life_idx = float(state.get("metrics", {}).get("life_index", 0.0) or 0.0)
    if life_idx >= 0.75:
        phase = "self-organizing"
    elif life_idx >= 0.45:
        phase = "adapting"
    else:
        phase = "forming"
    state["phase"] = phase
    return state
