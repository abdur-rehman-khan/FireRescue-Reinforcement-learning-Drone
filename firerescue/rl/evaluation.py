"""Deterministic evaluation of policies (SB3 models, random, or the Step 1 baseline controller)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..controllers import BaselineController

Policy = Callable[[np.ndarray], np.ndarray]


def model_policy(model, deterministic: bool = True) -> Policy:
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, 4).astype(np.float32)


def run_episode(env, policy: Optional[Policy], seed: int, baseline: bool = False) -> Dict[str, float]:
    """One episode; ``baseline=True`` drives the env with the Step 1 A* controller instead."""
    obs, _ = env.reset(seed=seed)
    ctl = BaselineController(env.world) if baseline else None
    info: Dict = {}
    done = False
    while not done:
        if ctl is not None:
            c = ctl.command()
            s = env._scale
            action = np.clip(np.array([c.right, c.up, c.forward, c.yaw]) / s, -1, 1)
        else:
            action = policy(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return dict(info["episode_metrics"])


def aggregate(episodes: List[Dict[str, float]]) -> Dict[str, float]:
    m = lambda k: float(np.mean([e[k] for e in episodes]))
    found, total = sum(e["victims_found"] for e in episodes), sum(e["victims_total"] for e in episodes)
    succ = [e for e in episodes if e["success"]]
    return {
        "episodes": len(episodes),
        "success_rate": m("success"),
        "mission_completion_rate": m("success"),
        "reward_mean": m("episode_reward"), "reward_std": float(np.std([e["episode_reward"] for e in episodes])),
        "victim_discovery_rate": found / max(1, total), "victims_found_mean": m("victims_found"),
        "collisions_mean": m("collisions"), "collision_free_rate": float(np.mean([e["collisions"] == 0 for e in episodes])),
        "coverage_mean": m("coverage"), "rooms_discovered_mean": m("rooms_discovered"),
        "fire_danger_steps_mean": m("fire_danger_steps"), "overheated_rate": m("overheated"),
        "steps_mean": m("steps"),
        "time_to_success_mean": float(np.mean([e["sim_time"] for e in succ])) if succ else float("nan"),
    }


def evaluate(env, policy: Optional[Policy], episodes: int = 10, seed: int = 1000, baseline: bool = False
             ) -> Dict[str, object]:
    """Episode i uses ``seed + i`` -> identical start states for every evaluated policy."""
    eps = [run_episode(env, policy, seed + i, baseline) for i in range(episodes)]
    return {"episodes": eps, "summary": aggregate(eps)}


def format_summary(name: str, s: Dict[str, float]) -> str:
    return (f"{name:<10} success {s['success_rate']:6.1%} | reward {s['reward_mean']:8.1f} ± {s['reward_std']:6.1f} | "
            f"victims {s['victims_found_mean']:.2f} ({s['victim_discovery_rate']:.0%}) | "
            f"collisions {s['collisions_mean']:5.1f} (free {s['collision_free_rate']:.0%}) | "
            f"coverage {s['coverage_mean']:5.1%} | rooms {s['rooms_discovered_mean']:.1f} | "
            f"fire-steps {s['fire_danger_steps_mean']:5.1f} | steps {s['steps_mean']:6.0f}")


def save_results(path, results: Dict[str, Dict[str, object]]) -> None:
    """Write JSON (all episodes + summaries) and a per-episode CSV next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    rows = [dict(policy=name, episode=i, **ep) for name, r in results.items() for i, ep in enumerate(r["episodes"])]
    if rows:
        with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
