"""Fixed-size, normalised numeric observation built from SimWorld state.

Frames: "body" = (x: drone right, y: up, z: drone forward), the same axes as the action.
Everything is clipped to [-1, 1] (most features are in [0, 1]).  Print the table with
``python -m firerescue.envs.observation``.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .rl_config import EnvConfig


class ObservationBuilder:
    def __init__(self, world, cfg: EnvConfig, coverage):
        self.world, self.cfg, self.coverage = world, cfg, coverage
        self.n_rooms = coverage.n_rooms
        self.names: List[str] = []
        self.docs: List[Tuple[str, int, str]] = []

        def add(name, n, doc):
            self.docs.append((name, n, doc))
            self.names += [f"{name}[{i}]" if n > 1 else name for i in range(n)]

        b = world.config.building
        self.size = np.array([b.size_x, b.height, b.size_z])
        add("pos", 3, "Position / building size, mapped to [-1, 1] (x, altitude, z)")
        add("vel_body", 3, "Velocity in body frame / drone speed limits (right, up, forward)")
        add("yaw_sin_cos", 2, "sin and cos of the yaw angle")
        add("goal_vec_body", 3, "Vector to the look-ahead point on the shortest path to the current target viewpoint "
                                "(body frame, / 10 m, clipped); y = goal altitude error")
        add("goal_dist", 1, "Shortest-path distance to the target viewpoint / 30 m (clipped to 1; 1 if none)")
        add("proximity_rays", cfg.n_rays + 2, "1 - clearance/ray_range for n_rays horizontal rays (clockwise from "
                                              "forward) + up + down; 1 = touching, 0 = nothing within range")
        k = cfg.n_fires_in_obs
        add("fires", 4 * k, f"{k} nearest fires: body-frame vector / 10 m (3) + danger level 1 - d/(1.5 R) in [0,1]; zero padded")
        add("thermal_peak", 1, "Strongest synthetic thermal signal among UNDISCOVERED victims with line of sight (0 if none)")
        add("thermal_bearing", 3, "Unit body-frame bearing to that peak (zeros if none)")
        add("thermal_interference", 1, "Fire-induced thermal interference at the drone / 0.5 (clipped)")
        g = cfg.thermal_grid
        add("thermal_grid", g * g, f"{g}x{g} body-aligned (row = forward, col = right) synthetic heat map of undiscovered "
                                    "victims + fires within sensor range, line-of-sight masked, / 1.2")
        add("room_explored", self.n_rooms, "Explored fraction of every room (config order)")
        add("coverage_global", 1, "Explored fraction of the whole building")
        add("victims_remaining", 1, "Undiscovered victims / total victims")
        add("time_left", 1, "Fraction of the episode step budget remaining")
        add("prev_action", 4, "Previous (sanitised) action [vx, vy, vz, yaw_rate]")
        add("collision", 1, "1 if the drone is touching a surface this step")
        self.size_total = len(self.names)
        ang = np.arange(cfg.n_rays) * (2 * math.pi / cfg.n_rays)
        self._ray_angles = ang

    # -------------------------------------------------------------------- helpers
    @staticmethod
    def _body(vec, fwd, right) -> np.ndarray:
        return np.array([float(np.dot(vec, right)), float(vec[1]), float(np.dot(vec, fwd))])

    def rays(self) -> np.ndarray:
        w, d, cfg = self.world, self.world.drone, self.cfg
        fwd, right = d.forward_vector(), d.right_vector()
        dirs = [fwd * math.cos(a) + right * math.sin(a) for a in self._ray_angles]
        dirs += [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]
        dist = w.obstacles.raycast(d.pos, np.array(dirs), cfg.ray_range + d.cfg.radius) - d.cfg.radius
        dist[-2] = min(dist[-2], d.cfg.max_altitude - d.pos[1])
        dist[-1] = min(dist[-1], d.pos[1] - d.cfg.min_altitude)
        return np.clip(1.0 - np.maximum(dist, 0.0) / cfg.ray_range, 0.0, 1.0)

    # ---------------------------------------------------------------------- build
    def build(self, goal_point, prev_action, steps: int) -> np.ndarray:
        w, d, cfg = self.world, self.world.drone, self.cfg
        fwd, right = d.forward_vector(), d.right_vector()
        out: List[np.ndarray] = []
        out.append(d.pos / self.size * 2.0 - 1.0)
        out.append(self._body(d.vel, fwd, right) / np.array([d.cfg.max_speed_xz, d.cfg.max_speed_y, d.cfg.max_speed_xz]))
        yr = math.radians(d.yaw)
        out.append(np.array([math.sin(yr), math.cos(yr)]))

        if goal_point is None:
            gv, gd = np.zeros(3), 1.0
        else:
            rel = np.array([goal_point[0] - d.pos[0], cfg.goal_altitude - d.pos[1], goal_point[1] - d.pos[2]])
            gv, gd = self._body(rel, fwd, right) / 10.0, min(1.0, self._goal_dist / 30.0)
        out.append(gv)
        out.append(np.array([gd]))
        out.append(self.rays())

        fires = sorted(w.fires, key=lambda f: f.distance_to(d.pos))[: cfg.n_fires_in_obs]
        fv = np.zeros(4 * cfg.n_fires_in_obs)
        for i, f in enumerate(fires):
            fv[4 * i:4 * i + 3] = self._body(f.position - d.pos, fwd, right) / 10.0
            fv[4 * i + 3] = max(0.0, 1.0 - f.distance_to(d.pos) / (f.danger_radius * w.config.mission.warning_factor))
        out.append(fv)

        th = w.thermal
        by_id = {v.id: v for v in w.victims}
        best, bsig = None, 0.0
        for r in th.victims:
            if not by_id[r.victim_id].discovered and r.line_of_sight and r.signal > bsig:
                best, bsig = by_id[r.victim_id], r.signal
        bearing = np.zeros(3)
        if best is not None:
            v = self._body(best.sensor_point - d.pos, fwd, right)
            bearing = v / (np.linalg.norm(v) + 1e-9)
        out.append(np.array([min(1.0, bsig)]))
        out.append(bearing)
        out.append(np.array([th.interference / 0.5]))

        n = cfg.thermal_grid * 2          # sample finer, then 2x2 average-pool
        undisc = [v for v in w.victims if not v.discovered]
        grid = w.sensor.heat_grid(d.pos, th, undisc, w.fires, w.obstacles, n=n, yaw_deg=d.yaw)
        pooled = grid.reshape(cfg.thermal_grid, 2, cfg.thermal_grid, 2).mean(axis=(1, 3))
        out.append(np.flipud(pooled).ravel() / 1.2)        # first row = farthest ahead

        out.append(self.coverage.room_fractions())
        out.append(np.array([self.coverage.global_fraction()]))
        total = max(1, len(w.victims))
        out.append(np.array([(total - w.discovered_count) / total]))
        out.append(np.array([max(0.0, 1.0 - steps / cfg.max_episode_steps)]))
        out.append(np.asarray(prev_action, float))
        out.append(np.array([1.0 if w.collision_active else 0.0]))
        obs = np.concatenate([np.asarray(o, float).ravel() for o in out])
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)

    _goal_dist = float("inf")

    def set_goal_distance(self, value: float) -> None:
        self._goal_dist = value if np.isfinite(value) else 30.0


def describe() -> str:
    from ..scenario import default_scenario
    from ..world import SimWorld
    from .coverage import CoverageTracker
    w = SimWorld(default_scenario())
    ob = ObservationBuilder(w, EnvConfig(), CoverageTracker(w))
    rows = ["| Feature | Size | Description |", "|---|---|---|"] + [f"| `{n}` | {s} | {d} |" for n, s, d in ob.docs]
    return "\n".join(rows) + f"\n\nTotal observation size: {ob.size_total}"


if __name__ == "__main__":
    print(describe())
