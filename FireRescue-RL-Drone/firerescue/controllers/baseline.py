"""Baseline autonomous controller: ground-truth-map A* search of every room.

This is intentionally a *simple baseline*, not a learned or perception-driven policy.
Assumptions (documented so that the RL agent in STEP 2 can be compared fairly):
  * it knows the building map (obstacle boxes) and fire positions but NOT the victims;
  * it visits coverage viewpoints room by room, flying at a fixed cruise altitude;
  * detection is entirely done by the synthetic thermal sensor inside SimWorld;
  * it avoids fire danger zones (plus margin) and obstacle clearance via the planner.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..drone import DroneCommand
from .planner import GridPlanner, XZ


class BaselineController:
    def __init__(self, world, cruise_altitude: float = 1.2, cruise_speed: float = 1.2,
                 viewpoint_spacing: float = 3.5, clearance_margin: float = 0.2):
        self.world = world
        self.cruise_altitude = cruise_altitude
        self.cruise_speed = cruise_speed          # m/s, capped by the drone limits
        cfg = world.config
        self.planner = GridPlanner(world.obstacles, cfg.building.size_x, cfg.building.size_z,
                                   cruise_altitude, cfg.drone.radius + clearance_margin)
        self.planner.set_fire_zones(world.fires)
        self.viewpoints_all = self._make_viewpoints(viewpoint_spacing)
        self.reset()

    # ----------------------------------------------------------------- set-up
    def _make_viewpoints(self, spacing: float) -> List[XZ]:
        pts: List[XZ] = []
        for room in self.world.config.building.rooms:
            if room.kind != "room":
                continue
            (x0, z0), (x1, z1) = room.min_xz, room.max_xz
            nx, nz = max(1, round((x1 - x0) / spacing)), max(1, round((z1 - z0) / spacing))
            for i in range(nx):
                for j in range(nz):
                    p = (x0 + (i + 0.5) * (x1 - x0) / nx, z0 + (j + 0.5) * (z1 - z0) / nz)
                    c = self.planner.nearest_free(self.planner.cell(p))
                    if c is not None:
                        pts.append(self.planner.centre(c))
        return pts

    def reset(self) -> None:
        self.todo: List[XZ] = list(self.viewpoints_all)
        self.path: List[XZ] = []
        self.target: Optional[XZ] = None
        self._progress_ref = (0.0, None)
        self.stuck_dropped = 0

    # ------------------------------------------------------------------ logic
    def _pick_next(self) -> None:
        pos = (float(self.world.drone.pos[0]), float(self.world.drone.pos[2]))
        best = None
        for vp in self.todo:
            res = self.planner.plan(pos, vp)
            if res is None:
                continue
            if best is None or res[1] < best[1]:
                best = (vp, res[1], res[0])
        if best is None:
            self.todo.clear()
            self.target, self.path = None, []
        else:
            self.target, self.path = best[0], best[2]

    def command(self) -> DroneCommand:
        w, d = self.world, self.world.drone
        if w.status.name != "RUNNING":
            return DroneCommand()
        if not self.path:
            if self.target is not None and self.target in self.todo:
                self.todo.remove(self.target)
            self.target = None
            if not self.todo:                       # nothing left to search: hover
                return self._altitude_only()
            self._pick_next()
            if not self.path:
                return self._altitude_only()

        pos = np.array([d.pos[0], d.pos[2]])
        wp = np.array(self.path[0])
        final = len(self.path) == 1
        dist = float(np.linalg.norm(wp - pos))
        if dist < (0.15 if final else 0.3):
            self.path.pop(0)
            return self._altitude_only()

        # stuck detection: no progress for 4 s -> drop this viewpoint
        t = w.time
        if self._progress_ref[1] is None or math.dist(self._progress_ref[1], pos) > 0.15:
            self._progress_ref = (t, tuple(pos))
        elif t - self._progress_ref[0] > 4.0:
            self.path, self.stuck_dropped = [], self.stuck_dropped + 1
            self._progress_ref = (t, None)
            return DroneCommand()

        speed = min(self.cruise_speed, d.cfg.max_speed_xz)
        if final:
            speed = min(speed, max(0.25, 1.2 * dist))
        elif len(self.path) > 1:
            nxt = np.array(self.path[1])
            a, b = wp - pos, nxt - wp
            cosang = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if cosang < 0.7:                        # sharp turn ahead: slow down
                speed = min(speed, max(0.35, 0.5 + 0.6 * dist))
        direction = (wp - pos) / (dist + 1e-9)
        v_world = np.array([direction[0] * speed, direction[1] * speed])

        fwd, right = d.forward_vector(), d.right_vector()
        vmax = d.cfg.max_speed_xz
        forward_cmd = (v_world[0] * fwd[0] + v_world[1] * fwd[2]) / vmax
        right_cmd = (v_world[0] * right[0] + v_world[1] * right[2]) / vmax

        desired_yaw = math.degrees(math.atan2(v_world[0], v_world[1]))
        err = (desired_yaw - d.yaw + 180.0) % 360.0 - 180.0
        yaw_cmd = max(-1.0, min(1.0, err / 45.0))
        up_cmd = self._altitude_only().up
        return DroneCommand(forward_cmd, right_cmd, up_cmd, yaw_cmd).clipped()

    def _altitude_only(self) -> DroneCommand:
        err = self.cruise_altitude - float(self.world.drone.pos[1])
        vy = self.world.drone.cfg.max_speed_y
        return DroneCommand(up=max(-1.0, min(1.0, 1.5 * err / vy)))
