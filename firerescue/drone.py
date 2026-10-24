"""Quadrotor kinematic model: velocity commands, acceleration limits, collision response.

This is a deliberately simple kinematic model (not rotor-level dynamics).  Commands are
normalised body-frame velocity set-points, which is also the action interface planned for
the Gymnasium environment in STEP 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .collision import ObstacleSet
from .config import BuildingConfig, DroneConfig


@dataclass
class DroneCommand:
    """Normalised body-frame command, each field in [-1, 1].

    forward: +1 = max speed along the drone heading; right: +1 = strafe right;
    up: +1 = climb; yaw: +1 = turn right (clockwise seen from above).
    """
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0

    def clipped(self) -> "DroneCommand":
        c = lambda v: float(min(1.0, max(-1.0, v)))
        return DroneCommand(c(self.forward), c(self.right), c(self.up), c(self.yaw))

    def is_zero(self) -> bool:
        return not (self.forward or self.right or self.up or self.yaw)

    def as_array(self) -> np.ndarray:
        return np.array([self.forward, self.right, self.up, self.yaw], dtype=float)


@dataclass
class CollisionInfo:
    collided: bool = False
    names: List[str] = field(default_factory=list)
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    impact_speed: float = 0.0


class Drone:
    def __init__(self, cfg: DroneConfig, building: BuildingConfig):
        self.cfg = cfg
        self.building = building
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0                      # degrees
        self.last_collision = CollisionInfo()
        self.reset()

    def reset(self) -> None:
        self.pos = np.array(self.cfg.start_position, dtype=float)
        self.vel = np.zeros(3)
        self.yaw = float(self.cfg.start_yaw_deg) % 360.0
        self.last_collision = CollisionInfo()

    # ------------------------------------------------------------------ helpers
    @property
    def altitude(self) -> float:
        return float(self.pos[1])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def landed(self) -> bool:
        return self.pos[1] <= self.cfg.min_altitude + 1e-6

    def forward_vector(self) -> np.ndarray:
        r = math.radians(self.yaw)
        return np.array([math.sin(r), 0.0, math.cos(r)])

    def right_vector(self) -> np.ndarray:
        r = math.radians(self.yaw)
        return np.array([math.cos(r), 0.0, -math.sin(r)])

    # --------------------------------------------------------------------- step
    def step(self, command: DroneCommand, dt: float, obstacles: ObstacleSet) -> CollisionInfo:
        c, cfg = command.clipped(), self.cfg
        self.yaw = (self.yaw + c.yaw * cfg.max_yaw_rate_deg * dt) % 360.0

        horiz = self.forward_vector() * c.forward + self.right_vector() * c.right
        n = float(np.linalg.norm(horiz))
        if n > 1.0:
            horiz /= n
        target = np.array([horiz[0] * cfg.max_speed_xz, c.up * cfg.max_speed_y, horiz[2] * cfg.max_speed_xz])

        dv = target - self.vel
        max_dv = cfg.max_accel * dt
        mag = float(np.linalg.norm(dv))
        if mag > max_dv:
            dv *= max_dv / mag
        self.vel = self.vel + dv
        pos = self.pos + self.vel * dt

        pos, vel, contacts, impact = obstacles.resolve_sphere(pos, self.vel, cfg.radius)
        names = [k.name for k in contacts]
        normal = np.zeros(3)
        for k in contacts:
            normal += k.normal
        nn = float(np.linalg.norm(normal))
        normal = normal / nn if nn > 1e-9 else normal

        # building envelope (failsafe, exterior walls normally stop the drone first)
        r = cfg.radius
        for axis, hi in ((0, self.building.size_x), (2, self.building.size_z)):
            if pos[axis] < r:
                pos[axis], vel[axis] = r, max(0.0, vel[axis])
            elif pos[axis] > hi - r:
                pos[axis], vel[axis] = hi - r, min(0.0, vel[axis])
        if pos[1] > cfg.max_altitude:                      # ceiling contact counts as a collision
            impact = max(impact, max(0.0, vel[1]))
            pos[1], vel[1] = cfg.max_altitude, min(0.0, vel[1])
            names.append("ceiling")
            normal = normal + np.array([0.0, -1.0, 0.0])
        if pos[1] < cfg.min_altitude:                      # floor = landing, not a collision
            pos[1], vel[1] = cfg.min_altitude, max(0.0, vel[1])

        self.pos, self.vel = pos, vel
        self.last_collision = CollisionInfo(bool(names), names, normal, impact)
        return self.last_collision
