"""Third-person orbit camera with zoom, reset, overview mode and wall-collision avoidance."""
from __future__ import annotations

import math

import numpy as np
from ursina import Vec3, camera, held_keys, mouse


class CameraRig:
    MIN_DIST, MAX_DIST = 1.2, 28.0

    def __init__(self, world):
        self.world = world
        self.overview = False
        self.reset()
        camera.fov = 75

    def reset(self) -> None:
        """Chase view: behind the drone, slightly above."""
        self.overview = False
        self.azimuth = float(self.world.drone.yaw)       # degrees, same convention as drone yaw
        self.pitch = 28.0
        self.dist = 3.6
        self._pos = None

    def toggle_overview(self) -> None:
        self.overview = not self.overview
        if self.overview:
            self.azimuth, self.pitch, self.dist = 0.0, 68.0, 24.0
        else:
            self.reset()

    def zoom(self, direction: int) -> None:
        self.dist = float(np.clip(self.dist * (0.88 if direction > 0 else 1.14), self.MIN_DIST, self.MAX_DIST))

    def orbit(self, d_az: float, d_pitch: float) -> None:
        self.azimuth = (self.azimuth + d_az) % 360.0
        self.pitch = float(np.clip(self.pitch + d_pitch, 5.0, 88.0))

    def update(self, dt: float, shake: float = 0.0) -> None:
        if held_keys["right mouse"] or held_keys["middle mouse"]:
            self.orbit(mouse.velocity[0] * 180.0, -mouse.velocity[1] * 120.0)
        self.orbit((held_keys["right arrow"] - held_keys["left arrow"]) * 90.0 * dt,
                   (held_keys["up arrow"] - held_keys["down arrow"]) * 60.0 * dt)

        if self.overview:
            b = self.world.config.building
            target = np.array([b.size_x / 2, 0.0, b.size_z / 2 - 1.0])
        else:
            target = self.world.drone.pos.copy()
        az, pt = math.radians(self.azimuth), math.radians(self.pitch)
        # camera sits opposite to the viewing direction
        direction = np.array([-math.sin(az) * math.cos(pt), math.sin(pt), -math.cos(az) * math.cos(pt)])
        dist = self.dist
        if not self.overview:                      # shorten the boom if a wall is in the way
            obs = self.world.obstacles
            if obs.segment_blocked(target, target + direction * dist):
                lo, hi = 0.0, dist
                for _ in range(7):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if not obs.segment_blocked(target, target + direction * mid) else (lo, mid)
                dist = max(0.7, lo - 0.15)
        want = target + direction * dist
        want[1] = max(want[1], 0.25)
        self._pos = want if self._pos is None else self._pos + (want - self._pos) * min(1.0, dt * 10.0)
        pos = self._pos.copy()
        if shake > 0:
            pos += np.array([math.sin(shake * 97), math.sin(shake * 131), math.sin(shake * 71)]) * 0.05 * min(shake, 1)
        camera.position = Vec3(*pos)
        camera.look_at(Vec3(*(target + np.array([0, 0.1, 0]))))
