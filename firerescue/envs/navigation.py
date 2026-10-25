"""Navigation goal generator: sticky coverage viewpoints + shortest-path distance fields.

The *goal* is a global-planner hint (like the output of a mission planner): the policy is
told where to go next (and the local path direction); it must learn the local flight control
(collision and fire avoidance, smooth motion).  See docs/RL_INTERFACE.md.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..controllers.planner import GridPlanner, make_viewpoints

XZ = Tuple[float, float]
FIELD_CLEARANCE = 0.1


class GoalManager:
    def __init__(self, world, cfg, coverage):
        self.world, self.cfg, self.coverage = world, cfg, coverage
        b = world.config.building
        self.planner = GridPlanner(world.obstacles, b.size_x, b.size_z, cfg.goal_altitude,
                                   world.config.drone.radius + cfg.planner_clearance_margin)
        self.planner.set_fire_zones(world.fires)
        self.viewpoints: List[XZ] = make_viewpoints(self.planner, b.rooms, cfg.viewpoint_spacing)
        # hard clearance well below the drone radius: every cell the drone can physically occupy has a
        # value (collision itself keeps it >= radius from walls); the soft wall cost centres the paths
        self.fields = [self.planner.distance_field(v, clearance=FIELD_CLEARANCE) for v in self.viewpoints]
        self.reset()

    def reset(self) -> None:
        self.done = [False] * len(self.viewpoints)
        self.current: Optional[int] = None
        self.sweeps = 0
        self._prev: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    def free_start_cells(self) -> np.ndarray:
        """(x, z) of cells where a random start is safe (free, outside fire zones)."""
        free = np.argwhere(~self.planner.blocked)
        return (free + 0.5) * self.planner.res

    def _value(self, field, xz) -> float:
        return float(field[self.planner.cell(xz)])

    def distance(self) -> float:
        """Shortest-path distance (m) from the drone to the current target (inf if none)."""
        if self.current is None:
            return float("inf")
        p = self.world.drone.pos
        return self._value(self.fields[self.current], (p[0], p[2]))

    @property
    def target(self) -> Optional[XZ]:
        return None if self.current is None else self.viewpoints[self.current]

    def lookahead_point(self) -> Optional[XZ]:
        """Point ``goal_lookahead`` metres ahead on the shortest path to the target."""
        if self.current is None:
            return None
        f, pl = self.fields[self.current], self.planner
        p = self.world.drone.pos
        c = pl.cell((p[0], p[2]))
        travelled = 0.0
        while travelled < self.cfg.goal_lookahead:
            best, bv = c, f[c]
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dz)
                    if (dx or dz) and 0 <= n[0] < pl.nx and 0 <= n[1] < pl.nz and f[n] < bv:
                        best, bv = n, f[n]
            if best == c:
                break
            travelled += pl.res * (1.414 if best[0] != c[0] and best[1] != c[1] else 1.0)
            c = best
        return pl.centre(c)

    # ------------------------------------------------------------------- update
    def update(self) -> float:
        """Advance goal selection; returns signed path-distance PROGRESS (m) this step.

        Positive = got closer to the (unchanged) goal.  Switching goals yields 0, so progress
        cannot be farmed by switching, and oscillating around yields ~0 net.
        """
        p = self.world.drone.pos
        xz = (float(p[0]), float(p[2]))
        first_sweep = self.sweeps == 0
        for i, v in enumerate(self.viewpoints):
            if not self.done[i]:
                reached = np.hypot(v[0] - xz[0], v[1] - xz[1]) <= self.cfg.goal_reach_radius
                if reached or (first_sweep and self.coverage.cell_covered_at(v)):
                    self.done[i] = True
        switched = False
        if self.current is None or self.done[self.current]:
            active = [i for i, d in enumerate(self.done) if not d and np.isfinite(self._value(self.fields[i], xz))]
            if not active and self.viewpoints:
                self.done, self.sweeps = [False] * len(self.viewpoints), self.sweeps + 1
                active = [i for i in range(len(self.viewpoints)) if np.isfinite(self._value(self.fields[i], xz))]
            self.current = min(active, key=lambda i: self._value(self.fields[i], xz)) if active else None
            switched = True
        dist = self.distance()
        if switched or self._prev is None or not np.isfinite(dist) or not np.isfinite(self._prev):
            progress = 0.0
        else:
            progress = self._prev - dist
        self._prev = dist if np.isfinite(dist) else None
        return float(progress)
