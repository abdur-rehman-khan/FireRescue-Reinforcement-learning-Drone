"""Room exploration / coverage tracking on a coarse grid (no engine objects)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

EYE_HEIGHT = 0.6      # height of the floor sample point used for line-of-sight


class CoverageTracker:
    """A cell is *explored* once the drone has been within ``radius`` of it with a clear
    line of sight.  Cells inside walls / door leaves are not walkable and are ignored.
    A room is *discovered* once ``room_fraction`` of its cells are explored."""

    def __init__(self, world, cell: float = 1.0, radius: float = 2.5, room_fraction: float = 0.3,
                 altitude: float = 1.2):
        b = world.config.building
        self.world, self.cell, self.radius, self.room_fraction = world, cell, radius, room_fraction
        self.rooms = list(b.rooms)
        self.nx, self.nz = int(np.ceil(b.size_x / cell)), int(np.ceil(b.size_z / cell))
        xs, zs = (np.arange(self.nx) + 0.5) * cell, (np.arange(self.nz) + 0.5) * cell
        self.cx, self.cz = np.meshgrid(xs, zs, indexing="ij")
        walkable = np.array([[not world.obstacles.point_inside(np.array([x, altitude, z]))
                              for z in zs] for x in xs])
        room_idx = np.full((self.nx, self.nz), -1, dtype=int)
        for r, room in enumerate(self.rooms):
            (x0, z0), (x1, z1) = room.min_xz, room.max_xz
            m = (self.cx >= x0) & (self.cx < x1) & (self.cz >= z0) & (self.cz < z1) & walkable
            room_idx[m] = r
        self.room_idx = room_idx
        self.valid = room_idx >= 0
        self.room_total = np.array([(room_idx == r).sum() for r in range(len(self.rooms))], dtype=float)
        self.reset()

    def reset(self) -> None:
        self.covered = np.zeros((self.nx, self.nz), dtype=bool)
        self.rooms_found = set()

    # ------------------------------------------------------------------ queries
    @property
    def n_rooms(self) -> int:
        return len(self.rooms)

    @property
    def total_cells(self) -> int:
        return int(self.valid.sum())

    def global_fraction(self) -> float:
        return float((self.covered & self.valid).sum()) / max(1, self.total_cells)

    def room_fractions(self) -> np.ndarray:
        out = np.zeros(self.n_rooms)
        for r in range(self.n_rooms):
            if self.room_total[r]:
                out[r] = (self.covered & (self.room_idx == r)).sum() / self.room_total[r]
        return out

    def cell_covered_at(self, xz: Tuple[float, float]) -> bool:
        i, j = min(self.nx - 1, int(xz[0] / self.cell)), min(self.nz - 1, int(xz[1] / self.cell))
        return bool(self.covered[i, j]) or not self.valid[i, j]

    # ------------------------------------------------------------------- update
    def update(self, pos) -> Tuple[int, List[int]]:
        """Mark newly explored cells; returns (n_new_cells, newly discovered room indices)."""
        pos = np.asarray(pos, float)
        near = (~self.covered) & self.valid & \
               ((self.cx - pos[0]) ** 2 + (self.cz - pos[2]) ** 2 <= self.radius ** 2)
        new = 0
        for i, j in np.argwhere(near):
            target = np.array([self.cx[i, j], EYE_HEIGHT, self.cz[i, j]])
            if not self.world.obstacles.segment_blocked(pos, target):
                self.covered[i, j] = True
                new += 1
        new_rooms: List[int] = []
        if new:
            fr = self.room_fractions()
            for r in range(self.n_rooms):
                if r not in self.rooms_found and self.room_total[r] and fr[r] >= self.room_fraction:
                    self.rooms_found.add(r)
                    new_rooms.append(r)
        return new, new_rooms
