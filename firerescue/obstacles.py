"""Axis-aligned solid obstacles generated from the building configuration.

Everything solid in the world (wall segments, door lintels, open door leaves,
furniture parts, debris) is an axis-aligned box so collision is exact and cheap.
Windows are visual glass panes set into solid exterior walls (not flyable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import BuildingConfig, FurnitureConfig, Color, Vec3

PALETTE = {
    "wall": (0.84, 0.82, 0.76),
    "lintel": (0.45, 0.34, 0.26),
    "door": (0.52, 0.33, 0.18),
    "table": (0.56, 0.36, 0.20),
    "chair": (0.62, 0.42, 0.26),
    "box": (0.72, 0.56, 0.34),
    "beam": (0.28, 0.22, 0.18),
    "rubble": (0.46, 0.46, 0.48),
}


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str
    center: Vec3
    size: Vec3
    color: Color

    @property
    def lo(self) -> np.ndarray:
        return np.array(self.center) - np.array(self.size) / 2.0

    @property
    def hi(self) -> np.ndarray:
        return np.array(self.center) + np.array(self.size) / 2.0


def _wall_box(axis: str, fixed: float, a: float, b: float, y0: float, y1: float, t: float
              ) -> Tuple[Vec3, Vec3]:
    h, cy, mid, length = y1 - y0, (y0 + y1) / 2, (a + b) / 2, b - a
    if axis == "x":
        return (mid, cy, fixed), (length, h, t)
    return (fixed, cy, mid), (t, h, length)


def build_wall_obstacles(b: BuildingConfig) -> List[Obstacle]:
    out: List[Obstacle] = []
    t, H = b.wall_thickness, b.height
    for wi, w in enumerate(b.walls):
        cursor = w.start
        segs = 0

        def add(kind, a, bb, y0, y1, suffix):
            c, s = _wall_box(w.axis, w.fixed, a, bb, y0, y1, t)
            out.append(Obstacle(f"wall{wi}_{suffix}", kind, c, s, PALETTE[kind]))

        for gi, (centre, width) in enumerate(sorted(w.gaps)):
            lo, hi = centre - width / 2, centre + width / 2
            if lo > cursor + 1e-9:
                add("wall", cursor, lo, 0.0, H, f"seg{segs}")
                segs += 1
            add("lintel", lo, hi, b.door_height, H, f"lintel{gi}")
            if not w.exterior:     # open door leaf hinged at the low edge, swung 90 degrees
                L, th = b.door_leaf_length, 0.05
                off = w.fixed + w.leaf_dir * (t / 2 + L / 2)
                if w.axis == "x":
                    c, s = (lo + th / 2, b.door_height / 2, off), (th, b.door_height, L)
                else:
                    c, s = (off, b.door_height / 2, lo + th / 2), (L, b.door_height, th)
                out.append(Obstacle(f"wall{wi}_door{gi}", "door", c, s, PALETTE["door"]))
            cursor = hi
        if cursor < w.end - 1e-9:
            add("wall", cursor, w.end, 0.0, H, f"seg{segs}")
    return out


def _rot90(dx: float, dz: float, k: int) -> Tuple[float, float]:
    for _ in range(k % 4):
        dx, dz = dz, -dx          # same handedness as drone yaw (yaw 90 maps +z to +x)
    return dx, dz


def _furniture_parts(kind: str, size):
    """Return [(suffix, (dx, dy, dz), (sx, sy, sz))] in local, unrotated coordinates.

    dy is the part's centre height above the piece's base elevation.
    """
    if kind == "table":
        sx, sy, sz = size or (1.4, 0.75, 0.8)
        parts = [("top", (0, sy - 0.03, 0), (sx, 0.06, sz))]
        for i, (a, c) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            parts.append((f"leg{i}", (a * (sx / 2 - 0.05), (sy - 0.06) / 2, c * (sz / 2 - 0.05)),
                          (0.06, sy - 0.06, 0.06)))
        return parts
    if kind == "chair":
        sx, sy, sz = size or (0.45, 0.9, 0.45)
        seat = 0.45
        parts = [("seat", (0, seat - 0.03, 0), (sx, 0.06, sz)),
                 ("back", (0, seat + (sy - seat) / 2, -(sz / 2 - 0.025)), (sx, sy - seat, 0.05))]
        for i, (a, c) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            parts.append((f"leg{i}", (a * (sx / 2 - 0.03), (seat - 0.06) / 2, c * (sz / 2 - 0.03)),
                          (0.05, seat - 0.06, 0.05)))
        return parts
    default = {"box": (0.6, 0.6, 0.6), "beam": (2.0, 0.25, 0.25), "rubble": (0.9, 0.5, 0.7)}[kind]
    sx, sy, sz = size or default
    return [("body", (0, sy / 2, 0), (sx, sy, sz))]


def build_furniture_obstacles(items: List[FurnitureConfig]) -> List[Obstacle]:
    out: List[Obstacle] = []
    for fi, f in enumerate(items):
        k = int(round(f.rotation_deg / 90.0)) % 4
        label = f.name or f"{f.kind}{fi}"
        for suffix, (dx, dy, dz), (sx, sy, sz) in _furniture_parts(f.kind, f.size):
            dx, dz = _rot90(dx, dz, k)
            if k % 2:
                sx, sz = sz, sx
            c = (f.position[0] + dx, f.position[1] + dy, f.position[2] + dz)
            out.append(Obstacle(f"{label}_{suffix}", f.kind, c, (sx, sy, sz), PALETTE[f.kind]))
    return out


def build_obstacles(b: BuildingConfig) -> List[Obstacle]:
    return build_wall_obstacles(b) + build_furniture_obstacles(b.furniture)
