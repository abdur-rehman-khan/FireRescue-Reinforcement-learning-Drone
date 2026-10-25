"""Engine-free top-down RGB renderer for RescueDroneGym.render('rgb_array')."""
from __future__ import annotations

import math

import numpy as np

PX = 20  # pixels per metre


def render_topdown(world, coverage, trajectory, goal_xz=None, px: int = PX) -> np.ndarray:
    b = world.config.building
    W, H = int(b.size_x * px), int(b.size_z * px)
    img = np.full((H, W, 3), 40, dtype=np.uint8)

    def rect(x0, z0, x1, z1, col):
        i0, i1 = max(0, int(z0 * px)), min(H, int(math.ceil(z1 * px)))
        j0, j1 = max(0, int(x0 * px)), min(W, int(math.ceil(x1 * px)))
        img[i0:i1, j0:j1] = col

    def disc(x, z, r, col, alpha=1.0):
        zz, xx = np.ogrid[:H, :W]
        m = (xx / px - x) ** 2 + (zz / px - z) ** 2 <= r * r
        img[m] = (img[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)

    for r in b.rooms:
        rect(r.min_xz[0], r.min_xz[1], r.max_xz[0], r.max_xz[1], tuple(int(255 * c * 0.8) for c in r.floor_color))
    for i in range(coverage.nx):                      # unexplored cells are darkened
        for j in range(coverage.nz):
            if coverage.valid[i, j] and not coverage.covered[i, j]:
                x0, z0 = i * coverage.cell, j * coverage.cell
                sub = img[int(z0 * px):int((z0 + coverage.cell) * px), int(x0 * px):int((x0 + coverage.cell) * px)]
                sub[:] = (sub * 0.55).astype(np.uint8)
    for o in world.obstacle_list:
        lo, hi = o.lo, o.hi
        col = (200, 200, 195) if o.kind in ("wall", "lintel") else (150, 100, 60) if o.kind != "door" else (120, 80, 40)
        if o.kind == "beam" and lo[1] > 1.0:
            col = (90, 70, 60)
        rect(lo[0], lo[2], hi[0], hi[2], col)
    for f in world.fires:
        disc(f.position[0], f.position[2], f.danger_radius, (230, 50, 30), 0.35)
        disc(f.position[0], f.position[2], 0.35, (255, 140, 0))
    for v in world.victims:
        disc(v.position[0], v.position[2], 0.3, (60, 255, 90) if v.discovered else (255, 230, 40))
    for p in trajectory:
        disc(p[0], p[2], 0.06, (60, 200, 255))
    if goal_xz is not None:
        disc(goal_xz[0], goal_xz[1], 0.25, (0, 255, 255), 0.8)
    d = world.drone
    disc(d.pos[0], d.pos[2], 0.28, (30, 90, 255))
    fwd = d.forward_vector()
    for k in np.linspace(0, 0.6, 12):
        disc(d.pos[0] + fwd[0] * k, d.pos[2] + fwd[2] * k, 0.05, (255, 255, 255))
    return np.flipud(img).copy()                      # +z up
