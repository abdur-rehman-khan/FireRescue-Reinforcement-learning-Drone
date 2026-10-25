"""2D occupancy-grid A* planner at a fixed flight altitude (used by the baseline controller)."""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..collision import ObstacleSet
from ..fire import FireSource

XZ = Tuple[float, float]
_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


class GridPlanner:
    def __init__(self, obstacles: ObstacleSet, size_x: float, size_z: float, altitude: float,
                 clearance: float, resolution: float = 0.2):
        self.res, self.altitude = resolution, altitude
        self.nx, self.nz = int(round(size_x / resolution)), int(round(size_z / resolution))
        xs = (np.arange(self.nx) + 0.5) * resolution
        zs = (np.arange(self.nz) + 0.5) * resolution
        self.gx, self.gz = np.meshgrid(xs, zs, indexing="ij")
        pts = np.stack([self.gx.ravel(), np.full(self.gx.size, altitude), self.gz.ravel()], axis=1)
        blocked = np.zeros(len(pts), dtype=bool)
        for s in range(0, len(pts), 1500):
            p = pts[s:s + 1500][:, None, :]
            closest = np.clip(p, obstacles.mins[None], obstacles.maxs[None])
            blocked[s:s + 1500] = (((p - closest) ** 2).sum(-1) < clearance ** 2).any(axis=1)
        blocked = blocked.reshape(self.nx, self.nz)
        blocked |= (self.gx < clearance) | (self.gx > size_x - clearance)
        blocked |= (self.gz < clearance) | (self.gz > size_z - clearance)
        self.static_blocked = blocked
        self.fire_blocked = np.zeros_like(blocked)
        self.fire_cost = np.zeros(blocked.shape)

    def set_fire_zones(self, fires: Sequence[FireSource], hard_margin: float = 0.4,
                       soft_margin: float = 1.2) -> None:
        self.fire_blocked[:] = False
        self.fire_cost[:] = 0.0
        for f in fires:
            d = np.sqrt((self.gx - f.position[0]) ** 2 + (self.altitude - f.position[1]) ** 2
                        + (self.gz - f.position[2]) ** 2)
            self.fire_blocked |= d < f.danger_radius + hard_margin
            self.fire_cost += np.clip((f.danger_radius + hard_margin + soft_margin - d) / soft_margin, 0, 1) * 4.0

    @property
    def blocked(self) -> np.ndarray:
        return self.static_blocked | self.fire_blocked

    def cell(self, p: XZ) -> Tuple[int, int]:
        return (min(self.nx - 1, max(0, int(p[0] / self.res))), min(self.nz - 1, max(0, int(p[1] / self.res))))

    def centre(self, c: Tuple[int, int]) -> XZ:
        return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)

    def nearest_free(self, c: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        blocked = self.blocked
        if not blocked[c]:
            return c
        free = np.argwhere(~blocked)
        if not len(free):
            return None
        i = int(np.argmin(((free - np.array(c)) ** 2).sum(1)))
        return int(free[i][0]), int(free[i][1])

    def line_free(self, a: XZ, b: XZ) -> bool:
        blocked = self.blocked
        n = max(2, int(math.dist(a, b) / (self.res / 2)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            if blocked[self.cell((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))]:
                return False
        return True

    def plan(self, start: XZ, goal: XZ) -> Optional[Tuple[List[XZ], float]]:
        """Return (waypoints excluding the start, path cost) or None if unreachable."""
        blocked = self.blocked
        s, g = self.nearest_free(self.cell(start)), self.nearest_free(self.cell(goal))
        if s is None or g is None:
            return None
        cost_map = self.fire_cost
        best = {s: 0.0}
        parent = {}
        heap = [(0.0, 0.0, s)]
        closed = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == g:
                break
            closed.add(cur)
            for dx, dz, step in _NEIGHBOURS:
                nx_, nz_ = cur[0] + dx, cur[1] + dz
                if not (0 <= nx_ < self.nx and 0 <= nz_ < self.nz) or blocked[nx_, nz_]:
                    continue
                if dx and dz and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dz]):
                    continue    # no diagonal corner cutting
                nxt = (nx_, nz_)
                ng = gc + step + cost_map[nx_, nz_] * 0.2
                if ng < best.get(nxt, float("inf")):
                    best[nxt] = ng
                    parent[nxt] = cur
                    h = math.hypot(g[0] - nx_, g[1] - nz_)
                    heapq.heappush(heap, (ng + h, ng, nxt))
        if g not in best:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = [self.centre(c) for c in cells]
        # greedy line-of-sight smoothing (string pulling)
        out: List[XZ] = []
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out, best[g] * self.res


def make_viewpoints(planner: "GridPlanner", rooms: Sequence, spacing: float) -> List[XZ]:
    """Generate one viewpoint per room cell grid (free cells only), at ``spacing`` metres."""
    viewpoints: List[XZ] = []
    step = max(spacing, planner.res)
    for room in rooms:
        x0, z0 = float(room.min_xz[0]), float(room.min_xz[1])
        x1, z1 = float(room.max_xz[0]), float(room.max_xz[1])
        xs = np.arange(x0 + step / 2, x1, step)
        zs = np.arange(z0 + step / 2, z1, step)
        for x in xs:
            for z in zs:
                c = planner.cell((float(x), float(z)))
                if not planner.blocked[c]:
                    viewpoints.append((float(x), float(z)))
    return viewpoints


# ------------------------------------------------------------------ distance fields
def _neighbour_offsets() -> List[Tuple[int, int, float]]:
    return [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def distance_field(self: "GridPlanner", target: XZ, clearance: float = 0.1) -> np.ndarray:
    """Shortest-path distance (m) from every free cell to ``target`` (inf if unreachable).

    Uses Dijkstra flood-fill over the planner's 8-neighbour grid.  ``clearance`` adds a small
    per-cell soft cost so paths hug walls slightly less.
    """
    blocked = self.blocked
    tc = self.cell(target)
    if blocked[tc]:
        return np.full((self.nx, self.nz), np.inf)
    dist = np.full((self.nx, self.nz), np.inf)
    dist[tc] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, tc[0], tc[1])]
    visited = np.zeros((self.nx, self.nz), dtype=bool)
    while heap:
        d, cx, cz = heapq.heappop(heap)
        if visited[cx, cz]:
            continue
        visited[cx, cz] = True
        for dx, dz, step in _neighbour_offsets():
            nx_, nz_ = cx + dx, cz + dz
            if not (0 <= nx_ < self.nx and 0 <= nz_ < self.nz) or blocked[nx_, nz_]:
                continue
            if dx and dz and (blocked[cx + dx, cz] or blocked[cx, cz + dz]):
                continue
            nd = d + step * self.res + self.fire_cost[nx_, nz_] * 0.2 * self.res + clearance * 0.05
            if nd < dist[nx_, nz_]:
                dist[nx_, nz_] = nd
                heapq.heappush(heap, (nd, nx_, nz_))
    return dist


GridPlanner.distance_field = distance_field
