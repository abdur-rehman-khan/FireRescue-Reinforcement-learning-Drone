"""Vectorised sphere-vs-AABB collision and segment line-of-sight queries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .obstacles import Obstacle


@dataclass
class Contact:
    index: int
    name: str
    normal: np.ndarray      # unit vector pointing from the obstacle toward the sphere
    depth: float


class ObstacleSet:
    def __init__(self, obstacles: Sequence[Obstacle]):
        self.obstacles: List[Obstacle] = list(obstacles)
        if self.obstacles:
            self.mins = np.array([o.lo for o in self.obstacles], dtype=float)
            self.maxs = np.array([o.hi for o in self.obstacles], dtype=float)
        else:
            self.mins = np.zeros((0, 3))
            self.maxs = np.zeros((0, 3))

    def __len__(self) -> int:
        return len(self.obstacles)

    # ------------------------------------------------------------------ sphere
    def sphere_contacts(self, pos: np.ndarray, radius: float) -> List[Contact]:
        if not len(self):
            return []
        closest = np.clip(pos, self.mins, self.maxs)
        delta = pos - closest
        dist2 = np.einsum("ij,ij->i", delta, delta)
        contacts: List[Contact] = []
        for i in np.nonzero(dist2 < radius * radius)[0]:
            dist = float(np.sqrt(dist2[i]))
            if dist > 1e-9:
                contacts.append(Contact(int(i), self.obstacles[i].name, delta[i] / dist, radius - dist))
            else:   # centre inside the box: push out through the nearest face
                to_lo, to_hi = pos - self.mins[i], self.maxs[i] - pos
                cand = np.concatenate([to_lo, to_hi])
                j = int(np.argmin(cand))
                normal = np.zeros(3)
                normal[j % 3] = -1.0 if j < 3 else 1.0
                contacts.append(Contact(int(i), self.obstacles[i].name, normal, float(cand[j]) + radius))
        return contacts

    def sphere_collides(self, pos, radius: float) -> bool:
        return bool(self.sphere_contacts(np.asarray(pos, dtype=float), radius))

    def resolve_sphere(self, pos: np.ndarray, vel: np.ndarray, radius: float, max_iter: int = 8
                       ) -> Tuple[np.ndarray, np.ndarray, List[Contact], float]:
        """Push the sphere out of every obstacle; remove velocity into the surface (sliding).

        Returns (pos, vel, contacts, impact_speed).
        """
        pos, vel = pos.copy(), vel.copy()
        hits: List[Contact] = []
        impact = 0.0
        for _ in range(max_iter):
            contacts = self.sphere_contacts(pos, radius)
            if not contacts:
                break
            c = max(contacts, key=lambda k: k.depth)
            pos += c.normal * (c.depth + 1e-6)
            vn = float(np.dot(vel, c.normal))
            if vn < 0:
                impact = max(impact, -vn)
                vel -= vn * c.normal
            hits.append(c)
        return pos, vel, hits, impact

    def point_inside(self, p, margin: float = 0.0) -> bool:
        if not len(self):
            return False
        p = np.asarray(p, float)
        return bool(np.any(np.all((p > self.mins - margin) & (p < self.maxs + margin), axis=1)))

    # ----------------------------------------------------------------- segment
    def segment_blocked(self, p0, p1) -> bool:
        """True if the segment p0->p1 intersects any obstacle (slab method)."""
        if not len(self):
            return False
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        parallel = np.abs(d) < 1e-12
        safe = np.where(parallel, 1.0, d)
        t1 = (self.mins - p0) / safe
        t2 = (self.maxs - p0) / safe
        tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
        inside = (p0 >= self.mins) & (p0 <= self.maxs)
        tlo = np.where(parallel, np.where(inside, -np.inf, np.inf), tlo)
        thi = np.where(parallel, np.where(inside, np.inf, -np.inf), thi)
        tnear, tfar = tlo.max(axis=1), thi.min(axis=1)
        return bool(np.any((tnear <= tfar) & (tfar >= 0.0) & (tnear <= 1.0)))
