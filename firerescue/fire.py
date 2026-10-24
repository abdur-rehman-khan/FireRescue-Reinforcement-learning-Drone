"""Fire sources: configurable positions with danger radii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import FireConfig

CLEAR, WARNING, DANGER = "CLEAR", "WARNING", "DANGER"
_ORDER = {CLEAR: 0, WARNING: 1, DANGER: 2}


@dataclass(frozen=True)
class FireProximity:
    level: str
    fire_id: Optional[str]
    distance: float          # to the nearest fire source (inf if none)


class FireSource:
    def __init__(self, cfg: FireConfig):
        self.id = cfg.id
        self.position = np.array(cfg.position, dtype=float)
        self.danger_radius = float(cfg.danger_radius)
        self.intensity = float(cfg.intensity)

    def distance_to(self, p) -> float:
        return float(np.linalg.norm(np.asarray(p, float) - self.position))

    def heat_at(self, p) -> float:
        """Smooth heat falloff used as thermal interference (peaks at the source)."""
        d = self.distance_to(p)
        return self.intensity / (1.0 + (d / self.danger_radius) ** 2)

    def level(self, p, warning_factor: float = 1.5) -> str:
        d = self.distance_to(p)
        if d < self.danger_radius:
            return DANGER
        if d < self.danger_radius * warning_factor:
            return WARNING
        return CLEAR


def proximity(fires: Sequence[FireSource], p, warning_factor: float = 1.5) -> FireProximity:
    """Worst proximity level over all fires (DANGER > WARNING > CLEAR); nearest fire reported."""
    if not fires:
        return FireProximity(CLEAR, None, float("inf"))
    nearest = min(fires, key=lambda f: f.distance_to(p))
    level = max((f.level(p, warning_factor) for f in fires), key=_ORDER.get)
    return FireProximity(level, nearest.id, nearest.distance_to(p))
