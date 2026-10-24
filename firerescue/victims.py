"""Simulated human victims with unique IDs and ground-truth poses."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .config import VictimConfig

CHEST_HEIGHT = {"standing": 1.1, "prone": 0.2}


class Victim:
    def __init__(self, cfg: VictimConfig):
        self.id = cfg.id
        self.position = np.array(cfg.position, dtype=float)    # feet / ground contact point
        self.pose = cfg.pose                                    # ground truth
        self.discovered = False
        self.discovered_at: Optional[float] = None

    @property
    def sensor_point(self) -> np.ndarray:
        """Torso point used for range and line-of-sight tests."""
        p = self.position.copy()
        p[1] += CHEST_HEIGHT[self.pose]
        return p

    def reset(self) -> None:
        self.discovered = False
        self.discovered_at = None


def nearest_undiscovered(victims: Sequence[Victim], p) -> Tuple[Optional[Victim], float]:
    best, best_d = None, float("inf")
    for v in victims:
        if v.discovered:
            continue
        d = float(np.linalg.norm(v.sensor_point - np.asarray(p, float)))
        if d < best_d:
            best, best_d = v, d
    return best, best_d
