"""Configurable reward calculation (engine-independent, unit-testable).

One-time rewards (rooms, victims, coverage cells) are granted by the *environment* only for
newly set IDs, so they cannot be re-earned within an episode; this module just prices events.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .rl_config import RewardConfig


@dataclass
class StepSignals:
    position: np.ndarray
    action: np.ndarray
    new_rooms: int = 0
    new_victims: int = 0
    new_cells: int = 0
    progress: float = 0.0               # signed metres of shortest-path progress
    new_collisions: int = 0
    in_contact: bool = False
    fire_level: str = "CLEAR"           # CLEAR | WARNING | DANGER
    success: bool = False
    fire_failure: bool = False
    collision_limit: bool = False
    truncated: bool = False
    episode_fraction_left: float = 0.0  # 1 at the start of the episode, 0 at the limit


class RewardCalculator:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._pos: Deque[np.ndarray] = deque(maxlen=self.cfg.dither_window + 1)
        self._prev_action: Optional[np.ndarray] = None
        self._idle = 0

    def compute(self, s: StepSignals) -> Tuple[float, Dict[str, float]]:
        c = self.cfg
        comp: Dict[str, float] = {"time": c.time_step}
        comp["new_room"] = c.new_room * s.new_rooms
        comp["new_victim"] = c.new_victim * s.new_victims
        comp["coverage"] = c.new_coverage_cell * s.new_cells
        comp["progress"] = c.progress * float(np.clip(s.progress, -c.max_progress_step, c.max_progress_step))
        comp["collision"] = c.collision * s.new_collisions + (c.collision_contact if s.in_contact else 0.0)
        comp["fire"] = {"WARNING": c.fire_warning, "DANGER": c.fire_danger}.get(s.fire_level, 0.0)

        # abrupt action reversals (oscillation)
        osc = 0.0
        a, pa = np.asarray(s.action, float), self._prev_action
        if pa is not None:
            na, npa = np.linalg.norm(a), np.linalg.norm(pa)
            if na > c.oscillation_min_action and npa > c.oscillation_min_action:
                if float(np.dot(a, pa) / (na * npa)) < c.oscillation_cos:
                    osc = c.oscillation
        comp["oscillation"] = osc
        self._prev_action = a.copy()

        # repeated unproductive movement: lots of path, little displacement
        pos = np.asarray(s.position, float)
        self._pos.append(pos.copy())
        dither = 0.0
        if len(self._pos) == self._pos.maxlen:
            pts = np.array(self._pos)
            path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            net = float(np.linalg.norm(pts[-1] - pts[0]))
            if path >= c.dither_min_path and net <= c.dither_max_net:
                dither = c.dither
        comp["dither"] = dither

        # stagnation: nothing productive for a while
        productive = s.new_cells > 0 or s.new_rooms > 0 or s.new_victims > 0 or s.progress > c.stagnation_progress_eps
        self._idle = 0 if productive else self._idle + 1
        comp["stagnation"] = c.stagnation if self._idle > c.stagnation_steps else 0.0

        # terminal terms
        comp["success"] = (c.mission_complete + c.success_time_bonus * max(0.0, s.episode_fraction_left)) if s.success else 0.0
        comp["fire_failure"] = c.fire_failure if s.fire_failure else 0.0
        comp["collision_limit"] = c.collision_limit_failure if s.collision_limit else 0.0
        comp["timeout"] = c.timeout if (s.truncated and not s.success) else 0.0
        return float(sum(comp.values())), comp
