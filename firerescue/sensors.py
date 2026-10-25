"""SYNTHETIC thermal sensor model.

*** This is NOT a physical thermal-camera simulation. ***
It is a simple analytic signal model used to produce plausible-looking detections:

    victim signal  = base * pose_factor / (1 + (d / d_ref)^2)        (0 beyond max_range)
    interference   = gain * sum_over_fires( intensity / (1 + (d_f / radius_f)^2) )
    score          = signal - interference (+ optional Gaussian noise)
    detected       = in range  AND  line of sight is clear  AND  score >= threshold

Fires therefore mask nearby victims, as strong background heat would in a real sensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .collision import ObstacleSet
from .config import SensorConfig
from .fire import FireSource
from .victims import Victim

SENSOR_LABEL = "SYNTHETIC THERMAL MODEL (not a real thermal camera)"


@dataclass
class VictimReading:
    victim_id: str
    distance: float
    line_of_sight: bool
    signal: float
    interference: float
    score: float
    detected: bool


@dataclass
class ThermalReading:
    victims: List[VictimReading] = field(default_factory=list)
    interference: float = 0.0
    peak_signal: float = 0.0
    peak_victim_id: Optional[str] = None
    visible_fire_ids: List[str] = field(default_factory=list)

    @property
    def detected_ids(self) -> List[str]:
        return [r.victim_id for r in self.victims if r.detected]


class SyntheticThermalSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)

    def victim_signal(self, victim: Victim, distance: float) -> float:
        c = self.cfg
        if distance > c.max_range:
            return 0.0
        pose = c.prone_factor if victim.pose == "prone" else 1.0
        return c.victim_base_signal * pose / (1.0 + (distance / c.reference_distance) ** 2)

    def interference_at(self, pos, fires: Sequence[FireSource]) -> float:
        return self.cfg.fire_interference_gain * sum(f.heat_at(pos) for f in fires)

    def read(self, pos, victims: Sequence[Victim], fires: Sequence[FireSource],
             obstacles: ObstacleSet) -> ThermalReading:
        pos = np.asarray(pos, float)
        c = self.cfg
        interf = self.interference_at(pos, fires)
        reading = ThermalReading(interference=interf)
        for v in victims:
            tgt = v.sensor_point
            d = float(np.linalg.norm(tgt - pos))
            los = d <= c.max_range and not obstacles.segment_blocked(pos, tgt)
            sig = self.victim_signal(v, d) if los else 0.0
            noise = float(self.rng.normal(0.0, c.noise_std)) if c.noise_std > 0 and los else 0.0
            score = sig - interf + noise
            detected = bool(los and score >= c.detection_threshold)
            reading.victims.append(VictimReading(v.id, d, los, sig, interf, score, detected))
            if sig > reading.peak_signal:
                reading.peak_signal, reading.peak_victim_id = sig, v.id
        reading.visible_fire_ids = [
            f.id for f in fires
            if np.linalg.norm(f.position - pos) <= c.max_range * 2 and not obstacles.segment_blocked(pos, f.position)]
        return reading

    def heat_grid(self, pos, reading: ThermalReading, victims: Sequence[Victim],
                  fires: Sequence[FireSource], obstacles: ObstacleSet, n: int = 32,
                  yaw_deg: float = 0.0) -> np.ndarray:
        """Body-aligned heat map (n x n, values 0..1+) centred on the drone.

        Indexing: grid[row, col] where col -> drone right and row -> drone
        forward (yaw 0 faces +z, yaw 90 faces +x).  Only heat sources with an
        unobstructed line of sight to the drone contribute.
        """
        ext = self.cfg.max_range
        axis = (np.arange(n) + 0.5) / n * 2 * ext - ext
        # sample in body axes, then rotate back to world axes to evaluate sources
        col_b, row_b = np.meshgrid(axis, axis)            # col_b = right, row_b = forward
        a = math.radians(yaw_deg)
        ca, sa = math.cos(a), math.sin(a)
        gx = pos[0] + ca * col_b - sa * row_b
        gz = pos[2] + sa * col_b + ca * row_b
        grid = np.zeros((n, n))
        vis = {r.victim_id for r in reading.victims if r.line_of_sight}
        for v in victims:
            if v.id in vis:
                pose = self.cfg.prone_factor if v.pose == "prone" else 1.0
                d2 = (gx - v.position[0]) ** 2 + (gz - v.position[2]) ** 2
                grid += self.cfg.victim_base_signal * pose * np.exp(-d2 / (2 * 0.45 ** 2))
        for f in fires:
            if f.id in reading.visible_fire_ids:
                d2 = (gx - f.position[0]) ** 2 + (gz - f.position[2]) ** 2
                grid += 0.7 * f.intensity * np.exp(-d2 / (2 * (f.danger_radius * 0.6) ** 2))
        return grid
