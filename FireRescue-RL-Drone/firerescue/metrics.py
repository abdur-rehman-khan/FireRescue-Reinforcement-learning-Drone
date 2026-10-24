"""Lightweight mission metrics recorder (plotting lives in scripts/plot_metrics.py)."""
from __future__ import annotations

from typing import Dict, List


class MetricsRecorder:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self._next = 0.0
        self.series: Dict[str, List[float]] = {k: [] for k in (
            "t", "x", "y", "z", "speed", "discovered", "nearest_victim", "nearest_fire", "collisions")}

    def record(self, world) -> None:
        if world.time + 1e-9 < self._next:
            return
        self._next = world.time + self.interval
        p, s = world.drone.pos, self.series
        s["t"].append(world.time)
        s["x"].append(float(p[0])); s["y"].append(float(p[1])); s["z"].append(float(p[2]))
        s["speed"].append(world.drone.speed)
        s["discovered"].append(world.discovered_count)
        s["nearest_victim"].append(world.nearest_victim_distance())
        s["nearest_fire"].append(world.fire_proximity().distance)
        s["collisions"].append(world.collision_count)

    def summary(self, world) -> Dict[str, float]:
        return {"time_s": world.time, "status": world.status.value,
                "discovered": world.discovered_count, "victims": len(world.victims),
                "collisions": world.collision_count, "fire_exposure_s": world.fire_exposure}
