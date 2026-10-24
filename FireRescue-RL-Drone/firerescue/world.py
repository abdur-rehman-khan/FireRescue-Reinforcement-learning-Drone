"""SimWorld: owns all simulation state and advances it.  No rendering, no engine imports."""
from __future__ import annotations

import math
import random
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .collision import ObstacleSet
from .config import ConfigError, FireConfig, ScenarioConfig, VictimConfig, validate_scenario
from .drone import CollisionInfo, Drone, DroneCommand
from .fire import DANGER, FireProximity, FireSource, proximity
from .metrics import MetricsRecorder
from .obstacles import build_obstacles
from .scenario import default_scenario
from .sensors import SyntheticThermalSensor, ThermalReading
from .victims import Victim, nearest_undiscovered


class MissionStatus(str, Enum):
    RUNNING = "IN PROGRESS"
    SUCCESS = "SUCCESS - all victims located"
    FAILED_FIRE = "FAILED - drone overheated"
    FAILED_TIME = "FAILED - time limit reached"


class SimWorld:
    def __init__(self, config: Optional[ScenarioConfig] = None, randomize: bool = False):
        self.config = config or default_scenario()
        self.randomize = randomize
        validate_scenario(self.config)
        cfg = self.config
        self.obstacle_list = build_obstacles(cfg.building)
        self.obstacles = ObstacleSet(self.obstacle_list)
        self.fires = [FireSource(f) for f in cfg.fires]
        self.victims = [Victim(v) for v in cfg.victims]
        self.sensor = SyntheticThermalSensor(cfg.sensor)
        self.drone = Drone(cfg.drone, cfg.building)
        self.metrics = MetricsRecorder()
        self._check_geometry()
        self.reset()

    # ------------------------------------------------------------- validation
    def _check_geometry(self) -> None:
        errs = []
        if self.obstacles.sphere_collides(np.array(self.config.drone.start_position), self.config.drone.radius):
            errs.append("drone start position intersects an obstacle")
        for v in self.victims:
            if self.obstacles.point_inside(v.sensor_point):
                errs.append(f"victim '{v.id}' is inside an obstacle")
        if errs:
            raise ConfigError("Invalid scenario geometry: " + "; ".join(errs))

    def _randomize_mission(self) -> None:
        cfg = self.config
        rng = random.Random()
        # random counts
        n_victims = rng.randint(3, 6)
        n_fires = rng.randint(2, 4)
        start = np.array(cfg.drone.start_position, dtype=float)
        # build victim configs
        victims: list = []
        attempts = 0
        while len(victims) < n_victims and attempts < 1000:
            attempts += 1
            x = rng.uniform(0.8, cfg.building.size_x - 0.8)
            z = rng.uniform(0.8, cfg.building.size_z - 0.8)
            pos = np.array([x, 0.0, z])
            if self.obstacles.point_inside(np.array([x, 1.1, z])) or self.obstacles.point_inside(np.array([x, 0.2, z])):
                continue
            if np.linalg.norm(pos[:2] - start[[0, 2]]) < 1.5:  # rough
                continue
            # distance to existing victims
            ok = True
            for v in victims:
                if math.hypot(v.position[0] - x, v.position[2] - z) < 1.5:
                    ok = False
                    break
            if not ok:
                continue
            victims.append(VictimConfig(id=f"V{len(victims)+1}", position=[float(x), 0.0, float(z)],
                                       pose=rng.choice(["standing", "prone"])))
        # build fire configs
        fires: list = []
        attempts = 0
        while len(fires) < n_fires and attempts < 1000:
            attempts += 1
            x = rng.uniform(0.8, cfg.building.size_x - 0.8)
            z = rng.uniform(0.8, cfg.building.size_z - 0.8)
            if self.obstacles.point_inside(np.array([x, 0.4, z])):
                continue
            ok = True
            for f in fires:
                if math.hypot(f.position[0] - x, f.position[2] - z) < 2.0:
                    ok = False
                    break
            if not ok:
                continue
            fires.append(FireConfig(id=f"F{len(fires)+1}", position=[float(x), 0.4, float(z)],
                                    danger_radius=rng.uniform(1.4, 2.2), intensity=1.0))
        cfg.victims = victims
        cfg.fires = fires
        self.fires = [FireSource(f) for f in cfg.fires]
        self.victims = [Victim(v) for v in cfg.victims]
        self._check_geometry()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Restart the whole mission (drone, victims, timers, counters)."""
        if self.randomize:
            self._randomize_mission()
        self.drone.reset()
        for v in self.victims:
            v.reset()
        self.sensor.reset()
        self.time = 0.0
        self.status = MissionStatus.RUNNING
        self.collision_count = 0
        self.drone_reset_count = 0
        self.fire_exposure = 0.0
        self.events: List[Dict] = []
        self._was_colliding = False
        self.metrics.reset()
        self.thermal: ThermalReading = ThermalReading()
        self._sense()

    def reset_drone(self) -> None:
        """Put the drone back at its start pose; mission progress (found victims, clock) is kept."""
        self.drone.reset()
        self._was_colliding = False
        self.drone_reset_count += 1
        self.events.append({"type": "drone_reset", "t": self.time})
        self._sense()

    # ------------------------------------------------------------------- step
    def step(self, command: Optional[DroneCommand] = None, dt: Optional[float] = None) -> None:
        base = self.config.mission.sim_dt
        dt = base if dt is None else float(dt)
        if dt <= 0:
            return
        n = max(1, math.ceil(dt / base - 1e-9))
        for _ in range(n):
            self._substep(command or DroneCommand(), dt / n)

    def _substep(self, command: DroneCommand, dt: float) -> None:
        running = self.status == MissionStatus.RUNNING
        info = self.drone.step(command if running else DroneCommand(), dt, self.obstacles)
        if not running:
            return
        self.time += dt
        if info.collided and not self._was_colliding:
            self.collision_count += 1
            self.events.append({"type": "collision", "t": self.time, "with": list(info.names),
                                "impact": info.impact_speed})
        self._was_colliding = info.collided

        if self.fire_proximity().level == DANGER:
            self.fire_exposure += dt
        self._sense()
        self.metrics.record(self)

        if self.discovered_count == len(self.victims):
            self.status = MissionStatus.SUCCESS
        elif self.fire_exposure >= self.config.mission.fire_exposure_limit_s:
            self.status = MissionStatus.FAILED_FIRE
        elif self.time >= self.config.mission.time_limit_s:
            self.status = MissionStatus.FAILED_TIME
        if self.status != MissionStatus.RUNNING:
            self.events.append({"type": "mission_end", "t": self.time, "status": self.status.name})

    def _sense(self) -> None:
        self.thermal = self.sensor.read(self.drone.pos, self.victims, self.fires, self.obstacles)
        by_id = {v.id: v for v in self.victims}
        for r in self.thermal.victims:
            v = by_id[r.victim_id]
            if r.detected and not v.discovered:
                v.discovered, v.discovered_at = True, self.time
                self.events.append({"type": "victim_discovered", "id": v.id, "t": self.time})

    # ---------------------------------------------------------------- queries
    @property
    def discovered_count(self) -> int:
        return sum(v.discovered for v in self.victims)

    @property
    def collision_active(self) -> bool:
        return self.drone.last_collision.collided

    def fire_proximity(self) -> FireProximity:
        return proximity(self.fires, self.drone.pos, self.config.mission.warning_factor)

    def nearest_victim(self):
        return nearest_undiscovered(self.victims, self.drone.pos)

    def nearest_victim_distance(self) -> float:
        """Ground-truth distance to the nearest undiscovered victim (inf if none left)."""
        return self.nearest_victim()[1]

    def pop_events(self) -> List[Dict]:
        ev, self.events = self.events, []
        return ev

    def heat_grid(self, n: int = 32) -> np.ndarray:
        return self.sensor.heat_grid(self.drone.pos, self.thermal, self.victims, self.fires,
                                     self.obstacles, n)
