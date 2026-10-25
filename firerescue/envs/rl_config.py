"""EnvConfig / RewardConfig dataclasses with YAML loading (weights documented in configs/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

from ..config import ConfigError, dataclass_from_dict


@dataclass
class EnvConfig:
    control_dt: float = 0.1
    physics_dt: float = 1.0 / 30.0
    max_episode_steps: int = 1500
    max_collisions: int = 30
    terminate_on_collision: bool = False
    max_speed_xz: Optional[float] = None
    max_speed_y: Optional[float] = None
    max_yaw_rate_deg: Optional[float] = None
    random_start: bool = True
    start_altitude_range: Tuple[float, float] = (1.0, 1.4)
    n_rays: int = 12
    ray_range: float = 4.0
    n_fires_in_obs: int = 3
    thermal_grid: int = 6
    coverage_cell: float = 1.0
    coverage_radius: float = 2.5
    room_discovery_fraction: float = 0.3
    goal_altitude: float = 1.2
    goal_lookahead: float = 1.5
    viewpoint_spacing: float = 3.5
    goal_reach_radius: float = 1.0
    planner_clearance_margin: float = 0.2

    def validate(self) -> None:
        e = []
        if self.control_dt <= 0 or self.physics_dt <= 0 or self.physics_dt > self.control_dt + 1e-9:
            e.append("need 0 < physics_dt <= control_dt")
        if self.max_episode_steps < 1 or self.max_collisions < 1:
            e.append("max_episode_steps and max_collisions must be >= 1")
        if self.n_rays < 4 or self.ray_range <= 0 or self.n_fires_in_obs < 0:
            e.append("n_rays >= 4, ray_range > 0, n_fires_in_obs >= 0 required")
        if self.thermal_grid < 1 or self.coverage_cell <= 0 or self.coverage_radius <= 0:
            e.append("thermal_grid, coverage_cell, coverage_radius must be positive")
        if not 0 < self.room_discovery_fraction <= 1:
            e.append("room_discovery_fraction must be in (0, 1]")
        if self.start_altitude_range[0] > self.start_altitude_range[1]:
            e.append("start_altitude_range must be (low, high)")
        for name in ("max_speed_xz", "max_speed_y", "max_yaw_rate_deg"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                e.append(f"{name} must be positive")
        if e:
            raise ConfigError("Invalid EnvConfig: " + "; ".join(e))


@dataclass
class RewardConfig:
    new_room: float = 5.0
    new_victim: float = 25.0
    new_coverage_cell: float = 0.1
    mission_complete: float = 60.0
    success_time_bonus: float = 20.0
    progress: float = 1.0
    max_progress_step: float = 0.4
    collision: float = -5.0
    collision_contact: float = -0.2
    fire_warning: float = -0.05
    fire_danger: float = -1.0
    fire_failure: float = -30.0
    collision_limit_failure: float = -20.0
    time_step: float = -0.02
    oscillation: float = -0.1
    oscillation_min_action: float = 0.3
    oscillation_cos: float = -0.3
    dither: float = -0.2
    dither_window: int = 20
    dither_min_path: float = 1.0
    dither_max_net: float = 0.3
    stagnation: float = -0.1
    stagnation_steps: int = 30
    stagnation_progress_eps: float = 0.02
    timeout: float = -10.0


def _load_yaml(path) -> dict:
    try:
        import yaml
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError("PyYAML is required: pip install pyyaml") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    return data


def load_env_config(path=None) -> EnvConfig:
    cfg = dataclass_from_dict(EnvConfig, _load_yaml(path)) if path else EnvConfig()
    cfg.validate()
    return cfg


def load_reward_config(path=None) -> RewardConfig:
    return dataclass_from_dict(RewardConfig, _load_yaml(path)) if path else RewardConfig()
