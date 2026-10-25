"""RescueDroneGym: Gymnasium environment wrapping the Step 1 SimWorld (no engine, no ROS).

Action  (Box(4,), [-1, 1]):  [vx, vy, vz, yaw_rate] in the drone BODY frame
    vx = right (+) / left (-), vy = up (+) / down (-), vz = forward (+) / backward (-),
    yaw_rate = turn right (+) / left (-).  Scaled by EnvConfig limits (default: the drone limits).
Observation: see firerescue.envs.observation (documented, normalised, fixed size).
Reward: see firerescue.envs.reward and configs/reward.yaml.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import ConfigError, ScenarioConfig
from ..drone import DroneCommand
from ..scenario import default_scenario
from ..world import MissionStatus, SimWorld
from .coverage import CoverageTracker
from .navigation import GoalManager
from .observation import ObservationBuilder
from .reward import RewardCalculator, StepSignals
from .rl_config import EnvConfig, RewardConfig
from .topdown import render_topdown


def _register() -> None:
    try:
        gym.register(
            id="RescueDrone-v0",
            entry_point="firerescue.envs.rescue_drone_gym:RescueDroneGym",
            nondeterministic=True,
        )
    except Exception:                       # already registered in a previous import
        pass


_register()


class RescueDroneGym(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, scenario: Optional[ScenarioConfig] = None, env_config: Optional[EnvConfig] = None,
                 reward_config: Optional[RewardConfig] = None, render_mode: Optional[str] = None,
                 world: Optional[SimWorld] = None):
        """``world`` lets the 3D viewer share its SimWorld with the environment (see PolicyRunner)."""
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        self.render_mode = render_mode
        self.cfg = env_config or EnvConfig()
        self.cfg.validate()
        self.reward_cfg = reward_config or RewardConfig()
        if world is None:
            sc = copy.deepcopy(scenario or default_scenario())
            sc.mission.sim_dt = self.cfg.physics_dt
            world = SimWorld(sc)
        self.world = world
        d = world.config.drone
        self.limits = np.array([self.cfg.max_speed_xz or d.max_speed_xz, self.cfg.max_speed_y or d.max_speed_y,
                                self.cfg.max_speed_xz or d.max_speed_xz, self.cfg.max_yaw_rate_deg or d.max_yaw_rate_deg])
        full = np.array([d.max_speed_xz, d.max_speed_y, d.max_speed_xz, d.max_yaw_rate_deg])
        if np.any(self.limits > full + 1e-9):
            raise ConfigError("action limits must not exceed the drone's own limits")
        self._scale = self.limits / full           # action -> DroneCommand fraction

        self.coverage = CoverageTracker(world, self.cfg.coverage_cell, self.cfg.coverage_radius,
                                        self.cfg.room_discovery_fraction, self.cfg.goal_altitude)
        self.goals = GoalManager(world, self.cfg, self.coverage)
        self.obs_builder = ObservationBuilder(world, self.cfg, self.coverage)
        self.reward_calc = RewardCalculator(self.reward_cfg)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_builder.size_total,), dtype=np.float32)
        self._init_state()

    # ===================================================================== state
    def _init_state(self) -> None:
        w = self.world
        self.steps = 0
        self.episode_reward = 0.0
        self.invalid_actions = 0
        self.fire_danger_steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.trajectory = [w.drone.pos.copy()]
        self.rooms_rewarded: set = set()
        self.victims_rewarded = {v.id for v in w.victims if v.discovered}   # found at spawn: no reward
        self._seen_collisions = w.collision_count
        self.coverage.reset()
        self.goals.reset()
        self.reward_calc.reset()
        self.coverage.update(w.drone.pos)
        self.rooms_rewarded |= set(self.coverage.rooms_found)               # spawn room: no reward
        self.goals.update()
        self.obs_builder.set_goal_distance(self.goals.distance())
        self.last_components: Dict[str, float] = {}
        self.finished = False

    def attach(self) -> np.ndarray:
        """Start tracking the CURRENT world state without resetting it (used by the 3D viewer)."""
        self._init_state()
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.obs_builder.build(self.goals.lookahead_point(), self.prev_action, self.steps)

    # ====================================================================== reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.world.reset()
        start = options.get("start")
        if start is not None:                                   # explicit (x, y, z, yaw_deg)
            self.world.set_drone_pose(start[:3], start[3])
        elif self.cfg.random_start:
            self._random_start()
        self._init_state()
        return self.observe(), self._info({})

    def _random_start(self) -> None:
        cells = self.goals.free_start_cells()
        d, lo_hi = self.world.drone.cfg, self.cfg.start_altitude_range
        for _ in range(50):
            x, z = cells[int(self.np_random.integers(len(cells)))]
            y = float(self.np_random.uniform(*lo_hi))
            pos = np.array([x, np.clip(y, d.min_altitude, d.max_altitude), z])
            if not self.world.obstacles.sphere_collides(pos, d.radius + 0.05):
                self.world.set_drone_pose(pos, float(self.np_random.uniform(0.0, 360.0)))
                return                                        # else: keep the default start

    # ======================================================================= step
    def action_to_command(self, action) -> Tuple[DroneCommand, np.ndarray, Dict[str, bool]]:
        """Sanitise an action and map it to a Step 1 DroneCommand.

        Wrong shape -> ValueError.  NaN/inf entries -> replaced by 0 (flag ``invalid``).
        Out-of-range values -> clipped to [-1, 1] (flag ``clipped``).
        """
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {a.shape}")
        flags = {"invalid": False, "clipped": False}
        bad = ~np.isfinite(a)
        if bad.any():
            a = np.where(bad, 0.0, a)
            flags["invalid"] = True
        if np.any(np.abs(a) > 1.0):
            a = np.clip(a, -1.0, 1.0)
            flags["clipped"] = True
        c = a * self._scale
        cmd = DroneCommand(forward=c[2], right=c[0], up=c[1], yaw=c[3])
        return cmd, a.astype(np.float32), flags

    def step(self, action):
        cmd, a, flags = self.action_to_command(action)
        self.world.step(cmd, self.cfg.control_dt)
        return self.finish_step(a, flags)

    def finish_step(self, action: np.ndarray, flags: Optional[Dict[str, bool]] = None):
        """Bookkeeping after the world advanced by ``control_dt`` (also used by the 3D viewer)."""
        w, cfg = self.world, self.cfg
        flags = flags or {}
        self.steps += 1
        if flags.get("invalid"):
            self.invalid_actions += 1
        self.trajectory.append(w.drone.pos.copy())

        new_cells, new_rooms = self.coverage.update(w.drone.pos)
        new_rooms = [r for r in new_rooms if r not in self.rooms_rewarded]
        self.rooms_rewarded.update(new_rooms)
        found = [v.id for v in w.victims if v.discovered and v.id not in self.victims_rewarded]
        self.victims_rewarded.update(found)
        progress = self.goals.update()
        self.obs_builder.set_goal_distance(self.goals.distance())
        new_col = w.collision_count - self._seen_collisions
        self._seen_collisions = w.collision_count
        fire = w.fire_proximity().level
        if fire == "DANGER":
            self.fire_danger_steps += 1

        success = w.status == MissionStatus.SUCCESS
        fire_fail = w.status == MissionStatus.FAILED_FIRE
        time_fail = w.status == MissionStatus.FAILED_TIME
        col_limit = w.collision_count >= cfg.max_collisions or (cfg.terminate_on_collision and new_col > 0)
        terminated = success or fire_fail or time_fail or col_limit
        truncated = (not terminated) and self.steps >= cfg.max_episode_steps

        reward, comp = self.reward_calc.compute(StepSignals(
            position=w.drone.pos, action=action, new_rooms=len(new_rooms), new_victims=len(found),
            new_cells=new_cells, progress=progress, new_collisions=new_col, in_contact=w.collision_active,
            fire_level=fire, success=success, fire_failure=fire_fail, collision_limit=col_limit and not success,
            truncated=truncated, episode_fraction_left=max(0.0, 1.0 - self.steps / cfg.max_episode_steps)))
        self.episode_reward += reward
        self.last_components = comp
        self.prev_action = np.asarray(action, dtype=np.float32)
        self.finished = terminated or truncated
        info = self._info(comp)
        info.update(new_victims=found, new_rooms=new_rooms, invalid_action=bool(flags.get("invalid")),
                    action_clipped=bool(flags.get("clipped")))
        if self.finished:
            info["episode_metrics"] = self.episode_metrics(success, truncated)
        return self.observe(), float(reward), terminated, truncated, info

    # ===================================================================== info
    def episode_metrics(self, success: bool, truncated: bool) -> Dict[str, float]:
        w = self.world
        return {
            "success": float(success), "episode_reward": float(self.episode_reward), "steps": float(self.steps),
            "victims_found": float(w.discovered_count), "victims_total": float(len(w.victims)),
            "collisions": float(w.collision_count), "coverage": self.coverage.global_fraction(),
            "rooms_discovered": float(len(self.coverage.rooms_found)), "rooms_total": float(self.coverage.n_rooms),
            "fire_danger_steps": float(self.fire_danger_steps), "overheated": float(w.status == MissionStatus.FAILED_FIRE),
            "truncated": float(truncated), "invalid_actions": float(self.invalid_actions),
            "sim_time": float(w.time),
        }

    def _info(self, comp) -> Dict[str, Any]:
        w = self.world
        tgt = self.goals.target
        return {"victims_found": w.discovered_count, "collisions": w.collision_count,
                "coverage": self.coverage.global_fraction(), "goal": tgt,
                "goal_distance": self.goals.distance(), "reward_components": dict(comp),
                "mission_status": w.status.name}

    # =================================================================== render
    def render(self):
        if self.render_mode == "rgb_array":
            tgt = self.goals.target
            return render_topdown(self.world, self.coverage, self.trajectory[-1500:], tgt)
        if self.render_mode == "ansi":
            w, p = self.world, self.world.drone.pos
            return (f"step {self.steps:4d} t={w.time:6.1f}s pos=({p[0]:5.1f},{p[1]:4.2f},{p[2]:5.1f}) "
                    f"victims {w.discovered_count}/{len(w.victims)} coverage {self.coverage.global_fraction():4.0%} "
                    f"collisions {w.collision_count} fire={w.fire_proximity().level} R={self.episode_reward:7.1f}")
        return None

    def close(self) -> None:
        self.trajectory = []
