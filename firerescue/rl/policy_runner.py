"""Runs a trained PPO policy inside the 3D viewer's SimWorld (engine-independent logic)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..drone import DroneCommand
from ..envs import RescueDroneGym, load_env_config, load_reward_config


class PolicyRunner:
    """Shares the viewer's world with a RescueDroneGym so rewards/coverage/goals are tracked live.

    The viewer steps the world every frame with ``command``; every ``control_dt`` of simulated
    time ``advance`` calls the policy, finishes the env step and picks a new action.
    """

    def __init__(self, world, model_path, env_config=None, reward_config=None, deterministic: bool = True):
        from stable_baselines3 import PPO
        mp = Path(model_path)
        run_dir = mp.parent if mp.parent.name != "checkpoints" else mp.parent.parent
        ec = env_config or (run_dir / "env.yaml" if (run_dir / "env.yaml").exists() else None)
        rc = reward_config or (run_dir / "reward.yaml" if (run_dir / "reward.yaml").exists() else None)
        cfg = load_env_config(ec)
        cfg.random_start = False                       # viewer: start where the world starts
        self.env = RescueDroneGym(env_config=cfg, reward_config=load_reward_config(rc), world=world)
        self.model = PPO.load(str(mp), device="cpu")
        self.name = mp.name
        self.deterministic = deterministic
        self.world = world
        self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """(Re)attach to the current world state without resetting it."""
        self.obs = self.env.attach()
        self.episode_reward = 0.0
        self.acc = 0.0
        self.done = False
        self.end_reason = ""
        self.last_components: Dict[str, float] = {}
        self._pick_action()

    def reset(self) -> None:
        """Full episode reset (also resets the world)."""
        self.obs, _ = self.env.reset()
        self.episode_reward, self.acc, self.done, self.end_reason = 0.0, 0.0, False, ""
        self.last_components = {}
        self._pick_action()

    def _pick_action(self) -> None:
        a = self.model.predict(self.obs, deterministic=self.deterministic)[0]
        self.cmd, self.action, self.flags = self.env.action_to_command(a)

    # ---------------------------------------------------------------------- per frame
    @property
    def command(self) -> DroneCommand:
        return DroneCommand() if self.done else self.cmd

    def advance(self, dt: float) -> Optional[Dict]:
        """Call after world.step(command, dt). Returns the env info when a control tick completed."""
        if self.done:
            return None
        self.acc += dt
        if self.acc + 1e-9 < self.env.cfg.control_dt:
            return None
        self.acc -= self.env.cfg.control_dt
        self.obs, r, term, trunc, info = self.env.finish_step(self.action, self.flags)
        self.episode_reward += r
        self.last_components = info["reward_components"]
        if term or trunc:
            self.done = True
            m = info["episode_metrics"]
            self.end_reason = ("SUCCESS" if m["success"] else "OVERHEATED" if m["overheated"]
                               else "STEP LIMIT" if trunc else "TERMINATED")
        else:
            self._pick_action()
        return info

    # ----------------------------------------------------------------------- display
    @property
    def goal_target(self):
        t = self.env.goals.target
        return None if t is None else (t[0], self.env.cfg.goal_altitude, t[1])

    @property
    def goal_waypoint(self):
        t = self.env.goals.lookahead_point()
        return None if t is None else (t[0], self.env.cfg.goal_altitude, t[1])

    def stats_text(self) -> str:
        e, w = self.env, self.world
        comp = "  ".join(f"{k} {v:+.2f}" for k, v in self.last_components.items()
                         if abs(v) > 1e-9 and k != "time")
        end = f"   ** EPISODE ENDED: {self.end_reason} **" if self.done else ""
        return (f"PPO policy: {self.name}   (deterministic){end}\n"
                f"Episode reward {self.episode_reward:8.1f}   step {e.steps}/{e.cfg.max_episode_steps}   "
                f"coverage {e.coverage.global_fraction():4.0%}   rooms {len(e.coverage.rooms_found)}/{e.coverage.n_rooms}\n"
                f"Goal distance {e.goals.distance():5.1f} m   victims {w.discovered_count}/{len(w.victims)}   "
                f"sweeps {e.goals.sweeps}\nlast reward terms: {comp or '-'}")
