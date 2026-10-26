"""Interactive Ursina application:  python -m firerescue.ui.app  (or `python -m firerescue`)."""
from __future__ import annotations

import argparse
import copy
import sys
import time as pytime
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import ConfigError, ScenarioConfig
from ..controllers import BaselineController, keys_to_command
from ..drone import DroneCommand
from ..scenario import default_scenario
from ..world import MissionStatus, SimWorld


def _find_default_model() -> Optional[Path]:
    """Locate a usable trained SB3 model zip for autonomous flight."""
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    for name in ("latest_model.zip", "best_model.zip", "final_model.zip"):
        for p in root.glob(f"runs/*/{name}"):
            if p.is_file():
                candidates.append(p)
    for name in ("latest_model.zip", "best_model.zip", "final_model.zip", "model.zip"):
        p = root / name
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    def key(p: Path):
        priority = {"latest_model.zip": 0, "best_model.zip": 1, "final_model.zip": 2}.get(p.name, 3)
        return (priority, -p.stat().st_mtime, len(str(p)))
    candidates.sort(key=key)
    return candidates[0]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="firerescue", description="FireRescue-RL-Drone 3D simulator (STEP 1)")
    p.add_argument("--scenario", type=Path, help="JSON scenario file (see README); default scenario if omitted")
    p.add_argument("--auto", action="store_true", help="start in autonomous (baseline controller) mode")
    p.add_argument("--model", "--policy", dest="model", type=Path, metavar="PATH",
                   help="path to trained PPO model zip (uses RL policy as agent)")
    p.add_argument("--deterministic", action="store_true", default=True, help="use deterministic policy actions")
    p.add_argument("--manual", action="store_true", help="force manual keyboard control (disable auto model loading)")
    p.add_argument("--demo-dir", type=Path, default=Path("runs/demonstrations"),
                   help="directory for asynchronously saved manual demonstration archives")
    p.add_argument("--flat", action="store_true", help="disable lighting shader (use if shading looks broken)")
    p.add_argument("--size", default="1280x720", help="window size, e.g. 1280x720")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--export-default-scenario", type=Path, metavar="PATH",
                   help="write the default scenario as JSON and exit")
    # headless / testing helpers
    p.add_argument("--overview", action="store_true", help="start in the top-down overview camera")
    p.add_argument("--frames", type=int, default=0, help="run N frames then exit (testing)")
    p.add_argument("--fixed-dt", type=float, default=0.0, help="advance the simulation by this dt per frame (testing)")
    p.add_argument("--screenshot", type=Path, help="save a screenshot at the end of --frames")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.export_default_scenario:
        default_scenario().to_json(args.export_default_scenario)
        print(f"wrote {args.export_default_scenario}")
        return 0
    try:
        cfg = ScenarioConfig.from_json(args.scenario) if args.scenario else default_scenario()
        world = SimWorld(cfg, randomize=True)
    except (ConfigError, OSError) as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2

    # Auto-load a trained model (if present) so the drone flies autonomously by default.
    if args.model is None and not args.manual and not args.auto:
        try:
            args.model = _find_default_model()
        except Exception:
            args.model = None

    # Import the RL stack (torch) BEFORE Ursina/Panda3D: Panda3D loads OpenGL
    # shared libraries that make torch's CPU DLLs fail to initialise afterwards.
    if args.model:
        try:
            import torch  # noqa: F401
            from stable_baselines3 import PPO  # noqa: F401
        except ImportError:
            print("The --model option needs Stable-Baselines3 + torch:  pip install -r requirements.txt",
                  file=sys.stderr)
            return 3

    try:
        from ursina import (Entity, Ursina, Vec3, Color, application, camera, window, held_keys, scene, Sky)
    except ImportError:
        print("The 3D front-end needs Ursina:  pip install ursina", file=sys.stderr)
        return 3

    w, h = (int(v) for v in args.size.lower().split("x"))
    app = Ursina(title="FireRescue-RL-Drone  (STEP 1 simulator)", borderless=False, fullscreen=args.fullscreen,
                 size=(w, h), development_mode=False, vsync=True)
    window.color = Color(0.45, 0.58, 0.72, 1)
    window.exit_button.visible = False
    window.fps_counter.enabled = False

    from .actors import DroneActor, FireActor, VictimActor
    from .camera_rig import CameraRig
    from .common import init_lighting
    from .hud import Hud, YELLOW, RED
    from .scene import build_static_scene
    from ursina import AmbientLight, DirectionalLight

    init_lighting(not args.flat)
    DirectionalLight(shadows=False, rotation=(55, -35, 0))
    AmbientLight(color=Color(0.45, 0.45, 0.5, 1))
    Sky(color=Color(0.55, 0.68, 0.82, 1), texture=None) if False else None

    build_static_scene(world)
    drone_actor = DroneActor(world)
    fire_actors = [FireActor(f, i) for i, f in enumerate(world.fires)]
    victim_actors = [VictimActor(v, i) for i, v in enumerate(world.victims)]
    rig = CameraRig(world)

    # optional RL policy
    policy_runner = None
    if args.model:
        try:
            from ..rl.policy_runner import PolicyRunner
            policy_runner = PolicyRunner(world, args.model, deterministic=args.deterministic)
        except Exception as exc:
            print(f"Failed to load policy: {exc}", file=sys.stderr)
            return 4

    class Sim(Entity):
        """Owns the control mode and drives world + visuals from Ursina's update loop."""

        def __init__(self):
            super().__init__()
            self.auto = False if args.manual else args.auto or policy_runner is not None
            self.controller: Optional[BaselineController] = None
            self.policy = policy_runner
            self.auto_controller = "rl" if self.policy is not None else "baseline"
            self.acc = 0.0
            self.clock = 0.0
            self.flash = 0.0
            self.shake = 0.0
            self.fps = 60.0
            self.hud = Hud(self.restart, self.reset_drone, self.toggle_mode, self.reset_camera, self.toggle_overview)
            from ..rl.demonstrations import DemonstrationRecorder
            self.demo_recorder = DemonstrationRecorder(args.demo_dir)
            self.demo_env = None
            self.demo_obs = None
            self.demo_elapsed = 0.0
            self.demo_action_sum = np.zeros(4, dtype=np.float64)
            self.demo_action_time = 0.0
            self._restarting = False
            self._pending_restart = False
            if self.auto and self.policy is None:
                self._ensure_controller()
            if args.overview:
                rig.toggle_overview()

        # ---- actions (keyboard + buttons)
        def _ensure_controller(self):
            if self.controller is None:
                self.controller = BaselineController(world)
            return self.controller

        def toggle_mode(self):
            if self.demo_recorder.active:
                self.finish_demo("auto_mode_selected")
            self.auto = not self.auto
            if self.auto and self.auto_controller == "baseline":
                self._ensure_controller().reset()
            elif self.auto and self.auto_controller == "rl" and self.policy is not None:
                self.policy.start()
            self.hud.show_toast("Control mode: " + self.mode_name, YELLOW, 2.0)

        def toggle_auto_controller(self):
            if self.policy is None:
                self.hud.show_toast("No PPO model loaded; baseline A* remains available", YELLOW, 3.0)
                return
            self.auto_controller = "baseline" if self.auto_controller == "rl" else "rl"
            if self.auto and self.auto_controller == "rl":
                self.policy.start()
            elif self.auto:
                self._ensure_controller().reset()
            self.hud.show_toast("Auto controller: " + self.mode_name, YELLOW, 2.0)

        def start_demo(self):
            if self.auto:
                self.hud.show_toast("Switch to MANUAL mode before recording a demonstration", YELLOW, 3.0)
                return
            from ..envs import RescueDroneGym
            env_config = copy.deepcopy(self.policy.env.cfg) if self.policy is not None else None
            reward_config = copy.deepcopy(self.policy.env.reward_cfg) if self.policy is not None else None
            self.demo_env = RescueDroneGym(world=world, env_config=env_config, reward_config=reward_config)
            self.demo_obs = self.demo_env.attach()
            self.demo_elapsed = self.demo_action_time = 0.0
            self.demo_action_sum[:] = 0.0
            episode_id = self.demo_recorder.start(self.demo_obs, {
                "environment_seed": None,
                "observation_version": 1,
                "action_order": ["right", "up", "forward", "yaw_rate"],
                "start_pose": [*world.drone.pos.tolist(), float(world.drone.yaw)],
                "env_config": self.demo_env.cfg.__dict__,
                "reward_config": self.demo_env.reward_cfg.__dict__,
            })
            self.hud.show_toast(f"Recording demonstration {episode_id[:8]} (G to finish, P to pause)", YELLOW, 3.0)

        def finish_demo(self, reason="manual_finish"):
            if not self.demo_recorder.active:
                return
            metrics = self.demo_env.episode_metrics(world.status == MissionStatus.SUCCESS, False)
            useful = metrics["victims_found"] > 0 or metrics["coverage"] >= 0.45
            safe = metrics["collisions"] == 0 and metrics["fire_danger_steps"] == 0
            path = self.demo_recorder.finish({
                "success": bool(metrics["success"]),
                "useful": bool(useful),
                "safe": bool(safe),
                "end_reason": reason,
                "mission_status": world.status.name,
                "episode_metrics": metrics,
                "episode_reward": self.demo_env.episode_reward,
            })
            self.demo_env = None
            self.demo_obs = None
            self.demo_elapsed = self.demo_action_time = 0.0
            self.demo_action_sum[:] = 0.0
            if path is None:
                self.hud.show_toast("Demonstration ended without any transitions", YELLOW, 3.0)
            else:
                self.hud.show_toast(f"Saving demonstration: {path} "
                                    f"(eligible={safe and (metrics['success'] or useful)})", YELLOW, 5.0)

        def toggle_demo(self):
            if self.demo_recorder.active:
                self.finish_demo()
            else:
                try:
                    self.start_demo()
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    print(f"Demonstration recording error: {exc}", file=sys.stderr)
                    self.hud.show_toast(f"Could not start recording: {exc}", RED, 5.0)

        def toggle_demo_pause(self):
            if not self.demo_recorder.active:
                self.hud.show_toast("No demonstration is being recorded", YELLOW, 2.0)
                return
            self.demo_recorder.paused = not self.demo_recorder.paused
            if not self.demo_recorder.paused:
                self.demo_obs = self.demo_env.attach()
                self.demo_elapsed = self.demo_action_time = 0.0
                self.demo_action_sum[:] = 0.0
            self.hud.show_toast("Demonstration " + ("paused" if self.demo_recorder.paused else "resumed"),
                                YELLOW, 2.0)

        def record_manual_step(self, dt, command):
            if not self.demo_recorder.active or self.demo_recorder.paused:
                return
            action = np.array([command.right, command.up, command.forward, command.yaw], dtype=np.float32)
            self.demo_elapsed += dt
            self.demo_action_sum += action * dt
            self.demo_action_time += dt
            control_dt = self.demo_env.cfg.control_dt
            while self.demo_elapsed + 1e-9 >= control_dt:
                sample = np.clip(self.demo_action_sum / max(self.demo_action_time, 1e-9), -1.0, 1.0)
                obs, reward, terminated, truncated, info = self.demo_env.finish_step(
                    sample.astype(np.float32), {})
                self.demo_recorder.add_transition(self.demo_obs, sample, obs, reward, terminated, truncated, info)
                self.demo_obs = obs
                self.demo_elapsed -= control_dt
                self.demo_action_sum[:] = 0.0
                self.demo_action_time = 0.0
                if terminated or truncated:
                    self.finish_demo("episode_terminated" if terminated else "episode_truncated")
                    break

        def restart(self):
            nonlocal fire_actors, victim_actors
            from ursina import destroy
            self.finish_demo("manual_restart")
            world.reset()
            for fa in fire_actors:
                destroy(fa.root)
            for va in victim_actors:
                destroy(va.root)
                destroy(va.label)
                destroy(va.beacon)
            fire_actors = [FireActor(f, i) for i, f in enumerate(world.fires)]
            victim_actors = [VictimActor(v, i) for i, v in enumerate(world.victims)]
            if self.controller:
                self.controller.reset()
            if self.policy:
                self.policy.reset()
            self.hud.clear()
            rig.reset()
            self.flash = self.shake = 0.0

        def reset_drone(self):
            self.finish_demo("drone_reset")
            world.reset_drone()
            if self.controller:
                self.controller.path, self.controller.target = [], None
            if self.policy:
                self.policy.start()
            self.hud.show_toast("Drone reset to start pose", YELLOW, 2.0)

        def reset_camera(self):
            rig.reset()

        def toggle_overview(self):
            rig.toggle_overview()

        @property
        def mode_name(self) -> str:
            if not self.auto:
                return "MANUAL  (keyboard)"
            return "AUTO  (RL PPO policy)" if self.auto_controller == "rl" else "AUTO  (baseline A* search controller)"

        # ---- Ursina callbacks
        def input(self, key):
            if key == "r": self.restart()
            elif key == "t": self.reset_drone()
            elif key == "m": self.toggle_mode()
            elif key == "b": self.toggle_auto_controller()
            elif key == "g": self.toggle_demo()
            elif key == "p": self.toggle_demo_pause()
            elif key == "c": self.reset_camera()
            elif key == "o": self.toggle_overview()
            elif key == "h": self.hud.help.enabled = not self.hud.help.enabled
            elif key == "scroll up": rig.zoom(+1)
            elif key == "scroll down": rig.zoom(-1)
            elif key == "escape":
                self.finish_demo("user_exit")
                try:
                    self.demo_recorder.close()
                except OSError as exc:
                    print(f"Demonstration save error: {exc}", file=sys.stderr)
                application.quit()

        def update(self):
            import time as t
            dt = args.fixed_dt if args.fixed_dt > 0 else min(t.dt, 0.1)
            if dt <= 0:
                return
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt if dt > 0 else 60.0)
            self.clock += dt

            # Deferred restart: process at the start of the next frame so Ursina
            # entities are not destroyed mid-update (which crashes the viewer).
            if self._pending_restart:
                self._pending_restart = False
                try:
                    self.restart()
                except Exception as exc:
                    print(f"Restart error: {exc}", file=sys.stderr)
                    self.policy = None
                    self.auto = False

            if self.auto:
                if self.auto_controller == "rl" and self.policy is not None:
                    cmd = self.policy.command
                else:
                    cmd = self._ensure_controller().command()
            else:
                cmd = keys_to_command(held_keys)
            world.step(cmd, dt)
            if not self.auto:
                self.record_manual_step(dt, cmd)
            if self.policy is not None and self.auto and self.auto_controller == "rl":
                try:
                    info = self.policy.advance(dt)
                except Exception as exc:
                    print(f"Policy error: {exc}", file=sys.stderr)
                    self.policy = None
                    self.auto = False
                    info = None
                if info is not None and self.policy.done and not self._restarting:
                    self._restarting = True
                    self._pending_restart = True
                    self._restarting = False
            for ev in world.pop_events():
                if ev["type"] == "collision":
                    self.flash, self.shake = 1.0, 1.0
                    drone_actor.trigger_collision()
                elif ev["type"] == "victim_discovered":
                    self.hud.show_toast(f"Victim {ev['id']} discovered by synthetic thermal sensor "
                                        f"({world.discovered_count}/{len(world.victims)})", None, 3.5)
            self.flash = max(0.0, self.flash - dt * 2.0)
            self.shake = max(0.0, self.shake - dt * 3.0)
            drone_actor.sync(dt)
            for f in fire_actors:
                f.sync(dt)
            for v in victim_actors:
                v.sync(dt, self.clock)
            rig.update(dt, self.shake)
            saved_paths, errors = self.demo_recorder.poll_saves()
            for path in saved_paths:
                self.hud.show_toast(f"Demonstration saved: {path}", YELLOW, 4.0)
            for path, exc in errors:
                print(f"Demonstration save error for {path}: {exc}", file=sys.stderr)
                self.hud.show_toast(f"Could not save demonstration: {exc}", RED, 5.0)
            recorder_status = " | DEMO PAUSED" if self.demo_recorder.paused else (
                " | DEMO RECORDING" if self.demo_recorder.active else "")
            self.hud.update(world, self.mode_name + recorder_status, dt, self.flash, self.fps)

    sim = Sim()

    if args.frames:        # bounded run for tests / screenshots
        for _ in range(args.frames):
            app.step()
        if args.screenshot:
            from panda3d.core import Filename
            application.base.win.saveScreenshot(Filename.from_os_specific(str(args.screenshot)))
            app.step()
        print(f"frames={args.frames} status={world.status.name} t={world.time:.1f}s "
              f"found={world.discovered_count}/{len(world.victims)} collisions={world.collision_count}")
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
