"""Interactive Ursina application:  python -m firerescue.ui.app  (or `python -m firerescue`)."""
from __future__ import annotations

import argparse
import sys
import time as pytime
from pathlib import Path
from typing import List, Optional

from ..config import ConfigError, ScenarioConfig
from ..controllers import BaselineController, keys_to_command
from ..drone import DroneCommand
from ..scenario import default_scenario
from ..world import MissionStatus, SimWorld


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="firerescue", description="FireRescue-RL-Drone 3D simulator (STEP 1)")
    p.add_argument("--scenario", type=Path, help="JSON scenario file (see README); default scenario if omitted")
    p.add_argument("--auto", action="store_true", help="start in autonomous (baseline controller) mode")
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

    class Sim(Entity):
        """Owns the control mode and drives world + visuals from Ursina's update loop."""

        def __init__(self):
            super().__init__()
            self.auto = args.auto
            self.controller: Optional[BaselineController] = None
            self.acc = 0.0
            self.clock = 0.0
            self.flash = 0.0
            self.shake = 0.0
            self.fps = 60.0
            self.hud = Hud(self.restart, self.reset_drone, self.toggle_mode, self.reset_camera, self.toggle_overview)
            if self.auto:
                self._ensure_controller()
            if args.overview:
                rig.toggle_overview()

        # ---- actions (keyboard + buttons)
        def _ensure_controller(self):
            if self.controller is None:
                self.controller = BaselineController(world)
            return self.controller

        def toggle_mode(self):
            self.auto = not self.auto
            if self.auto:
                self._ensure_controller().reset()
            self.hud.show_toast("Control mode: " + self.mode_name, YELLOW, 2.0)

        def restart(self):
            nonlocal fire_actors, victim_actors
            from ursina import destroy
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
            self.hud.clear()
            rig.reset()
            self.flash = self.shake = 0.0

        def reset_drone(self):
            world.reset_drone()
            if self.controller:
                self.controller.path, self.controller.target = [], None
            self.hud.show_toast("Drone reset to start pose", YELLOW, 2.0)

        def reset_camera(self):
            rig.reset()

        def toggle_overview(self):
            rig.toggle_overview()

        @property
        def mode_name(self) -> str:
            return "AUTO  (baseline A* search controller)" if self.auto else "MANUAL  (keyboard)"

        # ---- Ursina callbacks
        def input(self, key):
            if key == "r": self.restart()
            elif key == "t": self.reset_drone()
            elif key == "m": self.toggle_mode()
            elif key == "c": self.reset_camera()
            elif key == "o": self.toggle_overview()
            elif key == "h": self.hud.help.enabled = not self.hud.help.enabled
            elif key == "scroll up": rig.zoom(+1)
            elif key == "scroll down": rig.zoom(-1)
            elif key == "escape": application.quit()

        def update(self):
            import time as t
            dt = args.fixed_dt if args.fixed_dt > 0 else min(t.dt, 0.1)
            if dt <= 0:
                return
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt if dt > 0 else 60.0)
            self.clock += dt
            if self.auto:
                cmd = self._ensure_controller().command()
            else:
                cmd = keys_to_command(held_keys)
            world.step(cmd, dt)
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
            self.hud.update(world, self.mode_name, dt, self.flash, self.fps)

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
