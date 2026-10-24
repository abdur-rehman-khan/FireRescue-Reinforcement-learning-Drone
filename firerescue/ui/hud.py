"""Heads-up display: telemetry, warnings, synthetic thermal panel, buttons, toasts."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from ursina import Button, Color, Entity, Text, Texture, camera, window

from ..fire import CLEAR, DANGER
from ..sensors import SENSOR_LABEL
from .common import rgb

WHITE, GREY = rgb((0.95, 0.95, 0.95)), rgb((0.7, 0.72, 0.78))
GREEN, YELLOW, ORANGE, RED = rgb((0.35, 1, 0.5)), rgb((1, 0.9, 0.3)), rgb((1, 0.6, 0.15)), rgb((1, 0.3, 0.25))


def heat_to_rgb(grid: np.ndarray, drone_yaw_deg: float) -> np.ndarray:
    """Map heat (0..~1.5) to an inferno-like palette; north (+z) is up, east (+x) is right."""
    g = np.clip(grid / 1.2, 0.0, 1.0)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[8, 6, 24], [90, 20, 110], [200, 50, 60], [250, 160, 30], [255, 250, 200]], dtype=float)
    img = np.stack([np.interp(g, stops, cols[:, c]) for c in range(3)], axis=-1)
    n = img.shape[0]
    c = n // 2
    img[c - 1:c + 1, c - 3:c + 3] = (80, 255, 120)                  # drone marker (cross)
    img[c - 3:c + 3, c - 1:c + 1] = (80, 255, 120)
    r = math.radians(drone_yaw_deg)                                   # heading tick
    for k in range(3, 7):
        x, z = int(round(c + math.sin(r) * k)), int(round(c + math.cos(r) * k))
        if 0 <= x < n and 0 <= z < n:
            img[z, x] = (255, 255, 255)
    return np.flipud(img).astype(np.uint8)


class Hud:
    def __init__(self, on_restart, on_reset_drone, on_toggle_mode, on_camera_reset, on_overview):
        ui = camera.ui
        tl, tr, bl = window.top_left, window.top_right, window.bottom_left
        self.panel = Entity(parent=ui, model="quad", color=Color(0, 0, 0, 0.55), origin=(-.5, .5),
                            position=(tl[0] + 0.01, tl[1] - 0.01), scale=(0.62, 0.245), z=1)
        self.title = Text("FIRE-RESCUE DRONE SIMULATOR", parent=ui, position=(tl[0] + 0.02, tl[1] - 0.018),
                          scale=1.0, color=YELLOW)
        self.info = Text("", parent=ui, position=(tl[0] + 0.02, tl[1] - 0.055), scale=0.85, color=WHITE)
        self.status = Text("", parent=ui, position=(tl[0] + 0.02, tl[1] - 0.205), scale=1.0, color=GREEN)

        # synthetic thermal panel (top right)
        self.th_bg = Entity(parent=ui, model="quad", color=Color(0, 0, 0, 0.6), origin=(.5, .5),
                            position=(tr[0] - 0.01, tr[1] - 0.01), scale=(0.34, 0.455), z=1)
        self.th_title = Text("SYNTHETIC THERMAL SENSOR", parent=ui, origin=(.5, .5), scale=0.8,
                             position=(tr[0] - 0.02, tr[1] - 0.018), color=ORANGE)
        self.th_sub = Text("model - NOT a real thermal camera", parent=ui, origin=(.5, .5), scale=0.7,
                           position=(tr[0] - 0.02, tr[1] - 0.045), color=GREY)
        self.th_img = Entity(parent=ui, model="quad", origin=(.5, .5), position=(tr[0] - 0.02, tr[1] - 0.07),
                             scale=(0.30, 0.30), z=0.5)
        self.th_txt = Text("", parent=ui, origin=(.5, .5), scale=0.75, position=(tr[0] - 0.02, tr[1] - 0.38), color=WHITE)
        self._tex_timer = 0.0

        # banners
        self.warn = Text("", parent=ui, origin=(0, 0), position=(0, tl[1] - 0.06), scale=1.8, color=RED)
        self.toast = Text("", parent=ui, origin=(0, 0), position=(0, tl[1] - 0.12), scale=1.2, color=GREEN)
        self.banner = Text("", parent=ui, origin=(0, 0), position=(0, 0.12), scale=3.0, color=GREEN)
        self.banner_sub = Text("", parent=ui, origin=(0, 0), position=(0, 0.05), scale=1.3, color=WHITE)
        self.vignette = Entity(parent=ui, model="quad", color=Color(1, 0, 0, 0), scale=(3.5, 2), z=2)
        self._toast_t = 0.0

        # help + buttons
        self.help_bg = Entity(parent=ui, model="quad", color=Color(0, 0, 0, 0.45), origin=(-.5, -.5),
                              position=(bl[0], bl[1]), scale=(1.45, 0.07), z=1)
        self.help = Text("W/S fwd/back  A/D strafe  SPACE/LSHIFT up/down  Q/E yaw  |  RMB-drag/arrows orbit  wheel zoom\n"
                         "R restart  T reset drone  M manual/auto  C camera reset  O overview  H help  ESC quit",
                         parent=ui, origin=(-.5, -.5), position=(bl[0] + 0.015, bl[1] + 0.012), scale=0.75, color=GREY)
        y = bl[1] + 0.085
        self.buttons = []
        for i, (label, fn) in enumerate([("Restart (R)", on_restart), ("Reset drone (T)", on_reset_drone),
                                         ("Mode (M)", on_toggle_mode), ("Camera reset (C)", on_camera_reset),
                                         ("Overview (O)", on_overview)]):
            b = Button(text=label, parent=ui, scale=(0.2, 0.042), position=(bl[0] + 0.12 + i * 0.21, y),
                       color=Color(0.15, 0.2, 0.3, 0.9), highlight_color=Color(0.25, 0.35, 0.5, 1))
            b.on_click = fn
            self.buttons.append(b)
        self.mode_btn = self.buttons[2]

    # ------------------------------------------------------------------ helpers
    def show_toast(self, text: str, color=None, seconds: float = 3.0) -> None:
        self.toast.text, self.toast.color, self._toast_t = text, color or GREEN, seconds

    def clear(self) -> None:
        self.banner.text = self.banner_sub.text = self.toast.text = ""
        self._toast_t = 0.0

    # ------------------------------------------------------------------- update
    def update(self, world, mode: str, dt: float, collision_flash: float, fps: float) -> None:
        d = world.drone
        n, found = len(world.victims), world.discovered_count
        nv, dist = world.nearest_victim()
        fp = world.fire_proximity()
        dist_txt = "all found" if nv is None else f"{dist:5.2f} m  (ground truth, {nv.id})"
        fire_txt = "none" if fp.fire_id is None else f"{fp.level}  ({fp.fire_id} at {fp.distance:4.1f} m)"
        mm, ss = divmod(int(world.time), 60)
        self.info.text = (
            f"Mode:      {mode}\n"
            f"Position:  x {d.pos[0]:6.2f}   z {d.pos[2]:6.2f}   altitude {d.pos[1]:4.2f} m\n"
            f"Velocity:  vx {d.vel[0]:5.2f}  vy {d.vel[1]:5.2f}  vz {d.vel[2]:5.2f}   |v| {d.speed:4.2f} m/s   yaw {d.yaw:5.1f}\n"
            f"Victims discovered: {found} / {n}     Nearest undiscovered: {dist_txt}\n"
            f"Fire proximity: {fire_txt}     heat exposure {world.fire_exposure:3.1f}/{world.config.mission.fire_exposure_limit_s:.0f} s\n"
            f"Collisions: {world.collision_count}     mission time {mm:02d}:{ss:02d}     {fps:3.0f} fps")
        self.status.text = f"MISSION: {world.status.value}"
        self.status.color = GREEN if world.status.name in ("RUNNING", "SUCCESS") else RED
        label = "Mode: AUTO (M)" if mode.startswith("AUTO") else "Mode: MANUAL (M)"
        if self.mode_btn.text != label:
            self.mode_btn.text = label

        # warnings
        if fp.level == DANGER:
            self.warn.text, self.warn.color = "!! FIRE ZONE - DANGER !!", RED
        elif fp.level != CLEAR:
            self.warn.text, self.warn.color = "FIRE ZONE PROXIMITY WARNING", ORANGE
        elif world.collision_active:
            self.warn.text, self.warn.color = "COLLISION", RED
        else:
            self.warn.text = ""
        self.vignette.color = Color(1, 0, 0, 0.30 * collision_flash)

        if world.status.name != "RUNNING":
            ok = world.status.name == "SUCCESS"
            self.banner.text = "MISSION COMPLETE" if ok else "MISSION FAILED"
            self.banner.color = GREEN if ok else RED
            self.banner_sub.text = world.status.value + "   -   press R to restart"
        self._toast_t = max(0.0, self._toast_t - dt)
        if self._toast_t == 0.0:
            self.toast.text = ""

        # thermal readout
        th = world.thermal
        peak = "-" if th.peak_victim_id is None else f"{th.peak_signal:.2f} ({th.peak_victim_id})"
        self.th_txt.text = (f"peak victim signal {peak}\nfire interference {th.interference:.2f}   "
                            f"threshold {world.config.sensor.detection_threshold:.2f}\n"
                            f"view: top-down, north up, range {world.config.sensor.max_range:.0f} m")
        self._tex_timer -= dt
        if self._tex_timer <= 0:
            self._tex_timer = 0.2
            arr = heat_to_rgb(world.heat_grid(32), d.yaw)
            tex = Texture(Image.fromarray(arr, "RGB"), filtering=None)
            self.th_img.texture = tex
