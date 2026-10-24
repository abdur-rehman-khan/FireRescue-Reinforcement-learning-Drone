"""Animated actors: quadrotor, fires (+smoke), victims."""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np
from ursina import Circle, Cone, Cylinder, Entity, Text, Vec3, scene

from .common import LIGHTING, glow, rgb, solid


# =============================================================================== drone
class DroneActor(Entity):
    """Recognisable quadrotor: body, 4 arms/motors/propellers, 4 protective guard rings,
    front camera pod, top thermal-sensor mount, belly sensor and landing legs."""

    MOTORS = [(0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)]   # (x, z) in body frame

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.flash = 0.0
        self.t = 0.0
        body_col = (0.16, 0.17, 0.2)
        self.body = solid("cube", body_col, parent=self, scale=(0.20, 0.06, 0.20))
        solid("cube", (0.15, 0.35, 0.75), parent=self, y=0.04, scale=(0.13, 0.03, 0.13))   # top shell
        solid("cube", (0.1, 0.1, 0.12), parent=self, y=-0.05, scale=(0.12, 0.03, 0.12))    # battery
        for ang in (45, -45):                                                              # arms
            solid("cube", (0.1, 0.1, 0.12), parent=self, y=0.0, rotation_y=ang, scale=(0.44, 0.02, 0.03))
        self.props: List[Entity] = []
        for i, (mx, mz) in enumerate(self.MOTORS):
            solid(Cylinder(resolution=10, radius=0.5, height=1), (0.08, 0.08, 0.09), parent=self,
                  position=(mx, 0.0, mz), scale=(0.05, 0.05, 0.05))
            spinner = Entity(parent=self, position=(mx, 0.058, mz))
            for a in (0, 90):
                glow("cube", (0.85, 0.85, 0.9), 0.95, parent=spinner, rotation_y=a, scale=(0.16, 0.004, 0.018))
            glow("circle", (0.8, 0.8, 0.85), 0.16, parent=spinner, rotation_x=90, scale=0.17)
            self.props.append(spinner)
            # protective guard ring (frame)
            glow(Circle(resolution=20, radius=0.5, mode='line', thickness=3), (0.95, 0.55, 0.1), 1.0,
                 parent=self, position=(mx, 0.058, mz), rotation_x=90, scale=0.21)
            # motor LED (front = red, rear = green)
            glow("sphere", (1, 0.1, 0.1) if mz > 0 else (0.1, 1, 0.2), 1.0, parent=self,
                 position=(mx * 1.12, 0.01, mz * 1.12), scale=0.018)
        # sensor mounts
        solid("cube", (0.1, 0.1, 0.1), parent=self, position=(0, -0.01, 0.12), scale=(0.07, 0.05, 0.05))   # camera pod
        glow("sphere", (0.2, 0.5, 1.0), 1.0, parent=self, position=(0, -0.01, 0.15), scale=0.03)            # lens
        solid("cube", (0.3, 0.3, 0.3), parent=self, position=(0, 0.075, -0.02), scale=(0.012, 0.05, 0.012))  # stalk
        solid("cube", (0.95, 0.45, 0.1), parent=self, position=(0, 0.11, -0.02), scale=(0.085, 0.035, 0.06))  # thermal head
        glow("sphere", (0.1, 0.1, 0.1), 1.0, parent=self, position=(0, 0.11, 0.012), scale=0.026)            # thermal window
        solid(Cylinder(resolution=8, radius=0.5, height=1), (0.2, 0.2, 0.25), parent=self,
              position=(0, -0.09, 0), scale=(0.04, 0.03, 0.04))                                              # belly sensor
        for sx in (-1, 1):                                                                                   # landing legs
            for sz in (-1, 1):
                solid("cube", (0.12, 0.12, 0.14), parent=self, position=(0.07 * sx, -0.07, 0.07 * sz),
                      scale=(0.012, 0.07, 0.012))
        # ground shadow blob + collision spark
        self.shadow = glow("circle", (0, 0, 0), 0.35, rotation_x=90, scale=0.5)
        self.spark = glow("sphere", (1, 0.15, 0.1), 0.0, scale=0.05)

    def trigger_collision(self) -> None:
        self.flash = 1.0

    def sync(self, dt: float) -> None:
        d = self.world.drone
        self.position = Vec3(*d.pos)
        self.rotation_y = d.yaw
        # slight tilt into the direction of motion (cosmetic)
        local_f = float(np.dot(d.vel, d.forward_vector())) / max(d.cfg.max_speed_xz, 1e-6)
        local_r = float(np.dot(d.vel, d.right_vector())) / max(d.cfg.max_speed_xz, 1e-6)
        self.rotation_x = 12.0 * local_f
        self.rotation_z = -12.0 * local_r
        self.t += dt
        spin = 0 if d.landed and d.speed < 1e-3 else 2400.0 * dt
        for i, p in enumerate(self.props):
            p.rotation_y += spin * (1 if i in (0, 3) else -1)
        self.shadow.position = Vec3(d.pos[0], 0.02, d.pos[2])
        self.shadow.scale = 0.28 + 0.12 * min(d.pos[1], 2.0)
        self.flash = max(0.0, self.flash - dt * 2.5)
        self.body.color = rgb((0.16 + 0.8 * self.flash, 0.17 * (1 - self.flash), 0.2 * (1 - self.flash)), 1.0,
                              LIGHTING["gain"])
        c = self.world.drone.last_collision
        if self.flash > 0.01:
            self.spark.alpha = self.flash
            self.spark.position = self.position - Vec3(*(c.normal * 0.28)) if np.linalg.norm(c.normal) > 0 else self.position
            self.spark.scale = 0.06 + 0.1 * (1 - self.flash)
        else:
            self.spark.alpha = 0.0


# =================================================================================== fire
class FireActor:
    """Flickering flames, ember glow, danger-radius disc/ring and rising smoke particles."""

    SMOKE = 12

    def __init__(self, fire, index: int):
        self.fire = fire
        self.rng = random.Random(1000 + index)          # deterministic visuals
        p = fire.position
        base = Vec3(p[0], 0.0, p[2])
        self.root = Entity(position=base)
        r = fire.danger_radius
        glow("circle", (0.9, 0.12, 0.05), 0.16, parent=self.root, y=0.03, rotation_x=90, scale=2 * r)
        glow(Circle(resolution=48, radius=0.5, mode='line', thickness=3), (1, 0.25, 0.1), 0.9,
             parent=self.root, y=0.04, rotation_x=90, scale=2 * r)
        glow("circle", (0.07, 0.05, 0.04), 0.9, parent=self.root, y=0.03, rotation_x=90, scale=1.3)
        self.flames: List[tuple] = []
        palette = [((1.0, 0.18, 0.04), 1.0), ((1.0, 0.45, 0.05), 0.9), ((1.0, 0.8, 0.15), 0.65),
                   ((1.0, 0.3, 0.05), 0.85), ((1.0, 0.55, 0.08), 0.75), ((1.0, 0.9, 0.35), 0.45)]
        glow("sphere", (1.0, 0.45, 0.1), 0.22, parent=self.root, y=0.5, scale=1.9)           # heat glow
        for i, (col, hs) in enumerate(palette):
            ang = self.rng.uniform(0, 6.28)
            off = Vec3(math.cos(ang), 0, math.sin(ang)) * (0.0 if i in (2, 5) else self.rng.uniform(0.2, 0.5))
            e = glow(Cone(resolution=7, radius=0.5, height=1), col, 0.95, parent=self.root, position=off + Vec3(0, 0.02, 0),
                     scale=(0.9 * hs, 1.7 * hs, 0.9 * hs))
            self.flames.append((e, self.rng.uniform(0, 6.28), self.rng.uniform(5, 9), hs))
        self.smoke: List[tuple] = []
        for k in range(self.SMOKE):
            e = glow("circle", (0.42, 0.42, 0.45), 0.2, billboard=True, position=base + Vec3(0, 1.0, 0), scale=0.3)
            self.smoke.append((e, k / self.SMOKE, self.rng.uniform(-0.15, 0.15), self.rng.uniform(-0.15, 0.15)))
        self.t = self.rng.uniform(0, 10)

    def sync(self, dt: float) -> None:
        self.t += dt
        for e, ph, sp, hs in self.flames:
            f = 1.0 + 0.22 * math.sin(self.t * sp + ph) + 0.1 * math.sin(self.t * sp * 2.3 + ph * 1.7)
            e.scale = (0.9 * hs * (2 - f), 1.7 * hs * f, 0.9 * hs * (2 - f))
            e.rotation_y = math.degrees(math.sin(self.t * 1.3 + ph)) * 0.5
        p = self.fire.position
        for e, ph, dx, dz in self.smoke:
            age = (self.t * 0.18 + ph) % 1.0                      # 0..1 over a particle's life
            e.position = Vec3(p[0] + dx * 6 * age + 0.3 * math.sin(self.t + ph * 9) * age,
                              1.5 + age * 1.3, p[2] + dz * 6 * age)
            e.scale = 0.4 + 0.9 * age
            e.alpha = 0.26 * math.sin(math.pi * min(1.0, age * 1.1)) ** 1.3


# ================================================================================= victims
class VictimActor:
    SHIRTS = [(0.2, 0.45, 0.85), (0.85, 0.3, 0.3), (0.3, 0.7, 0.4), (0.85, 0.7, 0.2), (0.6, 0.35, 0.75), (0.2, 0.7, 0.7)]

    def __init__(self, victim, index: int):
        self.victim = victim
        p = victim.position
        self.root = Entity(position=Vec3(p[0], 0, p[2]), rotation_y=(index * 67 + 20) % 360)
        shirt = self.SHIRTS[index % len(self.SHIRTS)]
        skin, pants = (0.93, 0.76, 0.62), (0.15, 0.17, 0.3)
        fig = Entity(parent=self.root)
        solid("cube", pants, parent=fig, position=(-0.1, 0.4, 0), scale=(0.15, 0.8, 0.17))
        solid("cube", pants, parent=fig, position=(0.1, 0.4, 0), scale=(0.15, 0.8, 0.17))
        solid("cube", shirt, parent=fig, position=(0, 1.1, 0), scale=(0.42, 0.62, 0.24))
        solid("cube", shirt, parent=fig, position=(-0.28, 1.1, 0), scale=(0.11, 0.58, 0.13))
        solid("cube", shirt, parent=fig, position=(0.28, 1.1, 0), scale=(0.11, 0.58, 0.13))
        solid("sphere", skin, parent=fig, position=(0, 1.58, 0), scale=0.26)
        if victim.pose == "prone":
            fig.rotation_x = 90                  # lie down (head toward local +z), thickness stays on the floor
            fig.y = 0.12
            fig.position = Vec3(0, 0.12, -0.8)
        self.ring = glow("circle", (1.0, 0.85, 0.1), 0.35, parent=self.root, y=0.03, rotation_x=90, scale=1.5)
        self.label = Text(f"{victim.id}", parent=scene, billboard=True, scale=9, origin=(0, 0),
                          position=Vec3(p[0], 1.95 if victim.pose == "standing" else 0.9, p[2]),
                          color=rgb((1, 0.85, 0.1)))
        self.beacon = glow("cube", (0.3, 1.0, 0.4), 0.0, position=Vec3(p[0], 1.5, p[2]), scale=(0.04, 3.0, 0.04))
        self._shown = False

    def sync(self, dt: float, t: float) -> None:
        v = self.victim
        if v.discovered and not self._shown:
            self._shown = True
            self.ring.color = rgb((0.2, 1.0, 0.35), 0.5)
            self.label.text = f"{v.id} FOUND"
            self.label.color = rgb((0.3, 1.0, 0.45))
            self.beacon.alpha = 0.45
        elif not v.discovered and self._shown:
            self._shown = False
            self.ring.color = rgb((1.0, 0.85, 0.1), 0.35)
            self.label.text = f"{v.id}"
            self.label.color = rgb((1, 0.85, 0.1))
            self.beacon.alpha = 0.0
        if self._shown:
            self.beacon.alpha = 0.3 + 0.15 * math.sin(t * 4)
