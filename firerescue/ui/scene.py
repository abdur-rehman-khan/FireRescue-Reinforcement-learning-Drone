"""Static building geometry: floors, walls, doors, windows, furniture, debris, labels."""
from __future__ import annotations

from typing import List

from ursina import Entity, Text, scene

from ..config import WindowConfig
from .common import glow, rgb, solid


def build_static_scene(world) -> List[Entity]:
    cfg = world.config.building
    ents: List[Entity] = []

    # outdoor ground
    ents.append(solid("cube", (0.30, 0.34, 0.30), position=(cfg.size_x / 2, -0.30, cfg.size_z / 2),
                      scale=(cfg.size_x + 30, 0.5, cfg.size_z + 30)))
    # room floors
    for r in cfg.rooms:
        (x0, z0), (x1, z1) = r.min_xz, r.max_xz
        ents.append(solid("cube", r.floor_color, position=((x0 + x1) / 2, -0.025, (z0 + z1) / 2),
                          scale=(x1 - x0 - 0.02, 0.05, z1 - z0 - 0.02)))
        t = Text(r.name.upper(), parent=scene, position=((x0 + x1) / 2, 0.012, (z0 + z1) / 2 - 0.6),
                 rotation=(90, 0, 0), scale=14, origin=(0, 0), color=rgb((0.25, 0.25, 0.28), 0.8))
        ents.append(t)

    # every solid obstacle: walls, lintels, open door leaves, furniture, debris
    for o in world.obstacle_list:
        ents.append(solid("cube", o.color, position=o.center, scale=o.size))

    # windows (visual glass panes set into the exterior walls)
    for w in cfg.windows:
        ents.extend(_window(w, cfg))
    return ents


def _window(w: WindowConfig, cfg) -> List[Entity]:
    t = cfg.wall_thickness
    cy = w.sill + w.height / 2
    if w.wall in ("south", "north"):
        z = 0.0 if w.wall == "south" else cfg.size_z
        frame_pos, frame_scale = (w.offset, cy, z), (w.width + 0.14, w.height + 0.14, t + 0.02)
        glass_scale = (w.width, w.height, t + 0.06)
    else:
        x = 0.0 if w.wall == "west" else cfg.size_x
        frame_pos, frame_scale = (x, cy, w.offset), (t + 0.02, w.height + 0.14, w.width + 0.14)
        glass_scale = (t + 0.06, w.height, w.width)
    frame = solid("cube", (0.25, 0.22, 0.2), position=frame_pos, scale=frame_scale)
    glass = glow("cube", (0.55, 0.75, 0.95), 0.40, position=frame_pos, scale=glass_scale)
    return [frame, glass]
