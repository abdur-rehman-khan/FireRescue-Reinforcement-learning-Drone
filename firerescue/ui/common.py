"""Shared helpers for the Ursina front-end."""
from __future__ import annotations

from ursina import Color, Entity

LIGHTING = {"shader": None, "gain": 1.0}


def init_lighting(enabled: bool = True) -> None:
    """Select the shader used for solid geometry.

    `basic_lighting_shader` ships with Ursina and gives simple directional face shading.
    ``--flat`` falls back to Ursina's default unlit shader (useful on very old GPUs).
    """
    if enabled:
        from ursina.shaders import basic_lighting_shader
        LIGHTING["shader"], LIGHTING["gain"] = basic_lighting_shader, 1.2
    else:
        LIGHTING["shader"], LIGHTING["gain"] = None, 1.0


def rgb(c, a: float = 1.0, gain: float = 1.0) -> Color:
    return Color(min(1.0, c[0] * gain), min(1.0, c[1] * gain), min(1.0, c[2] * gain), a)


def solid(model="cube", color=(0.7, 0.7, 0.7), **kw) -> Entity:
    """A lit, opaque scene entity."""
    g = LIGHTING["gain"]
    if LIGHTING["shader"] is not None:
        kw.setdefault("shader", LIGHTING["shader"])
    return Entity(model=model, color=rgb(color, 1.0, g), **kw)


def glow(model="cube", color=(1, 0.5, 0.1), alpha=1.0, **kw) -> Entity:
    """A self-illuminated / translucent scene entity (always uses the unlit default shader)."""
    return Entity(model=model, color=rgb(color, alpha), **kw)
