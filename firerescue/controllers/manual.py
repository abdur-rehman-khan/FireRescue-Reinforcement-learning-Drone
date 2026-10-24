"""Keyboard -> DroneCommand mapping (pure function, engine independent, unit-testable)."""
from __future__ import annotations

from typing import Mapping

from ..drone import DroneCommand

KEY_HELP = ("W/S forward/back   A/D left/right   SPACE/LSHIFT up/down   Q/E yaw left/right")


def keys_to_command(held: Mapping[str, float]) -> DroneCommand:
    def axis(pos: str, neg: str) -> float:
        return (1.0 if held.get(pos) else 0.0) - (1.0 if held.get(neg) else 0.0)

    return DroneCommand(
        forward=axis("w", "s"),
        right=axis("d", "a"),
        up=axis("space", "left shift"),
        yaw=axis("e", "q"),
    )
