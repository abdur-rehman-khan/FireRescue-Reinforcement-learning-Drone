"""Typed, JSON-serialisable configuration for a FireRescue scenario.

Coordinate system (identical to Ursina so no conversion is needed):
    +x = right/east, +y = up, +z = forward/north.  Units are metres.
Yaw is in degrees about +y; yaw 0 faces +z, yaw 90 faces +x (clockwise seen from above).
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Color = Tuple[float, float, float]

FURNITURE_KINDS = ("table", "chair", "box", "beam", "rubble")
POSES = ("standing", "prone")
WALL_SIDES = ("south", "north", "west", "east")


class ConfigError(ValueError):
    """Raised when a scenario configuration is malformed or inconsistent."""


@dataclass
class DroneConfig:
    start_position: Vec3 = (1.5, 1.2, 7.0)
    start_yaw_deg: float = 90.0
    radius: float = 0.28                 # collision sphere (includes prop guards)
    max_speed_xz: float = 2.0            # m/s
    max_speed_y: float = 1.2             # m/s
    max_yaw_rate_deg: float = 90.0       # deg/s
    max_accel: float = 4.0               # m/s^2
    min_altitude: float = 0.15           # m (landed)
    max_altitude: float = 2.75           # m (below ceiling)


@dataclass
class SensorConfig:
    """Parameters of the SYNTHETIC thermal model (not a real thermal camera)."""
    max_range: float = 5.0
    victim_base_signal: float = 1.0
    reference_distance: float = 2.0      # signal = base / (1 + (d/ref)^2)
    prone_factor: float = 0.7            # prone victims present a smaller signature
    detection_threshold: float = 0.2
    fire_interference_gain: float = 0.25
    noise_std: float = 0.0               # 0 => fully deterministic
    seed: int = 0


@dataclass
class FireConfig:
    id: str
    position: Vec3
    danger_radius: float
    intensity: float = 1.0


@dataclass
class VictimConfig:
    id: str
    position: Vec3                       # feet position (y normally 0)
    pose: str = "standing"


@dataclass
class RoomConfig:
    name: str
    min_xz: Vec2
    max_xz: Vec2
    kind: str = "room"                   # "room" (searched) or "corridor"
    floor_color: Color = (0.7, 0.7, 0.7)


@dataclass
class WallConfig:
    axis: str                            # "x": wall runs along x at z=fixed; "z": runs along z at x=fixed
    fixed: float
    start: float
    end: float
    gaps: List[Vec2] = field(default_factory=list)   # doorways: (centre, width)
    exterior: bool = False
    leaf_dir: int = -1                   # side (-1/+1) toward which the open door leaf swings


@dataclass
class WindowConfig:
    wall: str                            # south | north | west | east (exterior only, visual glass pane)
    offset: float                        # position along that wall
    width: float = 1.6
    sill: float = 1.0
    height: float = 1.0


@dataclass
class FurnitureConfig:
    kind: str
    position: Vec3                       # base-centre of the piece (y = elevation of its underside)
    rotation_deg: float = 0.0            # multiples of 90 only (axis-aligned collision boxes)
    size: Optional[Vec3] = None
    name: str = ""


@dataclass
class BuildingConfig:
    size_x: float = 20.0
    size_z: float = 14.0
    height: float = 3.0
    wall_thickness: float = 0.2
    door_height: float = 2.1
    door_leaf_length: float = 0.9
    rooms: List[RoomConfig] = field(default_factory=list)
    walls: List[WallConfig] = field(default_factory=list)
    windows: List[WindowConfig] = field(default_factory=list)
    furniture: List[FurnitureConfig] = field(default_factory=list)


@dataclass
class MissionConfig:
    time_limit_s: float = 600.0
    fire_exposure_limit_s: float = 6.0   # cumulative seconds inside a danger zone before failure
    warning_factor: float = 1.5          # warning zone = factor * danger radius
    sim_dt: float = 1.0 / 60.0


@dataclass
class ScenarioConfig:
    name: str = "scenario"
    building: BuildingConfig = field(default_factory=BuildingConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    fires: List[FireConfig] = field(default_factory=list)
    victims: List[VictimConfig] = field(default_factory=list)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioConfig":
        cfg = dataclass_from_dict(cls, data)
        validate_scenario(cfg)
        return cfg

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "ScenarioConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
def _convert(tp, value):
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return dataclass_from_dict(tp, value)
    if origin is typing.Union:
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)][0]
        return _convert(inner, value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"Expected a list, got {type(value).__name__}")
        return [_convert(args[0], v) for v in value]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"Expected a sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        if len(args) != len(value):
            raise ConfigError(f"Expected {len(args)} values, got {len(value)}")
        return tuple(_convert(a, v) for a, v in zip(args, value))
    if tp is float:
        return float(value)
    if tp is int:
        return int(value)
    return value


def dataclass_from_dict(cls, data):
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} expects an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    try:
        kwargs = {k: _convert(hints[k], v) for k, v in data.items()}
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Bad {cls.__name__}: {exc}") from exc


def validate_scenario(cfg: ScenarioConfig) -> None:
    """Raise ConfigError listing every problem found."""
    e: List[str] = []
    b, d, s, m = cfg.building, cfg.drone, cfg.sensor, cfg.mission

    def inside(p, margin=0.0):
        return (margin <= p[0] <= b.size_x - margin and margin <= p[2] <= b.size_z - margin
                and 0.0 <= p[1] <= b.height)

    if min(b.size_x, b.size_z, b.height, b.wall_thickness, b.door_leaf_length) <= 0:
        e.append("building dimensions must be positive")
    if not 0 < b.door_height < b.height:
        e.append("door_height must be between 0 and building height")
    for i, w in enumerate(b.walls):
        if w.axis not in ("x", "z"):
            e.append(f"wall[{i}].axis must be 'x' or 'z'")
        if w.end <= w.start:
            e.append(f"wall[{i}] end must be greater than start")
        if w.leaf_dir not in (-1, 1):
            e.append(f"wall[{i}].leaf_dir must be -1 or 1")
        for c, width in w.gaps:
            if width <= 0 or c - width / 2 < w.start or c + width / 2 > w.end:
                e.append(f"wall[{i}] has an invalid doorway ({c}, {width})")
    for i, win in enumerate(b.windows):
        if win.wall not in WALL_SIDES:
            e.append(f"window[{i}].wall must be one of {WALL_SIDES}")
        if min(win.width, win.height) <= 0:
            e.append(f"window[{i}] size must be positive")
    for i, f in enumerate(b.furniture):
        if f.kind not in FURNITURE_KINDS:
            e.append(f"furniture[{i}].kind must be one of {FURNITURE_KINDS}")
        if f.rotation_deg % 90 != 0:
            e.append(f"furniture[{i}].rotation_deg must be a multiple of 90")
        if f.size is not None and min(f.size) <= 0:
            e.append(f"furniture[{i}].size must be positive")
    for r in b.rooms:
        if r.kind not in ("room", "corridor"):
            e.append(f"room '{r.name}' kind must be 'room' or 'corridor'")
        if r.max_xz[0] <= r.min_xz[0] or r.max_xz[1] <= r.min_xz[1]:
            e.append(f"room '{r.name}' has non-positive extent")

    if d.radius <= 0 or min(d.max_speed_xz, d.max_speed_y, d.max_yaw_rate_deg, d.max_accel) <= 0:
        e.append("drone radius and limits must be positive")
    if not 0 <= d.min_altitude < d.max_altitude <= b.height:
        e.append("drone altitude limits must satisfy 0 <= min < max <= building height")
    sp = d.start_position
    if not inside(sp, d.radius):
        e.append("drone start_position must be inside the building")
    elif not d.min_altitude <= sp[1] <= d.max_altitude:
        e.append("drone start altitude must be within altitude limits")

    if s.max_range <= 0 or s.reference_distance <= 0 or s.detection_threshold < 0:
        e.append("sensor range/reference distance must be positive and threshold >= 0")
    if not 0 < s.prone_factor <= 1:
        e.append("sensor.prone_factor must be in (0, 1]")
    if s.fire_interference_gain < 0 or s.noise_std < 0:
        e.append("sensor gain and noise must be >= 0")

    ids = [f.id for f in cfg.fires]
    if len(ids) != len(set(ids)):
        e.append("fire ids must be unique")
    for f in cfg.fires:
        if f.danger_radius <= 0 or f.intensity <= 0:
            e.append(f"fire '{f.id}' radius/intensity must be positive")
        if not inside(f.position):
            e.append(f"fire '{f.id}' lies outside the building")
    ids = [v.id for v in cfg.victims]
    if not ids:
        e.append("at least one victim is required")
    if len(ids) != len(set(ids)):
        e.append("victim ids must be unique")
    for v in cfg.victims:
        if v.pose not in POSES:
            e.append(f"victim '{v.id}' pose must be one of {POSES}")
        if not inside(v.position):
            e.append(f"victim '{v.id}' lies outside the building")

    if m.time_limit_s <= 0 or m.fire_exposure_limit_s <= 0 or m.sim_dt <= 0 or m.warning_factor < 1:
        e.append("mission limits must be positive and warning_factor >= 1")
    if e:
        raise ConfigError("Invalid scenario: " + "; ".join(e))
