import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firerescue import SimWorld, default_scenario, DroneCommand  # noqa: E402


@pytest.fixture
def cfg():
    return default_scenario()


@pytest.fixture
def world():
    return SimWorld(default_scenario())


def run(world, command, seconds):
    """Step the world with a constant command for `seconds` of simulated time."""
    n = int(round(seconds / world.config.mission.sim_dt))
    for _ in range(n):
        world.step(command)


def place(world, x, y, z, yaw=None):
    """Teleport the drone (test helper) and refresh sensing."""
    world.drone.pos = np.array([x, y, z], dtype=float)
    world.drone.vel = np.zeros(3)
    if yaw is not None:
        world.drone.yaw = yaw
    world._sense()


def single_scenario(victim_pos=(5.0, 0.0, 5.0), pose="standing", fire=None, **sensor):
    """Open 10x10 room, no furniture, one victim and an optional fire."""
    from firerescue.config import (BuildingConfig, DroneConfig, FireConfig, MissionConfig,
                                   ScenarioConfig, SensorConfig, VictimConfig, WallConfig)
    walls = [WallConfig("x", 0.0, -0.1, 10.1, exterior=True), WallConfig("x", 10.0, -0.1, 10.1, exterior=True),
             WallConfig("z", 0.0, 0.0, 10.0, exterior=True), WallConfig("z", 10.0, 0.0, 10.0, exterior=True)]
    b = BuildingConfig(size_x=10, size_z=10, walls=walls)
    fires = [FireConfig("F", fire[0], fire[1], fire[2] if len(fire) > 2 else 1.0)] if fire else []
    return ScenarioConfig(name="single", building=b, drone=DroneConfig(start_position=(2.0, 1.2, 2.0)),
                          sensor=SensorConfig(**sensor), mission=MissionConfig(), fires=fires,
                          victims=[VictimConfig("V1", victim_pos, pose)])
