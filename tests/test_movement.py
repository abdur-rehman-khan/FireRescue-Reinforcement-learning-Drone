import numpy as np
import pytest

from firerescue import DroneCommand, SimWorld, default_scenario
from conftest import place, run, single_scenario


def open_world():
    return SimWorld(single_scenario())


def test_start_state_matches_config(world):
    assert tuple(world.drone.pos) == world.config.drone.start_position
    assert world.drone.yaw == world.config.drone.start_yaw_deg
    assert world.drone.speed == 0


def test_forward_moves_along_heading():
    w = open_world(); w.drone.yaw = 90.0     # facing +x
    p0 = w.drone.pos.copy()
    run(w, DroneCommand(forward=1), 1.0)
    d = w.drone.pos - p0
    assert d[0] > 0.5 and abs(d[2]) < 1e-6 and abs(d[1]) < 1e-9


def test_backward_left_right_up_down():
    w = open_world(); w.drone.yaw = 0.0      # facing +z, right = +x
    p0 = w.drone.pos.copy()
    run(w, DroneCommand(forward=-1), 0.5); assert w.drone.pos[2] < p0[2]
    p1 = w.drone.pos.copy()
    run(w, DroneCommand(right=1), 0.5); assert w.drone.pos[0] > p1[0]
    p2 = w.drone.pos.copy()
    run(w, DroneCommand(right=-1), 1.0); assert w.drone.pos[0] < p2[0]
    y0 = w.drone.pos[1]
    run(w, DroneCommand(up=1), 0.5); assert w.drone.pos[1] > y0
    y1 = w.drone.pos[1]
    run(w, DroneCommand(up=-1), 1.5); assert w.drone.pos[1] < y1   # inertia first, then descent


def test_yaw_rotation_rate_and_direction():
    w = open_world(); w.drone.yaw = 0.0
    run(w, DroneCommand(yaw=1), 1.0)         # +yaw = turn right
    assert w.drone.yaw == pytest.approx(w.config.drone.max_yaw_rate_deg, abs=0.5)
    run(w, DroneCommand(yaw=-1), 2.0)
    assert w.drone.yaw == pytest.approx(360 - w.config.drone.max_yaw_rate_deg, abs=0.5)


def test_speed_limits_and_acceleration_limit():
    w = open_world()
    w.step(DroneCommand(forward=1))
    assert w.drone.speed <= w.config.drone.max_accel * w.config.mission.sim_dt + 1e-9
    run(w, DroneCommand(forward=1, right=1, up=1), 3.0)    # diagonal must not exceed the limit
    v = w.drone.vel
    assert np.hypot(v[0], v[2]) <= w.config.drone.max_speed_xz + 1e-6
    assert v[1] <= w.config.drone.max_speed_y + 1e-6


def test_command_values_are_clipped():
    w = open_world(); w.drone.yaw = 90.0
    run(w, DroneCommand(forward=50), 1.0)
    assert np.hypot(*w.drone.vel[[0, 2]]) <= w.config.drone.max_speed_xz + 1e-6


def test_altitude_limits():
    w = open_world()
    run(w, DroneCommand(up=1), 6.0)
    assert w.drone.pos[1] == pytest.approx(w.config.drone.max_altitude)
    run(w, DroneCommand(up=-1), 6.0)
    assert w.drone.pos[1] == pytest.approx(w.config.drone.min_altitude)
    assert w.drone.landed


def test_zero_command_decelerates_to_hover():
    w = open_world(); w.drone.yaw = 90
    run(w, DroneCommand(forward=1), 1.0)
    run(w, DroneCommand(), 1.0)
    assert w.drone.speed < 1e-6


def test_simulation_is_deterministic():
    def trajectory():
        w = SimWorld(default_scenario())
        for i in range(600):
            w.step(DroneCommand(forward=np.sin(i / 50), right=np.cos(i / 70), up=np.sin(i / 90), yaw=np.cos(i / 40)))
        return w.drone.pos.copy(), w.drone.yaw, w.collision_count
    a, b = trajectory(), trajectory()
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_large_dt_is_substepped_without_tunnelling(world):
    place(world, 1.5, 1.2, 6.5, yaw=180)          # facing the corridor wall at z=5.5
    for _ in range(10):
        world.step(DroneCommand(forward=1), dt=0.25)
    assert world.drone.pos[2] > 5.6
