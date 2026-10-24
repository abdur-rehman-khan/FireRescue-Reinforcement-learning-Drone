import numpy as np
import pytest

from firerescue import DroneCommand, SimWorld, default_scenario
from firerescue.collision import ObstacleSet
from firerescue.obstacles import Obstacle
from conftest import place, run


def box(c, s):
    return Obstacle("b", "box", c, s, (1, 1, 1))


def test_sphere_contact_and_pushout():
    s = ObstacleSet([box((0, 0, 0), (2, 2, 2))])
    assert not s.sphere_collides([2.0, 0, 0], 0.5)
    assert s.sphere_collides([1.3, 0, 0], 0.5)
    pos, vel, hits, impact = s.resolve_sphere(np.array([1.3, 0, 0]), np.array([-1.0, 0, 0]), 0.5)
    assert pos[0] == pytest.approx(1.5, abs=1e-5) and vel[0] == 0 and impact == pytest.approx(1.0)


def test_centre_inside_box_is_pushed_out():
    s = ObstacleSet([box((0, 0, 0), (2, 2, 2))])
    pos, _, hits, _ = s.resolve_sphere(np.array([0.9, 0.0, 0.0]), np.zeros(3), 0.3)
    assert hits and pos[0] >= 1.3 - 1e-5


def test_segment_blocked():
    s = ObstacleSet([box((0, 0, 0), (2, 2, 2))])
    assert s.segment_blocked([-3, 0, 0], [3, 0, 0])
    assert not s.segment_blocked([-3, 3, 0], [3, 3, 0])
    assert not s.segment_blocked([-3, 0, 0], [-1.5, 0, 0])     # stops short
    assert s.segment_blocked([0, 0, -3], [0, 0, 3])            # axis-parallel segment


def test_drone_cannot_pass_exterior_or_interior_walls(world):
    place(world, 1.5, 1.2, 7.0, yaw=180)       # facing south toward the solid part of the corridor wall
    run(world, DroneCommand(forward=1), 5.0)
    assert world.drone.pos[2] >= 5.5 + 0.1 + world.config.drone.radius - 1e-3
    assert world.collision_count == 1 and world.collision_active
    assert any(n.startswith("wall") for n in world.drone.last_collision.names)
    place(world, 1.5, 1.2, 7.0, yaw=270)       # west toward exterior wall
    run(world, DroneCommand(forward=1), 5.0)
    assert world.drone.pos[0] >= 0.1 + world.config.drone.radius - 1e-3


def test_sliding_along_wall_keeps_moving(world):
    place(world, 1.5, 1.2, 6.2, yaw=180)
    run(world, DroneCommand(forward=1, right=-1), 1.5)       # push into wall while strafing east
    assert world.drone.pos[0] > 2.0 or world.drone.pos[0] < 1.0


def test_doorway_is_passable_at_normal_altitude(world):
    place(world, 3.5, 1.2, 7.0, yaw=180)       # in front of the door at x=3.5, z=5.5
    n0 = world.collision_count
    run(world, DroneCommand(forward=0.6), 4.0)
    assert world.drone.pos[2] < 5.0, "drone should have flown through the doorway into Room A"
    assert world.collision_count == n0


def test_door_lintel_blocks_high_flight(world):
    place(world, 3.5, 2.7, 7.0, yaw=180)       # altitude above the 2.1 m door height
    run(world, DroneCommand(forward=0.6), 4.0)
    assert world.drone.pos[2] > 5.5 and world.collision_count >= 1


def test_ceiling_collision_feedback(world):
    run(world, DroneCommand(up=1), 3.0)
    assert world.drone.pos[1] == pytest.approx(world.config.drone.max_altitude)
    assert "ceiling" in world.drone.last_collision.names and world.collision_count >= 1


def test_furniture_and_debris_are_solid(world):
    # box in the corridor at (6.5, 0..0.6, 8.0): fly into it at low altitude, from the south
    place(world, 6.5, 0.3, 7.0, yaw=0)
    run(world, DroneCommand(forward=1), 3.0)
    assert world.drone.pos[2] < 8.0 - 0.3 + 1e-3 and world.collision_count == 1
    # the same box can be flown over
    place(world, 6.5, 1.3, 6.5, yaw=0); n = world.collision_count
    run(world, DroneCommand(forward=0.5), 1.5)
    assert world.drone.pos[2] > 7.8 and world.collision_count == n    # now above the box, no contact


def test_table_top_and_legs(world):
    # table A top at y=0.69..0.75, centred (4.6, 1.5); fly through at y=0.4 hits leg/top region? fly above ok
    place(world, 4.6, 1.3, 0.6, yaw=0)
    n = world.collision_count
    run(world, DroneCommand(forward=0.5), 3.0)
    assert world.drone.pos[2] > 2.3 and world.collision_count == n   # passed over the table
    place(world, 4.6, 0.7, 0.6, yaw=0)
    run(world, DroneCommand(forward=0.5), 3.0)
    assert world.drone.pos[2] < 1.5 and world.collision_count > n    # table top is solid


def test_hanging_beam_blocks_mid_height_not_low(world):
    place(world, 13.0, 1.8, 7.4, yaw=90); n = world.collision_count
    run(world, DroneCommand(forward=0.8), 3.0)
    assert world.drone.pos[0] < 14.9 and world.collision_count > n
    place(world, 13.0, 1.0, 7.4, yaw=90); n = world.collision_count
    run(world, DroneCommand(forward=0.8), 3.0)
    assert world.drone.pos[0] > 15.5 and world.collision_count == n


def test_random_flight_never_penetrates_solids():
    w = SimWorld(default_scenario())
    rng = np.random.default_rng(1234)
    cmd = DroneCommand()
    for i in range(6000):
        if i % 40 == 0:
            cmd = DroneCommand(*rng.uniform(-1, 1, 4))
        w.step(cmd)
        assert not w.obstacles.sphere_contacts(w.drone.pos, w.config.drone.radius - 1e-3), f"penetration at step {i}"
        p = w.drone.pos
        assert 0 < p[0] < 20 and 0 < p[2] < 14 and 0 <= p[1] <= 3.0
