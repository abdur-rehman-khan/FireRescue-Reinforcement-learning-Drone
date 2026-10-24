import numpy as np
import pytest

from firerescue import DroneCommand, MissionStatus, SimWorld
from firerescue.config import FireConfig
from conftest import place, run, single_scenario


def test_reset_restores_everything(world):
    place(world, 2.2, 1.2, 4.0)
    run(world, DroneCommand(forward=1, yaw=1), 2.0)
    assert world.time > 0 and world.discovered_count >= 1
    world.reset()
    assert tuple(world.drone.pos) == world.config.drone.start_position
    assert world.drone.speed == 0 and world.drone.yaw == world.config.drone.start_yaw_deg
    assert world.time == 0 and world.collision_count == 0 and world.fire_exposure == 0
    assert world.status == MissionStatus.RUNNING
    assert world.discovered_count == 0 and all(v.discovered_at is None for v in world.victims)
    assert world.metrics.series["t"] == [] or world.metrics.series["t"][0] == 0


def test_reset_drone_keeps_mission_progress(world):
    place(world, 2.2, 1.2, 4.0)
    assert world.discovered_count >= 1
    world.step(DroneCommand(forward=1)); t = world.time
    world.reset_drone()
    assert tuple(world.drone.pos) == world.config.drone.start_position and world.drone.speed == 0
    assert world.discovered_count >= 1 and world.time == t and world.drone_reset_count == 1


def test_reset_is_repeatable(world):
    def go():
        world.reset()
        for i in range(300):
            world.step(DroneCommand(forward=0.7, yaw=0.3, up=np.sin(i / 30)))
        return world.drone.pos.copy(), world.collision_count
    a, b = go(), go()
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_victim_discovery_with_line_of_sight(world):
    v1 = world.victims[0]                      # standing victim V1 at (2.2, 0, 2.8) inside Room A
    place(world, 2.2, 1.2, 6.5)                # corridor, 3.7 m away but behind a solid wall
    assert not v1.discovered
    place(world, 2.2, 1.2, 4.2)                # inside Room A, ~1.7 m away
    assert v1.discovered and v1.discovered_at is not None
    assert [e["id"] for e in world.pop_events() if e["type"] == "victim_discovered"] == ["V1"]
    assert world.discovered_count == 1


def test_discovery_is_not_repeated_and_range_limited():
    w = SimWorld(single_scenario(victim_pos=(8.0, 0, 8.0)))
    place(w, 2.0, 1.2, 2.0)                    # ~8.5 m: beyond sensor range
    assert w.discovered_count == 0
    place(w, 6.0, 1.2, 6.0)                    # ~2.9 m
    assert w.discovered_count == 1
    w.pop_events(); place(w, 6.0, 1.2, 6.0)
    assert not [e for e in w.pop_events() if e["type"] == "victim_discovered"]


def test_prone_victims_are_harder_to_detect():
    s = single_scenario(victim_pos=(5, 0, 5), pose="standing")
    p = single_scenario(victim_pos=(5, 0, 5), pose="prone")
    ws, wp = SimWorld(s), SimWorld(p)
    for w in (ws, wp):
        place(w, 5.0, 1.2, 8.3)                # horizontal 3.3 m
    assert ws.discovered_count == 1 and wp.discovered_count == 0
    place(wp, 5.0, 1.2, 7.0)
    assert wp.discovered_count == 1


def test_fire_interference_masks_victims():
    quiet = SimWorld(single_scenario(victim_pos=(5, 0, 5)))
    hot = SimWorld(single_scenario(victim_pos=(5, 0, 5), fire=((5.5, 0.4, 7.0), 1.5, 3.0)))
    for w in (quiet, hot):
        place(w, 5.0, 1.2, 8.3)                # same viewpoint, 3.3 m from the victim
    assert quiet.discovered_count == 1 and hot.discovered_count == 0


def test_nearest_undiscovered_distance(world):
    d0 = world.nearest_victim_distance()
    v, d = world.nearest_victim()
    assert d == pytest.approx(d0) and v is not None
    place(world, 2.2, 1.2, 4.2)
    assert world.victims[0].discovered
    assert world.nearest_victim()[0].id != "V1"
    for v in world.victims:
        v.discovered = True
    assert world.nearest_victim_distance() == float("inf")


def test_fire_proximity_levels(world):
    f = world.fires[0]                         # F1 radius 2.0
    place(world, f.position[0], f.position[1] + 1.0, f.position[2])
    assert world.fire_proximity().level == "DANGER"
    place(world, f.position[0] + 2.5, f.position[1], f.position[2])
    assert world.fire_proximity().level == "WARNING"
    place(world, 1.5, 1.2, 7.0)
    assert world.fire_proximity().level == "CLEAR"


def test_mission_success_and_freeze():
    w = SimWorld(single_scenario(victim_pos=(5, 0, 5)))
    place(w, 5.0, 1.2, 6.5)
    w.step(DroneCommand())
    assert w.status == MissionStatus.SUCCESS
    pos, t = w.drone.pos.copy(), w.time
    run(w, DroneCommand(forward=1), 1.0)       # commands are ignored after the mission ends
    assert w.time == t and np.allclose(w.drone.pos[[0, 2]], pos[[0, 2]], atol=0.05)


def test_mission_fails_in_fire_and_on_timeout():
    w = SimWorld(single_scenario(victim_pos=(9, 0, 9), fire=((5.0, 0.4, 5.0), 2.0)))
    place(w, 5.0, 1.0, 5.0)
    run(w, DroneCommand(), w.config.mission.fire_exposure_limit_s + 0.5)
    assert w.status == MissionStatus.FAILED_FIRE
    cfg = single_scenario(victim_pos=(9, 0, 9)); cfg.mission.time_limit_s = 1.0
    w2 = SimWorld(cfg)
    run(w2, DroneCommand(), 1.5)
    assert w2.status == MissionStatus.FAILED_TIME


def test_collision_event_logged_once_per_contact(world):
    place(world, 1.5, 1.2, 6.5, yaw=180)
    run(world, DroneCommand(forward=1), 3.0)
    ev = [e for e in world.pop_events() if e["type"] == "collision"]
    assert len(ev) == 1 and ev[0]["impact"] > 0
