import numpy as np

from firerescue import DroneCommand, MissionStatus, SimWorld, default_scenario
from firerescue.controllers import BaselineController, keys_to_command
from firerescue.controllers.planner import GridPlanner


def test_keys_to_command_mapping():
    assert keys_to_command({}) == DroneCommand()
    c = keys_to_command({"w": 1, "d": 1, "space": 1, "e": 1})
    assert (c.forward, c.right, c.up, c.yaw) == (1, 1, 1, 1)
    c = keys_to_command({"s": 1, "a": 1, "left shift": 1, "q": 1})
    assert (c.forward, c.right, c.up, c.yaw) == (-1, -1, -1, -1)
    assert keys_to_command({"w": 1, "s": 1}).forward == 0


def run_auto(max_time=400):
    w = SimWorld(default_scenario()); ctl = BaselineController(w)
    log = []
    while w.status == MissionStatus.RUNNING and w.time < max_time:
        w.step(ctl.command())
        log.append(tuple(np.round(w.drone.pos, 6)))
    return w, ctl, log


def test_baseline_controller_completes_default_mission_safely():
    w, ctl, _ = run_auto()
    assert w.status == MissionStatus.SUCCESS, w.status
    assert w.discovered_count == len(w.victims)
    assert w.collision_count == 0
    assert w.fire_exposure == 0.0
    assert ctl.stuck_dropped == 0


def test_baseline_controller_is_deterministic():
    a, b = run_auto(), run_auto()
    assert a[2] == b[2] and a[0].time == b[0].time


def test_planner_avoids_obstacles_and_fire():
    w = SimWorld(default_scenario())
    p = GridPlanner(w.obstacles, 20, 14, 1.2, 0.48)
    p.set_fire_zones(w.fires)
    res = p.plan((1.5, 7.0), (5.5, 1.3))             # corridor -> Room A through the doorway
    assert res is not None
    pts = [(1.5, 7.0)] + res[0]
    for a, b in zip(pts, pts[1:]):
        assert p.line_free(a, b)
    f = w.fires[0]
    for x, z in res[0]:
        assert np.hypot(x - f.position[0], z - f.position[2]) > f.danger_radius
    blocked = p.blocked
    assert blocked[p.cell((0.1, 0.1))] and not blocked[p.cell((1.5, 7.0))]


def test_controller_holds_altitude_when_idle():
    w = SimWorld(default_scenario()); ctl = BaselineController(w)
    ctl.todo.clear()
    for _ in range(300):
        w.step(ctl.command())
    assert abs(w.drone.pos[1] - ctl.cruise_altitude) < 0.05
