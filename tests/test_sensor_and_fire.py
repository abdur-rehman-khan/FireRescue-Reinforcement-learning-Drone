import numpy as np
import pytest

from firerescue import SimWorld
from firerescue.fire import CLEAR, DANGER, WARNING, FireSource
from firerescue.config import FireConfig
from conftest import place, single_scenario


def test_signal_decreases_with_distance_and_pose():
    w = SimWorld(single_scenario())
    v = w.victims[0]
    s = w.sensor
    sig = [s.victim_signal(v, d) for d in (0.5, 1, 2, 3, 4, 5)]
    assert all(a > b for a, b in zip(sig, sig[1:]))
    assert s.victim_signal(v, 5.01) == 0.0
    v.pose = "prone"
    assert s.victim_signal(v, 2.0) < sig[2]


def test_fire_source_levels_and_heat():
    f = FireSource(FireConfig("F", (0, 0, 0), 2.0, 1.0))
    assert f.level((1, 0, 0)) == DANGER and f.level((2.5, 0, 0)) == WARNING and f.level((4, 0, 0)) == CLEAR
    assert f.heat_at((0, 0, 0)) == pytest.approx(1.0) and f.heat_at((1, 0, 0)) > f.heat_at((3, 0, 0))


def test_thermal_reading_fields_and_los():
    w = SimWorld(default_scenario_fn())
    place(w, 2.2, 1.2, 6.5)                    # behind wall
    r = w.thermal.victims[0]
    assert not r.line_of_sight and r.signal == 0
    place(w, 2.2, 1.2, 4.0)
    r = w.thermal.victims[0]
    assert r.line_of_sight and r.signal > 0.2 and r.detected
    assert w.thermal.peak_victim_id == "V1"


def default_scenario_fn():
    from firerescue import default_scenario
    return default_scenario()


def test_heat_grid_peaks_at_visible_victim_only():
    w = SimWorld(single_scenario(victim_pos=(5, 0, 6)))
    place(w, 5.0, 1.2, 4.0)
    g = w.heat_grid(32)
    assert g.shape == (32, 32)
    row, col = np.unravel_index(np.argmax(g), g.shape)
    ext, n = w.config.sensor.max_range, 32
    x = 5.0 - ext + (col + 0.5) / n * 2 * ext
    z = 4.0 - ext + (row + 0.5) / n * 2 * ext
    assert abs(x - 5.0) < 0.5 and abs(z - 6.0) < 0.5
    # behind a wall no heat is shown
    w2 = SimWorld(default_scenario_fn()); place(w2, 2.2, 1.2, 6.5)
    assert w2.heat_grid(16).max() == pytest.approx(0.0, abs=1e-9)


def test_noise_is_seeded_and_resettable():
    cfg = single_scenario(noise_std=0.05, seed=7)
    a, b = SimWorld(cfg), SimWorld(single_scenario(noise_std=0.05, seed=7))
    place(a, 5, 1.2, 7); place(b, 5, 1.2, 7)
    assert a.thermal.victims[0].score == b.thermal.victims[0].score
    first = a.thermal.victims[0].score
    a.reset(); place(a, 5, 1.2, 7)
    assert a.thermal.victims[0].score == pytest.approx(first)
