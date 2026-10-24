import copy
import json

import pytest

from firerescue import ConfigError, ScenarioConfig, SimWorld, default_scenario
from firerescue.config import FireConfig


def test_default_scenario_is_valid_and_deterministic():
    a, b = default_scenario(), default_scenario()
    assert a == b
    assert len(a.victims) == 5 and len(a.fires) == 3
    assert {v.pose for v in a.victims} == {"standing", "prone"}
    SimWorld(a)   # builds without geometry errors


def test_dict_and_json_roundtrip(tmp_path):
    cfg = default_scenario()
    assert ScenarioConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    p = tmp_path / "s.json"
    cfg.to_json(p)
    assert ScenarioConfig.from_json(p) == cfg


def test_configurable_start_fire_and_victims():
    d = default_scenario().to_dict()
    d["drone"]["start_position"] = [10.0, 1.5, 7.0]
    d["fires"][0]["danger_radius"] = 3.3
    d["victims"][0]["position"] = [3.0, 0.0, 3.0]
    d["victims"][0]["pose"] = "prone"
    w = SimWorld(ScenarioConfig.from_dict(d))
    assert tuple(w.drone.pos) == (10.0, 1.5, 7.0)
    assert w.fires[0].danger_radius == 3.3
    assert w.victims[0].pose == "prone" and tuple(w.victims[0].position) == (3.0, 0.0, 3.0)


@pytest.mark.parametrize("mutate, text", [
    (lambda d: d["victims"][0].__setitem__("pose", "sitting"), "pose"),
    (lambda d: d["victims"][1].__setitem__("id", "V1"), "unique"),
    (lambda d: d["victims"][0].__setitem__("position", [99, 0, 1]), "outside"),
    (lambda d: d["drone"].__setitem__("start_position", [-5, 1, 1]), "start_position"),
    (lambda d: d["fires"][0].__setitem__("danger_radius", -1), "radius"),
    (lambda d: d["building"]["furniture"][0].__setitem__("rotation_deg", 45), "multiple of 90"),
    (lambda d: d["sensor"].__setitem__("prone_factor", 0), "prone_factor"),
    (lambda d: d.__setitem__("victims", []), "victim"),
])
def test_invalid_configs_are_rejected(mutate, text):
    d = copy.deepcopy(default_scenario().to_dict())
    mutate(d)
    with pytest.raises(ConfigError, match=text):
        ScenarioConfig.from_dict(d)


def test_unknown_key_and_bad_json(tmp_path):
    d = default_scenario().to_dict()
    d["drone"]["warp_speed"] = 9
    with pytest.raises(ConfigError, match="Unknown keys"):
        ScenarioConfig.from_dict(d)
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError):
        ScenarioConfig.from_json(p)


def test_start_inside_obstacle_is_rejected():
    cfg = default_scenario()
    cfg.drone.start_position = (6.5, 0.4, 8.0)   # inside corridor box
    with pytest.raises(ConfigError, match="intersects"):
        SimWorld(cfg)
