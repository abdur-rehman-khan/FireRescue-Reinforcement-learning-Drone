import numpy as np
import pytest

from firerescue.rl.demonstrations import DemonstrationRecorder, describe_demonstrations, load_demonstrations


def _record(recorder, *, success, useful, safe):
    observation = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    recorder.start(observation, {"environment_seed": 7})
    recorder.add_transition(
        observation,
        np.array([0.25, 0.0, -0.5, 0.1], dtype=np.float32),
        observation + 0.1,
        1.25,
        False,
        False,
        {"new_victims": ["v1"], "new_collisions": 0, "fire_level": "CLEAR"},
    )
    path = recorder.finish({"success": success, "useful": useful, "safe": safe})
    recorder.close()
    return path


def test_demo_archive_roundtrip_and_positive_filter(tmp_path):
    path = _record(DemonstrationRecorder(tmp_path), success=True, useful=True, safe=True)
    obs, actions, metadata = load_demonstrations(path)
    assert obs.shape == (1, 3)
    assert actions.shape == (1, 4)
    np.testing.assert_allclose(actions[0], [0.25, 0.0, -0.5, 0.1])
    assert metadata[0]["eligible_for_imitation"]
    assert describe_demonstrations(tmp_path)[0]["transitions"] == 1


def test_unsafe_and_unsuccessful_demo_are_diagnostic_only(tmp_path):
    safe_path = _record(DemonstrationRecorder(tmp_path), success=False, useful=True, safe=True)
    _record(DemonstrationRecorder(tmp_path), success=True, useful=True, safe=False)
    with pytest.raises(ValueError, match="No safe successful"):
        load_demonstrations(tmp_path)
    obs, actions, _ = load_demonstrations(safe_path, include_useful=True)
    assert obs.shape == (1, 3) and actions.shape == (1, 4)


def test_recorder_rejects_actions_outside_ppo_space(tmp_path):
    recorder = DemonstrationRecorder(tmp_path)
    recorder.start(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="PPO action range"):
        recorder.add_transition(np.zeros(2), np.array([2.0, 0.0, 0.0, 0.0]), np.zeros(2), 0, False, False, {})
    recorder.finish()
    recorder.close()
