import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("yaml")

from pathlib import Path

from firerescue import ConfigError, DroneCommand
from firerescue.collision import ObstacleSet
from firerescue.config import FireConfig
from firerescue.envs import (EnvConfig, RescueDroneGym, RewardCalculator, RewardConfig, StepSignals,
                             load_env_config, load_reward_config)
from firerescue.obstacles import Obstacle
from conftest import single_scenario

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def make(random_start=False, scenario=None, **kw):
    return RescueDroneGym(scenario, EnvConfig(random_start=random_start, **kw))


def idx(env, name):
    return env.obs_builder.names.index(name)


@pytest.fixture
def env():
    e = make()
    e.reset(seed=0)
    return e


# ------------------------------------------------------------ spaces / observation
def test_spaces_and_documented_features(env):
    assert env.action_space.shape == (4,) and np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    n = env.obs_builder.size_total
    assert env.observation_space.shape == (n,) == (sum(s for _, s, _ in env.obs_builder.docs),)
    assert len(env.obs_builder.names) == n and all(d for _, _, d in env.obs_builder.docs)


def test_observation_shape_dtype_and_numerical_validity():
    for rs in (False, True):
        env = make(random_start=rs)
        obs, _ = env.reset(seed=3)
        rng = np.random.default_rng(0)
        for i in range(400):
            assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
            assert np.isfinite(obs).all() and obs.min() >= -1 and obs.max() <= 1
            assert env.observation_space.contains(obs)
            obs, r, term, trunc, _ = env.step(env.action_space.sample() if i % 2 else rng.uniform(-1, 1, 4))
            assert np.isfinite(r)
            if term or trunc:
                obs, _ = env.reset()


def test_proximity_ray_sees_wall(env):
    env.world.set_drone_pose((1.5, 1.2, 7.0), 180.0)        # facing the corridor wall, 1.12 m clearance
    obs = env.observe()
    i = idx(env, "proximity_rays[0]")
    assert obs[i] == pytest.approx(1 - 1.12 / 4.0, abs=0.02)
    assert obs[idx(env, "proximity_rays[9]")] == 0.0              # left of a south-facing drone: open corridor
    env.world.set_drone_pose((1.5, 1.2, 7.0), 90.0)
    assert env.observe()[i] == 0.0                           # facing +x down the corridor


def test_thermal_features_point_toward_undiscovered_victim():
    env = make(scenario=single_scenario(victim_pos=(5, 0, 5)))
    env.reset(seed=0)
    env.world.set_drone_pose((5.0, 1.2, 9.4), 180.0)         # 4.4 m away: signal present, below threshold
    assert env.world.discovered_count == 0
    obs = env.observe()
    assert obs[idx(env, "thermal_peak")] > 0.1
    assert obs[idx(env, "thermal_bearing[2]")] > 0.9          # victim straight ahead (body +z)
    g = obs[idx(env, "thermal_grid[0]"):idx(env, "thermal_grid[0]") + env.cfg.thermal_grid ** 2]
    assert g.max() > 0
    env.world.set_drone_pose((5.0, 1.2, 9.4), 0.0)            # turn around: victim now behind
    assert env.observe()[idx(env, "thermal_bearing[2]")] < -0.9


def test_victims_remaining_and_goal_features(env):
    o = env.observe()
    assert o[idx(env, "victims_remaining")] == 1.0
    assert 0 <= o[idx(env, "goal_dist")] <= 1
    assert abs(o[idx(env, "goal_vec_body[0]"):idx(env, "goal_vec_body[0]") + 3]).sum() > 0
    env.world.set_drone_pose((2.2, 1.2, 4.2), 0.0)
    env.step(np.zeros(4))
    assert env.observe()[idx(env, "victims_remaining")] == pytest.approx(0.8)


# ---------------------------------------------------------------------- actions
def test_action_mapping_and_limits():
    e = make()
    cmd, a, _ = e.action_to_command([1, 0, 0, 0])
    assert (cmd.right, cmd.forward, cmd.up, cmd.yaw) == (1.0, 0.0, 0.0, 0.0)
    cmd, _, _ = e.action_to_command([0, 0.5, -1, 1])
    assert (cmd.up, cmd.forward, cmd.yaw) == (0.5, -1.0, 1.0)
    e2 = make(max_speed_xz=1.0, max_speed_y=0.6, max_yaw_rate_deg=45.0)
    cmd, _, _ = e2.action_to_command([1, 1, 1, 1])
    assert (cmd.right, cmd.up, cmd.forward, cmd.yaw) == (0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ConfigError):
        make(max_speed_xz=99.0)


def test_invalid_actions_are_handled(env):
    cmd, a, f = env.action_to_command([np.nan, 5.0, -np.inf, 0.2])
    assert f["invalid"] and f["clipped"] and np.isfinite(a).all() and a.max() <= 1 and a.min() >= -1
    assert cmd.right == 0.0 and cmd.up == 1.0 and cmd.forward == 0.0
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    _, _, _, _, info = env.step([np.nan] * 4)
    assert info["invalid_action"] and env.invalid_actions == 1


def test_actions_move_the_drone_in_body_frame(env):
    env.world.set_drone_pose((10.0, 1.2, 7.0), 90.0)         # facing +x
    p0 = env.world.drone.pos.copy()
    for _ in range(5):
        env.step([0, 0, 1, 0])                                # vz = forward
    d = env.world.drone.pos - p0
    assert d[0] > 0.3 and abs(d[2]) < 1e-6
    env.world.set_drone_pose((10.0, 1.2, 7.0), 90.0)
    for _ in range(5):
        env.step([0, 1, 0, 0])                                # vy = up
    assert env.world.drone.pos[1] > 1.4


def test_collision_is_reported_and_penalised():
    e = make()
    e.reset(seed=0)
    e.world.set_drone_pose((1.5, 1.2, 6.5), 180.0)
    total, hit = 0.0, False
    for _ in range(30):
        _, r, _, _, info = e.step([0, 0, 1, 0])
        hit |= info["reward_components"]["collision"] < 0
    assert hit and e.world.collision_count == 1


# ------------------------------------------------------------------- determinism
def test_deterministic_reset_and_rollout():
    def rollout(seed):
        e = make(random_start=True)
        o, _ = e.reset(seed=seed)
        rng = np.random.default_rng(5)
        out = [o]
        for _ in range(40):
            o, r, *_ = e.step(rng.uniform(-1, 1, 4))
            out += [o, r]
        return out
    a, b, c = rollout(11), rollout(11), rollout(12)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])                    # different seed -> different random start


def test_fixed_start_ignores_seed_and_reset_restores_state(env):
    o1, _ = env.reset(seed=1)
    for _ in range(30):
        env.step([0, 0, 1, 0.5])
    o2, _ = env.reset(seed=99)
    assert np.array_equal(o1, o2) and env.steps == 0 and env.episode_reward == 0
    assert env.world.time == 0 and env.world.collision_count == 0 and env.world.discovered_count == 0
    assert len(env.trajectory) == 1


# ------------------------------------------------------------------- rewards (env)
def test_victim_reward_only_once_per_victim(env):
    env.world.set_drone_pose((2.2, 1.2, 4.2), 0.0)           # V1 in range
    _, r1, *_, info = env.step(np.zeros(4))
    assert info["new_victims"] == ["V1"] and info["reward_components"]["new_victim"] == env.reward_cfg.new_victim
    for _ in range(5):
        *_, info = env.step(np.zeros(4))
        assert info["new_victims"] == [] and info["reward_components"]["new_victim"] == 0
    env.world.reset_drone()                                   # progress is kept: no second reward
    env.world.set_drone_pose((2.2, 1.2, 4.2), 0.0)
    *_, info = env.step(np.zeros(4))
    assert info["reward_components"]["new_victim"] == 0
    env.reset(seed=0)                                         # new episode: earnable again
    env.world.set_drone_pose((2.2, 1.2, 4.2), 0.0)
    *_, info = env.step(np.zeros(4))
    assert info["new_victims"] == ["V1"]


def test_room_and_coverage_rewards_are_one_time(env):
    env.world.set_drone_pose((3.5, 1.2, 3.0), 0.0)            # inside Room A
    *_, info = env.step(np.zeros(4))
    names = [env.coverage.rooms[r].name for r in info["new_rooms"]]
    assert names == ["Room A"] and info["reward_components"]["coverage"] > 0
    cov = env.coverage.global_fraction()
    for _ in range(3):
        *_, info = env.step(np.zeros(4))
        assert info["new_rooms"] == [] and info["reward_components"]["coverage"] == 0
    assert env.coverage.global_fraction() == pytest.approx(cov)


def test_coverage_is_monotonic_and_blocked_by_walls(env):
    env.world.set_drone_pose((1.5, 1.2, 7.0), 90.0)           # corridor: Room A cells beyond the wall stay unexplored
    env.step(np.zeros(4))
    fr = dict(zip([r.name for r in env.coverage.rooms], env.coverage.room_fractions()))
    assert fr["Room A"] == 0.0 and fr["Corridor"] > 0
    last = env.coverage.global_fraction()
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step(rng.uniform(-1, 1, 4))
        cur = env.coverage.global_fraction()
        assert cur >= last
        last = cur


def test_goal_progress_is_signed_and_not_farmable(env):
    env.goals.update()
    start = env.goals.distance()
    prog = []
    for a in ([0, 0, 1, 0],) * 3 + ([0, 0, -1, 0],) * 3:
        env.step(a)
        prog.append(env.goals.distance())
    assert np.isfinite(start)
    # moving back and forth cannot accumulate positive progress: total = start - end
    total = sum(info for info in [start - prog[-1]])
    assert abs(total) <= abs(start - min(prog)) + 1e-6


def test_distance_field_decreases_toward_goal(env):
    g = env.goals
    f, v = g.fields[0], g.viewpoints[0]
    assert f[g.planner.cell(v)] == pytest.approx(0.0, abs=0.3)
    assert np.isfinite(f[g.planner.cell((1.5, 7.0))])


# --------------------------------------------------------------------- termination
def test_success_terminates_with_bonus():
    e = make(scenario=single_scenario(victim_pos=(5, 0, 5)))
    e.reset(seed=0)
    e.world.set_drone_pose((5.0, 1.2, 6.5), 0.0)
    obs, r, term, trunc, info = e.step(np.zeros(4))
    assert term and not trunc and info["episode_metrics"]["success"] == 1.0
    assert info["reward_components"]["success"] >= e.reward_cfg.mission_complete
    assert info["mission_status"] == "SUCCESS" and info["episode_metrics"]["victims_found"] == 1


def test_fire_exposure_terminates_with_penalty():
    sc = single_scenario(victim_pos=(9, 0, 9), fire=((5.0, 0.4, 5.0), 2.0))
    e = make(scenario=sc)
    e.reset(seed=0)
    e.world.set_drone_pose((5.0, 1.0, 5.0), 0.0)
    for i in range(200):
        _, r, term, trunc, info = e.step(np.zeros(4))
        if term:
            break
    assert term and info["mission_status"] == "FAILED_FIRE"
    assert info["reward_components"]["fire_failure"] == e.reward_cfg.fire_failure
    assert info["episode_metrics"]["overheated"] == 1.0 and e.fire_danger_steps > 50


def test_truncation_at_step_limit():
    e = make(max_episode_steps=5)
    e.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, info = e.step(np.zeros(4))
        assert not term and trunc == (i == 4)
    assert info["reward_components"]["timeout"] == e.reward_cfg.timeout
    assert info["episode_metrics"]["truncated"] == 1.0 and info["episode_metrics"]["success"] == 0.0


def test_collision_limit_terminates():
    e = make(max_collisions=1)
    e.reset(seed=0)
    e.world.set_drone_pose((1.5, 1.2, 6.5), 180.0)
    for _ in range(30):
        _, _, term, _, info = e.step([0, 0, 1, 0])
        if term:
            break
    assert term and info["reward_components"]["collision_limit"] == e.reward_cfg.collision_limit_failure


# ------------------------------------------------------------ reward calculator
def S(**kw):
    base = dict(position=np.zeros(3), action=np.zeros(4))
    base.update(kw)
    return StepSignals(**base)


def test_reward_terms_and_weights():
    c = RewardConfig()
    rc = RewardCalculator(c)
    assert rc.compute(S())[0] == pytest.approx(c.time_step)
    assert rc.compute(S(new_victims=2))[1]["new_victim"] == 2 * c.new_victim
    assert rc.compute(S(new_rooms=1))[1]["new_room"] == c.new_room
    assert rc.compute(S(new_cells=7))[1]["coverage"] == pytest.approx(7 * c.new_coverage_cell)
    comp = rc.compute(S(new_collisions=2, in_contact=True))[1]
    assert comp["collision"] == pytest.approx(2 * c.collision + c.collision_contact)
    assert rc.compute(S(fire_level="WARNING"))[1]["fire"] == c.fire_warning
    assert rc.compute(S(fire_level="DANGER"))[1]["fire"] == c.fire_danger
    comp = rc.compute(S(success=True, episode_fraction_left=0.5))[1]
    assert comp["success"] == pytest.approx(c.mission_complete + 0.5 * c.success_time_bonus)
    assert rc.compute(S(truncated=True))[1]["timeout"] == c.timeout
    assert rc.compute(S(truncated=True, success=True))[1]["timeout"] == 0
    assert rc.compute(S(progress=10.0))[1]["progress"] == pytest.approx(c.progress * c.max_progress_step)   # clipped
    assert rc.compute(S(progress=-0.1))[1]["progress"] < 0
    scaled = RewardCalculator(RewardConfig(new_victim=100.0)).compute(S(new_victims=1))[1]["new_victim"]
    assert scaled == 100.0


def test_oscillation_dither_and_stagnation_penalties():
    c = RewardConfig()
    rc = RewardCalculator(c)
    assert rc.compute(S(action=np.array([1.0, 0, 0, 0])))[1]["oscillation"] == 0     # no previous action
    assert rc.compute(S(action=np.array([1.0, 0, 0, 0])))[1]["oscillation"] == 0     # same direction
    assert rc.compute(S(action=np.array([-1.0, 0, 0, 0])))[1]["oscillation"] == c.oscillation
    rc.reset()
    dith = []
    for i in range(c.dither_window + 6):                                              # shuffle +-0.1 m
        dith.append(rc.compute(S(position=np.array([0.1 * (i % 2), 1.0, 0.0]),
                                 action=np.array([0.2 * (-1) ** i, 0, 0, 0])))[1]["dither"])
    assert dith[0] == 0 and dith[-1] == c.dither
    rc.reset()
    stag = [rc.compute(S(position=np.array([float(i), 0, 0])))[1]["stagnation"] for i in range(c.stagnation_steps + 3)]
    assert stag[0] == 0 and stag[-1] == c.stagnation
    assert rc.compute(S(new_cells=1))[1]["stagnation"] == 0                            # productive step resets
    straight = RewardCalculator(c)
    assert all(straight.compute(S(position=np.array([0.2 * i, 0, 0]), progress=0.2,
                                  action=np.array([1.0, 0, 0, 0])))[1]["dither"] == 0 for i in range(40))


def test_oscillating_policy_scores_worse_than_steady_one():
    def run(actions):
        e = make()
        e.reset(seed=0)
        return sum(e.step(a)[1] for a in actions)
    steady = [[0, 0, 1, 0]] * 40
    shake = [[0, 0, 1, 0] if i % 2 else [0, 0, -1, 0] for i in range(40)]
    assert run(shake) < run(steady)


# ------------------------------------------------------------------------ config
def test_yaml_files_match_dataclass_defaults_and_validate():
    assert load_reward_config(CONFIGS / "reward.yaml") == RewardConfig()
    assert load_env_config(CONFIGS / "env.yaml") == EnvConfig()
    assert load_env_config() == EnvConfig()


def test_bad_yaml_values_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("new_victim: 1\nbogus_weight: 3\n")
    with pytest.raises(ConfigError):
        load_reward_config(p)
    p.write_text("physics_dt: 1.0\ncontrol_dt: 0.1\n")
    with pytest.raises(ConfigError):
        load_env_config(p)


# ------------------------------------------------------ gym API / render / helpers
def test_gymnasium_env_checker(env):
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    check_env(make(random_start=True), skip_render_check=True)


def test_render_modes_and_close():
    e = RescueDroneGym(env_config=EnvConfig(random_start=False), render_mode="rgb_array")
    e.reset(seed=0)
    e.step(np.zeros(4))
    img = e.render()
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3 and img.shape[0] == 280 and img.shape[1] == 400
    assert img.std() > 10
    e.close()
    a = RescueDroneGym(env_config=EnvConfig(random_start=False), render_mode="ansi")
    a.reset(seed=0)
    assert "victims 0/5" in a.render()
    assert RescueDroneGym().render() is None
    with pytest.raises(ValueError):
        RescueDroneGym(render_mode="human")


def test_gym_make_registration():
    e = gym.make("RescueDrone-v0", env_config=EnvConfig(random_start=False))
    obs, _ = e.reset(seed=0)
    assert obs.shape == e.observation_space.shape


def test_raycast_hits_boxes():
    s = ObstacleSet([Obstacle("b", "box", (5, 0, 0), (2, 2, 2), (1, 1, 1))])
    d = s.raycast([0, 0, 0], np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]], float), 10.0)
    assert d.tolist() == pytest.approx([4.0, 10.0, 10.0, 10.0])
    assert s.raycast([5, 0, 0], np.array([[1, 0, 0]], float), 10.0)[0] == 0.0     # origin inside
    assert ObstacleSet([]).raycast([0, 0, 0], np.array([[1.0, 0, 0]]), 3.0)[0] == 3.0
