import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("yaml")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from firerescue import SimWorld
from firerescue.envs import EnvConfig, RescueDroneGym
from firerescue.rl.evaluation import (aggregate, evaluate, model_policy, random_policy, save_results)
from firerescue.rl.demonstrations import DemonstrationRecorder
from firerescue.rl.policy_runner import PolicyRunner
from firerescue.rl.training import load_model, load_run_config, train, train_repetitions

ROOT = Path(__file__).resolve().parents[1]

TINY = {"seed": 0, "total_timesteps": 128, "n_envs": 1, "vec_env": "dummy", "device": "cpu",
        "env_config": str(ROOT / "configs" / "env.yaml"), "reward_config": str(ROOT / "configs" / "reward.yaml"),
        "scenario": None, "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        "policy": {"net_arch": [16, 16], "activation": "tanh"},
        "checkpoint_freq": 64, "eval_freq": 10 ** 9, "eval_episodes": 1, "eval_seed": 1000}


def small_env(**kw):
    return RescueDroneGym(env_config=EnvConfig(random_start=True, max_episode_steps=60, **kw))


def tiny_model(env=None, steps=64):
    venv = DummyVecEnv([lambda: Monitor(small_env())])
    m = PPO("MlpPolicy", venv, n_steps=32, batch_size=16, n_epochs=1, seed=0, device="cpu",
            policy_kwargs=dict(net_arch=[16, 16]))
    m.learn(steps)
    return m


def test_sb3_environment_checker_and_vec_env():
    check_env(small_env(), warn=True, skip_render_check=True)
    m = tiny_model()
    obs = m.env.reset()
    a, _ = m.predict(obs, deterministic=True)
    assert a.shape == (1, 4) and np.all(np.abs(a) <= 1.0)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    m = tiny_model()
    path = tmp_path / "m"
    m.save(str(path))
    loaded = PPO.load(str(path), device="cpu")
    obs = np.random.default_rng(0).uniform(-1, 1, (8, m.observation_space.shape[0])).astype(np.float32)
    a1, _ = m.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.array_equal(a1, a2)
    # a loaded model can continue training
    loaded.set_env(DummyVecEnv([lambda: Monitor(small_env())]))
    loaded.learn(32, reset_num_timesteps=False)


def test_train_function_writes_all_artifacts(tmp_path):
    out = tmp_path / "run"
    s = train(dict(TINY, _config_dir=str(ROOT / "configs")), out, verbose=0)
    for f in ("final_model.zip", "latest_model.zip", "best_model.zip", "summary.json", "model_metadata.json",
              "training_metrics.csv", "env.yaml", "reward.yaml", "run_config.json"):
        assert (out / f).exists(), f
    assert any((out / "checkpoints").glob("*.zip"))
    assert any((out / "tb").rglob("events.out.tfevents*")), "TensorBoard event file missing"
    assert s["final_eval"] is not None and s["total_timesteps"] == 128
    assert json.loads((out / "summary.json").read_text())["n_envs"] == 1
    m = load_model(out / "final_model.zip")
    obs, _ = small_env().reset(seed=1)
    assert model_policy(m)(obs).shape == (4,)


def test_repetition_training_updates_after_exact_complete_episodes(tmp_path):
    import yaml

    env_cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text())
    env_cfg["max_episode_steps"] = 2
    env_cfg["random_start"] = False
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(env_cfg))
    cfg = dict(TINY, n_envs=1, env_config=str(env_path), eval_episodes=1, eval_seed=500)
    initial = tiny_model()
    initial_path = tmp_path / "initial.zip"
    initial.save(str(initial_path))
    before = {key: value.clone() for key, value in initial.policy.state_dict().items()}

    out = tmp_path / "repetition-run"
    summary = train_repetitions(cfg, out, repetitions=2, seed=3, resume=str(initial_path), verbose=0)
    assert summary["completed_episodes"] == 2
    assert summary["ppo_updates"] == 1 and summary["training_steps"] == 4
    assert len((out / "training_metrics.csv").read_text().splitlines()) == 3
    assert (out / "original_model.zip").exists()
    updated = load_model(out / "final_model.zip")
    assert any(not np.array_equal(before[k].cpu().numpy(), value.cpu().numpy())
               for k, value in updated.policy.state_dict().items())


def test_behavior_cloning_cli_preserves_source_and_evaluates(tmp_path):
    env = RescueDroneGym(env_config=EnvConfig(random_start=False, max_episode_steps=2))
    observation, _ = env.reset(seed=1)
    env.close()
    demos = tmp_path / "demos"
    recorder = DemonstrationRecorder(demos)
    recorder.start(observation, {"environment_seed": 1})
    for step in range(3):
        next_obs = observation.copy()
        next_obs[0] = min(1.0, next_obs[0] + 0.01 * (step + 1))
        recorder.add_transition(observation, np.zeros(4, dtype=np.float32), next_obs, 0.1,
                                step == 2, False, {"new_collisions": 0})
        observation = next_obs
    recorder.finish({"success": True, "useful": True, "safe": True})
    recorder.close()

    model_path = tmp_path / "source.zip"
    tiny_model().save(str(model_path))
    env_cfg = {"max_episode_steps": 2, "random_start": False}
    env_cfg_path = tmp_path / "env.yaml"
    import yaml
    env_cfg_path.write_text(yaml.safe_dump(env_cfg))
    out = tmp_path / "clone"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "train_from_demos.py"),
        "--model", str(model_path), "--demos", str(demos), "--out", str(out),
        "--env-config", str(env_cfg_path), "--epochs", "1", "--batch-size", "2",
        "--eval-episodes", "1",
    ], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (out / "original_model.zip").exists()
    assert (out / "imitation_model.zip").exists()
    assert (out / "model_metadata.json").exists()
    assert json.loads((out / "imitation_summary.json").read_text())["evaluation_episodes"] == 1


def test_evaluation_is_deterministic_and_reports_metrics():
    m = tiny_model()
    env = small_env()
    r1 = evaluate(env, model_policy(m), episodes=3, seed=500)
    r2 = evaluate(small_env(), model_policy(m), episodes=3, seed=500)
    assert r1["episodes"] == r2["episodes"]                      # identical per-episode metrics
    np.testing.assert_equal(r1["summary"], r2["summary"])        # NaN-aware comparison
    s = r1["summary"]
    for k in ("success_rate", "reward_mean", "collisions_mean", "victims_found_mean", "coverage_mean",
              "mission_completion_rate", "victim_discovery_rate", "rooms_discovered_mean"):
        assert k in s and np.isfinite(s[k])
    assert 0 <= s["success_rate"] <= 1 and 0 <= s["coverage_mean"] <= 1
    rr = evaluate(small_env(), random_policy(0), episodes=2, seed=500)
    assert rr["summary"]["episodes"] == 2


def test_aggregate_math_and_result_files(tmp_path):
    eps = [dict(success=1.0, episode_reward=10.0, victims_found=5.0, victims_total=5.0, collisions=0.0, coverage=0.9,
                rooms_discovered=6.0, fire_danger_steps=0.0, overheated=0.0, steps=100.0, sim_time=10.0),
           dict(success=0.0, episode_reward=-10.0, victims_found=2.0, victims_total=5.0, collisions=4.0, coverage=0.3,
                rooms_discovered=2.0, fire_danger_steps=10.0, overheated=0.0, steps=200.0, sim_time=20.0)]
    a = aggregate(eps)
    assert a["success_rate"] == 0.5 and a["reward_mean"] == 0 and a["victim_discovery_rate"] == 0.7
    assert a["collisions_mean"] == 2.0 and a["collision_free_rate"] == 0.5 and a["time_to_success_mean"] == 10.0
    save_results(tmp_path / "r.json", {"ppo": {"episodes": eps, "summary": a}})
    assert (tmp_path / "r.csv").read_text().count("\n") == 3


def test_baseline_controller_through_the_env_interface():
    env = RescueDroneGym(env_config=EnvConfig(random_start=False, max_episode_steps=400))
    r = evaluate(env, None, episodes=1, seed=0, baseline=True)["summary"]
    assert r["victims_found_mean"] >= 1 and r["collisions_mean"] == 0


def test_policy_runner_drives_a_shared_world(tmp_path):
    m = tiny_model()
    m.save(str(tmp_path / "model"))
    world = SimWorld()
    runner = PolicyRunner(world, tmp_path / "model.zip")
    ticks = 0
    for _ in range(120):                                   # 2 s at 60 Hz -> 20 control ticks
        world.step(runner.command, 1 / 60)
        ticks += runner.advance(1 / 60) is not None
    assert 18 <= ticks <= 21 and runner.env.steps == ticks
    assert np.isfinite(runner.episode_reward) and "Episode reward" in runner.stats_text()
    assert runner.goal_target is not None
    runner.reset()
    assert runner.env.steps == 0 and world.time == 0


def test_cli_scripts_end_to_end(tmp_path):
    import yaml
    cfg = {k: v for k, v in TINY.items()}
    cfg["total_timesteps"] = 96
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_agent.py"), "--config", str(tmp_path / "tiny.yaml"),
                        "--out", str(out), "--quiet"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_agent.py"), "--model", str(out / "final_model.zip"),
                        "--episodes", "2", "--compare", "random", "--out", str(tmp_path / "res.json")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "ppo" in r.stdout and "random" in r.stdout and (tmp_path / "res.json").exists()
