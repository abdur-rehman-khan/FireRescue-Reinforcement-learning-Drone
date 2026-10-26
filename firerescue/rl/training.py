"""PPO training pipeline (Stable-Baselines3): vector envs, checkpoints, TensorBoard, eval."""
from __future__ import annotations

import csv
import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ScenarioConfig
from ..envs import RescueDroneGym, load_env_config, load_reward_config
from .evaluation import evaluate, format_summary, model_policy, save_results

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_REPETITIONS = 20


# ------------------------------------------------------------------------ config
def load_run_config(path) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    return cfg


def resolve(cfg: Dict[str, Any], key: str) -> Optional[Path]:
    """Resolve a path from the run config relative to cwd, the config's folder, or the repo root."""
    v = cfg.get(key)
    if not v:
        return None
    p = Path(v)
    for base in (Path.cwd(), Path(cfg.get("_config_dir", ".")), Path(cfg.get("_config_dir", ".")).parent, REPO_ROOT):
        if (base / p).exists():
            return base / p
    raise FileNotFoundError(f"{key}: {v} not found")


def write_model_metadata(out: Path, model, env) -> None:
    """Persist the policy's environment-facing space contract beside its checkpoints."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    observation_names = getattr(getattr(base, "obs_builder", None), "names", None)
    metadata = {
        "metadata_version": 1,
        "observation_shape": list(model.observation_space.shape),
        "observation_features": observation_names,
        "action_shape": list(model.action_space.shape),
        "action_order": ["right", "up", "forward", "yaw_rate"],
        "action_range": [-1.0, 1.0],
        "policy_class": type(model).__name__,
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_env_factory(cfg: Dict[str, Any], seed: int = 0, monitor_dir: Optional[Path] = None, rank: int = 0,
                     random_start: Optional[bool] = None):
    env_path, rew_path, sc_path = resolve(cfg, "env_config"), resolve(cfg, "reward_config"), resolve(cfg, "scenario")

    def _init():
        from stable_baselines3.common.monitor import Monitor
        ec = load_env_config(env_path)
        if random_start is not None:
            ec.random_start = random_start
        env = RescueDroneGym(ScenarioConfig.from_json(sc_path) if sc_path else None, ec, load_reward_config(rew_path))
        mon = str(monitor_dir / f"monitor_{rank}") if monitor_dir else None
        env = Monitor(env, mon)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(cfg, n_envs, seed, monitor_dir=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    fns = [make_env_factory(cfg, seed, monitor_dir, i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if (cfg.get("vec_env") == "subproc" and n_envs > 1) else DummyVecEnv(fns)


# --------------------------------------------------------------------- callbacks
def make_callbacks(cfg, out: Path, n_envs: int):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

    class MetricsCallback(BaseCallback):
        """Logs episode metrics to TensorBoard and an append-only CSV."""

        def __init__(self, window: int = 50):
            super().__init__()
            self.buf = {k: deque(maxlen=window) for k in
                        ("success", "collisions", "victims_found", "coverage", "rooms_discovered", "fire_danger_steps")}
            self.csv_path = out / "training_metrics.csv"
            self.csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            self.csv_fields = ("episode", "episode_seed", "repetition_target", "timesteps", "episode_reward", "success",
                               "victims_found", "victims_total", "coverage", "rooms_discovered", "rooms_total",
                               "collisions", "fire_danger_steps", "fire_exposure", "steps", "overheated",
                               "truncated", "invalid_actions", "sim_time")
            self.episodes = 0

        def _on_training_start(self) -> None:
            if not self.csv_exists:
                with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=self.csv_fields).writeheader()
                self.csv_exists = True

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                m = info.get("episode_metrics")
                if m:
                    self.episodes += 1
                    for k in self.buf:
                        self.buf[k].append(m[k])
                    for k, v in self.buf.items():
                        self.logger.record(f"rescue/{k}", float(np.mean(v)))
                    with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
                        csv.DictWriter(stream, fieldnames=self.csv_fields).writerow(
                            {"episode": self.episodes, "episode_seed": "", "repetition_target": "",
                             "timesteps": self.num_timesteps, **m})
            return True

    class PeriodicEval(BaseCallback):
        """Deterministic evaluation every ``freq`` steps; keeps best_model.zip."""

        def __init__(self, freq: int, episodes: int, seed: int):
            super().__init__()
            self.freq, self.episodes, self.seed, self.next, self.best = freq, episodes, seed, freq, None
            self.history = []
            self.env = None

        def _on_training_start(self) -> None:
            self.env = make_env_factory(cfg, self.seed)()
            self.next = self.model.num_timesteps + self.freq
            previous_best = out / "best_model.zip"
            if previous_best.is_file():
                from stable_baselines3 import PPO
                best_model = PPO.load(str(previous_best), device=self.model.device)
                result = evaluate(self.env, model_policy(best_model), self.episodes, self.seed)["summary"]
                self.best = (result["success_rate"], result["victims_found_mean"], result["reward_mean"])

        def _on_step(self) -> bool:
            if self.model.num_timesteps >= self.next:
                self.next += self.freq
                self.run()
            return True

        def run(self) -> None:
            res = evaluate(self.env, model_policy(self.model), self.episodes, self.seed)["summary"]
            for k in ("success_rate", "reward_mean", "victims_found_mean", "collisions_mean", "coverage_mean"):
                self.logger.record(f"eval/{k}", res[k])
            self.history.append({"timesteps": int(self.model.num_timesteps), **res})
            print(f"[eval @ {self.model.num_timesteps}] " + format_summary("policy", res), flush=True)
            score = (res["success_rate"], res["victims_found_mean"], res["reward_mean"])
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(str(out / "best_model"))

        def _on_training_end(self) -> None:
            if self.env is not None:
                self.env.close()

    ckpt = CheckpointCallback(save_freq=max(1, int(cfg.get("checkpoint_freq", 10000)) // n_envs),
                              save_path=str(out / "checkpoints"), name_prefix="ppo_rescue")
    ev = PeriodicEval(int(cfg.get("eval_freq", 10000)), int(cfg.get("eval_episodes", 5)), int(cfg.get("eval_seed", 1000)))
    return CallbackList([ckpt, MetricsCallback(), ev]), ev


# ------------------------------------------------------------------------- train
def train(cfg: Dict[str, Any], out_dir, total_timesteps: Optional[int] = None, seed: Optional[int] = None,
          resume: Optional[str] = None, verbose: int = 1) -> Dict[str, Any]:
    import torch
    from stable_baselines3 import PPO

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    total = int(total_timesteps or cfg.get("total_timesteps", 30000))
    n_envs = int(cfg.get("n_envs", 1))
    device = cfg.get("device", "cpu")

    # keep the exact configs next to the model for reproducible evaluation
    for key, name in (("env_config", "env.yaml"), ("reward_config", "reward.yaml")):
        p = resolve(cfg, key)
        if p:
            shutil.copy(p, out / name)
    (out / "run_config.json").write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2))

    try:                                    # TensorBoard is optional at runtime
        import tensorboard  # noqa: F401
        tb = str(out / "tb")
    except ImportError:
        print("WARNING: tensorboard not installed - logging disabled (pip install tensorboard)")
        tb = None

    venv = build_vec_env(cfg, n_envs, seed)
    act = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[cfg.get("policy", {}).get("activation", "tanh")]
    net = list(cfg.get("policy", {}).get("net_arch", [128, 128]))
    ppo_kwargs = dict(cfg.get("ppo", {}))
    if resume:
        model = PPO.load(resume, env=venv, device=device, tensorboard_log=tb)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device=device, tensorboard_log=tb, verbose=verbose,
                    policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=act,
                                       log_std_init=float(cfg.get("policy", {}).get("log_std_init", 0.0))), **ppo_kwargs)
    write_model_metadata(out, model, venv.envs[0])
    callbacks, evcb = make_callbacks(cfg, out, n_envs)
    t0 = time.time()
    model.learn(total_timesteps=total, callback=callbacks, tb_log_name="ppo", reset_num_timesteps=not resume,
                progress_bar=False)
    elapsed = time.time() - t0
    model.save(str(out / "final_model"))
    model.save(str(out / "latest_model"))
    evcb.run()                                   # final deterministic evaluation of the final model
    venv.close()
    summary = {"total_timesteps": total, "seed": seed, "n_envs": n_envs, "train_seconds": elapsed,
               "steps_per_second": total / max(elapsed, 1e-9), "eval_history": evcb.history,
               "final_eval": evcb.history[-1] if evcb.history else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def load_model(path, device: str = "cpu"):
    from stable_baselines3 import PPO
    return PPO.load(str(path), device=device)


def train_repetitions(cfg: Dict[str, Any], out_dir, repetitions: int = TRAINING_REPETITIONS,
                      seed: Optional[int] = None, resume: Optional[str] = None,
                      verbose: int = 1) -> Dict[str, Any]:
    """Train from exactly N complete episodes, then make one valid PPO rollout update."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import configure_logger, obs_as_tensor

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    device = cfg.get("device", "cpu")
    for key, name in (("env_config", "env.yaml"), ("reward_config", "reward.yaml")):
        p = resolve(cfg, key)
        if p:
            shutil.copy(p, out / name)
    saved_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    saved_cfg["training_repetitions"] = repetitions
    saved_cfg["n_envs"] = 1
    saved_cfg["rollout_method"] = "complete episodic on-policy PPO buffer"
    (out / "run_config.json").write_text(json.dumps(saved_cfg, indent=2), encoding="utf-8")

    env_path, rew_path, scenario_path = (resolve(cfg, key) for key in ("env_config", "reward_config", "scenario"))
    training_env = RescueDroneGym(
        ScenarioConfig.from_json(scenario_path) if scenario_path else None,
        load_env_config(env_path),
        load_reward_config(rew_path),
    )
    eval_env = make_env_factory(cfg, int(cfg.get("eval_seed", 1000)))()
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([make_env_factory(cfg, seed, rank=0)])
    act = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[cfg.get("policy", {}).get("activation", "tanh")]
    net = list(cfg.get("policy", {}).get("net_arch", [128, 128]))
    ppo_kwargs = dict(cfg.get("ppo", {}))
    if resume:
        resume_path = Path(resume)
        original_copy = out / "original_model.zip"
        if (resume_path.is_file() and not original_copy.exists()
                and resume_path.resolve() != original_copy.resolve()):
            shutil.copy2(resume_path, original_copy)
        model = PPO.load(str(resume), env=venv, device=device)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device=device, verbose=verbose,
                    policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=act,
                                       log_std_init=float(cfg.get("policy", {}).get("log_std_init", 0.0))),
                    **ppo_kwargs)
    write_model_metadata(out, model, training_env)

    metrics_path = out / "training_metrics.csv"
    metrics_fields = ("episode", "episode_seed", "repetition_target", "timesteps", "episode_reward", "success",
                      "victims_found", "victims_total", "coverage", "rooms_discovered", "rooms_total",
                      "collisions", "fire_danger_steps", "fire_exposure", "steps", "overheated", "truncated",
                      "invalid_actions", "sim_time")
    eval_episodes = max(20, int(cfg.get("eval_episodes", 20)))
    eval_seed = int(cfg.get("eval_seed", 1000))
    best_path = out / "best_model.zip"
    best_score = None
    history = []
    if best_path.is_file():
        previous_best = PPO.load(str(best_path), env=eval_env, device=device)
        previous_summary = evaluate(eval_env, model_policy(previous_best), eval_episodes, eval_seed)["summary"]
        best_score = (previous_summary["success_rate"], previous_summary["victims_found_mean"],
                      previous_summary["reward_mean"])

    def evaluate_and_save(label: str):
        nonlocal best_score
        result = evaluate(eval_env, model_policy(model), eval_episodes, eval_seed)
        summary = result["summary"]
        history.append({"batch": label, "timesteps": int(model.num_timesteps), **summary})
        score = (summary["success_rate"], summary["victims_found_mean"], summary["reward_mean"])
        if best_score is None or score > best_score:
            model.save(str(out / "best_model"))
            best_score = score
        model.save(str(out / "latest_model"))
        save_results(out / f"evaluation_{label}.json", {"ppo": result})
        print(f"[evaluation {label}] " + format_summary("ppo", summary), flush=True)

    try:
        if resume or best_path.is_file():
            evaluate_and_save("before")
        started = time.time()
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        transitions = []
        episode_rows = []
        episode_step_count = 0
        recent_rewards = deque(maxlen=10)
        model.policy.set_training_mode(False)
        for episode in range(repetitions):
            observation, _ = training_env.reset(seed=seed + episode)
            episode_start = True
            while True:
                obs_batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
                obs_tensor = obs_as_tensor(obs_batch, model.device)
                with torch.no_grad():
                    actions_tensor, values, log_probs = model.policy(obs_tensor)
                raw_action = actions_tensor.cpu().numpy()
                env_action = np.clip(raw_action[0], training_env.action_space.low, training_env.action_space.high)
                next_observation, reward, terminated, truncated, info = training_env.step(env_action)
                done = terminated or truncated
                if truncated and not terminated:
                    with torch.no_grad():
                        terminal_value = model.policy.predict_values(
                            obs_as_tensor(np.asarray(next_observation, dtype=np.float32).reshape(1, -1),
                                          model.device))
                    reward += model.gamma * float(terminal_value.cpu().numpy().reshape(-1)[0])
                transitions.append((obs_batch, raw_action, np.asarray([reward], dtype=np.float32),
                                    np.asarray([episode_start], dtype=np.float32), values.cpu(), log_probs.cpu()))
                episode_step_count += 1
                observation = next_observation
                episode_start = done
                if done:
                    metrics = info["episode_metrics"]
                    row = {"episode": episode + 1, "episode_seed": seed + episode,
                           "repetition_target": repetitions,
                           "timesteps": episode_step_count, **metrics}
                    episode_rows.append(row)
                    recent_rewards.append(metrics["episode_reward"])
                    print(f"[training episode {episode + 1}/{repetitions}] reward "
                          f"{metrics['episode_reward']:.1f} avg{len(recent_rewards)}="
                          f"{np.mean(recent_rewards):.1f} success={bool(metrics['success'])} "
                          f"victims={metrics['victims_found']:.0f}/{metrics['victims_total']:.0f} "
                          f"coverage={metrics['coverage']:.1%} "
                          f"rooms={metrics['rooms_discovered']:.0f}/{metrics['rooms_total']:.0f} "
                          f"collisions={metrics['collisions']:.0f} fire={metrics['fire_exposure']:.1f}s/"
                          f"{metrics['fire_danger_steps']:.0f} steps total_steps={metrics['steps']:.0f}",
                          flush=True)
                    break
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=metrics_fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in metrics_fields} for row in episode_rows)

        rollout = RolloutBuffer(len(transitions), training_env.observation_space, training_env.action_space,
                                device=model.device, gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=1)
        for transition in transitions:
            rollout.add(*transition)
        rollout.compute_returns_and_advantage(last_values=torch.zeros(1, device=model.device), dones=np.array([True]))
        model.rollout_buffer = rollout
        model.num_timesteps += episode_step_count
        model._total_timesteps = model.num_timesteps
        model._current_progress_remaining = 1.0
        model.set_logger(configure_logger(verbose=verbose, tb_log_name="ppo_repetitions",
                                          reset_num_timesteps=False))
        model.train()
        model.policy.set_training_mode(False)
        evaluate_and_save(f"after_{repetitions}_episodes")
        elapsed = time.time() - started
        model.save(str(out / "final_model"))
        model.save(str(checkpoints / f"ppo_rescue_{model.num_timesteps}_steps"))
        summary = {
            "seed": seed,
            "requested_repetitions": repetitions,
            "completed_episodes": repetitions,
            "ppo_updates": 1,
            "training_steps": episode_step_count,
            "total_timesteps": int(model.num_timesteps),
            "training_seconds": elapsed,
            "evaluation_episodes": eval_episodes,
            "eval_history": history,
            "final_eval": history[-1],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        return summary
    finally:
        training_env.close()
        eval_env.close()
        venv.close()
