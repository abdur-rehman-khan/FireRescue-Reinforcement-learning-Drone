"""PPO training pipeline (Stable-Baselines3): vector envs, checkpoints, TensorBoard, eval."""
from __future__ import annotations

import copy
import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ScenarioConfig
from ..envs import RescueDroneGym, load_env_config, load_reward_config
from .evaluation import evaluate, format_summary, model_policy

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        """Logs rolling episode metrics (success, collisions, victims, coverage) to TensorBoard."""

        def __init__(self, window: int = 50):
            super().__init__()
            self.buf = {k: deque(maxlen=window) for k in
                        ("success", "collisions", "victims_found", "coverage", "rooms_discovered", "fire_danger_steps")}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                m = info.get("episode_metrics")
                if m:
                    for k in self.buf:
                        self.buf[k].append(m[k])
                    for k, v in self.buf.items():
                        self.logger.record(f"rescue/{k}", float(np.mean(v)))
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
    callbacks, evcb = make_callbacks(cfg, out, n_envs)
    t0 = time.time()
    model.learn(total_timesteps=total, callback=callbacks, tb_log_name="ppo", reset_num_timesteps=not resume,
                progress_bar=False)
    elapsed = time.time() - t0
    model.save(str(out / "final_model"))
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
