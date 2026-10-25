"""Deterministic evaluation of a trained agent (optionally vs. random and the Step 1 baseline).

    python scripts/evaluate_agent.py --model runs/<run>/best_model.zip --episodes 20
    python scripts/evaluate_agent.py --model runs/<run>/best_model.zip --compare random baseline
    python scripts/evaluate_agent.py --model ... --fixed-start     # always start at the scenario start
Episode i uses seed (--seed + i), so every compared policy sees identical start states.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="SB3 PPO .zip (omit to evaluate only --compare policies)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--env-config", type=Path, help="default: env.yaml stored next to the model, else configs/env.yaml")
    ap.add_argument("--reward-config", type=Path)
    ap.add_argument("--scenario", type=Path, help="Step 1 JSON scenario")
    ap.add_argument("--fixed-start", action="store_true", help="disable random starts")
    ap.add_argument("--compare", nargs="*", choices=["random", "baseline"], default=[])
    ap.add_argument("--out", type=Path, help="results JSON (a CSV is written next to it)")
    a = ap.parse_args()
    if not a.model and not a.compare:
        ap.error("give --model and/or --compare")
    try:
        from firerescue.config import ScenarioConfig
        from firerescue.envs import RescueDroneGym, load_env_config, load_reward_config
        from firerescue.rl.evaluation import evaluate, format_summary, model_policy, random_policy, save_results
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install -r requirements.txt", file=sys.stderr)
        return 3

    run_dir = a.model.parent.parent if a.model and a.model.parent.name == "checkpoints" else (a.model.parent if a.model else None)
    pick = lambda given, name, default: given or (run_dir / name if run_dir and (run_dir / name).exists() else default)
    env_cfg = load_env_config(pick(a.env_config, "env.yaml", ROOT / "configs" / "env.yaml"))
    if a.fixed_start:
        env_cfg.random_start = False
    rew = load_reward_config(pick(a.reward_config, "reward.yaml", ROOT / "configs" / "reward.yaml"))
    env = RescueDroneGym(ScenarioConfig.from_json(a.scenario) if a.scenario else None, env_cfg, rew)

    results = {}
    if a.model:
        from firerescue.rl.training import load_model
        results["ppo"] = evaluate(env, model_policy(load_model(a.model), deterministic=True), a.episodes, a.seed)
    if "random" in a.compare:
        results["random"] = evaluate(env, random_policy(a.seed), a.episodes, a.seed)
    if "baseline" in a.compare:
        results["baseline"] = evaluate(env, None, a.episodes, a.seed, baseline=True)
    print(f"\nEvaluation: {a.episodes} deterministic episodes, seeds {a.seed}..{a.seed + a.episodes - 1}, "
          f"random_start={env_cfg.random_start}")
    for name, r in results.items():
        print(format_summary(name, r["summary"]))
    if a.out:
        save_results(a.out, results)
        print(f"saved {a.out} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
