"""Train a PPO agent on RescueDroneGym.

    python scripts/train_agent.py                                   # configs/ppo_small.yaml
    python scripts/train_agent.py --config configs/ppo_long.yaml --out runs/long
    python scripts/train_agent.py --timesteps 5000 --out runs/smoke # quick check
    tensorboard --logdir runs                                       # view logs
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from firerescue.rl.training import TRAINING_REPETITIONS, load_run_config, train, train_repetitions
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install -r requirements.txt", file=sys.stderr)
        return 3
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo_small.yaml")
    ap.add_argument("--out", type=Path, help="output folder (default runs/<config>-<timestamp>)")
    ap.add_argument("--timesteps", type=int, help="override total_timesteps")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--resume", type=Path, help="continue from a saved model .zip")
    ap.add_argument("--repetitions", type=int, nargs="?", const=TRAINING_REPETITIONS,
                    help=f"train until this many complete episodes finish using full PPO rollouts "
                         f"(default when used without a value: {TRAINING_REPETITIONS})")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    out = a.out or ROOT / "runs" / f"{a.config.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"config={a.config}  out={out}")
    if a.repetitions is not None:
        s = train_repetitions(cfg, out, a.repetitions, a.seed, a.resume, verbose=0 if a.quiet else 1)
        print(f"\nDone: requested {s['requested_repetitions']} repetitions; "
              f"completed {s['completed_episodes']} episodes in {s['total_timesteps']} steps")
    else:
        s = train(cfg, out, a.timesteps, a.seed, a.resume, verbose=0 if a.quiet else 1)
        print(f"\nDone: {s['total_timesteps']} steps in {s['train_seconds']:.0f}s "
              f"({s['steps_per_second']:.0f} steps/s)")
    print(f"Models: {out/'final_model.zip'}, {out/'best_model.zip'}, {out/'latest_model.zip'}")
    print(f"Metrics: {out/'training_metrics.csv'}")
    print("NOTE: judge the policy by scripts/evaluate_agent.py, not by training reward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
