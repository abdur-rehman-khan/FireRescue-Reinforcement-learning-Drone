"""Behavior-clone a PPO policy from safe manual demonstration archives.

    python scripts/train_from_demos.py --model runs/medium/best_model.zip
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="compatible Stable-Baselines3 PPO model")
    parser.add_argument("--demos", type=Path, default=ROOT / "runs" / "demonstrations")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "imitation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--env-config", type=Path, help="environment YAML (default: beside source model or project config)")
    parser.add_argument("--reward-config", type=Path, help="reward YAML (default: beside source model or project config)")
    parser.add_argument("--scenario", type=Path, help="optional JSON mission scenario")
    parser.add_argument("--include-useful", action="store_true",
                        help="also imitate safe demonstrations marked useful but not successful")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--list-demos", action="store_true", help="list archives, including unsuccessful diagnostics")
    args = parser.parse_args()
    if args.list_demos:
        from firerescue.rl.demonstrations import describe_demonstrations
        try:
            for demo in describe_demonstrations(args.demos):
                metrics = demo.get("episode_metrics", {})
                print(f"{demo['file']}: {demo.get('transitions', 0)} transitions, "
                      f"success={demo.get('success', False)}, useful={demo.get('useful', False)}, "
                      f"safe={demo.get('safe', False)}, victims={metrics.get('victims_found', 0)}, "
                      f"coverage={metrics.get('coverage', 0):.1%}, eligible={demo.get('eligible_for_imitation', False)}")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Cannot list demonstrations: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.model is None:
        parser.error("--model is required unless --list-demos is used")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.eval_episodes < 1:
        parser.error("epochs, batch size and evaluation episodes must be positive; learning rate must be > 0")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    try:
        import torch
        from firerescue.rl.demonstrations import load_demonstrations
        from firerescue.rl.training import load_model
        from firerescue.envs import RescueDroneGym, load_env_config, load_reward_config
        from firerescue.config import ScenarioConfig
        from firerescue.rl.training import write_model_metadata
        from firerescue.rl.evaluation import evaluate, model_policy, save_results
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install -r requirements.txt", file=sys.stderr)
        return 3

    try:
        observations, actions, demos = load_demonstrations(args.demos, include_useful=args.include_useful)
        model = load_model(args.model, device="cpu")
        obs_size = int(model.observation_space.shape[0])
        if observations.shape[1] != obs_size or model.action_space.shape != (4,):
            raise ValueError(f"dataset/model space mismatch: demos use {observations.shape[1]} observations and "
                             f"4 actions; model expects {obs_size} observations and "
                             f"{model.action_space.shape} actions")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot load demonstrations/model: {exc}", file=sys.stderr)
        return 2

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "original_model.zip"
    if args.model.resolve() != backup.resolve() and not backup.exists():
        shutil.copy2(args.model, backup)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(observations))
    split = max(1, int(len(order) * 0.9))
    train_indices = order[:split]
    validation_indices = order[split:]
    obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    acts = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    losses = []
    validation_loss = None
    model.policy.set_training_mode(True)
    for _ in range(args.epochs):
        rng.shuffle(train_indices)
        epoch_loss = []
        for start in range(0, len(train_indices), args.batch_size):
            batch = train_indices[start:start + args.batch_size]
            predicted = model.policy.get_distribution(obs[batch]).distribution.mean
            loss = torch.nn.functional.mse_loss(torch.clamp(predicted, -1.0, 1.0), acts[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_loss)))
        if len(validation_indices):
            with torch.no_grad():
                predicted = model.policy.get_distribution(obs[validation_indices]).distribution.mean
                validation_loss = float(torch.nn.functional.mse_loss(
                    torch.clamp(predicted, -1.0, 1.0), acts[validation_indices]).cpu())
    model.policy.set_training_mode(False)

    model.save(str(out / "imitation_model"))
    model.save(str(out / "latest_model"))
    model_run = args.model.parent.parent if args.model.parent.name == "checkpoints" else args.model.parent
    env_source = args.env_config or (model_run / "env.yaml" if (model_run / "env.yaml").exists()
                                     else ROOT / "configs" / "env.yaml")
    reward_source = args.reward_config or (model_run / "reward.yaml" if (model_run / "reward.yaml").exists()
                                           else ROOT / "configs" / "reward.yaml")
    for source, name in ((env_source, "env.yaml"), (reward_source, "reward.yaml")):
        if source.resolve() != (out / name).resolve():
            shutil.copy2(source, out / name)
    env_cfg = load_env_config(env_source)
    reward_cfg = load_reward_config(reward_source)
    env = RescueDroneGym(ScenarioConfig.from_json(args.scenario) if args.scenario else None,
                         env_config=env_cfg, reward_config=reward_cfg)
    write_model_metadata(out, model, env)
    try:
        evaluation = evaluate(env, model_policy(model, deterministic=True), args.eval_episodes, args.seed)
        save_results(out / "imitation_evaluation.json", {"imitation_ppo": evaluation})
    finally:
        env.close()
    report = {
        "method": "behavioral cloning of the PPO policy's deterministic action mean",
        "source_model": str(args.model.resolve()),
        "backup_model": str(backup.resolve()),
        "demonstrations": demos,
        "transition_count": int(len(observations)),
        "observation_size": obs_size,
        "action_size": 4,
        "epochs": args.epochs,
        "training_loss": losses,
        "heldout_transition_loss": validation_loss,
        "evaluation_episodes": args.eval_episodes,
        "evaluation_summary": evaluation["summary"],
    }
    (out / "imitation_summary.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"Cloned {len(observations)} transitions from {len(demos)} safe demonstration(s).")
    print(f"Model: {out / 'imitation_model.zip'}; original backup: {backup}")
    print(f"Evaluation: {evaluation['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
