# Demonstrations, imitation, and PPO

Install the RL extras if they are not already present:

```powershell
python -m pip install -e ".[rl]"
```

## Record manual demonstrations

Start the viewer in manual mode:

```powershell
python -m firerescue --manual
```

Press **G** to start or finish a demonstration and **P** to pause or resume it. Recording is active only in manual mode. The HUD shows recording/paused state. Compressed, versioned archives are written asynchronously under `runs/demonstrations` by default; use `--demo-dir PATH` to choose another directory.

Each transition contains the environment observation, normalized PPO action in `[right, up, forward, yaw_rate]` order, next observation, reward and reward components, collision/fire/victim/room events, and terminal flags. Actions and observations are sampled at the RL control interval. Episode metadata includes the start pose, config, timestamp, mission outcome, and safety/usefulness labels.

List saved demonstrations, including unsuccessful diagnostic data:

```powershell
python scripts/train_from_demos.py --list-demos
```

## Behavior cloning

Clone only safe successful demonstrations by default:

```powershell
python scripts/train_from_demos.py --model runs/medium/best_model.zip
```

Safe useful-but-incomplete demonstrations can be added explicitly:

```powershell
python scripts/train_from_demos.py --model runs/medium/best_model.zip --include-useful
```

The script validates the dataset against the PPO observation/action dimensions, trains the policy's deterministic action mean with supervised loss, and evaluates it on separate seeded episodes. It preserves the source at `runs/imitation/original_model.zip` and writes `imitation_model.zip`, `latest_model.zip`, and evaluation metrics in `runs/imitation/`. This is supervised pretraining; PPO is not trained by replaying demonstration transitions.

## PPO training batches

Train exactly 20 episodes of one selected mission, then make a PPO update from the complete on-policy episodic rollout:

```powershell
python scripts/train_agent.py --config configs/ppo_small.yaml --out runs/mission-batch --resume runs/imitation/imitation_model.zip --repetitions 20
```

Change `20` to configure the episode count. Using `--repetitions` without a value selects the default of 20. The batch uses one environment and controlled, varied seeds. PPO parameters are updated once after all requested episodes complete; truncations are bootstrapped before returns/advantages are computed. It does not replay old PPO trajectories. Evaluation is separate and runs at least 20 episodes without policy updates.

Artifacts include an `original_model.zip` backup when resuming, `latest_model.zip`, `best_model.zip`, `final_model.zip`, a checkpoint, `training_metrics.csv`, and evaluation JSON/CSV. An existing best policy is evaluated first and is retained unless the new model scores better by success rate, then victims found, then reward.

## Fair evaluation and automatic control

Compare original/improved PPO models and the baseline on the same evaluation seeds:

```powershell
python scripts/evaluate_agent.py --model runs/medium/best_model.zip --compare-model runs/imitation/imitation_model.zip --compare baseline --episodes 20 --seed 1000 --out runs/comparison.json
```

Evaluation reports reward mean/std, mission success rate, victims found, coverage, room exploration, collisions, fire-exposure seconds/steps, and episode length. It does not update weights.

In the viewer, **M** switches between manual and automatic control. **B** switches the automatic controller between PPO and the existing baseline A* controller when a compatible PPO model is loaded. The app prefers the most recently saved `latest_model.zip`, then `best_model.zip`, then `final_model.zip`; `--model PATH` selects an explicit policy. Without a compatible PPO file, baseline auto mode remains available.
