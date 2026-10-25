# RescueDroneGym interface (STEP 2)

Module: `firerescue/envs/rescue_drone_gym.py` — `RescueDroneGym(scenario=None, env_config=None, reward_config=None, render_mode=None, world=None)`.
It wraps the Step 1 `SimWorld` (drone, obstacles, fires, victims, synthetic thermal sensor) directly: no engine objects,
no ROS 2, no Gazebo. Also registered as `gym.make("RescueDrone-v0")`.

## Episode

* One `step` = `control_dt` (0.1 s) of simulated time, executed as physics sub-steps of `physics_dt` (1/30 s).
* **Terminated:** all victims found (success), heat-exposure limit exceeded (6 s inside danger spheres),
  `max_collisions` reached, or the Step 1 600 s mission limit. **Truncated:** `max_episode_steps` (default 1500 = 150 s).
* `reset(seed=s)` is deterministic. With `random_start: true` the start position/yaw is drawn from the env's seeded RNG
  among collision-free, fire-free cells; with `false` it is always the scenario start. `reset(options={"start": (x, y, z, yaw)})` forces a start.
* `info["episode_metrics"]` (on the last step): success, episode_reward, steps, victims_found/total, collisions, coverage,
  rooms_discovered/total, fire_danger_steps, overheated, truncated, invalid_actions, sim_time.

## Action space — `Box(-1, 1, (4,))`: `[vx, vy, vz, yaw_rate]`

Body-frame, using the simulator axes: `vx` right(+)/left(−), `vy` up(+)/down(−), `vz` forward(+)/backward(−),
`yaw_rate` turn right(+)/left(−). Scaled by `max_speed_xz` / `max_speed_y` / `max_yaw_rate_deg` from `configs/env.yaml`
(default: the drone's own limits; larger values are rejected). The Step 1 physics still applies acceleration limits,
altitude limits and collision response (sliding along surfaces).
Invalid input: wrong shape → `ValueError`; NaN/inf entries → replaced by 0 (`info["invalid_action"]`, counted);
out-of-range values → clipped (`info["action_clipped"]`).

## Observation space — `Box(-1, 1, ({N},), float32)`

Built by `ObservationBuilder` (no engine objects). "Body frame" = (x: drone right, y: up, z: drone forward).

| Feature | Size | Description |
|---|---|---|
| `pos` | 3 | Position / building size, mapped to [-1, 1] (x, altitude, z) |
| `vel_body` | 3 | Velocity in body frame / drone speed limits (right, up, forward) |
| `yaw_sin_cos` | 2 | sin and cos of the yaw angle |
| `goal_vec_body` | 3 | Vector to the look-ahead point on the shortest path to the current target viewpoint (body frame, / 10 m, clipped); y = goal altitude error |
| `goal_dist` | 1 | Shortest-path distance to the target viewpoint / 30 m (clipped to 1; 1 if none) |
| `proximity_rays` | 14 | 1 - clearance/ray_range for n_rays horizontal rays (clockwise from forward) + up + down; 1 = touching, 0 = nothing within range |
| `fires` | 12 | 3 nearest fires: body-frame vector / 10 m (3) + danger level 1 - d/(1.5 R) in [0,1]; zero padded |
| `thermal_peak` | 1 | Strongest synthetic thermal signal among UNDISCOVERED victims with line of sight (0 if none) |
| `thermal_bearing` | 3 | Unit body-frame bearing to that peak (zeros if none) |
| `thermal_interference` | 1 | Fire-induced thermal interference at the drone / 0.5 (clipped) |
| `thermal_grid` | 36 | 6x6 body-aligned (row = forward, col = right) synthetic heat map of undiscovered victims + fires within sensor range, line-of-sight masked, / 1.2 |
| `room_explored` | 6 | Explored fraction of every room (config order) |
| `coverage_global` | 1 | Explored fraction of the whole building |
| `victims_remaining` | 1 | Undiscovered victims / total victims |
| `time_left` | 1 | Fraction of the episode step budget remaining |
| `prev_action` | 4 | Previous (sanitised) action [vx, vy, vz, yaw_rate] |
| `collision` | 1 | 1 if the drone is touching a surface this step |

Total observation size: 93


Print it yourself: `python -c "from firerescue.envs.observation import describe; print(describe())"`.
The size depends on the scenario (number of rooms) and `EnvConfig` (rays, fires, thermal grid).

### The navigation goal (read this before interpreting results)

`GoalManager` keeps a list of coverage *viewpoints* (a coarse grid of free points inside every room) and a
shortest-path distance field to each (Dijkstra on the Step 1 planner grid, fire zones excluded). The current goal is the
nearest not-yet-explored viewpoint (sticky until reached or explored). The policy sees the direction of the shortest path
(look-ahead point) and its length. **This is a global-planner hint**: the agent is *not* learning to explore from scratch or
to find victims without guidance; it must learn local flight control (stay clear of walls/furniture and fire, handle doorways,
altitude, yaw) and use the thermal features to react to victims. Comparisons with the Step 1 baseline should keep this in mind.

## Reward — `configs/reward.yaml` (`RewardConfig`)

| Term | Default | Applied |
|---|---|---|
| `new_room` | +5 | once per room (≥30 % of its cells explored) |
| `new_victim` | +25 | once per victim ID |
| `new_coverage_cell` | +0.1 | once per explored 1 m cell |
| `mission_complete` (+ `success_time_bonus` · fraction of time left) | +60 (+20) | on success |
| `progress` | +1 / m | signed shortest-path progress to the current goal, clipped to ±0.4 m/step |
| `collision` / `collision_contact` | −5 / −0.2 | per new collision event / per step touching a surface |
| `fire_warning` / `fire_danger` | −0.05 / −1 | per step in the warning / danger zone |
| `fire_failure`, `collision_limit_failure` | −30, −20 | terminal failures |
| `time_step` | −0.02 | every step |
| `oscillation` | −0.1 | abrupt action reversal (cos < −0.3, both action norms > 0.3) |
| `dither` | −0.2 | moved ≥1 m within 2 s but net displacement ≤0.3 m |
| `stagnation` | −0.1 | after 30 steps without progress or new coverage |
| `timeout` | −10 | episode truncated without success |

**Exploit prevention:** rooms, victims and cells are tracked in per-episode sets (spawn-time discoveries earn nothing);
discoveries persist across `reset_drone()` in the viewer; progress is *signed* (back-and-forth nets ≈0) and is 0 on the
step a goal changes; oscillation/dither/stagnation penalties cover shaking and looping. All are unit-tested.

## Rendering

`render_mode="rgb_array"` returns a 400×280 top-down image (rooms, unexplored cells darkened, fires, victims, trajectory,
goal, drone); `"ansi"` returns a one-line status string. For the interactive 3D view use
`python -m firerescue --policy runs/<run>/best_model.zip`.
