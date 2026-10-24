# FireRescue-RL-Drone — STEP 1: standalone 3D fire-rescue drone simulator

An interactive 3D indoor fire-rescue simulator (Python + Ursina + NumPy). A quadrotor flies through a
20 m × 14 m, six-space building, avoids obstacles, observes fire zones and finds simulated victims with a
**synthetic** thermal sensor. It runs independently of ROS 2 and Gazebo and is structured for STEP 2
(Gymnasium + PPO).

> **The thermal sensor is an analytic model, not a thermal-camera simulation.** The baseline controller uses
> ground-truth map knowledge (obstacles and fire positions, not victims). The drone is kinematic (velocity
> set-points with acceleration limits), not a rotor-level dynamics model.

## Install (Windows + VS Code)

Use Python **3.10–3.12** (64-bit). Newer Python versions work only if `panda3d` ships a wheel for them.

```powershell
cd FireRescue-RL-Drone
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .venv*. Launch configs are in `.vscode/launch.json`.

## Run

```powershell
python -m firerescue                    # manual start
python -m firerescue --auto             # start with the baseline autonomous controller
python -m firerescue --scenario my.json # custom scenario (see below)
python -m firerescue --flat             # unlit fallback if shading looks wrong on your GPU
python -m firerescue --export-default-scenario default.json
python scripts/headless_demo.py --plot metrics.png   # no window; prints/plots mission metrics
```

## Tests

```powershell
pytest -q                               # 57 core tests, headless (~20 s)
$env:FIRERESCUE_UI_TESTS="1"; pytest -q # also runs the Ursina smoke test (needs a GL context)
```

## Controls

| Input | Action |
|---|---|
| W / S | forward / backward (drone heading) |
| A / D | strafe left / right |
| Space / Left Shift | up / down |
| Q / E | yaw left / right |
| Right-mouse drag, arrow keys | orbit camera |
| Mouse wheel | zoom |
| C / O | reset camera / top-down overview |
| M | manual ⇄ autonomous baseline |
| T | reset drone to start (mission progress kept) |
| R | restart whole mission |
| H, Esc | toggle help, quit |

HUD shows position/altitude, velocity, victims found, distance to nearest undiscovered victim (ground truth),
fire proximity warning (CLEAR/WARNING/DANGER), control mode, mission status, collisions, and the synthetic
thermal panel. Collisions flash the drone red, tint the screen, shake the camera and are counted.

Mission rules: success when all victims are found; failure after 6 s cumulative inside a fire danger sphere or
after 600 s. Door lintels (2.1 m), a hanging beam and the ceiling (2.75 m limit) punish careless altitude.

## Architecture

```
firerescue/
  config.py       dataclass config, JSON load/save, validation (ConfigError)
  scenario.py     deterministic default scenario
  obstacles.py    walls/doors/lintels/furniture/debris -> axis-aligned boxes
  collision.py    sphere-vs-AABB resolution (sliding), segment line-of-sight
  drone.py        DroneCommand + kinematic quadrotor + collision response
  fire.py         fire sources, danger radii, proximity levels
  victims.py      victims (ID, position, standing/prone pose)
  sensors.py      SYNTHETIC thermal model + heat grid for the UI
  world.py        SimWorld: step/reset/reset_drone, mission status, events (no rendering)
  metrics.py      time-series recorder
  controllers/    manual key mapping, A* planner, baseline autonomous controller
  ui/             Ursina front-end (scene, actors, camera rig, HUD, app)
  envs/           placeholder for STEP 2
tests/            pytest suite
```
Coordinates: +x right, +y up, +z forward (same as Ursina), metres; yaw 0 faces +z, yaw 90 faces +x.

### Synthetic thermal model
`signal = base · pose_factor / (1 + (d/d_ref)²)` (prone = 0.7), `interference = gain · Σ fire heat at the drone`,
detected when `signal − interference ≥ threshold`, within 5 m and with clear line of sight. Fires therefore
mask nearby victims. Optional seeded noise (`sensor.noise_std`).

### Custom scenarios
Export the default, edit drone `start_position`, `fires` (position, `danger_radius`), `victims`
(position, pose `standing|prone`), furniture, walls; load with `--scenario`. Invalid files raise a clear error.

## STEP 2 readiness
`DroneCommand.as_array()` is the 4-D action in [-1, 1]; `SimWorld.step/reset` are render-free and
deterministic. Events, thermal readings and metrics are available for observations and rewards.

## Known limitations
- Windows are visual only (exterior walls are solid); there is no ceiling mesh (cut-away view).
- Furniture is axis-aligned (rotations in multiples of 90°). Smoke/fire are visual only (no airflow/damage).
- Requires OpenGL 3.x; remote-desktop/VM sessions may fail to open the window. Audio is unused.
- Exact framerate on your GPU is untested by the author (see report); `--flat` is the fallback.

MIT licensed.
