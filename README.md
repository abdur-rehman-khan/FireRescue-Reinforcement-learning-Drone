# FireRescue RL Drone

## Step 1: Standalone 3D Fire-Rescue Drone Simulator

An interactive 3D indoor fire-rescue simulation built with **Python, Ursina, Panda3D, and NumPy**. The project simulates a quadrotor navigating a fire-affected building, avoiding obstacles, monitoring fire hazards, and discovering simulated human victims using a synthetic thermal-sensing model.

The standalone simulator provides the foundation for future autonomous robotics and reinforcement learning research.

See [docs/LEARNING_WORKFLOW.md](docs/LEARNING_WORKFLOW.md) for manual recording, behavioral cloning, repeated PPO training, and fair model evaluation.

## 🚀 Features

- **3D Building Environment:** A 20 m × 14 m building containing six spaces and connected passages.
- **Quadrotor Simulation:** Four-rotor visual model with kinematic movement and acceleration limits.
- **Manual Flight Controls:** Navigate the building using keyboard inputs.
- **Collision Detection:** Detect collisions with walls, furniture, and structural obstacles.
- **Fire Hazard Monitoring:** Configurable fire sources, danger zones, and proximity warnings.
- **Synthetic Thermal Sensing:** Simulated heat signatures with distance attenuation, fire interference, and optional sensor noise.
- **Human Victim Simulation:** Standing and prone victim poses with discovery tracking.
- **Autonomous Baseline Controller:** Conventional A* path planning using map-based obstacle information.
- **Interactive HUD:** Drone telemetry, victim discovery, collision counts, mission status, and thermal readings.
- **Custom Scenarios:** Export and load JSON-based environment configurations.
- **Headless Simulation:** Run mission demonstrations and collect metrics without opening the 3D interface.
- **Learning Workflow:** Record safe manual demonstrations, behavior-clone PPO policies, train repeated on-policy mission batches, and compare models fairly.

## 🛠️ Technology Stack

- Python 3.10–3.12
- Ursina
- Panda3D
- NumPy
- A* Path Planning
- JSON Configuration
- pytest

## 📂 Project Structure

```text
FireRescue-RL-Drone/
├── firerescue/
│   ├── config.py
│   ├── scenario.py
│   ├── obstacles.py
│   ├── collision.py
│   ├── drone.py
│   ├── fire.py
│   ├── victims.py
│   ├── sensors.py
│   ├── world.py
│   ├── metrics.py
│   ├── controllers/
│   ├── ui/
│   └── envs/
├── scripts/
│   └── headless_demo.py
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
