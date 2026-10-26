
# FireRescue RL Drone
### Dynamic 3D Fire-Rescue Simulation with User Guided Reinforcement Learning

An interactive 3D indoor fire rescue drone simulation developed using **Python, Ursina, Panda3D, and NumPy**. The project simulates a quadrotor navigating a fire-affected building, avoiding obstacles, detecting simulated victims, and responding to changing emergency scenarios.

The long-term objective is to develop an autonomous AI agent that learns from **user demonstrations, imitation learning, and reinforcement learning** to perform rescue missions independently.

<p align="center">
  <img src="assets/images/firerescue-simulation.png" alt="FireRescue RL Drone Simulation" width="90%">
</p>

<p align="center">
  <strong>Dynamic Fire & Victim Positioning | Manual Flight | AI Agent Learning | Autonomous Rescue</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10--3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/3D-Ursina-orange" alt="Ursina">
  <img src="https://img.shields.io/badge/AI-Reinforcement%20Learning-purple" alt="Reinforcement Learning">
  <img src="https://img.shields.io/badge/Domain-Autonomous%20Robotics-red" alt="Autonomous Robotics">
</p>

---

## 🎬 Simulation Demo

Watch the drone navigate the environment and interact with simulated fire hazards and victims.

<p align="center">
  <a href="assets/videos/firerescue-demo.mp4">
    <img src="assets/images/firerescue-video-thumbnail.png" alt="FireRescue Drone Demo" width="85%">
  </a>
</p>

<p align="center">
  <a href="assets/videos/firerescue-demo.mp4">▶ Watch the Full Simulation Video</a>
</p>

---

## 🚀 Key Features

### 🌐 1. Interactive 3D Environment
- Indoor building environment measuring approximately 20 m × 14 m.
- Six spaces connected through passages.
- Furniture, walls, and structural obstacles.
- Interactive visualization and mission telemetry.
- Configurable scenarios for repeatable experiments.

### 🔥 2. Dynamic Fire Positioning
- Configure fire sources at different locations within the building.
- Change fire locations between missions and training episodes.
- Simulate hazard zones and proximity-based warnings.
- Test autonomous navigation under different fire configurations.
- Support thermal interference and optional sensor noise.

### 🧍 3. Dynamic Human Victim Positioning
- Place simulated victims at different locations and in different rooms.
- Support standing and prone victim representations.
- Change victim locations between missions to prevent memorization of fixed positions.
- Track discovered victims and mission progress.
- Use synthetic thermal observations to assist victim discovery.

### 🚁 4. Manual Drone Control
- Four-rotor quadrotor visualization.
- Keyboard-controlled drone movement.
- Kinematic motion and acceleration limits.
- Collision detection for walls, furniture, and obstacles.
- Manual exploration and user-guided rescue missions.

### 🧠 5. AI Agent Learning from User Actions
The learning system is designed to learn drone-control behavior from user demonstrations.

- **Demonstration recording:** Record user actions and corresponding environment observations.
- **Behavioral cloning:** Train an initial policy to imitate demonstrated actions.
- **PPO reinforcement learning:** Fine-tune the policy through interaction with the environment.
- **Repeated training episodes:** Improve the policy through valid training rollouts.
- **Persistent model checkpoints:** Save and reload trained policies.
- **Autonomous execution:** Use the trained policy to control the drone in Auto Mode.
- **Performance evaluation:** Compare the learned agent with manual control and a conventional navigation baseline.

The objective is to learn general rescue behavior rather than simply replaying a fixed sequence of movements.

### 🧭 6. Navigation and Obstacle Avoidance
- A* path-planning baseline.
- Collision and obstacle monitoring.
- Fire aware navigation experiments.
- Exploration and victim-search objectives.
- Evaluation across different emergency scenarios.

### 📊 7. Mission Monitoring and Evaluation
- Drone position and telemetry.
- Victim discovery tracking.
- Collision counts.
- Mission completion status.
- Environment coverage.
- Episode reward and success rate, where supported by the evaluation code.

---



## 🧠 AI Learning Workflow

The intended training pipeline combines user demonstrations with reinforcement learning.

```text
          3D Fire-Rescue Environment
                       |
                       v
             User Controls Drone
                       |
                       v
           Record Observations/Actions
                       |
                       v
             Demonstration Dataset
                       |
                       v
               Behavioral Cloning
                       |
                       v
                 Initial Policy
                       |
                       v
               PPO Fine-Tuning
                       |
                       v
             Repeated Training Episodes
                       |
                       v
              Evaluate Model Performance
                       |
                       v
              Save Best Model Checkpoint
                       |
                       v
             Autonomous Rescue Missions
                       |
                       v
          Test New Fire/Victim Locations
```

### How the Agent Learns

1. The environment provides observations about the drone, obstacles, hazards, and sensor readings.
2. The user manually controls the drone during a rescue mission.
3. The application records observations, selected actions, rewards, next observations, and episode termination information.
4. Behavioral cloning trains a policy to imitate the demonstrated actions.
5. PPO fine-tunes the policy through additional interaction with the environment.
6. The trained model is evaluated on consistent test scenarios.
7. The model checkpoint is saved and loaded for autonomous missions.
8. New fire and victim configurations are used to assess generalization.

**Note:** Recording user actions alone does not train an AI model. Demonstration storage, policy training, PPO updates, checkpoint management, and autonomous inference must be implemented in the application.

---

## 🎯 Dynamic Scenario Testing

Dynamic positioning allows the agent to be tested in situations it has not previously encountered.

| Scenario variable | Purpose |
|---|---|
| Fire locations | Test hazard-aware navigation |
| Victim locations | Test generalization in victim search |
| Drone starting position | Reduce dependence on a fixed starting point |
| Obstacle configurations | Evaluate collision avoidance |
| Sensor noise | Test robustness to imperfect observations |
| Mission conditions | Compare rescue performance across episodes |

Scenario randomization should ensure that fires, victims, obstacles, and starting positions remain within valid regions of the environment.

---

## 📈 Evaluation Metrics

The following metrics can be used to measure learning progress:

- **Mission success rate:** Percentage of completed rescue missions.
- **Victim discovery rate:** Percentage of victims located.
- **Collision count:** Number of collisions per episode.
- **Environment coverage:** Percentage of the environment explored.
- **Episode reward:** Total reward accumulated during a mission.
- **Mission duration:** Number of simulation steps required.
- **Generalization performance:** Success rate on unseen scenarios.

Compare manual control, the A* baseline, the imitation-learning policy, and the PPO-trained agent under the same evaluation conditions.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core simulation and learning logic |
| Ursina | Interactive 3D environment |
| Panda3D | 3D rendering |
| NumPy | Numerical computation |
| A* | Baseline path planning |
| Stable-Baselines3 PPO | Reinforcement learning, if implemented |
| PyTorch | Neural network training, if required |
| JSON | Scenario configuration |
| pytest | Automated testing |

---



## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abdur-rehman-khan/FireRescue-Reinforcement-learning-Drone.git
cd FireRescue-Reinforcement-learning-Drone
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Simulator

Run the actual entry-point script in your project. For example:

```bash
python main.py
```

### 5. Run the Headless Demonstration

```bash
python scripts/headless_demo.py
```

### 6. Run Tests

```bash
pytest
```

*Adjust commands to match the entry points and dependencies actually provided.*

---

## 🔬 Research Objectives

This project supports research into:

1. Learning drone-control policies from user demonstrations.
2. Combining behavioral cloning with PPO reinforcement learning.
3. Autonomous navigation in dynamic indoor rescue environments.
4. Fire-aware obstacle avoidance and victim discovery.
5. Generalization to previously unseen fire and victim positions.
6. Comparing learned policies with conventional navigation algorithms.
7. Evaluating autonomous emergency-response strategies in simulation.

The environment uses simulated victims and synthetic sensing. Successful simulation results do not by themselves establish real-world flight safety or rescue capability.

---


## 🤝 Contributing

Contributions, bug reports, and research discussions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing your contribution.

---


## 👨‍💻 Author

**Abdur Rehman Khan**

AI Research Scientist | Machine Learning | Reinforcement Learning | Autonomous Robotics

- **GitHub:** [abdur-rehman-khan](https://github.com/abdur-rehman-khan)
- **Google Scholar:** [Research Publications](https://scholar.google.com/citations?user=VJGeMo4AAAAJ)
- **LinkedIn:** [Professional Profile](https://www.linkedin.com/in/abdur-rehman-khan-msai/)

---

<p align="center">
  <strong>FireRescue RL Drone — Learning to Navigate Dynamic Emergency Environments</strong>
</p>
