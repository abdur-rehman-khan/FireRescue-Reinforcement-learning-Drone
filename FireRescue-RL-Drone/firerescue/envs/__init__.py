"""PLACEHOLDER for STEP 2 (Gymnasium + PPO).  Intentionally empty in STEP 1.

Planned mapping: action = DroneCommand.as_array() (4 floats in [-1, 1]);
observation built from SimWorld (drone pose/velocity, thermal readings, fire proximity);
reward from victim discovery, collisions and fire exposure; ``SimWorld.step`` / ``reset``
are already side-effect free of any rendering.
"""
