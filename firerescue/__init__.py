"""FireRescue-RL-Drone: standalone 3D indoor fire-rescue quadrotor simulator (STEP 1).

The simulation core (everything except ``firerescue.ui``) depends only on NumPy and
can run headless, which is what the automated tests and the future Gymnasium
environment use.  The interactive 3D front-end lives in ``firerescue.ui`` (Ursina).
"""

__version__ = "0.1.0"

from .config import ScenarioConfig, ConfigError  # noqa: F401
from .scenario import default_scenario  # noqa: F401
from .world import SimWorld, MissionStatus  # noqa: F401
from .drone import DroneCommand  # noqa: F401
