"""Run the baseline controller headless and (optionally) plot mission metrics.

    python scripts/headless_demo.py [--plot out.png]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firerescue import MissionStatus, SimWorld  # noqa: E402
from firerescue.controllers import BaselineController  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", type=Path)
    a = ap.parse_args()
    w = SimWorld(); ctl = BaselineController(w)
    while w.status == MissionStatus.RUNNING:
        w.step(ctl.command())
    print(w.metrics.summary(w))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = w.metrics.series
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(s["x"], s["z"]); ax[0].set(title="Drone path (top view)", xlabel="x [m]", ylabel="z [m]", aspect="equal")
        for f in w.fires: ax[0].add_patch(plt.Circle((f.position[0], f.position[2]), f.danger_radius, color="r", alpha=.2))
        for v in w.victims: ax[0].plot(v.position[0], v.position[2], "g^")
        ax[1].plot(s["t"], s["discovered"]); ax[1].set(title="Victims discovered", xlabel="t [s]")
        ax[2].plot(s["t"], s["nearest_fire"], label="nearest fire"); ax[2].plot(s["t"], s["nearest_victim"], label="nearest undiscovered victim")
        ax[2].set(title="Distances [m]", xlabel="t [s]"); ax[2].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=110); print("saved", a.plot)


if __name__ == "__main__":
    main()
