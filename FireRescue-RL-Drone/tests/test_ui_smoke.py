"""Optional end-to-end UI smoke test (needs Ursina and a GL context).

Enable with:  FIRERESCUE_UI_TESTS=1 pytest tests/test_ui_smoke.py
(on Linux CI wrap with `xvfb-run -a` and LIBGL_ALWAYS_SOFTWARE=1).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FIRERESCUE_UI_TESTS") != "1",
                                reason="set FIRERESCUE_UI_TESTS=1 to run the Ursina smoke test")


def test_ui_runs_and_saves_screenshot(tmp_path):
    shot = tmp_path / "shot.png"
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "firerescue", "--auto", "--size", "640x360", "--frames", "40",
                        "--fixed-dt", "0.05", "--screenshot", str(shot)],
                       cwd=root, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "frames=40" in r.stdout and shot.exists() and shot.stat().st_size > 5000
