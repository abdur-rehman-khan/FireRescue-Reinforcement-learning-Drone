"""Versioned storage and loading for manual flight demonstrations."""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np


DEMONSTRATION_VERSION = 1


class DemonstrationRecorder:
    """Collect transitions in memory and write compressed archives off the UI thread."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firerescue-demo")
        self._lock = threading.Lock()
        self._pending: List[Tuple[Path, Future]] = []
        self.active = False
        self.paused = False
        self._transitions: List[tuple] = []
        self._metadata: Dict = {}

    def start(self, observation, metadata: Optional[Dict] = None) -> str:
        if self.active:
            raise RuntimeError("A demonstration is already being recorded")
        self.active, self.paused = True, False
        self._transitions = []
        self._metadata = dict(metadata or {})
        self._metadata.update(
            schema_version=DEMONSTRATION_VERSION,
            episode_id=uuid4().hex,
            started_at=datetime.now(timezone.utc).isoformat(),
            observation_size=int(np.asarray(observation).size),
            action_size=4,
        )
        self._metadata.setdefault("observation_version", 1)
        self._metadata.setdefault("action_order", ["right", "up", "forward", "yaw_rate"])
        return self._metadata["episode_id"]

    def add_transition(self, observation, action, next_observation, reward, terminated, truncated, info):
        if not self.active:
            return
        obs = np.asarray(observation, dtype=np.float32)
        act = np.asarray(action, dtype=np.float32)
        nxt = np.asarray(next_observation, dtype=np.float32)
        if obs.ndim != 1 or nxt.shape != obs.shape:
            raise ValueError("Demonstration observations must be matching one-dimensional arrays")
        if act.shape != (4,) or not np.all(np.isfinite(act)) or np.any(np.abs(act) > 1.00001):
            raise ValueError("Demonstration actions must be four finite values inside the PPO action range [-1, 1]")
        self._transitions.append((
            obs.copy(), act.copy(), nxt.copy(), float(reward), bool(terminated), bool(truncated),
            json.dumps(info, default=_json_default, sort_keys=True),
        ))

    def finish(self, metadata: Optional[Dict] = None) -> Optional[Path]:
        if not self.active:
            return None
        self.active, self.paused = False, False
        if not self._transitions:
            self._transitions = []
            return None
        details = dict(self._metadata)
        details.update(metadata or {})
        details["finished_at"] = datetime.now(timezone.utc).isoformat()
        details["transitions"] = len(self._transitions)
        details["eligible_for_imitation"] = bool(
            details.get("safe", False) and (details.get("success", False) or details.get("useful", False))
        )
        rows = self._transitions
        self._transitions = []
        self._metadata = {}
        arrays = {
            "observations": np.stack([r[0] for r in rows]),
            "actions": np.stack([r[1] for r in rows]),
            "next_observations": np.stack([r[2] for r in rows]),
            "rewards": np.asarray([r[3] for r in rows], dtype=np.float32),
            "terminated": np.asarray([r[4] for r in rows], dtype=np.bool_),
            "truncated": np.asarray([r[5] for r in rows], dtype=np.bool_),
            "info_json": np.asarray([r[6] for r in rows], dtype=np.str_),
            "metadata_json": np.asarray(json.dumps(details, default=_json_default, sort_keys=True)),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"demo-{details['episode_id']}.npz"
        with self._lock:
            self._pending.append((path, self._executor.submit(_write_archive, path, arrays)))
        return path

    def poll_saves(self) -> Tuple[List[Path], List[Tuple[Path, Exception]]]:
        completed = []
        errors = []
        with self._lock:
            remaining = []
            for path, future in self._pending:
                if not future.done():
                    remaining.append((path, future))
                    continue
                try:
                    future.result()
                    completed.append(path)
                except Exception as exc:
                    errors.append((path, exc))
            self._pending = remaining
        return completed, errors

    def poll_errors(self) -> List[Tuple[Path, Exception]]:
        return self.poll_saves()[1]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        _, errors = self.poll_saves()
        if errors:
            path, error = errors[0]
            raise OSError(f"Could not save demonstration {path}: {error}") from error


def _write_archive(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_demonstrations(paths, include_useful: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load safe, successful demonstrations (plus useful ones when explicitly requested)."""
    candidates = _archive_paths(paths)
    observations, actions, metadata = [], [], []
    expected_obs = None
    for path in candidates:
        with np.load(path, allow_pickle=False) as archive:
            required = {"observations", "actions", "next_observations", "rewards", "terminated",
                        "truncated", "info_json", "metadata_json"}
            if not required.issubset(archive.files):
                raise ValueError(f"{path}: missing demonstration arrays {sorted(required - set(archive.files))}")
            details = json.loads(str(archive["metadata_json"].item()))
            if details.get("schema_version") != DEMONSTRATION_VERSION:
                raise ValueError(f"{path}: unsupported demonstration schema version")
            if details.get("action_size") != 4 or details.get("action_order") != [
                    "right", "up", "forward", "yaw_rate"]:
                raise ValueError(f"{path}: action representation does not match the PPO interface")
            eligible = details.get("success", False) or (include_useful and details.get("useful", False))
            if not details.get("safe", False) or not eligible:
                continue
            obs = np.asarray(archive["observations"], dtype=np.float32)
            act = np.asarray(archive["actions"], dtype=np.float32)
            if obs.ndim != 2:
                raise ValueError(f"{path}: invalid observation/action array shapes")
            n = obs.shape[0]
            if (act.shape != (n, 4) or n == 0
                    or archive["next_observations"].shape != obs.shape
                    or archive["rewards"].shape != (n,)
                    or archive["terminated"].shape != (n,)
                    or archive["truncated"].shape != (n,)
                    or archive["info_json"].shape != (n,)):
                raise ValueError(f"{path}: invalid observation/action array shapes")
            if details.get("observation_size") != obs.shape[1]:
                raise ValueError(f"{path}: metadata observation size does not match its arrays")
            if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act)):
                raise ValueError(f"{path}: demonstration contains non-finite data")
            if np.any(np.abs(act) > 1.00001):
                raise ValueError(f"{path}: actions are outside the PPO action range")
            if expected_obs is not None and obs.shape[1] != expected_obs:
                raise ValueError(f"{path}: observation size differs from previously loaded demonstrations")
            expected_obs = obs.shape[1]
            observations.append(obs)
            actions.append(act)
            metadata.append(details)
    if not observations:
        raise ValueError("No safe successful demonstrations found; record a successful or useful safe episode first")
    return np.concatenate(observations), np.concatenate(actions), metadata


def _archive_paths(paths) -> List[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    found: List[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            found.extend(sorted(path.glob("demo-*.npz")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return found


def describe_demonstrations(paths) -> List[Dict]:
    """Return metadata for all archives, including unsuccessful diagnostic recordings."""
    results = []
    for path in _archive_paths(paths):
        with np.load(path, allow_pickle=False) as archive:
            details = json.loads(str(archive["metadata_json"].item()))
        results.append({"file": str(path), **details})
    return results
