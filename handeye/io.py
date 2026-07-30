"""Pose-set data model and JSON / CSV readers and writers.

Conventions (the same as OpenCV's ``calibrateHandEye``):

* ``robot_poses[i]``  = base_T_gripper (gripper pose expressed in the robot base frame)
* ``target_poses[i]`` = camera_T_target (target pose expressed in the camera frame)
* eye-in-hand result X = gripper_T_camera; eye-to-hand result X = base_T_camera.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import se3

SETUPS = ("eye_in_hand", "eye_to_hand")


@dataclass
class PoseSet:
    robot: np.ndarray            # (N,4,4) base_T_gripper
    target: np.ndarray           # (N,4,4) camera_T_target
    setup: str = "eye_in_hand"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.robot = np.asarray(self.robot, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        if self.setup not in SETUPS:
            raise ValueError(f"setup must be one of {SETUPS}, got {self.setup!r}")
        if self.robot.shape != self.target.shape or self.robot.shape[1:] != (4, 4):
            raise ValueError("robot and target poses must both be (N,4,4) with the same N")
        if len(self.robot) < 3:
            raise ValueError("at least 3 poses are needed")

    def __len__(self) -> int:
        return len(self.robot)

    def solver_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (H, C) such that H_i X C_i is constant for the true X."""
        H = self.robot if self.setup == "eye_in_hand" else se3.inv(self.robot)
        return H, self.target


def _pose_from_obj(obj) -> np.ndarray:
    if isinstance(obj, dict):
        if "matrix" in obj:
            return np.asarray(obj["matrix"], dtype=float).reshape(4, 4)
        t = np.asarray(obj["t"], dtype=float)
        if "q" in obj:  # quaternion x, y, z, w
            R = Rotation.from_quat(obj["q"]).as_matrix()
        elif "rvec" in obj:
            R = Rotation.from_rotvec(obj["rvec"]).as_matrix()
        else:
            raise ValueError("pose needs 'matrix', or 't' with 'q' (xyzw) or 'rvec'")
        return se3.make(R, t)
    return np.asarray(obj, dtype=float).reshape(4, 4)


def load_json(path: str | Path) -> PoseSet:
    data = json.loads(Path(path).read_text())
    robot = np.stack([_pose_from_obj(p) for p in data["robot_poses"]])
    target = np.stack([_pose_from_obj(p) for p in data["target_poses"]])
    scale = float(data.get("translation_scale", 1.0))  # e.g. 0.001 for millimetres
    robot[:, :3, 3] *= scale
    target[:, :3, 3] *= scale
    return PoseSet(robot, target, data.get("setup", "eye_in_hand"))


CSV_COLUMNS = [f"{frame}_{c}" for frame in ("robot", "target")
               for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]


def load_csv(path: str | Path, setup: str = "eye_in_hand") -> PoseSet:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in CSV_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    arr = {c: np.array([float(r[c]) for r in rows]) for c in CSV_COLUMNS}
    poses = []
    for frame in ("robot", "target"):
        t = np.stack([arr[f"{frame}_{c}"] for c in "xyz"], axis=1)
        q = np.stack([arr[f"{frame}_q{c}"] for c in "xyzw"], axis=1)
        poses.append(se3.make(Rotation.from_quat(q).as_matrix(), t))
    return PoseSet(poses[0], poses[1], setup)


def load(path: str | Path, setup: str | None = None) -> PoseSet:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path, setup or "eye_in_hand")
    ps = load_json(path)
    if setup:
        ps.setup = setup
    return ps


def _pose_to_obj(T: np.ndarray) -> dict:
    return {"t": [round(float(v), 9) for v in T[:3, 3]],
            "q": [round(float(v), 12) for v in Rotation.from_matrix(T[:3, :3]).as_quat()]}


def dump_json(ps: PoseSet, path: str | Path) -> None:
    data = {"setup": ps.setup,
            "robot_poses": [_pose_to_obj(T) for T in ps.robot],
            "target_poses": [_pose_to_obj(T) for T in ps.target]}
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def dump_csv(ps: PoseSet, path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for G, C in zip(ps.robot, ps.target):
            row = []
            for T in (G, C):
                row += list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())
            w.writerow([f"{v:.9f}" for v in row])
