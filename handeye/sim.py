"""Synthetic eye-in-hand / eye-to-hand scenarios with known ground truth and planted faults."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import se3
from .io import PoseSet

FAULTS = ("none", "robot_inverted", "target_inverted", "robot_transposed", "target_transposed")


@dataclass
class ScenarioConfig:
    setup: str = "eye_in_hand"
    n_poses: int = 15
    rot_noise_deg: float = 0.1       # per-pose noise on robot and target rotations
    trans_noise_mm: float = 0.5      # per-pose noise on robot and target translations
    outlier_frac: float = 0.0        # fraction of target poses replaced by gross errors
    diversity_deg: float = 30.0      # cone half-angle / roll range of viewpoints
    parallel_axes: bool = False      # plant near-parallel rotation axes (degenerate set)
    axis_jitter_deg: float = 0.5     # off-axis tilt allowed when parallel_axes is set
    fault: str = "none"              # planted convention fault (see FAULTS)
    family: str = "custom"


@dataclass
class Scenario:
    config: ScenarioConfig
    poses: PoseSet                   # what the user would record (noise, outliers, faults applied)
    X_true: np.ndarray
    outlier_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def _look_at(cam_pos: np.ndarray, target: np.ndarray, roll: float) -> np.ndarray:
    """Rotation of a camera at cam_pos whose optical (z) axis points at target."""
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return R @ se3.exp_rot(np.array([0.0, 0.0, roll]))


def _viewpoints(rng, n, diversity_deg, parallel, jitter_deg):
    """World poses of a 'moving' frame looking at the origin from above, with given diversity."""
    poses = []
    if parallel:
        # Orbit about a single fixed axis: every relative rotation shares (nearly) one axis.
        tilt = np.radians(rng.uniform(15, 40))
        base = _look_at(np.array([0.0, -np.sin(tilt), np.cos(tilt)]) * 0.5, np.zeros(3), 0.0)
        axis = np.array([0.0, 0.0, 1.0])
        for _ in range(n):
            phi = np.radians(rng.uniform(-max(diversity_deg, 20), max(diversity_deg, 20)) * 2)
            wob = rng.normal(size=3)
            wob -= wob.dot(axis) * axis
            wob *= np.radians(jitter_deg) / max(np.linalg.norm(wob), 1e-12) * rng.uniform(0, 1)
            R = se3.exp_rot(axis * phi + wob) @ base
            t = R @ np.array([0.0, 0.0, -0.5]) + rng.normal(scale=0.01, size=3)
            poses.append(se3.make(R, t))
        return np.stack(poses)
    div = np.radians(diversity_deg)
    for _ in range(n):
        theta = div * np.sqrt(rng.uniform())
        az = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.35, 0.6)
        pos = r * np.array([np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), np.cos(theta)])
        R = _look_at(pos, rng.normal(scale=0.03, size=3), rng.uniform(-div, div))
        poses.append(se3.make(R, pos))
    return np.stack(poses)


def apply_fault(ps: PoseSet, fault: str) -> PoseSet:
    G, C = ps.robot.copy(), ps.target.copy()
    if fault == "robot_inverted":
        G = se3.inv(G)
    elif fault == "target_inverted":
        C = se3.inv(C)
    elif fault == "robot_transposed":
        G = se3.transpose_rotation(G)
    elif fault == "target_transposed":
        C = se3.transpose_rotation(C)
    elif fault != "none":
        raise ValueError(f"unknown fault {fault!r}")
    return PoseSet(G, C, ps.setup, dict(ps.meta))


def generate(cfg: ScenarioConfig, rng: np.random.Generator) -> Scenario:
    n = cfg.n_poses
    # Ground-truth hand-eye transform.
    if cfg.setup == "eye_in_hand":   # X = gripper_T_camera: small offset on the flange
        X = se3.make(se3.exp_rot(rng.normal(scale=0.6, size=3)), rng.uniform(-0.1, 0.1, 3))
        # fixed = base_T_target: calibration board on the table in front of the robot
        fixed = se3.make(se3.random_rotation(rng), rng.uniform(-0.3, 0.3, 3) + [0.6, 0.0, 0.0])
    else:                            # X = base_T_camera: camera on a stand looking at the workspace
        d = rng.normal(size=3)
        X = se3.make(se3.random_rotation(rng), d / np.linalg.norm(d) * rng.uniform(0.8, 1.5))
        # fixed = gripper_T_target: board held by the gripper
        fixed = se3.make(se3.random_rotation(rng), rng.uniform(-0.1, 0.1, 3))
    W = _viewpoints(rng, n, cfg.diversity_deg, cfg.parallel_axes, cfg.axis_jitter_deg)
    if cfg.setup == "eye_in_hand":
        # Moving camera: base_T_cam_i = fixed_target @ W_i (target at origin of the viewpoint frame).
        cam = fixed @ W                           # base_T_camera
        G = cam @ se3.inv(X)                      # base_T_gripper
        C = se3.inv(cam) @ fixed                  # camera_T_target
    else:
        # Moving target in front of a static camera; viewpoint frame expressed in camera frame.
        tgt_in_cam = se3.inv(W)                   # camera_T_target
        g_T_t = fixed                             # target mounted on the gripper
        C = tgt_in_cam
        G = X @ C @ se3.inv(g_T_t)                # base_T_gripper
    rs, ts = np.radians(cfg.rot_noise_deg), cfg.trans_noise_mm * 1e-3
    G = se3.perturb(G, rng, rs, ts)
    C = se3.perturb(C, rng, rs, ts)
    mask = np.zeros(n, bool)
    k = int(round(cfg.outlier_frac * n))
    if k:
        idx = rng.choice(n, k, replace=False)
        mask[idx] = True
        for i in idx:  # gross target-pose errors, e.g. a mis-detected or flipped marker
            ang = np.radians(rng.uniform(5, 60))
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            dt = rng.normal(size=3)
            dt *= rng.uniform(0.02, 0.15) / np.linalg.norm(dt)
            C[i] = C[i] @ se3.make(se3.exp_rot(ax * ang), dt)
    ps = apply_fault(PoseSet(G, C, cfg.setup), cfg.fault)
    return Scenario(cfg, ps, X, mask)


def config_dict(cfg: ScenarioConfig) -> dict:
    return asdict(cfg)


def sweep_configs(n: int, seed: int) -> list[ScenarioConfig]:
    """Deterministic mix of benchmark families: noise, outliers, diversity, degeneracy, faults."""
    rng = np.random.default_rng(seed)
    families = ["noise", "outliers", "diversity", "convention", "degenerate"]
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    out = []
    for k in range(n):
        fam = families[rng.choice(len(families), p=weights)]
        cfg = ScenarioConfig(setup=("eye_in_hand", "eye_to_hand")[k % 2], family=fam,
                             n_poses=int(rng.integers(10, 26)))
        if fam == "noise":
            cfg.rot_noise_deg = float(rng.choice([0.0, 0.05, 0.1, 0.25, 0.5, 1.0]))
            cfg.trans_noise_mm = float(cfg.rot_noise_deg * 4)
        elif fam == "outliers":
            cfg.outlier_frac = float(rng.choice([0.1, 0.2, 0.3, 0.4]))
        elif fam == "diversity":
            cfg.diversity_deg = float(rng.choice([5.0, 10.0, 20.0, 40.0]))
        elif fam == "convention":
            cfg.fault = str(rng.choice(FAULTS[1:]))
        elif fam == "degenerate":
            cfg.parallel_axes = True
            cfg.axis_jitter_deg = float(rng.choice([0.0, 0.5, 1.0]))
        out.append(cfg)
    return out
