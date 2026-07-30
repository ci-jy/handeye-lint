"""Lint checks for AX=XB inputs: degeneracy, convention errors and outlier samples.

Each check produces a numeric feature; a warning fires when the feature crosses
a threshold.  The default thresholds were chosen on a calibration set of
synthetic scenarios (``handeye bench`` re-derives them on a separate seed and
reports precision / recall on held-out scenarios).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import se3, solver
from .io import PoseSet

# Chosen on the calibration split by `handeye bench` (results/lint_thresholds.json).
DEFAULT_THRESHOLDS = {
    "axis_diversity": 0.05,       # warn if s2/s1 of stacked rotation vectors is below
    "convention_log_ratio": 0.5,  # warn if log10(cost(declared) / cost(best)) is above
    "outlier_score": 6.0,         # flag samples whose robust normalised residual is above
}

ROBOT_FIXES = {"as_given": lambda T: T, "inverted": se3.inv, "transposed": se3.transpose_rotation}
TARGET_FIXES = ROBOT_FIXES

CONVENTION_HINTS = {
    ("inverted", "as_given"): "robot poses look inverted: pass base_T_gripper, not gripper_T_base "
                              "(or the setup is eye-to-hand and labelled eye-in-hand, or vice versa)",
    ("as_given", "inverted"): "target poses look inverted: pass camera_T_target, not target_T_camera",
    ("transposed", "as_given"): "robot rotation matrices look transposed (row/column-major mix-up "
                                "or a passive/active rotation convention)",
    ("as_given", "transposed"): "target rotation matrices look transposed (row/column-major mix-up "
                                "or a passive/active rotation convention)",
}


@dataclass
class Warning_:
    code: str
    message: str
    value: float
    threshold: float
    details: dict = field(default_factory=dict)

    def as_dict(self):
        return {"code": self.code, "message": self.message, "value": self.value,
                "threshold": self.threshold, **({"details": self.details} if self.details else {})}


@dataclass
class LintReport:
    n_poses: int
    setup: str
    axis_diversity: float
    rotation_condition: float
    median_rotation_deg: float
    convention_costs: dict
    convention_log_ratio: float
    best_convention: tuple
    outlier_scores: np.ndarray
    warnings: list
    fixed: PoseSet
    solution: solver.SolveResult

    @property
    def codes(self) -> set:
        return {w.code for w in self.warnings}

    def as_dict(self) -> dict:
        return {
            "n_poses": self.n_poses,
            "setup": self.setup,
            "metrics": {
                "axis_diversity": self.axis_diversity,
                "rotation_condition": self.rotation_condition,
                "median_relative_rotation_deg": self.median_rotation_deg,
                "convention_log_ratio": self.convention_log_ratio,
                "best_convention": {"robot": self.best_convention[0],
                                    "target": self.best_convention[1]},
                "convention_costs": self.convention_costs,
                "outlier_scores": [round(float(s), 3) for s in self.outlier_scores],
            },
            "warnings": [w.as_dict() for w in self.warnings],
        }


def axis_metrics(H: np.ndarray, C: np.ndarray) -> tuple[float, float, float]:
    """Rotation-axis diversity s2/s1 of the stacked robot rotation vectors, its condition
    number s1/s2, and the median relative rotation angle (deg)."""
    A, _, _, _ = solver.motion_pairs(H, C)
    v = se3.rotvec(A[:, :3, :3])
    s = np.linalg.svd(v, compute_uv=False)
    ratio = float(s[1] / s[0]) if s[0] > 0 else 0.0
    cond = float(s[0] / s[1]) if s[1] > 0 else float("inf")
    return ratio, cond, float(np.degrees(np.median(np.linalg.norm(v, axis=1))))


def apply_convention(ps: PoseSet, robot_fix: str, target_fix: str) -> PoseSet:
    return PoseSet(ROBOT_FIXES[robot_fix](ps.robot), TARGET_FIXES[target_fix](ps.target),
                   ps.setup, dict(ps.meta))


def convention_scan(ps: PoseSet, n_hyp: int = 96, seed: int = 0, tie_factor: float = 2.0):
    """Fit every robot/target convention and return the consistency cost of each.

    Screw invariants (rotation angle, pitch) of relative motions are unchanged by
    inverting or transposing all poses, so a fault cannot be seen pair by pair;
    instead each candidate convention is solved with RANSAC and the one whose
    motion pairs agree best on a single X wins.

    Inverting *both* inputs is an exact symmetry of H_i X C_i = T (it swaps the
    roles of X and T), so one inverted input has two equally consistent fixes.
    Near-ties are broken with a physical prior: an eye-in-hand camera sits close
    to the flange (small |t_X|); an eye-to-hand target sits close to the
    gripper (small |t_T|).
    """
    H0, C0 = ps.solver_inputs()
    A0, _, _, _ = solver.motion_pairs(H0, C0)
    length = solver.characteristic_length(A0)
    costs, lever = {}, {}
    for rf, tf in itertools.product(ROBOT_FIXES, TARGET_FIXES):
        H, C = apply_convention(ps, rf, tf).solver_inputs()
        rs = solver.ransac(H, C, n_hyp=n_hyp, rng=np.random.default_rng(seed), length=length)
        costs[(rf, tf)] = max(rs.score, 1e-12)
        if ps.setup == "eye_in_hand":
            lever[(rf, tf)] = float(np.linalg.norm(rs.X[:3, 3]))
        else:
            lever[(rf, tf)] = float(np.linalg.norm(solver.medoid_T(rs.X, H, C, length)[:3, 3]))
    best_cost = min(costs.values())
    near = [k for k, c in costs.items() if c <= tie_factor * best_cost]
    best = min(near, key=lambda k: (lever[k], costs[k]))
    log_ratio = float(np.log10(costs[("as_given", "as_given")] / best_cost))
    return costs, best, log_ratio


def lint(ps: PoseSet, thresholds: dict | None = None, autofix: bool = True,
         seed: int = 0) -> LintReport:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings = []

    costs, best, log_ratio = convention_scan(ps, seed=seed)
    fixed = ps
    if log_ratio > th["convention_log_ratio"] and best != ("as_given", "as_given"):
        hint = CONVENTION_HINTS.get(best, f"data is consistent with robot={best[0]}, target={best[1]}")
        warnings.append(Warning_(
            "convention-mismatch", hint, log_ratio, th["convention_log_ratio"],
            {"robot": best[0], "target": best[1]}))
        if autofix:
            fixed = apply_convention(ps, *best)

    H, C = fixed.solver_inputs()
    ratio, cond, med_rot = axis_metrics(H, C)
    if ratio < th["axis_diversity"]:
        warnings.append(Warning_(
            "degenerate-rotation-axes",
            f"robot rotation axes are nearly parallel (s2/s1 = {ratio:.3g}, condition {cond:.3g}); "
            "translation along the common axis is unobservable - add poses that rotate about "
            "different axes", ratio, th["axis_diversity"]))

    sol = solver.solve(H, C, seed=seed)
    flagged = np.flatnonzero(sol.outlier_score > th["outlier_score"])
    if len(flagged):
        warnings.append(Warning_(
            "outlier-samples",
            f"{len(flagged)} of {len(ps)} samples disagree with the consensus solution: "
            f"indices {flagged.tolist()}", float(sol.outlier_score.max()), th["outlier_score"],
            {"indices": flagged.tolist(),
             "scores": [round(float(sol.outlier_score[i]), 2) for i in flagged]}))

    return LintReport(len(ps), ps.setup, ratio, cond, med_rot,
                      {f"{k[0]}/{k[1]}": float(v) for k, v in costs.items()},
                      log_ratio, best, sol.outlier_score, warnings, fixed, sol)


def format_report(rep: LintReport) -> str:
    lines = [f"{rep.n_poses} poses, setup={rep.setup}",
             f"  rotation-axis diversity s2/s1 = {rep.axis_diversity:.3f} "
             f"(condition {rep.rotation_condition:.1f}), median relative rotation "
             f"{rep.median_rotation_deg:.1f} deg",
             f"  convention: best robot={rep.best_convention[0]}, target={rep.best_convention[1]}, "
             f"log10 cost ratio vs as-given = {rep.convention_log_ratio:.2f}",
             f"  robust residual scale: {rep.solution.rot_sigma_deg:.3f} deg, "
             f"{rep.solution.trans_sigma_mm:.2f} mm"]
    if not rep.warnings:
        lines.append("OK: no warnings")
    for w in rep.warnings:
        lines.append(f"WARNING [{w.code}] {w.message}")
    return "\n".join(lines)
