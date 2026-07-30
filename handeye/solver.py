"""Robust AX=XB solver: RANSAC over motion pairs, then robust refinement on SE(3).

Stage 1 draws two motion pairs at a time and solves the closed-form minimal
problem (rotation by aligning the screw axes of A and B, translation by linear
least squares).  Hypotheses are ranked by a low quantile of the per-pair cost,
which tolerates more than half of the motion pairs being contaminated.

Stage 2 refines X jointly with the constant transform T = H_i X C_i
(robot-world / target-base) by minimising per-sample SE(3) residuals with a
Cauchy loss in ``scipy.optimize.least_squares``.  Working on samples rather
than pairs keeps the residuals independent and gives one outlier score per
recorded pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import se3


def motion_pairs(H: np.ndarray, C: np.ndarray, max_pairs: int | None = 400,
                 rng: np.random.Generator | None = None):
    """All relative motions A = H_j^-1 H_i, B = C_j C_i^-1 for i < j."""
    n = len(H)
    i, j = np.triu_indices(n, k=1)
    if max_pairs is not None and len(i) > max_pairs:
        rng = rng or np.random.default_rng(0)
        keep = np.sort(rng.choice(len(i), max_pairs, replace=False))
        i, j = i[keep], j[keep]
    A = se3.inv(H[j]) @ H[i]
    B = C[j] @ se3.inv(C[i])
    return A, B, i, j


def minimal_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Closed-form X from motion pairs, batched: A, B are (..., k, 4, 4) with k >= 2.

    Rotation aligns the rotation axes (and their cross product) of B onto A;
    translation solves (R_A - I) t_X = R_X t_B - t_A in least squares.
    """
    a = se3.rotvec(A[..., :3, :3])
    b = se3.rotvec(B[..., :3, :3])
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    M = np.einsum("...ki,...kj->...ij", a, b)
    ac = np.cross(a[..., 0, :], a[..., 1, :])
    bc = np.cross(b[..., 0, :], b[..., 1, :])
    M = M + np.einsum("...i,...j->...ij", ac, bc)
    RX = se3.project_to_so3(M)
    I3 = np.eye(3)
    L = A[..., :3, :3] - I3                                         # (...,k,3,3)
    rhs = np.einsum("...ij,...kj->...ki", RX, B[..., :3, 3]) - A[..., :3, 3]
    LtL = np.einsum("...kji,...kjl->...il", L, L) + 1e-12 * I3
    Ltr = np.einsum("...kji,...kj->...i", L, rhs)
    tX = np.linalg.solve(LtL, Ltr[..., None])[..., 0]
    return se3.make(RX, tX)


def pair_costs(X: np.ndarray, A: np.ndarray, B: np.ndarray, length: float):
    """Rotation (rad, approx.) and translation (m) residuals of AX=XB; X (...,4,4) vs pairs (P,4,4)."""
    RX = X[..., None, :3, :3]
    tX = X[..., None, :3, 3]
    RA, tA, RB, tB = A[:, :3, :3], A[:, :3, 3], B[:, :3, :3], B[:, :3, 3]
    rot = np.linalg.norm(RA @ RX - RX @ RB, axis=(-2, -1)) / np.sqrt(2.0)
    lhs = np.einsum("pij,...pj->...pi", RA, np.broadcast_to(tX, tX.shape[:-2] + (len(A), 3))) + tA
    rhs = np.einsum("...pij,pj->...pi", np.broadcast_to(RX, RX.shape[:-3] + (len(A), 3, 3)), tB) + tX
    trans = np.linalg.norm(lhs - rhs, axis=-1)
    return rot, trans, rot + trans / length


@dataclass
class RansacResult:
    X: np.ndarray
    score: float          # quantile of the combined per-pair cost of the best hypothesis
    pair_cost: np.ndarray


def characteristic_length(A: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(A[:, :3, 3], axis=-1))) + 1e-3


def ransac(H: np.ndarray, C: np.ndarray, n_hyp: int = 256, quantile: float = 0.25,
           rng: np.random.Generator | None = None, length: float | None = None,
           min_angle_deg: float = 3.0, min_axis_sep_deg: float = 8.0) -> RansacResult:
    rng = rng or np.random.default_rng(0)
    A, B, _, _ = motion_pairs(H, C, rng=rng)
    if length is None:
        length = characteristic_length(A)
    angA = se3.rot_angle(A[:, :3, :3])
    usable = np.flatnonzero(angA > np.radians(min_angle_deg))
    if len(usable) < 2:
        usable = np.arange(len(A))
    p = usable[rng.integers(len(usable), size=4 * n_hyp)]
    q = usable[rng.integers(len(usable), size=4 * n_hyp)]
    ax = se3.rotvec(A[:, :3, :3])
    ax = ax / np.maximum(np.linalg.norm(ax, axis=-1, keepdims=True), 1e-12)
    sep = np.abs(np.einsum("ki,ki->k", ax[p], ax[q]))
    ok = (p != q) & (sep < np.cos(np.radians(min_axis_sep_deg)))
    if ok.sum() >= 8:
        p, q = p[ok][:n_hyp], q[ok][:n_hyp]
    else:  # (near-)degenerate set: still return the best available guess
        p, q = p[p != q][:n_hyp], q[p != q][:n_hyp]
        if len(p) == 0:
            p, q = np.array([0]), np.array([1 % len(A)])
    idx = np.stack([p, q], axis=1)
    Xs = minimal_solve(A[idx], B[idx])
    _, _, cost = pair_costs(Xs, A, B, length)
    scores = np.quantile(cost, quantile, axis=-1)
    scores = np.where(np.isfinite(scores), scores, np.inf)
    best = int(np.argmin(scores))
    return RansacResult(Xs[best], float(scores[best]), cost[best])


def sample_residuals(X: np.ndarray, T: np.ndarray, H: np.ndarray, C: np.ndarray):
    """Per-sample rotation (rad) and translation (m) residual of T^-1 H_i X C_i."""
    E = se3.inv(T) @ H @ X @ C
    return se3.rotvec(E[:, :3, :3]), E[:, :3, 3]


def medoid_T(X, H, C, length):
    """Robust estimate of the constant T = H_i X C_i: the sample closest to all others."""
    Ts = H @ X @ C
    rot = np.linalg.norm(Ts[:, None, :3, :3] - Ts[None, :, :3, :3], axis=(-2, -1))
    tr = np.linalg.norm(Ts[:, None, :3, 3] - Ts[None, :, :3, 3], axis=-1)
    return Ts[int(np.argmin(np.median(rot + tr / length, axis=1)))]


@dataclass
class SolveResult:
    X: np.ndarray                 # (4,4) hand-eye transform
    T: np.ndarray                 # (4,4) constant world/target transform
    outlier_score: np.ndarray     # (N,) robust normalised residual per sample
    rot_sigma_deg: float          # robust residual scale of inliers
    trans_sigma_mm: float
    ransac_score: float


def _refine(X0, T0, H, C, s_r, s_t, loss, f_scale):
    RX0, RT0 = X0[:3, :3], T0[:3, :3]

    def unpack(p):
        X = se3.make(se3.exp_rot(p[0:3]) @ RX0, p[3:6])
        T = se3.make(se3.exp_rot(p[6:9]) @ RT0, p[9:12])
        return X, T

    def fun(p):
        X, T = unpack(p)
        r, t = sample_residuals(X, T, H, C)
        return np.concatenate([(r / s_r).ravel(), (t / s_t).ravel()])

    p0 = np.concatenate([np.zeros(3), X0[:3, 3], np.zeros(3), T0[:3, 3]])
    res = least_squares(fun, p0, loss=loss, f_scale=f_scale, method="trf",
                        x_scale="jac", max_nfev=200)
    return unpack(res.x)


def _scales(X, T, H, C, floor_r=1e-7, floor_t=1e-8):
    r, t = sample_residuals(X, T, H, C)
    nr, nt = np.linalg.norm(r, axis=1), np.linalg.norm(t, axis=1)
    return nr, nt, max(float(np.median(nr)), floor_r), max(float(np.median(nt)), floor_t)


def solve(H: np.ndarray, C: np.ndarray, n_hyp: int = 256, seed: int = 0) -> SolveResult:
    """Robustly solve H_i X C_i = T for X (and T). Returns X plus per-sample outlier scores."""
    rng = np.random.default_rng(seed)
    rs = ransac(H, C, n_hyp=n_hyp, rng=rng)
    A, _, _, _ = motion_pairs(H, C, rng=np.random.default_rng(seed))
    length = characteristic_length(A)
    X, T = rs.X, medoid_T(rs.X, H, C, length)
    _, _, s_r, s_t = _scales(X, T, H, C)
    X, T = _refine(X, T, H, C, s_r, s_t, "soft_l1", 2.0)
    _, _, s_r, s_t = _scales(X, T, H, C)
    X, T = _refine(X, T, H, C, s_r, s_t, "cauchy", 2.0)
    nr, nt, s_r, s_t = _scales(X, T, H, C)
    score = np.sqrt(((nr / s_r) ** 2 + (nt / s_t) ** 2) / 2.0)
    return SolveResult(X, T, score, float(np.degrees(s_r)), float(s_t * 1e3), rs.score)


def solve_poseset(ps, **kw) -> SolveResult:
    H, C = ps.solver_inputs()
    return solve(H, C, **kw)
