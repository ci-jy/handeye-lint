"""Small SE(3) helpers on 4x4 homogeneous matrices, vectorised over leading axes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build homogeneous transforms from rotations (...,3,3) and translations (...,3)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transforms (...,4,4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def transpose_rotation(T: np.ndarray) -> np.ndarray:
    """Transpose the rotation block and keep the translation (a row/column-major mix-up)."""
    return make(np.swapaxes(T[..., :3, :3], -1, -2), T[..., :3, 3])


def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vectors (...,3) of rotation matrices (...,3,3)."""
    shape = R.shape[:-2]
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(shape + (3,))


def exp_rot(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    shape = v.shape[:-1]
    return Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(shape + (3, 3))


def rot_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle in radians, robust near 0 and pi."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    return np.arctan2(0.5 * np.linalg.norm(w, axis=-1), c)


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> np.ndarray:
    return np.degrees(rot_angle(np.swapaxes(R_true, -1, -2) @ R_est))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (Frobenius) via SVD, vectorised."""
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


def random_rotation(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    return Rotation.random(n, random_state=rng).as_matrix()


def perturb(T: np.ndarray, rng: np.random.Generator, rot_sigma_rad: float,
            trans_sigma_m: float) -> np.ndarray:
    """Right-multiply by an isotropic Gaussian perturbation."""
    shape = T.shape[:-2]
    dR = exp_rot(rng.normal(scale=rot_sigma_rad, size=shape + (3,)))
    dt = rng.normal(scale=trans_sigma_m, size=shape + (3,))
    return T @ make(dR, dt)
