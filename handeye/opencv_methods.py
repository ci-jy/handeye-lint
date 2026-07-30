"""Thin wrapper around the five ``cv2.calibrateHandEye`` methods."""
from __future__ import annotations

import cv2
import numpy as np

from . import se3
from .io import PoseSet

METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def calibrate(ps: PoseSet, method: str) -> np.ndarray | None:
    """Return X (4x4) or None if OpenCV fails or returns a non-finite / non-rotation result.

    Eye-to-hand is handled the standard way: pass gripper_T_base instead of
    base_T_gripper, and the result is base_T_camera.
    """
    H, C = ps.solver_inputs()
    try:
        R, t = cv2.calibrateHandEye(list(H[:, :3, :3]), list(H[:, :3, 3].reshape(-1, 3, 1)),
                                    list(C[:, :3, :3]), list(C[:, :3, 3].reshape(-1, 3, 1)),
                                    method=METHODS[method])
    except cv2.error:
        return None
    if R is None or t is None or not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return None
    R = np.asarray(R, dtype=float)
    if abs(np.linalg.det(R) - 1.0) > 1e-3:
        R = se3.project_to_so3(R)
    return se3.make(R, np.asarray(t, dtype=float).ravel())
