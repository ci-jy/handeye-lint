import numpy as np
from hypothesis import given, settings, strategies as st

from handeye import se3

seeds = st.integers(0, 2**32 - 1)


@given(seeds)
@settings(max_examples=50, deadline=None)
def test_inverse_roundtrip(seed):
    rng = np.random.default_rng(seed)
    T = se3.make(se3.random_rotation(rng, 5), rng.normal(size=(5, 3)))
    np.testing.assert_allclose(T @ se3.inv(T), np.broadcast_to(np.eye(4), (5, 4, 4)), atol=1e-12)


@given(seeds)
@settings(max_examples=50, deadline=None)
def test_rotvec_exp_roundtrip(seed):
    rng = np.random.default_rng(seed)
    R = se3.random_rotation(rng, 4)
    np.testing.assert_allclose(se3.exp_rot(se3.rotvec(R)), R, atol=1e-10)


def test_rotation_error_of_known_rotation():
    R = se3.exp_rot(np.array([0.0, 0.0, np.radians(10.0)]))
    assert abs(se3.rotation_error_deg(R, np.eye(3)) - 10.0) < 1e-9


def test_project_to_so3_returns_proper_rotation():
    rng = np.random.default_rng(0)
    R = se3.project_to_so3(rng.normal(size=(10, 3, 3)))
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), (10, 3, 3)),
                               atol=1e-12)
