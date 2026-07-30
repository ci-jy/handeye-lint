import numpy as np
from hypothesis import given, settings, strategies as st

from handeye import opencv_methods, se3, sim, solver
from handeye.io import PoseSet

seeds = st.integers(0, 2**32 - 1)
setups = st.sampled_from(["eye_in_hand", "eye_to_hand"])


def errors(X, X_true):
    return (float(se3.rotation_error_deg(X[:3, :3], X_true[:3, :3])),
            float(np.linalg.norm(X[:3, 3] - X_true[:3, 3])))


def noise_free(seed, setup, n=10):
    cfg = sim.ScenarioConfig(setup=setup, n_poses=n, rot_noise_deg=0, trans_noise_mm=0)
    return sim.generate(cfg, np.random.default_rng(seed))


@given(seeds, setups, st.integers(4, 20))
@settings(max_examples=40, deadline=None)
def test_exact_recovery_noise_free(seed, setup, n):
    sc = noise_free(seed, setup, n)
    rot, tr = errors(solver.solve_poseset(sc.poses).X, sc.X_true)
    assert rot < 1e-4 and tr < 1e-6


@given(seeds)
@settings(max_examples=30, deadline=None)
def test_minimal_solver_exact_on_two_pairs(seed):
    sc = noise_free(seed, "eye_in_hand", 3)
    H, C = sc.poses.solver_inputs()
    A, B, _, _ = solver.motion_pairs(H, C)
    rot, tr = errors(solver.minimal_solve(A[:2], B[:2]), sc.X_true)
    assert rot < 1e-6 and tr < 1e-8


@given(seeds, setups)
@settings(max_examples=25, deadline=None)
def test_invariance_to_frame_relabelling(seed, setup):
    """Re-expressing the robot base and the target in other frames must not change X."""
    sc = noise_free(seed, setup)
    rng = np.random.default_rng(seed + 1)
    U = se3.make(se3.random_rotation(rng), rng.normal(size=3))   # new base frame
    V = se3.make(se3.random_rotation(rng), rng.normal(size=3))   # new target frame
    G, C = sc.poses.robot, sc.poses.target
    if setup == "eye_in_hand":
        relabelled = PoseSet(U @ G, C @ V, setup)
        X_expected = sc.X_true
    else:   # moving the base frame moves the answer base_T_camera with it
        relabelled = PoseSet(U @ G, C @ V, setup)
        X_expected = U @ sc.X_true
    rot, tr = errors(solver.solve_poseset(relabelled).X, X_expected)
    assert rot < 1e-4 and tr < 1e-6


@given(seeds, st.sampled_from([0.1, 0.2, 0.3]))
@settings(max_examples=15, deadline=None)
def test_robust_to_outliers(seed, frac):
    cfg = sim.ScenarioConfig(n_poses=20, outlier_frac=frac)
    sc = sim.generate(cfg, np.random.default_rng(seed))
    res = solver.solve_poseset(sc.poses)
    rot, tr = errors(res.X, sc.X_true)
    assert rot < 1.0 and tr < 0.01
    # every planted outlier scores above every inlier
    assert res.outlier_score[sc.outlier_mask].min() > res.outlier_score[~sc.outlier_mask].max()


def test_matches_opencv_on_noise_free_data(clean_eih):
    for m in opencv_methods.METHODS:
        X = opencv_methods.calibrate(clean_eih.poses, m)
        rot, tr = errors(X, clean_eih.X_true)
        assert rot < 1e-3 and tr < 1e-5, m


def test_noise_accuracy_comparable_to_opencv():
    rng = np.random.default_rng(3)
    ours, park = [], []
    for _ in range(20):
        sc = sim.generate(sim.ScenarioConfig(rot_noise_deg=0.2, trans_noise_mm=1.0), rng)
        ours.append(errors(solver.solve_poseset(sc.poses).X, sc.X_true)[0])
        park.append(errors(opencv_methods.calibrate(sc.poses, "park"), sc.X_true)[0])
    assert np.median(ours) < 1.5 * np.median(park)
