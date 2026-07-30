import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from handeye import lint, se3, sim
from handeye.bench import FAULT_FIX

seeds = st.integers(0, 2**32 - 1)
setups = st.sampled_from(["eye_in_hand", "eye_to_hand"])


@given(seeds, setups)
@settings(max_examples=15, deadline=None)
def test_clean_data_has_no_warnings(seed, setup):
    sc = sim.generate(sim.ScenarioConfig(setup=setup, n_poses=15), np.random.default_rng(seed))
    assert lint.lint(sc.poses).warnings == []


@given(seeds, setups, st.sampled_from(["robot_inverted", "target_inverted"]))
@settings(max_examples=20, deadline=None)
def test_detects_planted_inversions(seed, setup, fault):
    sc = sim.generate(sim.ScenarioConfig(setup=setup, fault=fault), np.random.default_rng(seed))
    rep = lint.lint(sc.poses)
    assert "convention-mismatch" in rep.codes
    assert rep.best_convention == FAULT_FIX[fault]
    # after the automatic fix the solution is right again
    err = se3.rotation_error_deg(rep.solution.X[:3, :3], sc.X_true[:3, :3])
    assert err < 1.0


@pytest.mark.parametrize("fault", ["robot_transposed", "target_transposed"])
def test_detects_transposed_rotations(fault):
    sc = sim.generate(sim.ScenarioConfig(fault=fault), np.random.default_rng(11))
    rep = lint.lint(sc.poses)
    assert "convention-mismatch" in rep.codes
    assert rep.best_convention == FAULT_FIX[fault]


def test_inverting_both_inputs_is_a_symmetry(clean_eih):
    """Inverting robot and target poses together is an exactly consistent system."""
    costs, _, _ = lint.convention_scan(clean_eih.poses)
    assert costs[("inverted", "inverted")] < 1e-6
    assert costs[("inverted", "as_given")] > 1e-3


def test_degenerate_parallel_axes_warned():
    sc = sim.generate(sim.ScenarioConfig(parallel_axes=True, axis_jitter_deg=0.5),
                      np.random.default_rng(5))
    rep = lint.lint(sc.poses)
    assert "degenerate-rotation-axes" in rep.codes
    assert rep.axis_diversity < lint.DEFAULT_THRESHOLDS["axis_diversity"]


def test_outliers_flagged_by_index():
    sc = sim.generate(sim.ScenarioConfig(n_poses=20, outlier_frac=0.2), np.random.default_rng(9))
    rep = lint.lint(sc.poses)
    w = [w for w in rep.warnings if w.code == "outlier-samples"][0]
    assert set(w.details["indices"]) == set(np.flatnonzero(sc.outlier_mask).tolist())


def test_report_serialises(clean_eih):
    d = lint.lint(clean_eih.poses).as_dict()
    assert set(d["metrics"]) >= {"axis_diversity", "convention_log_ratio", "outlier_scores"}
    assert d["warnings"] == []
