import json

import numpy as np
import pytest

from handeye import bench, cli, io, sim


@pytest.mark.parametrize("ext", ["json", "csv"])
def test_roundtrip(tmp_path, clean_eih, ext):
    path = tmp_path / f"poses.{ext}"
    (io.dump_json if ext == "json" else io.dump_csv)(clean_eih.poses, path)
    ps = io.load(path)
    np.testing.assert_allclose(ps.robot, clean_eih.poses.robot, atol=1e-8)
    np.testing.assert_allclose(ps.target, clean_eih.poses.target, atol=1e-8)


def test_json_matrix_and_millimetres(tmp_path, clean_eih):
    p = clean_eih.poses
    data = {"setup": "eye_in_hand", "translation_scale": 0.001,
            "robot_poses": [{"matrix": (T * [[1], [1], [1], [1]]).tolist()} for T in p.robot],
            "target_poses": [{"matrix": T.tolist()} for T in p.target]}
    for key in ("robot_poses", "target_poses"):
        for d in data[key]:
            for r in range(3):
                d["matrix"][r][3] *= 1000.0
    (tmp_path / "mm.json").write_text(json.dumps(data))
    ps = io.load(tmp_path / "mm.json")
    np.testing.assert_allclose(ps.robot, p.robot, atol=1e-9)


def test_rejects_mismatched_lengths(clean_eih):
    with pytest.raises(ValueError):
        io.PoseSet(clean_eih.poses.robot, clean_eih.poses.target[:-1])


def test_cli_lint_and_solve(tmp_path, capsys):
    path = tmp_path / "p.json"
    assert cli.main(["simulate", "--out", str(path), "--fault", "robot_inverted",
                     "--seed", "3"]) == 0
    truth = json.loads(capsys.readouterr().out)
    assert cli.main(["lint", str(path), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert [w["code"] for w in rep["warnings"]] == ["convention-mismatch"]
    assert cli.main(["lint", str(path), "--strict"]) == 1
    capsys.readouterr()
    assert cli.main(["solve", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    X = np.array(out["gripper_T_camera"]["matrix"])
    np.testing.assert_allclose(X[:3, 3], np.array(truth["X_true"]["matrix"])[:3, 3], atol=3e-3)


def test_threshold_choice_separates_classes():
    x = np.array([0.1, 0.2, 0.3, 2.0, 2.5])
    y = np.array([False, False, False, True, True])
    t = bench.choose_threshold(x, y, "above")
    assert 0.3 < t < 2.0


def test_small_benchmark_runs(tmp_path):
    code = bench.run(30, tmp_path, n_calib=30, workers=1, log=lambda *_: None)
    assert code == 0
    assert (tmp_path / "error_sweeps.png").stat().st_size > 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {e["warning"] for e in summary["lint"]} == {
        "degenerate-rotation-axes", "convention-mismatch", "outlier-samples"}
