# handeye-lint

When 10-40% of the recorded poses were bad, the best of OpenCV's five `calibrateHandEye` methods had a 95th-percentile rotation error of 24.7°. The solver in this repo kept it at 1.23°, on the same 2,000 held-out synthetic scenarios. handeye-lint is a Python library and CLI that checks robot and camera poses for the usual reasons a hand-eye calibration goes wrong (rotations that all share one axis, poses that are inverted or transposed, a few bad marker detections) and then solves AX=XB in a way that survives them.

I wrote it because OpenCV's methods often disagree on real data and give no hint why. Forum threads show [answers spread across methods](https://forum.opencv.org/t/hand-to-eye-calibration-results-from-different-methods/7393) and [huge numbers on a real robot after the code worked in simulation](https://forum.opencv.org/t/calibratehandeye-gives-back-huge-or-wrong-numbers-for-real-robot-but-worked-in-simulated/15894).

![95th-percentile error vs pose noise, outlier fraction and viewpoint diversity for the five OpenCV methods and this solver](docs/error_sweeps.png)

## Inverted poses are invisible, pair by pair

The convention mistakes were the interesting part. If every robot pose is inverted (gripper_T_base instead of base_T_gripper), the rotation angle and screw pitch of each relative motion stay exactly the same. No check on single motion pairs can see it. So the linter solves the whole problem under 9 candidate conventions (robot and target each as given, inverted or rotation-transposed) and keeps the one where the motion pairs agree on one X.

Then a second catch. Inverting robot and target poses together is an exact symmetry: it just swaps X with the constant base-to-target transform. That means a single inverted input has two equally good fixes, and the data alone can't pick. I break the tie with a physical prior, that an eye-in-hand camera sits near the flange and an eye-to-hand board sits near the gripper. On the convention-fault scenarios this took the worst-case rotation error from 166° (OpenCV) to 0.27°.

## The solver

It starts with RANSAC over all motion pairs. Each hypothesis takes two pairs with well-separated rotation axes and solves the minimal problem in closed form. Hypotheses are ranked by a low quantile of the per-pair cost instead of an inlier threshold, because noise on real rigs varies by orders of magnitude and I didn't want a threshold to tune. This needs to tolerate a lot: 30% bad samples already spoil 51% of the pairs.

After that, X and the constant target transform are refined together with SciPy's `least_squares` on per-sample residuals, first with a soft-L1 loss and then a Cauchy loss. Using one residual per recorded pose (not per pair) is what lets `solve` name the bad poses by index. Details of each lint check and the thresholds are in [docs/method.md](docs/method.md).

## Running it

```bash
pip install -e ".[dev]"
handeye lint examples/eye_in_hand_swapped_frames.json
handeye solve examples/eye_to_hand_outliers.csv --setup eye_to_hand
```

The first file has its robot poses stored as gripper_T_base, and `lint` says so:

```
WARNING [convention-mismatch] robot poses look inverted ...
```

The second has 4 corrupted samples out of 20. `solve` names them (`indices [3, 5, 6, 9]`) and returns base_T_camera within 0.2° and 0.5 mm of the ground truth in `examples/eye_to_hand_outliers.truth.json`.

Input files use OpenCV's conventions and can be JSON or CSV; the formats and every flag are in [docs/usage.md](docs/usage.md). `handeye bench --scenarios 2000 --out results --check` reruns the full comparison against OpenCV, and the results tables are in [docs/benchmark.md](docs/benchmark.md). Tests run with `python3 -m pytest -q`. If your interpreter is missing only some dependencies (usually OpenCV), `scripts/install_local_deps.sh [python]` puts the missing ones in a git-ignored `.deps/` folder. OpenCV is pinned below 5.0, since `calibrateHandEye` is no longer at the top level of the `cv2` 5.0 module.

## Known issues and next steps

- Every number here comes from synthetic scenarios with gross outliers (5-60°, 2-15 cm) and whole-dataset convention faults. Subtler problems like a single mis-ordered pose or time-sync offsets will be harder to separate from noise. Testing on recorded data, such as the dataset released with [Hydra](https://arxiv.org/abs/2504.20584v1), is next.
- A rig that breaks the tie-break prior (an eye-in-hand camera far from the flange, say) can get the dual fix. The warning prints the costs so you can check.
- Quaternion order mix-ups (wxyz vs xyzw), unit mistakes and partial faults are not detected.
- The degeneracy check only looks at rotation-axis diversity. It won't warn about small total rotation or badly spread translations, and the solver doesn't report a covariance.

Project period: 2026-07-30 to 2026-09-06.
