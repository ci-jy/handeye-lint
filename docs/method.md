# Method

Every pose pair (i, j) gives a motion pair A = H_j^-1 H_i, B = C_j C_i^-1 with AX = XB, where H is the robot pose (inverted for eye-to-hand) and C the target pose.

## Lint checks

| Warning | Feature | Fires when |
|---|---|---|
| `degenerate-rotation-axes` | s2/s1 of the stacked rotation vectors of all robot motions (s1/s2 is reported as the condition number) | below 0.052: rotations share one axis, so translation along it is unobservable |
| `convention-mismatch` | log10 of the consistency cost of the data as given over the best of 9 conventions (robot and target each as given, inverted, or rotation transposed) | above 0.63 |
| `outlier-samples` | per-sample residual of the fit, normalised by the median residual | above 4.17, reported by index |

Thresholds were picked to maximise F1 on a calibration split and are stored in `results/lint_thresholds.json`; the defaults in `handeye/lint.py` are those values.

Convention faults cannot be seen pair by pair, because the rotation angle and screw pitch of each relative motion are unchanged when every pose is inverted or transposed. The linter solves the problem under each candidate convention and keeps the one whose motion pairs agree on a single X. Inverting robot and target poses together is an exact symmetry (it swaps X with the constant base/target transform), so a single inverted input has two equally consistent fixes. The tie is broken with a physical prior: an eye-in-hand camera is near the flange, an eye-to-hand target is near the gripper.

## Solver

1. RANSAC over motion pairs. Each hypothesis takes two motion pairs with well-separated rotation axes and solves the closed-form minimal problem: R_X aligns the axes of B (and their cross product) onto those of A via SVD, and t_X solves (R_A - I) t_X = R_X t_B - t_A in least squares. 256 hypotheses are scored in one vectorised NumPy pass by the 25th percentile of the per-pair cost, which tolerates more than half of the pairs being contaminated (30% bad samples already spoil 51% of the pairs). No noise threshold is needed.
2. Refinement on SE(3). X and the constant transform T = H_i X C_i are refined jointly with `scipy.optimize.least_squares` on per-sample residuals log(T^-1 H_i X C_i), first with a soft-L1 loss and then a Cauchy loss, each rescaled by the median residual.

All N(N-1)/2 pairs are used for RANSAC, but refining on pairs would double-count each pose and make residuals correlated. Refining X and T jointly on N independent per-sample residuals (the AX=YB form) gives one meaningful outlier score per recorded pose.

A solver alone cannot repair a convention fault: the `robust` row on the convention family in [benchmark.md](benchmark.md) is as wrong as OpenCV. Only testing alternative conventions can, which is why lint runs before solve.
