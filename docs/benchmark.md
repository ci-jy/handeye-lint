# Benchmark

`handeye bench --scenarios 2000 --out results --check` generates a 1,000-scenario calibration split (seed 1) for the lint thresholds and a 2,000-scenario test split (seed 0) mixing five families. All numbers below are from the test split. Per-scenario rows are in `results/scenarios.csv`, sweep tables in `results/sweeps.csv`, and every family's table in `results/RESULTS.md`.

`lint+robust` applies the lint's convention fix before solving; `robust` solves the data as given.

## Scenarios

Eye-in-hand (camera 0-17 cm from the flange, board on the table) and eye-to-hand (camera 0.8-1.5 m from the base, board in the gripper), viewpoints on a cone looking at the board, Gaussian noise on both robot and target poses, gross target-pose outliers (5-60° and 2-15 cm), near-parallel rotation axes (an orbit about one axis with up to 1° wobble) and the four convention faults (robot or target poses inverted or rotation-transposed).

## 95th-percentile error per family

Rotation in degrees / translation in mm.

| method | outliers (n=485) | convention fault (n=315) | noise only (n=619) | low diversity (n=405) | parallel axes (n=176) | all (n=2000) |
|---|---|---|---|---|---|---|
| Tsai | 98.8 / 503 | 166 / 1290 | 2.49 / 15.5 | 44.5 / 100 | 125 / 1250 | 127 / 704 |
| Park | 25.4 / 178 | 173 / 1290 | 1.77 / 13.0 | 0.857 / 6.52 | 146 / 5210 | 132 / 768 |
| Horaud | 24.7 / 176 | 173 / 1290 | 1.80 / 13.0 | 0.858 / 6.52 | 144 / 2160 | 133 / 664 |
| Andreff | 30.5 / 494 | 174 / 1310 | 1.67 / 179 | 0.767 / 7.31 | 3.20 / 551 | 127 / 512 |
| Daniilidis | 147 / 2450 | 173 / 2000 | 5.11 / 69.6 | 0.868 / 10.5 | 14.9 / 12400 | 143 / 1690 |
| robust | 1.23 / 12.9 | 171 / 1350 | 1.67 / 11.8 | 0.518 / 5.85 | 1.84 / 541 | 121 / 547 |
| lint+robust | 1.23 / 12.9 | 0.274 / 2.01 | 1.67 / 11.8 | 0.518 / 5.85 | 1.84 / 541 | 1.30 / 35.4 |

On noise-only scenarios there is no accuracy cost: 95th-percentile rotation error 1.67° vs 1.67° for the best OpenCV method (Andreff) and translation 11.8 mm vs 13.0 mm (Park). With noise-free poses the solver recovers X to better than 1e-3° and 1e-3 mm. On parallel-axis sets every method, this one included, has large translation error, which is why that case is a lint warning rather than something a solver can fix.

## Lint warnings on the test split

| warning | unit | threshold | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| degenerate-rotation-axes | scenario | 0.0522 | 1.000 | 1.000 | 176 | 0 | 0 |
| convention-mismatch | scenario | 0.626 | 1.000 | 1.000 | 315 | 0 | 0 |
| outlier-samples | sample | 4.17 | 1.000 | 0.997 | 2060 | 0 | 7 |

Among detected convention faults, the specific fault (which input, inverted or transposed) was identified correctly in 100% of cases.

These figures reflect faults of the planted size. Subtler errors will be harder to separate from noise.

## What --check requires

The command exits non-zero unless:

- lint+robust beats every OpenCV method's p95 rotation and translation error on the outlier and convention families;
- its median error on the noise family is within 1.5x of the best OpenCV method;
- noise-free recovery is exact;
- every lint warning reaches precision and recall of at least 0.9, with at least 90% correct fault identification.

## Tests

`python3 -m pytest -q` runs 25 tests, including Hypothesis property tests over random seeds and setups: exact recovery from noise-free random poses (4-20 poses, both setups), exactness of the two-pair minimal solver, invariance of X to re-labelling the robot base and target frames, robustness to 10-30% outliers with every planted outlier scoring above every inlier, detection and identification of planted inversions, and no warnings on clean data. Further tests cover transposed rotations, the inversion symmetry, the parallel-axis warning, JSON/CSV round trips, the CLI and a small benchmark run.
