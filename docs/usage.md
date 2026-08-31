# Usage

## Conventions

Input conventions match OpenCV: `robot_poses[i]` is base_T_gripper and `target_poses[i]` is camera_T_target. Eye-in-hand returns gripper_T_camera; eye-to-hand returns base_T_camera. Translations in the output are in the same units as the input.

## Input formats

JSON. Each pose can be `{"t": [x,y,z], "q": [qx,qy,qz,qw]}`, `{"t": ..., "rvec": [...]}` or `{"matrix": 4x4}`. Add `"translation_scale": 0.001` if translations are in millimetres.

```json
{"setup": "eye_in_hand",
 "robot_poses":  [{"t": [0.41, -0.12, 0.33], "q": [0.01, 0.92, -0.05, 0.38]}, ...],
 "target_poses": [{"t": [0.02, 0.05, 0.47],  "q": [0.99, 0.03, 0.02, -0.08]}, ...]}
```

CSV with columns `robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw,target_x,...,target_qw`. CSV files carry no setup, so pass `--setup eye_to_hand` when needed.

## Commands

### handeye lint POSES

- `--setup {eye_in_hand,eye_to_hand}`: override the setup in the file
- `--json`: machine-readable metrics and warnings
- `--strict`: exit code 1 if any warning fires (useful in CI or capture scripts)

### handeye solve POSES

Lints first, then solves. If a convention fault is detected, the fix is applied before solving.

- `--setup`, `--json`: as for `lint`
- `--no-autofix`: solve the data as given, without applying a detected convention fix
- `--out FILE`: write the result JSON (transform as matrix, translation and xyzw quaternion, residuals, the convention applied, and warnings)

### handeye simulate --out FILE

Writes a synthetic pose set (`.json` or `.csv`) and prints the ground truth.

- `--setup`, `--n` (poses, default 15), `--rot-noise` (deg, default 0.1), `--trans-noise` (mm, default 0.5)
- `--outliers` (fraction, default 0), `--fault` (one of the convention faults, default `none`), `--parallel-axes`, `--seed`

Example: `handeye simulate --out p.json --fault target_inverted --outliers 0.2`

### handeye bench

Runs the benchmark against OpenCV's five methods (about 3 minutes on 4 cores at the default size).

- `--scenarios` (default 2000), `--calib` (calibration-split size, default N/2), `--out` (default `results`), `--seed` (default 0), `--workers`
- `--check`: exit 1 if an acceptance check fails (see [benchmark.md](benchmark.md))

## Regenerating files

- Examples: `python3 scripts/make_examples.py`
- The README figure, from benchmark output: `python3 scripts/make_figure.py`
