"""Regenerate the example pose files in examples/ (synthetic, with known ground truth)."""
from pathlib import Path

from handeye import cli

OUT = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = {
    "eye_in_hand_clean.json": ["--setup", "eye_in_hand", "--seed", "1", "--n", "15"],
    "eye_in_hand_swapped_frames.json": ["--setup", "eye_in_hand", "--seed", "2", "--n", "15",
                                        "--fault", "robot_inverted"],
    "eye_to_hand_outliers.csv": ["--setup", "eye_to_hand", "--seed", "3", "--n", "20",
                                 "--outliers", "0.2", "--rot-noise", "0.2", "--trans-noise", "1.0"],
    "eye_in_hand_parallel_axes.json": ["--setup", "eye_in_hand", "--seed", "4", "--n", "15",
                                       "--parallel-axes"],
}

if __name__ == "__main__":
    import contextlib
    import io

    for name, args in EXAMPLES.items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["simulate", "--out", str(OUT / name), *args])
        (OUT / (Path(name).stem + ".truth.json")).write_text(buf.getvalue())
        print("wrote", OUT / name)
