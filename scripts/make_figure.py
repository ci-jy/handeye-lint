"""Draw the README figure (docs/error_sweeps.png) from results/sweeps.csv written by `handeye bench`."""
import csv
import sys
from pathlib import Path

from handeye.plots import plot_sweeps

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "sweeps.csv"
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    (ROOT / "docs").mkdir(exist_ok=True)
    plot_sweeps(rows, ROOT / "docs" / "error_sweeps.png")
    print("wrote", ROOT / "docs" / "error_sweeps.png")
