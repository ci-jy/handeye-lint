"""Headless plots of benchmark sweeps."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TITLES = {"noise": "pose noise (deg; mm = 4x)", "outliers": "outlier fraction",
          "diversity": "viewpoint diversity (deg)"}
STYLE = {"robust": dict(color="black", lw=2.5, ls="--"), "lint+robust": dict(color="black", lw=2.5)}


def plot_sweeps(sweeps: list[dict], path) -> None:
    """2x3 grid: p95 rotation / translation error against each swept parameter."""
    fams = list(TITLES)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    methods = list(dict.fromkeys(r["method"] for r in sweeps))
    for j, fam in enumerate(fams):
        sub = [r for r in sweeps if r["sweep"] == fam]
        for i, (key, label) in enumerate([("p95_rot_deg", "p95 rotation error (deg)"),
                                          ("p95_trans_mm", "p95 translation error (mm)")]):
            ax = axes[i, j]
            for m in methods:
                # the noise-free level is exact (checked separately) and would flatten a log axis
                pts = sorted((float(r["level"]), float(r[key])) for r in sub
                             if r["method"] == m and not (fam == "noise" and float(r["level"]) == 0))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, [max(y, 1e-6) for y in ys], marker="o", label=m,
                        **STYLE.get(m, dict(lw=1.2, alpha=0.85)))
            ax.set_yscale("log")
            ax.set_xlabel(TITLES[fam])
            ax.set_ylabel(label)
            ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Hand-eye error (95th percentile) vs OpenCV calibrateHandEye methods")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
