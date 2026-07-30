"""Benchmark: robust solver and linter vs the five OpenCV calibrateHandEye methods.

Two disjoint scenario sets are generated from the seed: a *calibration* split,
on which each lint threshold is chosen to maximise F1, and a *test* split, on
which solver errors and lint precision / recall are reported.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import lint, opencv_methods, se3, sim, solver

METHODS = list(opencv_methods.METHODS) + ["robust", "lint+robust"]
OPENCV = list(opencv_methods.METHODS)
FAULT_FIX = {"robot_inverted": ("inverted", "as_given"), "target_inverted": ("as_given", "inverted"),
             "robot_transposed": ("transposed", "as_given"),
             "target_transposed": ("as_given", "transposed")}
FAIL_ROT, FAIL_TRANS = 180.0, 1e6   # error recorded when a method returns nothing usable


def _errors(X, X_true):
    if X is None:
        return FAIL_ROT, FAIL_TRANS
    return (float(se3.rotation_error_deg(X[:3, :3], X_true[:3, :3])),
            float(np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1e3))


def run_scenario(args) -> dict:
    cfg, seed, k = args
    rng = np.random.default_rng([seed, k])
    sc = sim.generate(cfg, rng)
    row = {"id": k, **asdict(cfg)}
    for m in OPENCV:
        row[f"{m}_rot"], row[f"{m}_trans"] = _errors(opencv_methods.calibrate(sc.poses, m), sc.X_true)
    rep = lint.lint(sc.poses, autofix=True, seed=k)
    row["lint+robust_rot"], row["lint+robust_trans"] = _errors(rep.solution.X, sc.X_true)
    if rep.fixed is sc.poses:
        row["robust_rot"], row["robust_trans"] = row["lint+robust_rot"], row["lint+robust_trans"]
    else:
        raw = solver.solve_poseset(sc.poses, seed=k)
        row["robust_rot"], row["robust_trans"] = _errors(raw.X, sc.X_true)
    row["axis_diversity"] = rep.axis_diversity
    row["convention_log_ratio"] = rep.convention_log_ratio
    row["best_convention"] = "/".join(rep.best_convention)
    row["outlier_scores"] = " ".join(f"{s:.4g}" for s in rep.outlier_scores)
    row["outlier_mask"] = "".join("1" if b else "0" for b in sc.outlier_mask)
    return row


def run_split(configs, seed, workers):
    jobs = [(c, seed, k) for k, c in enumerate(configs)]
    if workers <= 1:
        return [run_scenario(j) for j in jobs]
    with get_context("fork").Pool(workers) as pool:
        return list(pool.imap(run_scenario, jobs, chunksize=8))


# ---------------------------------------------------------------- lint scoring

def lint_samples(rows):
    """(feature, label) arrays for each warning; outliers are scored per sample."""
    deg = (np.array([r["axis_diversity"] for r in rows]),
           np.array([bool(r["parallel_axes"]) for r in rows]))
    conv = (np.array([r["convention_log_ratio"] for r in rows]),
            np.array([r["fault"] != "none" for r in rows]))
    sc, lab = [], []
    for r in rows:
        sc += [float(v) for v in r["outlier_scores"].split()]
        lab += [c == "1" for c in r["outlier_mask"]]
    return {"degenerate-rotation-axes": (deg[0], deg[1], "below"),
            "convention-mismatch": (conv[0], conv[1], "above"),
            "outlier-samples": (np.array(sc), np.array(lab), "above")}


def _pr(fire, label):
    tp = int(np.sum(fire & label))
    fp = int(np.sum(fire & ~label))
    fn = int(np.sum(~fire & label))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, tp, fp, fn


def choose_threshold(x, y, direction):
    """Threshold maximising F1 on the calibration split (midpoints between sorted values)."""
    v = np.unique(x[np.isfinite(x)])
    cands = (v[:-1] + v[1:]) / 2 if len(v) > 1 else v
    best, best_f1 = cands[0], -1.0
    for t in cands:
        fire = x < t if direction == "below" else x > t
        p, r, *_ = _pr(fire, y)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1 + 1e-12:
            best, best_f1 = t, f1
    return float(best)


THRESHOLD_KEYS = {"degenerate-rotation-axes": "axis_diversity",
                  "convention-mismatch": "convention_log_ratio",
                  "outlier-samples": "outlier_score"}


def calibrate_thresholds(rows):
    return {THRESHOLD_KEYS[name]: choose_threshold(x, y, d)
            for name, (x, y, d) in lint_samples(rows).items()}


def evaluate_lint(rows, thresholds):
    out = []
    for name, (x, y, d) in lint_samples(rows).items():
        t = thresholds[THRESHOLD_KEYS[name]]
        fire = x < t if d == "below" else x > t
        p, r, tp, fp, fn = _pr(fire, y)
        out.append({"warning": name, "unit": "sample" if name == "outlier-samples" else "scenario",
                    "threshold": t, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
                    "positives": int(y.sum()), "negatives": int((~y).sum())})
    fault_rows = [r for r in rows if r["fault"] != "none"
                  and r["convention_log_ratio"] > thresholds["convention_log_ratio"]]
    ident = np.mean([r["best_convention"] == "/".join(FAULT_FIX[r["fault"]]) for r in fault_rows]) \
        if fault_rows else float("nan")
    return out, float(ident)


# ---------------------------------------------------------------- error tables

def _stats(rows, m):
    rot = np.array([r[f"{m}_rot"] for r in rows])
    tr = np.array([r[f"{m}_trans"] for r in rows])
    fails = int(np.sum(rot >= FAIL_ROT))
    return {"method": m, "n": len(rows),
            "median_rot_deg": float(np.median(rot)), "p95_rot_deg": float(np.percentile(rot, 95)),
            "median_trans_mm": float(np.median(tr)), "p95_trans_mm": float(np.percentile(tr, 95)),
            "failures": fails}


def summarise(rows):
    fams = ["all"] + sorted({r["family"] for r in rows})
    table = []
    for f in fams:
        sub = rows if f == "all" else [r for r in rows if r["family"] == f]
        for m in METHODS:
            table.append({"family": f, **_stats(sub, m)})
    return table


SWEEPS = {"noise": "rot_noise_deg", "outliers": "outlier_frac", "diversity": "diversity_deg"}


def sweep_tables(rows):
    out = []
    for fam, key in SWEEPS.items():
        sub = [r for r in rows if r["family"] == fam]
        for level in sorted({r[key] for r in sub}):
            lv = [r for r in sub if r[key] == level]
            for m in METHODS:
                out.append({"sweep": fam, "parameter": key, "level": level, **_stats(lv, m)})
    return out


# ---------------------------------------------------------------- outputs

def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def headline(table, family):
    rows = {r["method"]: r for r in table if r["family"] == family}
    best_rot = min(OPENCV, key=lambda m: rows[m]["p95_rot_deg"])
    best_tr = min(OPENCV, key=lambda m: rows[m]["p95_trans_mm"])
    ours = rows["lint+robust"]
    return {"family": family,
            "best_opencv_rot": best_rot, "opencv_p95_rot_deg": rows[best_rot]["p95_rot_deg"],
            "best_opencv_trans": best_tr, "opencv_p95_trans_mm": rows[best_tr]["p95_trans_mm"],
            "ours_p95_rot_deg": ours["p95_rot_deg"], "ours_p95_trans_mm": ours["p95_trans_mm"]}


def checks(table, lint_eval, ident, test_rows):
    rows = {(r["family"], r["method"]): r for r in table}
    res = []
    for fam in ("outliers", "convention"):
        h = headline(table, fam)
        res.append((f"{fam}: robust p95 rotation below best OpenCV",
                    h["ours_p95_rot_deg"] < h["opencv_p95_rot_deg"]))
        res.append((f"{fam}: robust p95 translation below best OpenCV",
                    h["ours_p95_trans_mm"] < h["opencv_p95_trans_mm"]))
    noise = {m: rows[("noise", m)] for m in METHODS}
    best_med_rot = min(noise[m]["median_rot_deg"] for m in OPENCV)
    best_med_tr = min(noise[m]["median_trans_mm"] for m in OPENCV)
    res.append(("noise: robust median rotation within 1.5x of best OpenCV",
                noise["lint+robust"]["median_rot_deg"] <= 1.5 * best_med_rot))
    res.append(("noise: robust median translation within 1.5x of best OpenCV",
                noise["lint+robust"]["median_trans_mm"] <= 1.5 * best_med_tr))
    clean = [r for r in test_rows if r["family"] == "noise" and r["rot_noise_deg"] == 0]
    if clean:
        res.append(("noise-free: robust max error < 1e-3 deg and 1e-3 mm",
                    max(r["lint+robust_rot"] for r in clean) < 1e-3
                    and max(r["lint+robust_trans"] for r in clean) < 1e-3))
    for e in lint_eval:
        res.append((f"lint {e['warning']}: precision and recall >= 0.9",
                    e["precision"] >= 0.9 and e["recall"] >= 0.9))
    res.append(("lint convention fault identified correctly >= 90%", ident >= 0.9))
    return res


def run(n_scenarios: int = 2000, out: str | Path = "results", seed: int = 0,
        n_calib: int | None = None, workers: int | None = None, check: bool = False,
        log=print) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(4, os.cpu_count() or 1)
    n_calib = n_calib or max(n_scenarios // 2, 50)
    log(f"calibration split: {n_calib} scenarios (seed {seed + 1})")
    calib = run_split(sim.sweep_configs(n_calib, seed + 1), seed + 1, workers)
    th = calibrate_thresholds(calib)
    (out / "lint_thresholds.json").write_text(json.dumps(th, indent=2) + "\n")
    log(f"thresholds chosen on calibration split: {th}")
    log(f"test split: {n_scenarios} scenarios (seed {seed})")
    test = run_split(sim.sweep_configs(n_scenarios, seed), seed, workers)
    _write_csv(out / "scenarios.csv", test)
    table = summarise(test)
    _write_csv(out / "benchmark_summary.csv", table)
    sweeps = sweep_tables(test)
    _write_csv(out / "sweeps.csv", sweeps)
    lint_eval, ident = evaluate_lint(test, th)
    _write_csv(out / "lint_precision_recall.csv", lint_eval)
    heads = [headline(table, f) for f in ("outliers", "convention", "noise")]

    from .plots import plot_sweeps
    plot_sweeps(sweeps, out / "error_sweeps.png")

    cols = ["method", "median_rot_deg", "p95_rot_deg", "median_trans_mm", "p95_trans_mm", "failures"]
    f = {c: "{:.3g}" for c in cols[1:5]}
    md = ["# Benchmark results", "",
          f"Test split: {n_scenarios} scenarios (seed {seed}); lint thresholds chosen on a separate "
          f"calibration split of {n_calib} scenarios (seed {seed + 1}).", ""]
    for fam in ["all"] + sorted({r["family"] for r in test}):
        sub = [r for r in table if r["family"] == fam]
        md += [f"## Family: {fam} (n={sub[0]['n']})", "", _md_table(sub, cols, f), ""]
    md += ["## Lint warnings on held-out planted faults", "",
           _md_table(lint_eval, ["warning", "unit", "threshold", "precision", "recall", "tp", "fp",
                                 "fn", "positives", "negatives"],
                     {"threshold": "{:.3g}", "precision": "{:.3f}", "recall": "{:.3f}"}), "",
           f"Convention fault identified correctly (detected cases): {ident:.3f}", ""]
    (out / "RESULTS.md").write_text("\n".join(md))

    check_res = checks(table, lint_eval, ident, test)
    summary = {"n_test": n_scenarios, "n_calib": n_calib, "seed": seed, "thresholds": th,
               "headlines": heads, "lint": lint_eval, "convention_identification": ident,
               "checks": [{"check": c, "pass": bool(ok)} for c, ok in check_res]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for h in heads:
        log(f"{h['family']}: p95 rot {h['ours_p95_rot_deg']:.3g} deg vs best OpenCV "
            f"({h['best_opencv_rot']}) {h['opencv_p95_rot_deg']:.3g}; p95 trans "
            f"{h['ours_p95_trans_mm']:.3g} mm vs ({h['best_opencv_trans']}) "
            f"{h['opencv_p95_trans_mm']:.3g}")
    for e in lint_eval:
        log(f"lint {e['warning']}: precision {e['precision']:.3f} recall {e['recall']:.3f}")
    log(f"convention fault identification: {ident:.3f}")
    failed = [c for c, ok in check_res if not ok]
    for c, ok in check_res:
        log(f"[{'PASS' if ok else 'FAIL'}] {c}")
    if check and failed:
        return 1
    return 0
