"""Command-line interface: handeye lint | solve | bench | simulate."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import bench, io, lint, sim


def _load(args):
    return io.load(args.poses, args.setup)


def _transform_dict(T):
    return {"matrix": np.round(T, 9).tolist(), "t": np.round(T[:3, 3], 9).tolist(),
            "q_xyzw": np.round(Rotation.from_matrix(T[:3, :3]).as_quat(), 12).tolist()}


def cmd_lint(args):
    rep = lint.lint(_load(args))
    print(json.dumps(rep.as_dict(), indent=2) if args.json else lint.format_report(rep))
    return 1 if (args.strict and rep.warnings) else 0


def cmd_solve(args):
    ps = _load(args)
    rep = lint.lint(ps, autofix=not args.no_autofix)
    sol = rep.solution
    name = "gripper_T_camera" if ps.setup == "eye_in_hand" else "base_T_camera"
    out = {"setup": ps.setup, name: _transform_dict(sol.X),
           "residual_rot_deg": sol.rot_sigma_deg, "residual_trans_mm": sol.trans_sigma_mm,
           "convention_applied": {"robot": "as_given", "target": "as_given"}
           if rep.fixed is ps else {"robot": rep.best_convention[0], "target": rep.best_convention[1]},
           "warnings": [w.as_dict() for w in rep.warnings]}
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    if args.json:
        print(text)
    else:
        print(lint.format_report(rep))
        print(f"\n{name} (translation in input units):")
        print(np.array2string(sol.X, precision=6, suppress_small=True))
        if rep.fixed is not ps:
            print(f"(solved after applying robot={rep.best_convention[0]}, "
                  f"target={rep.best_convention[1]}; use --no-autofix to disable)")
    return 0


def cmd_bench(args):
    return bench.run(args.scenarios, args.out, seed=args.seed, n_calib=args.calib,
                     workers=args.workers, check=args.check)


def cmd_simulate(args):
    cfg = sim.ScenarioConfig(setup=args.setup or "eye_in_hand", n_poses=args.n,
                             rot_noise_deg=args.rot_noise, trans_noise_mm=args.trans_noise,
                             outlier_frac=args.outliers, fault=args.fault,
                             parallel_axes=args.parallel_axes)
    sc = sim.generate(cfg, np.random.default_rng(args.seed))
    (io.dump_csv if args.out.endswith(".csv") else io.dump_json)(sc.poses, args.out)
    truth = {"X_true": _transform_dict(sc.X_true),
             "outlier_indices": np.flatnonzero(sc.outlier_mask).tolist(), "fault": args.fault}
    print(json.dumps(truth, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="handeye", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in [("lint", cmd_lint, "check a pose set for degeneracy, "
                                "convention errors and outliers"),
                               ("solve", cmd_solve, "robust hand-eye solve (lints first)")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("poses", help="JSON or CSV file of robot and target poses")
        s.add_argument("--setup", choices=io.SETUPS, help="override the setup in the file")
        s.add_argument("--json", action="store_true", help="machine-readable output")
        s.set_defaults(fn=fn)
        if name == "lint":
            s.add_argument("--strict", action="store_true", help="exit 1 if any warning fires")
        else:
            s.add_argument("--no-autofix", action="store_true",
                           help="do not apply a detected convention fix before solving")
            s.add_argument("--out", help="write the result JSON here")
    b = sub.add_parser("bench", help="benchmark against OpenCV's five methods")
    b.add_argument("--scenarios", type=int, default=2000)
    b.add_argument("--calib", type=int, default=None, help="calibration-split size (default N/2)")
    b.add_argument("--out", default="results")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--check", action="store_true", help="exit 1 if an acceptance check fails")
    b.set_defaults(fn=cmd_bench)
    g = sub.add_parser("simulate", help="write a synthetic pose set with known ground truth")
    g.add_argument("--out", required=True, help=".json or .csv")
    g.add_argument("--setup", choices=io.SETUPS)
    g.add_argument("--n", type=int, default=15)
    g.add_argument("--rot-noise", type=float, default=0.1, help="deg")
    g.add_argument("--trans-noise", type=float, default=0.5, help="mm")
    g.add_argument("--outliers", type=float, default=0.0, help="fraction")
    g.add_argument("--fault", choices=sim.FAULTS, default="none")
    g.add_argument("--parallel-axes", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_simulate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
