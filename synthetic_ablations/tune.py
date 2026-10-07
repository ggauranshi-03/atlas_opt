"""Learning-rate tuning for the base (non-SAM) optimizers, per alpha.

Protocol (as in Zhong, Milsom & Murray 2026): tune the outer optimizer without SAM, then every
SAM variant inherits the learning rate of its outer optimizer and only rho is swept. Tuning uses
seeds disjoint from the evaluation seeds, so the reported numbers carry no selection bias.

    python -m synthetic_ablations.tune --config configs/single_matrix.yaml --workers 12
"""
import argparse
import json
import math
from collections import defaultdict

from .experiment import (CsvAppender, load_config, parse_alpha, resolve_out_dir, run_jobs)
from .runner import alpha_key

OUTER_TO_ALGORITHM = {"sgd": "sgd", "adam": "adam", "muon": "muon", "nsgdm": "nsgdm", "clip_sgd": "clip-sgd"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--out")
    parser.add_argument("--outers", nargs="+", choices=sorted(OUTER_TO_ALGORITHM))
    parser.add_argument("--alphas", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tuning = cfg.get("tuning", {})
    grids = tuning["lr_grid"]
    outers = args.outers or [o for o in OUTER_TO_ALGORITHM if o in grids]
    alphas = [parse_alpha(a) for a in args.alphas] if args.alphas else cfg["noise"]["alphas"]
    seeds = int(tuning.get("seeds", 16))
    seed_offset = int(tuning.get("seed_offset", 100_000))
    steps = int(tuning.get("steps", cfg["run"]["steps"]))
    max_divergence = float(tuning.get("max_divergence_rate", 0.25))

    jobs = [dict(name=OUTER_TO_ALGORITHM[outer], mode=None, alpha=alpha, rho=None, lr=float(lr),
                 seeds=seeds, seed_offset=seed_offset, steps=steps)
            for outer in outers for alpha in alphas for lr in grids[outer]]
    out = resolve_out_dir(cfg, args.out)
    table = CsvAppender(out / "tuning.csv")
    scores = defaultdict(list)
    print(f"Tuning {len(jobs)} runs ({seeds} tuning seeds, offset {seed_offset}) -> {out}", flush=True)

    def on_result(index, job, curves, finals):
        gaps = sorted(r["final_gap"] for r in finals)
        median = gaps[len(gaps) // 2]
        diverged = sum(r["diverged"] for r in finals) / len(finals)
        outer = next(o for o, a in OUTER_TO_ALGORITHM.items() if a == job["name"])
        scores[(outer, alpha_key(job["alpha"]))].append((job["lr"], median, diverged))
        table.write([{"outer": outer, "alpha": alpha_key(job["alpha"]), "lr": job["lr"],
                      "median_final_gap": median, "divergence_rate": diverged}])
        print(f"  {outer:9s} alpha={job['alpha']:g} lr={job['lr']:<9g} median gap={median:.3e} "
              f"diverged={diverged:.0%}", flush=True)

    run_jobs(jobs, cfg, args.workers, on_result)

    path = out / "tuned_lrs.json"
    tuned = json.load(open(path)) if path.exists() else {}
    for (outer, key), rows in sorted(scores.items()):
        eligible = [r for r in rows if r[2] <= max_divergence and math.isfinite(r[1])] or rows
        lr, median, _ = min(eligible, key=lambda r: r[1])
        grid = sorted(grids[outer])
        edge = " (grid edge: consider extending)" if lr in (grid[0], grid[-1]) else ""
        print(f"best {outer:9s} alpha={key:5s}: lr={lr:g}  median gap={median:.3e}{edge}")
        tuned.setdefault(outer, {})[key] = lr
    with open(path, "w") as output:
        json.dump(tuned, output, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
