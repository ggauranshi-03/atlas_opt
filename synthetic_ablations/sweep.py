"""Main experiment: every variant x alpha x rho, many seeds per configuration.

    python -m synthetic_ablations.sweep --config configs/single_matrix.yaml --workers 12
"""
import argparse
import json
import math
import time

from .experiment import (CsvAppender, build_jobs, completed_jobs, job_key, load_config,
                         load_tuned_lrs, parse_alpha, problem_info, resolve_out_dir,
                         resolved_config, run_jobs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--out", help="results directory (default: synthetic_ablations/results/<name>)")
    parser.add_argument("--algorithms", nargs="+", help="subset of algorithm names")
    parser.add_argument("--alphas", nargs="+", help="override noise.alphas (use 'inf' for Gaussian)")
    parser.add_argument("--rhos", nargs="+", type=float, help="override sam.rhos")
    parser.add_argument("--rho-units", choices=("nominal", "matched_frobenius"))
    parser.add_argument("--seeds", type=int, help="override run.seeds")
    parser.add_argument("--steps", type=int, help="override run.steps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="ignore existing results instead of resuming")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seeds:
        cfg["run"]["seeds"] = args.seeds
    if args.steps:
        cfg["run"]["steps"] = args.steps
    if args.rho_units:
        cfg["sam"]["rho_units"] = args.rho_units
    if args.algorithms:
        cfg["algorithms"] = args.algorithms
    alphas = [parse_alpha(a) for a in args.alphas] if args.alphas else None
    if alphas:
        cfg["noise"]["alphas"] = alphas

    out = resolve_out_dir(cfg, args.out)
    finals_path, curves_path = out / "finals.csv", out / "curves.csv"
    if args.fresh:
        for path in (finals_path, curves_path):
            if path.exists():
                path.unlink()
    tuned = load_tuned_lrs(cfg, out)
    jobs = build_jobs(cfg, tuned, rhos=args.rhos)
    done = completed_jobs(finals_path)
    pending = [j for j in jobs if job_key(j["name"], j["mode"], j["alpha"], j["rho"]) not in done]

    with open(out / "config.json", "w") as output:
        json.dump(resolved_config(cfg) | {"tuned_lrs_used": tuned}, output, indent=2, default=str)
    with open(out / "problem.json", "w") as output:
        json.dump(problem_info(cfg), output, indent=2)
    print(f"{len(jobs)} jobs ({len(jobs) - len(pending)} already done) -> {out}", flush=True)
    if not tuned:
        print("  note: no tuned_lrs.json found; using optim.lr defaults (run tune.py first for fair comparisons)")

    finals_csv, curves_csv = CsvAppender(finals_path), CsvAppender(curves_path)
    start = time.time()

    def on_result(index, job, curves, finals):
        finals_csv.write(finals)
        curves_csv.write(curves)
        gaps = sorted(r["final_gap"] for r in finals)
        median = gaps[len(gaps) // 2]
        diverged = sum(r["diverged"] for r in finals) / len(finals)
        rho = "-" if job["rho"] is None else f"{job['rho']:g}"
        mode = f"@{job['mode']}" if job["mode"] else ""
        print(f"[{time.time() - start:7.0f}s] {job['name']}{mode} alpha={job['alpha']:g} rho={rho} "
              f"lr={job['lr']:g}: median gap={median:.3e} diverged={diverged:.0%}", flush=True)

    run_jobs(pending, cfg, args.workers, on_result)
    if not args.no_plots:
        from .plots import make_all
        make_all(out)


if __name__ == "__main__":
    main()
