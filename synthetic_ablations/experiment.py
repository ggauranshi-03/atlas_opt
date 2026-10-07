"""Config loading, job construction and parallel execution shared by sweep.py and tune.py."""
import copy
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from .algorithms import SPEC_BY_NAME, SPECS, variants
from .objective import make_problem, full_objective
from .runner import alpha_key, describe, run_variant, summarize

PACKAGE_DIR = Path(__file__).resolve().parent
DTYPES = {"float64": torch.float64, "float32": torch.float32}


def parse_alpha(value):
    if isinstance(value, str) and value.strip().lower() in ("inf", "infinity", "gaussian"):
        return math.inf
    return float(value)


def load_config(path):
    path = Path(path)
    if not path.exists() and (PACKAGE_DIR / path).exists():
        path = PACKAGE_DIR / path
    with open(path) as source:
        cfg = yaml.safe_load(source)
    cfg.setdefault("sam", {})
    cfg.setdefault("optim", {})
    cfg.setdefault("noise", {})
    cfg["noise"]["alphas"] = [parse_alpha(a) for a in cfg["noise"].get("alphas", [1.6])]
    for alpha in cfg["noise"]["alphas"]:
        if alpha <= 1.0 and not cfg["noise"].get("allow_undefined_mean", False):
            raise ValueError("alpha <= 1 has no finite mean (biased oracle); set "
                             "noise.allow_undefined_mean: true for a labelled stress test")
    names = cfg.get("algorithms", "all")
    cfg["algorithms"] = [s.name for s in SPECS] if names == "all" else list(names)
    unknown = [n for n in cfg["algorithms"] if n not in SPEC_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")
    return cfg


def resolve_out_dir(cfg, override=None):
    out = Path(override) if override else PACKAGE_DIR / "results" / cfg.get("name", "run")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_tuned_lrs(cfg, out_dir):
    candidates = [cfg["optim"].get("tuned_lrs"), out_dir / "tuned_lrs.json"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            with open(candidate) as source:
                return json.load(source)
    return {}


def learning_rate(cfg, tuned, outer, alpha):
    per_alpha = tuned.get(outer, {})
    if alpha_key(alpha) in per_alpha:
        return float(per_alpha[alpha_key(alpha)])
    defaults = cfg["optim"].get("lr", {})
    if outer not in defaults:
        raise KeyError(f"No learning rate for outer optimizer '{outer}' (set optim.lr.{outer} or run tune.py)")
    return float(defaults[outer])


def build_jobs(cfg, tuned, names=None, alphas=None, rhos=None):
    depth = int(cfg["problem"]["depth"])
    names = names or cfg["algorithms"]
    alphas = alphas or cfg["noise"]["alphas"]
    rhos = rhos or [float(r) for r in cfg["sam"].get("rhos", [0.05])]
    modes = cfg["sam"].get("frobenius_modes", ["global", "per-layer"])
    jobs = []
    for alpha in alphas:
        for spec, mode in variants(names, modes, depth):
            lr = learning_rate(cfg, tuned, spec.outer, alpha)
            for rho in (rhos if spec.is_sam else [None]):
                jobs.append(dict(name=spec.name, mode=mode, alpha=alpha, rho=rho, lr=lr))
    return jobs


_PROBLEM_CACHE = {}


def _problem(cfg):
    key = json.dumps(cfg["problem"], sort_keys=True) + cfg["run"].get("dtype", "float64")
    if key not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[key] = make_problem(cfg["problem"], DTYPES[cfg["run"].get("dtype", "float64")])
    return _PROBLEM_CACHE[key]


def execute(job, cfg):
    spec = SPEC_BY_NAME[job["name"]]
    problem = _problem(cfg)
    log, alive, death = run_variant(spec, job["mode"], job["alpha"], job["rho"], job["lr"], problem, cfg,
                                    seeds=job.get("seeds"), seed_offset=job.get("seed_offset"),
                                    steps=job.get("steps"))
    meta = describe(spec, job["mode"], job["alpha"], job["rho"], job["lr"], problem, cfg)
    meta.update(job.get("tag", {}))
    return summarize(log, alive, death, meta)


def _init_worker():
    torch.set_num_threads(1)


def run_jobs(jobs, cfg, workers, on_result):
    """Run jobs (in parallel when workers > 1) and hand each (job, curves, finals) to on_result."""
    if workers <= 1:
        for index, job in enumerate(jobs):
            on_result(index, job, *execute(job, cfg))
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(execute, job, cfg): (index, job) for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            index, job = futures[future]
            on_result(index, job, *future.result())


class CsvAppender:
    """Append rows to a CSV, writing the header once. Tolerates new columns by rewriting."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = None
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, newline="") as source:
                self.fields = next(csv.reader(source))

    def write(self, rows):
        if not rows:
            return
        if self.fields is None:
            self.fields = list(rows[0].keys())
            with open(self.path, "w", newline="") as output:
                csv.DictWriter(output, fieldnames=self.fields).writeheader()
        with open(self.path, "a", newline="") as output:
            csv.DictWriter(output, fieldnames=self.fields, extrasaction="ignore").writerows(rows)


def job_key(name, mode, alpha, rho):
    return (name, mode or "", alpha_key(alpha), "" if rho is None or (isinstance(rho, float) and math.isnan(rho))
            else f"{float(rho):g}")


def completed_jobs(finals_path):
    done = set()
    if not Path(finals_path).exists():
        return done
    with open(finals_path, newline="") as source:
        for row in csv.DictReader(source):
            rho = row["rho"]
            rho = None if rho in ("", "nan") else float(rho)
            done.add(job_key(row["algorithm"], row["frobenius_mode"], parse_alpha(row["alpha"]), rho))
    return done


def resolved_config(cfg):
    out = copy.deepcopy(cfg)
    out["noise"]["alphas"] = [alpha_key(a) for a in cfg["noise"]["alphas"]]
    return out


def problem_info(cfg):
    problem = _problem(cfg)
    info = {
        "depth": problem.depth, "shapes": problem.shapes, "n_train": problem.n,
        "F_star": problem.F_star, "rank_budget": problem.rank_budget,
        "constraint": problem.constraint, "radius": problem.radius,
        "sigma_condition_number": (lambda e: (e.max() / e.min()).item())(torch.linalg.eigvalsh(problem.sigma)),
        "F_init": float(full_objective([w.unsqueeze(0) for w in problem.init], problem).item()),
        "cpu_count": os.cpu_count(),
    }
    if problem.P_star is not None:
        # Only defined for the linear objectives (mse, l1): P_star is the single data-generating
        # matrix there. The nonlinear objective (nn_ce) has no such single matrix -- see Problem's
        # docstring note -- so this key is omitted rather than filled with a placeholder.
        s = torch.linalg.svdvals(problem.P_star)
        info["teacher_singular_values"] = [round(v, 6) for v in s[s > 1e-12].tolist()]
    return info


def _init_product(problem):
    product = problem.init[0]
    for w in problem.init[1:]:
        product = w @ product
    return product
