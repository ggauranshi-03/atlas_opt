"""v2 hyperparameter tuning and final-run queueing (identical search budget for every optimizer).

Protocol
  Stage 1 (lr):  sweep the learning rate of the three base optimizers (AdamW, SGD, Muon) on a fixed grid.
                 Every sharpness-aware method inherits the tuned lr / momentum / weight decay of the base optimizer whose
                 update rule it wraps (standard SAM practice: tune the base optimizer, then rho).
  Stage 2 (rho): sweep rho of the six sharpness-aware methods on one shared grid.
  Selection:     lowest final held-out validation loss (CIFAR-10: highest validation accuracy), tuning seed 0
                 (the reported runs use seeds 42, 43, 44, so selection never sees a reported seed).
  Grid edges:    if the best value is on the edge of the grid, the grid is extended by one point (x sqrt(10)) in that
                 direction and the new point is run; at most MAX_EXT extensions per optimizer. Remaining edges are recorded.
  Budget:        language models are tuned on a 3-epoch proxy (300 optimizer steps, same schedule shape);
                 CIFAR-10 runs are short, so they are tuned at full length (7 epochs).

  python tools/tune.py lr    --gpus 0,1,2   # stage 1 for all three models, writes configs/tuned/*.yaml
  python tools/tune.py rho   --gpus 0,1,2   # stage 2, updates configs/tuned/*.yaml
  python tools/tune.py final --gpus 0,1,2   # 9 optimizers x 3 models x 3 seeds, full length, INT4 PTQ of each
"""
import os
import sys
import csv
import json
import math
import copy
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import experiments as E  # noqa: E402

MODELS = {  # key -> (base config, proxy epochs for tuning, final epochs, is_lm)
    "nanogpt": ("configs/nanogpt_v2.yaml", 3, 15, True),
    "pythia70m": ("configs/pythia70m_v2.yaml", 3, 14, True),
    "cifar10": ("configs/cifar10_v2.yaml", 7, 7, False),
}
BASE_OPTS = ["adam", "sgd", "muon"]
INHERIT = {"sam": "sgd", "fsam": "sgd", "muon_sam": "muon", "fsam_ortho_muon": "muon",
           "fsam_ortho_muon_stale_momentum": "muon", "randsam_muon": "muon",
           "soma_prens5": "muon", "op_soma_postns5": "muon"}
INHERITED_KEYS = ["lr", "momentum", "weight_decay", "nesterov", "adam_lr"]
LR_GRID = {
    True: {"adam": [2e-5, 6e-5, 2e-4, 6e-4, 2e-3], "sgd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
           "muon": [6e-4, 2e-3, 6e-3, 2e-2, 6e-2]},
    False: {"adam": [3e-4, 1e-3, 3e-3, 1e-2], "sgd": [1e-2, 3e-2, 1e-1, 3e-1], "muon": [3e-3, 1e-2, 3e-2, 1e-1]},
}
RHO_GRID = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
MAX_EXT = 2
TUNE_SEED = 0
FINAL_SEEDS = [42, 43, 44]
ONLY = None   # optional optimizer subset (--optimizers); then queues and sweep tables get the suffix TAG
TAG = ""
STEP = math.sqrt(10)
OUTDIR = "results/tuning"


def bid(model):
    return E.exp_id(E.load_cfg(MODELS[model][0]), MODELS[model][0])


def tuned_path(model):
    return f"configs/tuned/{bid(model)}.yaml"


def r6(x):
    return float(f"{x:.6g}")


def point_cfg(model, base_cfg, opt, entry, tag):
    return E.sweep_cfg(base_cfg, bid(model), opt, entry, tag)


def score(path, opt, epochs, lm):
    cid = E.exp_id(E.load_cfg(path), path)
    row = E.final_row(f"logs/{cid}_logs.csv", opt, TUNE_SEED, epochs)
    if row is None:
        return math.inf, None  # crashed / diverged before the last epoch
    v = float(row["val_loss"]) if lm else -float(row["val_accuracy"])
    return (v if math.isfinite(v) else math.inf), row


def run_queue(queue, gpus):
    procs = []
    for g in gpus:
        log = open(f"logs/queue/worker_{os.path.basename(queue)}_g{g}.log", "a")
        procs.append(subprocess.Popen([sys.executable, "-u", "tools/experiments.py", "work", "--queue", queue, "--gpu", str(g)],
                                      stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    jobs = E.read_queue(queue)
    bad = [j["id"] for j in jobs if j["status"] != "done"]
    print(f"[{time.strftime('%F %T')}] queue {queue}: {len(jobs) - len(bad)}/{len(jobs)} done; not done: {bad or 'none'}", flush=True)


def sweep(stage, gpus):
    """stage 'lr' or 'rho'. Runs the grid for all models, extends edges, writes the tuned configs."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("logs/queue", exist_ok=True)
    queue = f"queues/tune_{stage}{TAG}.json"
    plan = {}  # (model, opt) -> sorted list of values
    for model, (cfg_path, epochs, _, lm) in MODELS.items():
        cfg = E.load_cfg(cfg_path if stage == "lr" else tuned_path(model))
        opts = BASE_OPTS if stage == "lr" else list(INHERIT)
        opts = [o for o in opts if ONLY is None or o in ONLY]
        for opt in opts:
            plan[(model, opt)] = list(LR_GRID[lm][opt] if stage == "lr" else RHO_GRID)
    ext = {k: 0 for k in plan}
    while True:
        jobs = []
        for (model, opt), values in plan.items():
            cfg_path, epochs, _, lm = MODELS[model]
            cfg = E.load_cfg(cfg_path if stage == "lr" else tuned_path(model))
            for v in values:
                entry = copy.deepcopy(cfg["optimizers"][opt])
                if stage == "lr":
                    entry["lr"] = r6(v)
                    if opt == "muon":
                        entry["adam_lr"] = r6(v / 25)
                else:
                    entry["rho"] = entry["rho_vector"] = r6(v)
                path = point_cfg(model, cfg, opt, entry, f"{stage}{r6(v):g}")
                jobs.append(E.make_job(path, opt, TUNE_SEED, epochs, ptq=False, keep_ckpt=False))
        jobs.sort(key=lambda j: (0 if "nanogpt" in j["id"] else 1 if "pythia" in j["id"] else 2))  # longest first
        E.add_jobs(queue, jobs)
        run_queue(queue, gpus)
        extended = False
        for (model, opt), values in plan.items():
            _, epochs, _, lm = MODELS[model]
            best = select(model, opt, stage, values, epochs, lm)[0]
            vs = sorted(values)
            if ext[(model, opt)] < MAX_EXT and best in (vs[0], vs[-1]) and len(vs) > 1:
                new = r6(vs[0] / STEP) if best == vs[0] else r6(vs[-1] * STEP)
                values.append(new)
                ext[(model, opt)] += 1
                extended = True
                print(f"  edge: {model}/{opt} best {stage}={best:g} on grid edge -> adding {new:g}", flush=True)
        if not extended:
            break
    write_tuned(stage, plan)


def select(model, opt, stage, values, epochs, lm):
    cfg_path = MODELS[model][0]
    rows = []
    for v in sorted(values):
        path = os.path.join("configs", "generated", bid(model), f"{bid(model)}__{opt}__{stage}{r6(v):g}.yaml")
        s, row = score(path, opt, epochs, lm)
        rows.append((v, s, row))
    finite = [(v, s) for v, s, _ in rows if math.isfinite(s)]
    if not finite:
        raise SystemExit(f"[{model}/{opt}] every {stage} grid point diverged or crashed")
    best = min(finite, key=lambda t: t[1])[0]
    return best, rows


def write_tuned(stage, plan):
    record = []
    for model in MODELS:
        cfg_path, epochs, _, lm = MODELS[model]
        tuned = E.load_cfg(cfg_path if stage == "lr" else tuned_path(model))
        for (m, opt), values in plan.items():
            if m != model:
                continue
            best, rows = select(model, opt, stage, values, epochs, lm)
            vs = sorted(values)
            edge = best in (vs[0], vs[-1])
            if stage == "lr":
                tuned["optimizers"][opt]["lr"] = r6(best)
                if opt == "muon":
                    tuned["optimizers"][opt]["adam_lr"] = r6(best / 25)
            else:
                tuned["optimizers"][opt]["rho"] = tuned["optimizers"][opt]["rho_vector"] = r6(best)
            for v, s, row in rows:
                record.append(dict(model=model, optimizer=opt, stage=stage, value=v, selected=(v == best),
                                   metric=("val_loss" if lm else "val_accuracy"),
                                   score=(s if lm else -s) if math.isfinite(s) else "diverged/crashed",
                                   val_perplexity=row["val_perplexity"] if row and lm else "",
                                   still_on_grid_edge=edge if v == best else ""))
            print(f"  {model:<9} {opt:<32} {stage}: " + "  ".join(
                f"{v:g}:{((s if lm else -s) if math.isfinite(s) else float('nan')):.4f}" for v, s, _ in rows)
                  + f"  -> {best:g}{'  [still on grid edge]' if edge else ''}", flush=True)
        if stage == "lr":
            for opt, parent in INHERIT.items():
                for k in INHERITED_KEYS:
                    if k in tuned["optimizers"][parent]:
                        tuned["optimizers"][opt][k] = tuned["optimizers"][parent][k]
        E.write_cfg(tuned, tuned_path(model))
    with open(os.path.join(OUTDIR, f"{stage}_sweep{TAG}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(record[0]))
        w.writeheader()
        w.writerows(record)
    print(f"wrote configs/tuned/*.yaml and {OUTDIR}/{stage}_sweep{TAG}.csv", flush=True)


def final(gpus):
    cost = {"sam": 2, "fsam": 2, "muon_sam": 2, "fsam_ortho_muon": 2}
    jobs = []
    for model, (_, _, epochs, _) in MODELS.items():
        cfg = E.load_cfg(tuned_path(model))
        for seed in FINAL_SEEDS:
            for opt in cfg["optimizers"]:
                if ONLY is not None and opt not in ONLY:
                    continue
                j = E.make_job(tuned_path(model), opt, seed, epochs, ptq=True, keep_ckpt=True)
                j["_cost"] = epochs * cost.get(opt, 1) * (2 if model == "nanogpt" else 1 if model == "pythia70m" else 0.05)
                jobs.append(j)
    jobs.sort(key=lambda j: -j.pop("_cost"))  # longest first keeps the GPUs busy until the end
    E.add_jobs(f"queues/final{TAG}.json", jobs)
    run_queue(f"queues/final{TAG}.json", gpus)
    with open(os.path.join(OUTDIR, "selected_hyperparameters.json"), "w") as f:
        json.dump({m: E.load_cfg(tuned_path(m))["optimizers"] for m in MODELS}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lr", "rho", "final"])
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--optimizers", default=None, help="comma list: only these optimizers")
    ap.add_argument("--tag", default="", help="suffix for queue / sweep-table files when --optimizers is used")
    a = ap.parse_args()
    ONLY = set(a.optimizers.split(",")) if a.optimizers else None
    TAG = a.tag
    gpus = [int(g) for g in a.gpus.split(",")]
    final(gpus) if a.stage == "final" else sweep(a.stage, gpus)
