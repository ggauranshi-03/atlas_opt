"""Hyperparameter sweeps, final multi-seed runs, and a GPU job queue.

Tuning protocol (identical effort for every method, following Watts et al. 2026 / standard SAM practice):
  1. `sweep-lr`   : sweep the LR of the three base optimizers (adam, sgd, muon).
  2. `select-lr`  : pick each base optimizer's best LR by final held-out validation loss (CIFAR: val acc);
                    every SAM-family method inherits the tuned LR/momentum/weight decay of its base optimizer.
  3. `sweep-rho`  : sweep rho over ONE shared grid for all ten SAM-family methods (Atlas included).
  4. `select-rho` : pick each method's best rho.
  5. `final`      : queue every optimizer x seed with the tuned config, then INT4-AWQ PTQ of each checkpoint.
Workers (`work`, one per GPU) pull jobs from a shared queue file; jobs are resumable and idempotent.
"""
import os
import sys
import csv
import json
import math
import copy
import time
import fcntl
import signal
import socket
import argparse
import subprocess
import glob
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_OPTS = ["adam", "sgd", "muon"]
# SAM-family method -> base optimizer whose update rule it wraps (and whose tuned LR it inherits).
INHERIT = {
    "atlas": "muon", "atlas_raw": "muon", "atlas_random": "muon",
    "muon_sam": "muon", "muon_sam_frob": "muon", "muon_sam_stale": "muon",
    "fsam_muon": "muon", "fsam_ortho_muon": "muon",
    "fsam": "sgd", "fsam_ortho": "sgd",
}
INHERITED_KEYS = ["lr", "momentum", "weight_decay", "nesterov"]
LR_MULTS = [0.25, 0.5, 1.0, 2.0]
RHO_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


# ----------------------------------------------------------------------------- config helpers

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def exp_id(cfg, path=None):
    return cfg.get("experiment", {}).get("experiment_id", os.path.splitext(os.path.basename(path or ""))[0])


def is_cifar(cfg):
    return cfg.get("experiment", {}).get("task_type") == "image_classification"


def write_cfg(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def sweep_cfg(base_cfg, base_id, opt, entry, tag):
    cfg = copy.deepcopy(base_cfg)
    sid = f"{base_id}__{opt}__{tag}"
    cfg["experiment"]["experiment_id"] = sid
    cfg["experiment"]["wandb_project"] = cfg["experiment"].get("wandb_project", base_id) + "-sweep"
    cfg["optimizers"] = {opt: entry}
    path = os.path.join("configs", "generated", base_id, f"{sid}.yaml")
    write_cfg(cfg, path)
    return path


def tuned_path(base_id):
    return os.path.join("configs", "tuned", f"{base_id}.yaml")


# ----------------------------------------------------------------------------- queue file

def _locked(queue_path):
    lock = open(queue_path + ".lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def read_queue(queue_path):
    if not os.path.exists(queue_path):
        return []
    with open(queue_path) as f:
        return json.load(f)


def write_queue(queue_path, jobs):
    tmp = queue_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(jobs, f, indent=1)
    os.replace(tmp, queue_path)


def add_jobs(queue_path, new_jobs):
    os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)
    lock = _locked(queue_path)
    try:
        jobs = read_queue(queue_path)
        known = {j["id"] for j in jobs}
        added = [j for j in new_jobs if j["id"] not in known]
        write_queue(queue_path, jobs + added)
    finally:
        lock.close()
    print(f"queue {queue_path}: +{len(added)} jobs ({len(new_jobs) - len(added)} already present), total {len(jobs) + len(added)}")


def make_job(config_path, optimizer, seed, epochs, ptq, keep_ckpt):
    cid = exp_id(load_cfg(config_path), config_path)
    return {"id": f"{cid}|{optimizer}|seed{seed}", "config": config_path, "optimizer": optimizer, "seed": seed,
            "epochs": epochs, "ptq": ptq, "keep_ckpt": keep_ckpt, "status": "pending", "attempts": 0}


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(queue_path, gpu):
    lock = _locked(queue_path)
    try:
        jobs = read_queue(queue_path)
        host = socket.gethostname()
        for j in jobs:  # recover jobs whose worker on this host died
            if j["status"] == "running" and j.get("host") == host and not _pid_alive(j.get("pid", -1)):
                j["status"] = "pending"
        job = next((j for j in jobs if j["status"] == "pending"), None)
        if job is not None:
            job.update(status="running", host=host, pid=os.getpid(), gpu=gpu, started=time.strftime("%F %T"))
        write_queue(queue_path, jobs)
        return copy.deepcopy(job) if job else None
    finally:
        lock.close()


def finish(queue_path, job_id, ok, max_attempts, message=""):
    lock = _locked(queue_path)
    try:
        jobs = read_queue(queue_path)
        for j in jobs:
            if j["id"] == job_id:
                j["attempts"] += 1
                j["finished"] = time.strftime("%F %T")
                j["message"] = message
                j["status"] = "done" if ok else ("pending" if j["attempts"] < max_attempts else "failed")
        write_queue(queue_path, jobs)
    finally:
        lock.close()


# ----------------------------------------------------------------------------- running one job

def paths_for(job):
    cid = exp_id(load_cfg(job["config"]), job["config"])
    opt, seed, e = job["optimizer"], job["seed"], job["epochs"]
    return {
        "cid": cid,
        "ckpt": f"checkpoints/{cid}_{opt}_seed{seed}_epoch{e}.pt",
        "ckpt_glob": f"checkpoints/{cid}_{opt}_seed{seed}_epoch*.pt",
        "train_log": f"logs/queue/{cid}_{opt}_seed{seed}_train.log",
        "ptq_log": f"logs/queue/{cid}_{opt}_seed{seed}_ptq.log",
        "ptq_json": f"ptq_results/{cid}/{opt}_seed{seed}.json",
        "csv": f"logs/{cid}_logs.csv",
    }


def final_row(csv_path, opt, seed, epochs):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["optimizer"] == opt and r.get("seed") == str(seed) and int(r["epoch"]) == epochs:
                return r
    return None


_child = None


def _run(cmd, log_path, gpu):
    global _child
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log_path, "a") as log:
        log.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
        log.flush()
        _child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        rc = _child.wait()
        _child = None
    return rc


def run_job(job, gpu):
    p = paths_for(job)
    opt, seed, e = job["optimizer"], job["seed"], job["epochs"]
    if final_row(p["csv"], opt, seed, e) is None or not os.path.exists(p["ckpt"]):
        _run([sys.executable, "-u", "main_experiment.py", "-c", job["config"], "-o", opt, "-e", str(e), "--seed", str(seed)],
             p["train_log"], gpu)
        if final_row(p["csv"], opt, seed, e) is None or not os.path.exists(p["ckpt"]):
            return False, f"training did not produce {p['ckpt']} and a final-epoch log row (see {p['train_log']})"
    if job["ptq"] and not os.path.exists(p["ptq_json"]):
        _run([sys.executable, "-u", "run_ptq.py", "-c", job["config"], "-o", opt, "--seed", str(seed), "-ckpt", p["ckpt"]],
             p["ptq_log"], gpu)
        if not os.path.exists(p["ptq_json"]):
            return False, f"PTQ did not produce {p['ptq_json']} (see {p['ptq_log']})"
    if not job["keep_ckpt"]:
        for f in glob.glob(p["ckpt_glob"]):
            os.remove(f)
    return True, ""


# ----------------------------------------------------------------------------- selection

def _score(row, cifar):
    """Lower is better. Diverged runs (NaN / inf) rank last."""
    if row is None:
        return math.inf
    v = float(row["val_accuracy"]) if cifar else float(row["val_loss"])
    if not math.isfinite(v):
        return math.inf
    return -v if cifar else v


def _pick(candidates, cifar, label):
    """candidates: list of (value, config_path, opt, epochs). Returns best value; warns on grid edges."""
    scored = []
    for value, path, opt, epochs in candidates:
        cid = exp_id(load_cfg(path), path)
        row = final_row(f"logs/{cid}_logs.csv", opt, 42, epochs)
        scored.append((_score(row, cifar), value, row))
    missing = [v for s, v, r in scored if r is None]
    if missing:
        raise SystemExit(f"[{label}] missing sweep results for {missing}: finish the sweep queue first")
    scored.sort(key=lambda t: t[0])
    grid = sorted(v for _, v, _ in scored)
    best_s, best_v, _ = scored[0]
    metric = "val_acc" if cifar else "val_loss"
    print(f"  {label:<16} " + "  ".join(f"{v:g}:{(-s if cifar else s):.4f}" for s, v, _ in sorted(scored, key=lambda t: t[1]))
          + f"  -> best {best_v:g} ({metric})")
    if not math.isfinite(best_s):
        raise SystemExit(f"[{label}] every grid point diverged")
    if best_v in (grid[0], grid[-1]):
        print(f"  [WARNING] {label}: best value {best_v:g} is on the edge of the grid {grid}; extend the grid and re-run")
    return best_v


def sweep_points(base_id, stage):
    """Returns {opt: [(value, config_path)]} from the generated sweep configs."""
    out = {}
    for path in sorted(glob.glob(os.path.join("configs", "generated", base_id, f"{base_id}__*__{stage}*.yaml"))):
        cfg = load_cfg(path)
        (opt, entry), = cfg["optimizers"].items()
        out.setdefault(opt, []).append((entry["lr"] if stage == "lr" else entry["rho"], path))
    return out


# ----------------------------------------------------------------------------- commands

def cmd_sweep_lr(a):
    base = load_cfg(a.config)
    bid = exp_id(base, a.config)
    jobs = []
    for opt in BASE_OPTS:
        for m in LR_MULTS:
            entry = copy.deepcopy(base["optimizers"][opt])
            entry["lr"] = float(f"{entry['lr'] * m:.6g}")
            path = sweep_cfg(base, bid, opt, entry, f"lr{entry['lr']:g}")
            jobs.append(make_job(path, opt, 42, a.epochs, ptq=False, keep_ckpt=False))
    add_jobs(a.queue, jobs)


def cmd_select_lr(a):
    base = load_cfg(a.config)
    bid = exp_id(base, a.config)
    cifar = is_cifar(base)
    pts = sweep_points(bid, "lr")
    tuned = copy.deepcopy(base)
    print(f"[{bid}] LR selection (final-epoch held-out validation, seed 42)")
    for opt in BASE_OPTS:
        tuned["optimizers"][opt]["lr"] = _pick([(v, p, opt, a.epochs) for v, p in pts.get(opt, [])], cifar, opt)
    for opt, parent in INHERIT.items():
        for k in INHERITED_KEYS:
            if k in tuned["optimizers"][parent]:
                tuned["optimizers"][opt][k] = tuned["optimizers"][parent][k]
    write_cfg(tuned, tuned_path(bid))
    print(f"wrote {tuned_path(bid)} (SAM-family methods inherit their base optimizer's tuned LR/momentum/weight decay)")


def cmd_sweep_rho(a):
    tuned = load_cfg(tuned_path(a.base_id))
    jobs = []
    for opt in INHERIT:
        for rho in RHO_GRID:
            entry = copy.deepcopy(tuned["optimizers"][opt])
            entry["rho"] = rho
            entry["rho_vector"] = rho
            path = sweep_cfg(tuned, a.base_id, opt, entry, f"rho{rho:g}")
            jobs.append(make_job(path, opt, 42, a.epochs, ptq=False, keep_ckpt=False))
    add_jobs(a.queue, jobs)


def cmd_select_rho(a):
    tuned = load_cfg(tuned_path(a.base_id))
    cifar = is_cifar(tuned)
    pts = sweep_points(a.base_id, "rho")
    print(f"[{a.base_id}] rho selection (final-epoch held-out validation, seed 42)")
    for opt in INHERIT:
        best = _pick([(v, p, opt, a.epochs) for v, p in pts.get(opt, [])], cifar, opt)
        tuned["optimizers"][opt]["rho"] = best
        tuned["optimizers"][opt]["rho_vector"] = best
    write_cfg(tuned, tuned_path(a.base_id))
    print(f"updated {tuned_path(a.base_id)}")


def cmd_final(a):
    cfg = load_cfg(a.config)
    opts = a.optimizers.split(",") if a.optimizers else list(cfg["optimizers"].keys())
    seeds = [int(s) for s in a.seeds.split(",")]
    add_jobs(a.queue, [make_job(a.config, o, s, a.epochs, ptq=True, keep_ckpt=True) for s in seeds for o in opts])


def cmd_work(a):
    def stop(signum, frame):
        if _child is not None:
            _child.terminate()
            try:
                _child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                _child.kill()
        if current is not None:
            lock = _locked(a.queue)
            try:
                jobs = read_queue(a.queue)
                for j in jobs:
                    if j["id"] == current["id"] and j["status"] == "running":
                        j["status"] = "pending"
                write_queue(a.queue, jobs)
            finally:
                lock.close()
        sys.exit(1)

    current = None
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        current = claim(a.queue, a.gpu)
        if current is None:
            print(f"[{time.strftime('%F %T')}] GPU {a.gpu}: queue empty, exiting")
            return
        print(f"[{time.strftime('%F %T')}] GPU {a.gpu}: start {current['id']}", flush=True)
        ok, msg = run_job(current, a.gpu)
        finish(a.queue, current["id"], ok, a.max_attempts, msg)
        print(f"[{time.strftime('%F %T')}] GPU {a.gpu}: {'done' if ok else 'FAILED'} {current['id']} {msg}", flush=True)
        current = None


def cmd_status(a):
    jobs = read_queue(a.queue)
    counts = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print(f"{a.queue}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f" (total {len(jobs)})")
    for j in jobs:
        if j["status"] in ("running", "failed"):
            print(f"  {j['status']:<8} gpu={j.get('gpu', '-')} {j['id']} {j.get('started', '')} {j.get('message', '')}")


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep-lr"); s.add_argument("--config", required=True); s.add_argument("--epochs", type=int, required=True); s.add_argument("--queue", required=True)
    s = sub.add_parser("select-lr"); s.add_argument("--config", required=True); s.add_argument("--epochs", type=int, required=True)
    s = sub.add_parser("sweep-rho"); s.add_argument("--base-id", required=True); s.add_argument("--epochs", type=int, required=True); s.add_argument("--queue", required=True)
    s = sub.add_parser("select-rho"); s.add_argument("--base-id", required=True); s.add_argument("--epochs", type=int, required=True)
    s = sub.add_parser("final"); s.add_argument("--config", required=True); s.add_argument("--epochs", type=int, required=True)
    s.add_argument("--seeds", default="42,43,44"); s.add_argument("--optimizers", default=None); s.add_argument("--queue", required=True)
    s = sub.add_parser("work"); s.add_argument("--queue", required=True); s.add_argument("--gpu", type=int, required=True); s.add_argument("--max-attempts", type=int, default=2)
    s = sub.add_parser("status"); s.add_argument("--queue", required=True)

    a = ap.parse_args()
    {"sweep-lr": cmd_sweep_lr, "select-lr": cmd_select_lr, "sweep-rho": cmd_sweep_rho, "select-rho": cmd_select_rho,
     "final": cmd_final, "work": cmd_work, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
