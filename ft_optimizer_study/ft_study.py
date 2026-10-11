"""Forgetting study with a FIXED pretrained base model and a VARYING fine-tuning optimizer.

The earlier forgetting experiment fixed the fine-tuning optimizer (AdamW) and varied the optimizer that pretrained the model.
Here it is the other way round:  L_before is the FineWeb-Edu loss of ONE pretrained base model (pretrained with AdamW, Muon or SOMA),
and each of the 11 optimizers is used to fine-tune that same model on a new dataset; L_after is the FineWeb-Edu loss after that fine-tuning.

Everything else follows tools/forgetting.py (same data, 64 steps of 64 x 512 tokens, 10% warmup + cosine, clipping 1.0, same held-out sets, same
matched-loss rule). Per fine-tuning optimizer the tuned pretraining hyper-parameters (configs/tuned/*_v2.yaml: momentum, weight decay, rho, lambda,
sigma, ns_steps, auxiliary-AdamW ratio lr/25) are kept; only the learning rate is swept, on a 11-point log grid from lr_best/100 to 10 x lr_best, where lr_best is
found per (model, optimizer) in a calibration sweep (base = AdamW, seed 42, datasets codeparrot and tulu3, 19 learning rates from 1e-8 to 10).

  python ft_optimizer_study/ft_study.py setup
  python ft_optimizer_study/ft_study.py selftest --gpu 0
  python ft_optimizer_study/ft_study.py calibrate --gpus 0,1,2     # sweeps, then writes results/lr_grids.json
  python ft_optimizer_study/ft_study.py launch --gpus 0,1,2        # 2 models x 3 bases x 3 seeds x 11 optimizers
  python ft_optimizer_study/ft_study.py analyze                    # 6 tables
"""
import argparse
import csv
import json
import math
import os
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
os.chdir(REPO)                      # the repository code uses repo-relative paths (configs/, forgetting_cache/, ptq_cache/)
import forgetting as F             # noqa: E402  (constants, data caches, mean_loss, lr_lambda)

BASES = {"adam": "AdamW", "muon": "Muon", "fsam_ortho_muon_stale_momentum": "SOMA"}
FT_OPTS = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon", "fsam_ortho_muon": "FP-SOMA",
           "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon", "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}
CLOSURE = {"sam", "fsam", "muon_sam", "fsam_ortho_muon", "fsam_ortho_muon_stale_momentum", "randsam_muon", "soma_prens5", "op_soma_postns5"}
MODELS = {"nanogpt": "nanoGPT (GPT-2 small)", "pythia70m": "Pythia-70M"}
WANDB = {"nanogpt": "FTOpt-Forgetting-NanoGPT", "pythia70m": "FTOpt-Forgetting-Pythia70M"}
SEEDS = [42, 43, 44]
DATASETS = list(F.DATASETS)
CAL_DATASETS = ["codeparrot", "tulu3"]
CAL_LRS = [float(f"{x:.3g}") for x in 10 ** np.linspace(-8, 1, 19)]
GRID_EXPONENTS = np.linspace(-2, 1, 11)
CKPT_SRC = os.environ.get("FT_BASE_CKPT_DIR", "/home/anant/atlas_opt_v2/checkpoints")
CKPT_DIR = os.path.join(HERE, "base_checkpoints")
RES = os.path.join(HERE, "results")


# ----------------------------------------------------------------------------- shared pieces

def tuned(m):
    return yaml.safe_load(open(os.path.join(REPO, F.MODELS[f"{m}_s42"]["config"])))["optimizers"]


def ckpt_name(m, base, seed):
    mm = F.MODELS[f"{m}_s{seed}"]
    return f"{mm['cid']}_{base}_seed{seed}_epoch{mm['epoch']}.pt"


def ft_params(m, name, lr, wd=None):
    e = dict(tuned(m)[name])
    e["lr"] = lr
    if "adam_lr" in e:
        e["adam_lr"] = lr / 25.0       # same ratio as in pretraining
    if wd is not None:
        e["weight_decay"] = wd
    return e


def load_base(m, base, seed, device):
    from utils.helpers import build_model
    mk = f"{m}_s{seed}"
    cfg = yaml.safe_load(open(os.path.join(REPO, F.MODELS[mk]["config"])))
    model, _, tok = build_model(cfg, device)
    path = os.path.join(CKPT_DIR, ckpt_name(m, base, seed))
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.float(), tok, mk


def finetune(model, base, train, order, name, params, device):
    """One fine-tuning run with optimizer `name`; returns (train-loss curve, ||theta - theta_base||)."""
    from utils.helpers import make_optimizer, amp_context
    with torch.no_grad():
        for p, b in zip(model.parameters(), base):
            p.copy_(b)
    torch.manual_seed(F.DATA_SEED)
    opt, _ = make_optimizer(name, model, params, epochs=1, steps_per_epoch=F.FT_STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, F.lr_lambda)     # fine-tuning schedule (replaces the pretraining WSD schedule)
    model.train()
    curve = []
    for step in range(F.FT_STEPS):
        def closure():
            opt.zero_grad()
            total = 0.0
            for j in range(0, F.FT_BATCH, F.FT_MICRO):
                ids = train[order[step, j:j + F.FT_MICRO]].to(device, dtype=torch.long)
                with amp_context(model):
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                (loss * (F.FT_MICRO / F.FT_BATCH)).backward()
                total += loss.item() * F.FT_MICRO / F.FT_BATCH
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            return torch.tensor(total)
        if name in CLOSURE:
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()
        sched.step()
        curve.append(float(loss))
    with torch.no_grad():
        dist = math.sqrt(sum(((p - b) ** 2).sum().item() for p, b in zip(model.parameters(), base)))
    return curve, dist


def sweep(model, base, mk, train, heldout, pt_eval, order, name, m, lrs, device, wd=None):
    rows = []
    for lr in lrs:
        t0 = time.time()
        curve, dist = finetune(model, base, train, order, name, ft_params(m, name, lr, wd), device)
        ok = all(math.isfinite(x) for x in curve)
        ft = F.mean_loss(model, heldout, device) if ok else float("nan")
        pt = F.mean_loss(model, pt_eval, device) if ok and math.isfinite(ft) else float("nan")
        ok = ok and math.isfinite(ft) and math.isfinite(pt)
        rows.append(dict(lr=lr, ft_loss=ft, pt_loss=pt, final_train_loss=float(np.mean(curve[-8:])) if ok else float("nan"),
                         delta_norm=dist, status="ok" if ok else "diverged", time_s=round(time.time() - t0, 1)))
    return rows


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


# ----------------------------------------------------------------------------- setup / selftest

def cmd_setup(a):
    os.makedirs(CKPT_DIR, exist_ok=True)
    n = 0
    for m in MODELS:
        for base in BASES:
            for s in SEEDS:
                src, dst = os.path.join(CKPT_SRC, ckpt_name(m, base, s)), os.path.join(CKPT_DIR, ckpt_name(m, base, s))
                assert os.path.exists(src), f"missing base checkpoint {src}"
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
                n += 1
    print(f"{n} base checkpoints linked into {CKPT_DIR}")
    missing = [c for m in MODELS for d in DATASETS if not os.path.exists(F.cache_path(f"{m}_s42", d)) for c in [(m, d)]]
    assert not missing, f"missing fine-tuning data caches (python tools/forgetting.py prepare ...): {missing}"
    print("fine-tuning data caches present")


def cmd_selftest(a):
    from utils.helpers import device
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    # (1) equivalence: AdamW fine-tuning of the AdamW-pretrained nanoGPT (weight decay 0 as in the old experiment) must reproduce the old result
    m, seed, ds, lr = "nanogpt", 42, "codeparrot", 0.000178
    model, tok, mk = load_base(m, "adam", seed, device)
    from ptq import data as pdata
    base = [p.detach().clone() for p in model.parameters()]
    data = torch.load(F.cache_path(mk, ds)); train, heldout = data["train"], data["heldout"]
    order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(F.DATA_SEED)).view(F.FT_STEPS, F.FT_BATCH)
    pt_eval = pdata.lm_eval_sets(tok, F.SEQ)["fineweb"]
    r = sweep(model, base, mk, train, heldout, pt_eval, order, "adam", m, [lr], device, wd=0.0)[0]
    old = next(x for x in csv.DictReader(open(os.path.join(REPO, "artifacts", "forgetting_results", f"{mk}.csv"))) if x["optimizer"] == "adam" and x["dataset"] == ds
               and abs(float(x["lr"]) - lr) < 1e-9)
    print(f"(1) equivalence  new: ft {r['ft_loss']:.5f} pt {r['pt_loss']:.5f} | old: ft {float(old['ft_loss']):.5f} pt {float(old['pt_loss']):.5f}")
    assert abs(r["ft_loss"] - float(old["ft_loss"])) < 5e-3 and abs(r["pt_loss"] - float(old["pt_loss"])) < 5e-3
    # (2) every fine-tuning optimizer runs and gives finite losses (Pythia, 64 steps, a mid-range learning rate)
    m = "pythia70m"
    model, tok, mk = load_base(m, "adam", seed, device)
    base = [p.detach().clone() for p in model.parameters()]
    data = torch.load(F.cache_path(mk, ds)); train, heldout = data["train"], data["heldout"]
    order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(F.DATA_SEED)).view(F.FT_STEPS, F.FT_BATCH)
    pt_eval = pdata.lm_eval_sets(tok, F.SEQ)["fineweb"]
    for name in FT_OPTS:
        lr = tuned(m)[name]["lr"] * 0.3
        t0 = time.time()
        r = sweep(model, base, mk, train, heldout, pt_eval, order, name, m, [lr], device)[0]
        print(f"(2) {name:<32} lr {lr:.3g}: ft {r['ft_loss']:.4f} pt {r['pt_loss']:.4f} status {r['status']} ({time.time() - t0:.0f}s)")
        assert r["status"] == "ok", name
    print("SELFTEST OK")
    F._exit()


# ----------------------------------------------------------------------------- calibration of the learning-rate grids

def cmd_calibrate_run(a):
    from utils.helpers import device
    from ptq import data as pdata
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    out = os.path.join(RES, "calibration", f"{a.model}_{a.opt}.json")
    if os.path.exists(out):
        F._exit()
    model, tok, mk = load_base(a.model, "adam", 42, device)
    base = [p.detach().clone() for p in model.parameters()]
    pt_eval = pdata.lm_eval_sets(tok, F.SEQ)["fineweb"]
    res = {}
    for ds in CAL_DATASETS:
        data = torch.load(F.cache_path(mk, ds)); train, heldout = data["train"], data["heldout"]
        order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(F.DATA_SEED)).view(F.FT_STEPS, F.FT_BATCH)
        res[ds] = sweep(model, base, mk, train, heldout, pt_eval, order, a.opt, a.model, CAL_LRS, device)
        print(f"[{a.model}|{a.opt}|{ds}] best ft_loss {min((r['ft_loss'] for r in res[ds] if r['status']=='ok'), default=float('nan')):.4f}", flush=True)
    atomic_json(out, res)
    F._exit()


def cmd_grids(a):
    grids = {}
    for m in MODELS:
        grids[m] = {}
        for o in FT_OPTS:
            res = json.load(open(os.path.join(RES, "calibration", f"{m}_{o}.json")))
            mean_ft = []
            for i, lr in enumerate(CAL_LRS):
                vals = [res[ds][i]["ft_loss"] for ds in CAL_DATASETS if res[ds][i]["status"] == "ok"]
                mean_ft.append(np.mean(vals) if len(vals) == len(CAL_DATASETS) else float("inf"))
            j = int(np.argmin(mean_ft))
            best = CAL_LRS[j]
            grid = [float(f"{best * 10 ** e:.3g}") for e in GRID_EXPONENTS]
            grids[m][o] = dict(best_lr=best, grid=grid, at_calibration_edge=bool(j in (0, len(CAL_LRS) - 1)), mean_ft_loss_at_best=float(mean_ft[j]))
            flag = "  [calibration edge!]" if grids[m][o]["at_calibration_edge"] else ""
            print(f"{m:<10}{o:<32} best lr {best:.3g} (mean ft loss {mean_ft[j]:.4f}) -> grid {grid[0]:.3g} .. {grid[-1]:.3g}{flag}")
    atomic_json(os.path.join(RES, "lr_grids.json"), grids)


# ----------------------------------------------------------------------------- main runs

def wandb_log(m, base, o, seed, out):
    try:
        import wandb
        run = wandb.init(project=WANDB[m], name=f"base-{BASES[base]}__ft-{FT_OPTS[o]}__s{seed}", group=f"base-{BASES[base]}", job_type="ft_sweep",
                         tags=[f"base:{BASES[base]}", f"ft:{FT_OPTS[o]}", m], reinit=True,
                         config=dict(model=m, base_optimizer=base, ft_optimizer=o, seed=seed, protocol="fixed base model, varied fine-tuning optimizer"))
        for ds, d in out["datasets"].items():
            for i, r in enumerate(d["runs"]):
                wandb.log({f"{ds}/point": i, f"{ds}/lr": r["lr"], f"{ds}/ft_loss": r["ft_loss"], f"{ds}/pt_loss": r["pt_loss"]})
        run.summary["pt_loss_base"] = out["pt_loss_base"]
        wandb.finish()
    except Exception as e:                                   # logging must never break a run
        print("[wandb skipped]", repr(e)[:120], flush=True)


def cmd_run(a):
    from utils.helpers import device
    from ptq import data as pdata
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    out_path = os.path.join(RES, "runs", a.model, a.base, f"{a.opt}_s{a.seed}.json")
    if os.path.exists(out_path):
        F._exit()
    grid = json.load(open(os.path.join(RES, "lr_grids.json")))[a.model][a.opt]["grid"]
    model, tok, mk = load_base(a.model, a.base, a.seed, device)
    base = [p.detach().clone() for p in model.parameters()]
    pt_eval = pdata.lm_eval_sets(tok, F.SEQ)["fineweb"]
    pt_base = F.mean_loss(model, pt_eval, device)
    out = dict(meta=dict(model=a.model, base=a.base, ft_optimizer=a.opt, seed=a.seed, grid=grid), pt_loss_base=pt_base, datasets={})
    for ds in DATASETS:
        data = torch.load(F.cache_path(mk, ds)); train, heldout = data["train"], data["heldout"]
        order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(F.DATA_SEED)).view(F.FT_STEPS, F.FT_BATCH)
        ft_base = F.mean_loss(model, heldout, device)
        runs = sweep(model, base, mk, train, heldout, pt_eval, order, a.opt, a.model, grid, device)
        out["datasets"][ds] = dict(ft_loss_base=ft_base, runs=runs)
        print(f"[{a.model}|base {a.base}|ft {a.opt}|s{a.seed}|{ds}] best ft {min((r['ft_loss'] for r in runs if r['status']=='ok'), default=float('nan')):.4f} "
              f"| ok runs {sum(r['status']=='ok' for r in runs)}/{len(runs)}", flush=True)
    atomic_json(out_path, out)
    wandb_log(a.model, a.base, a.opt, a.seed, out)
    F._exit()


def launch(jobs, gpus, timeout, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    free, running, attempts, pending = list(gpus), {}, {}, list(jobs)
    print(f"{len(pending)} jobs on GPUs {gpus}", flush=True)
    while pending or running:
        while free and pending:
            gpu, job = free.pop(0), pending.pop(0)
            name, args, done = job
            log = open(os.path.join(log_dir, name + ".log"), "a")
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__)] + args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            running[p] = (gpu, job, time.time(), log)
            attempts[name] = attempts.get(name, 0) + 1
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: start {name} (attempt {attempts[name]})", flush=True)
        time.sleep(10)
        for p, (gpu, job, t0, log) in list(running.items()):
            timed_out = p.poll() is None and time.time() - t0 > timeout
            if p.poll() is None and not timed_out:
                continue
            if timed_out:
                p.kill(); p.wait()
            log.close(); del running[p]; free.append(gpu)
            ok = os.path.exists(job[2])
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: {'done' if ok else ('TIMEOUT' if timed_out else 'FAILED rc=%s' % p.returncode)} {job[0]} ({time.time() - t0:.0f}s)", flush=True)
            if not ok and attempts[job[0]] < 2:
                pending.append(job)
    missing = [j[0] for j in jobs if not os.path.exists(j[2])]
    print("finished; jobs without result:", missing or "none", flush=True)


def cost(m, o):
    return (2.0 if m == "nanogpt" else 1.0) * (1.9 if o in ("sam", "fsam", "muon_sam", "fsam_ortho_muon") else 1.3 if o == "op_soma_postns5" else 1.0)


def cmd_calibrate(a):
    gpus = a.gpus.split(",")
    jobs = [(f"cal_{m}_{o}", ["calibrate-run", "--model", m, "--opt", o], os.path.join(RES, "calibration", f"{m}_{o}.json")) for m in MODELS for o in FT_OPTS]
    jobs.sort(key=lambda j: -cost(j[1][2], j[1][4]))
    launch(jobs, gpus, 3 * 3600, os.path.join(HERE, "logs", "calibration"))
    cmd_grids(a)


def cmd_launch(a):
    gpus = a.gpus.split(",")
    jobs = [(f"{m}_{b}_{o}_s{s}", ["run", "--model", m, "--base", b, "--opt", o, "--seed", str(s)], os.path.join(RES, "runs", m, b, f"{o}_s{s}.json"))
            for m in MODELS for b in BASES for s in SEEDS for o in FT_OPTS]
    jobs.sort(key=lambda j: -cost(j[1][2], j[1][6]))
    launch(jobs, gpus, 4 * 3600, os.path.join(HERE, "logs", "runs"))


# ----------------------------------------------------------------------------- analysis and tables

def load_runs(m, base, seed):
    return {o: json.load(open(os.path.join(RES, "runs", m, base, f"{o}_s{seed}.json"))) for o in FT_OPTS}


def analyze_group(m, base):
    """Per seed: common target tau per dataset (largest of the fine-tuning optimizers' best held-out fine-tuning losses); L_after of an optimizer =
    lowest FineWeb loss among its runs that reach tau. Returns per-optimizer lists over seeds."""
    before = []
    after = {o: {d: [] for d in DATASETS} for o in FT_OPTS}
    taus = {d: [] for d in DATASETS}
    sel = {o: {d: [] for d in DATASETS} for o in FT_OPTS}
    for s in SEEDS:
        runs = load_runs(m, base, s)
        before.append(runs["adam"]["pt_loss_base"])
        for d in DATASETS:
            ok = {o: [r for r in runs[o]["datasets"][d]["runs"] if r["status"] == "ok"] for o in FT_OPTS}
            tau = max(min(r["ft_loss"] for r in v) for v in ok.values())
            taus[d].append(tau)
            for o, v in ok.items():
                best = min((r for r in v if r["ft_loss"] <= tau), key=lambda r: r["pt_loss"])
                after[o][d].append(best["pt_loss"])
                grid = runs[o]["meta"]["grid"]
                sel[o][d].append("low" if best["lr"] == grid[0] else "high" if best["lr"] == grid[-1] else "mid")
    return before, after, taus, sel


def cmd_analyze(a):
    from decimal import Decimal, ROUND_HALF_UP
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
    rnd = lambda x, nd: Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    os.makedirs(os.path.join(HERE, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "paper_tables"), exist_ok=True)
    md = []
    for m, long in MODELS.items():
        for base, bname in BASES.items():
            before, after, taus, sel = analyze_group(m, base)
            k = st.mean(before)
            rows = {}
            for o in FT_OPTS:
                avg_seed = [st.mean(after[o][d][i] for d in DATASETS) for i in range(len(SEEDS))]
                adj = [100 * (x - b) / (b + k) for x, b in zip(avg_seed, before)]
                rows[o] = dict(ds={d: (st.mean(after[o][d]), sd(after[o][d])) for d in DATASETS}, avg=(st.mean(avg_seed), sd(avg_seed)), adj=(st.mean(adj), sd(adj)),
                               edge={d: sel[o][d].count("low") + sel[o][d].count("high") for d in DATASETS})
            low = {d: min(rnd(rows[o]["ds"][d][0], 3) for o in FT_OPTS) for d in DATASETS}
            low_avg, low_adj = min(rnd(rows[o]["avg"][0], 3) for o in FT_OPTS), min(rnd(rows[o]["adj"][0], 2) for o in FT_OPTS)
            cell = lambda mu, s_, nd, b: (r"\textbf{%s}" % rnd(mu, nd) if b else str(rnd(mu, nd))) + r"{\scriptsize$\pm$" + str(rnd(s_, nd)) + "}"
            body, csv_rows = [], []
            for o, n in FT_OPTS.items():
                r = rows[o]
                cells = [n, cell(st.mean(before), sd(before), 3, False)]
                cells += [cell(*r["ds"][d], 3, rnd(r["ds"][d][0], 3) == low[d]) for d in DATASETS]
                cells += [cell(*r["avg"], 3, rnd(r["avg"][0], 3) == low_avg), cell(*r["adj"], 2, rnd(r["adj"][0], 2) == low_adj)]
                body.append("    " + " & ".join(cells) + r" \\")
                csv_rows.append([m, bname, n, st.mean(before), sd(before)] + [x for d in DATASETS for x in r["ds"][d]] + list(r["avg"]) + list(r["adj"])
                                + [sum(r["edge"].values())])
            with open(os.path.join(HERE, "analysis", f"{m}_base_{bname}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["model", "base", "ft_optimizer", "L_before", "L_before_sd"] + [f"{d}{s}" for d in DATASETS for s in ("", "_sd")] + ["avg", "avg_sd", "adj_pct", "adj_pct_sd",
                                                                                                                                                 "n_selected_lr_at_grid_edge(of 12)"])
                w.writerows(csv_rows)
            taus_txt = ", ".join(f"{d} {st.mean(taus[d]):.3f}" for d in DATASETS)
            tex = r"""%% Forgetting with a fixed base model and different fine-tuning optimizers. Values are hard-coded; needs \usepackage{booktabs} and \usepackage{graphicx}.
\begin{table*}[t]
  \centering
  \caption{Forgetting when fine-tuning one %s-pretrained %s model with different optimizers: held-out FineWeb-Edu loss before fine-tuning
  ($L_{\mathrm{before}}$, the same for every row) and after fine-tuning on four datasets with the optimizer named in the row ($L_{\mathrm{after}}$); lower is better.
  $L_{\mathrm{after}}$ is read at the same fine-tuning loss for all optimizers (%s). $\Delta\%%_{\mathrm{adj}}=100\,(L_{\mathrm{after}}-L_{\mathrm{before}})/(L_{\mathrm{before}}+k)$ for the average
  $L_{\mathrm{after}}$, with $k=%.3f$ the mean of all $L_{\mathrm{before}}$. Mean $\pm$ std over 3 base-model seeds; bold: lowest in column.}
  \label{tab:ftopt-%s-%s}
  \vspace{2pt}
  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%%
  \footnotesize\setlength{\tabcolsep}{5pt}
  \begin{tabular}{lccccccc}
    \toprule
    & & \multicolumn{4}{c}{$L_{\mathrm{after}}$ by fine-tuning dataset} & & \\
    \cmidrule(lr){3-6}
    Fine-tuning optimizer & $L_{\mathrm{before}}$ & codeparrot & stackmathqa & musicpile & tulu3 & Average & $\Delta\%%_{\mathrm{adj}}$ \\
    \midrule
%s
    \bottomrule
  \end{tabular}}
\end{table*}
""" % (bname, long, taus_txt, k, bname.lower(), m, "\n".join(body))
            open(os.path.join(HERE, "paper_tables", f"table_ftopt_{m}_base_{bname}.tex"), "w").write(tex)
            md.append(f"### {long}, base = {bname}  (k = {k:.3f}; targets: {taus_txt})")
            md.append("| Fine-tuning optimizer | L_before | " + " | ".join(DATASETS) + " | Average | adj. Δ% | selected lr at grid edge |")
            md.append("|---|---|" + "---|" * (len(DATASETS) + 3))
            for o, n in FT_OPTS.items():
                r = rows[o]
                md.append(f"| {n} | {st.mean(before):.3f} ± {sd(before):.3f} | " + " | ".join(f"{r['ds'][d][0]:.3f} ± {r['ds'][d][1]:.3f}" for d in DATASETS)
                          + f" | {r['avg'][0]:.3f} ± {r['avg'][1]:.3f} | {r['adj'][0]:.2f} ± {r['adj'][1]:.2f} | {sum(r['edge'].values())}/12 |")
            md.append("")
    open(os.path.join(HERE, "analysis", "all_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    s = sub.add_parser("selftest"); s.add_argument("--gpu", default="0")
    s = sub.add_parser("calibrate"); s.add_argument("--gpus", required=True)
    s = sub.add_parser("calibrate-run"); s.add_argument("--model", required=True, choices=list(MODELS)); s.add_argument("--opt", required=True, choices=list(FT_OPTS))
    sub.add_parser("grids")
    s = sub.add_parser("launch"); s.add_argument("--gpus", required=True)
    s = sub.add_parser("run"); s.add_argument("--model", required=True, choices=list(MODELS)); s.add_argument("--base", required=True, choices=list(BASES))
    s.add_argument("--opt", required=True, choices=list(FT_OPTS)); s.add_argument("--seed", type=int, required=True)
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"setup": cmd_setup, "selftest": cmd_selftest, "calibrate": cmd_calibrate, "calibrate-run": cmd_calibrate_run, "grids": cmd_grids,
     "launch": cmd_launch, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
