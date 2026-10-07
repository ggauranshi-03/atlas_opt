"""E1: learning-forgetting frontier of the continued-pretrained LM checkpoints (Watts et al. 2026, Sec. 3.2).

Each base checkpoint (one per pretraining optimizer) is fine-tuned on a new-domain dataset with one fixed
recipe (AdamW, cosine, 10% warmup, batch 64 x 512, wd 0) over a log-spaced LR grid. Per run we record the
held-out fine-tuning loss (learning), the held-out FineWeb-Edu loss (forgetting) and ||theta_FT - theta_base||.
The base optimizer is the only variable: every base model is fine-tuned with the identical AdamW recipe,
the identical data and the identical batch order.

  python tools/forgetting.py prepare --dataset stackmathqa      # tokenize + split once (both tokenizers)
  python tools/forgetting.py launch --gpus 0,1,2,3,4,6,7         # every (model, optimizer, dataset) sweep
  python tools/forgetting.py analyze                             # frontiers, matched-loss forgetting, W&B
"""
import os
import sys
import csv
import json
import math
import time
import fcntl
import argparse
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "forgetting_cache")
OUT = os.path.join(ROOT, "forgetting_results")

SEQ = 512                        # = pretraining context
FT_BATCH = 64                    # sequences per optimizer step
FT_MICRO = 32                    # micro-batch; 2 micro-batches per step, identical math to batch 64
FT_STEPS = 64                    # 64 x 64 x 512 = 2,097,152 fine-tuning tokens
HELDOUT_WINDOWS = 1024           # 524,288 held-out fine-tuning tokens (document-disjoint from train)
DOC_CAP = 4 * SEQ                # tokens kept per document, so no single long file dominates a split
WARMUP_FRAC = 0.10
LRS = [float(f"{x:.3g}") for x in np.geomspace(1e-5, 3e-3, 12)]
DATA_SEED = 0
OPTIMIZERS = ["muon", "muon_sam", "muon_sam_frob", "fsam_muon", "fsam_ortho_muon", "fsam_ortho_muon_stale",
              "fsam_frob_muon_stale", "fsam_ortho_muon_stale_momentum", "fsam_frob_muon_stale_momentum",
              "muon_sam_gfrob", "fsam_gfrob_muon_stale", "fsam_gfrob_muon_stale_momentum", "randsam_muon",
              "sgd", "sam", "sam_ortho", "fsam", "fsam_ortho", "adam"]
# Each sharpness-aware method is compared with the base optimizer whose update rule it wraps, and with AdamW
# (the paper's baseline). Muon-family: Muon descent step; SGD-family: SGD-momentum descent step.
FAMILY_BASE = {"muon": "muon", "muon_sam": "muon", "muon_sam_frob": "muon", "fsam_muon": "muon",
               "fsam_ortho_muon": "muon", "fsam_ortho_muon_stale": "muon", "fsam_frob_muon_stale": "muon",
               "fsam_ortho_muon_stale_momentum": "muon", "fsam_frob_muon_stale_momentum": "muon",
               "muon_sam_gfrob": "muon", "randsam_muon": "muon", "fsam_gfrob_muon_stale": "muon", "fsam_gfrob_muon_stale_momentum": "muon",
               "sgd": "sgd", "sam": "sgd", "sam_ortho": "sgd", "fsam": "sgd", "fsam_ortho": "sgd", "adam": "adam"}
# (Frobenius, spectral) versions of the same method: only the perturbation norm differs.
NORM_PAIRS = [("sam", "sam_ortho"), ("fsam", "fsam_ortho"), ("muon_sam_frob", "muon_sam"),
              ("fsam_muon", "fsam_ortho_muon"), ("fsam_frob_muon_stale", "fsam_ortho_muon_stale"),
              ("fsam_frob_muon_stale_momentum", "fsam_ortho_muon_stale_momentum"),
              # global-Frobenius versions of the per-layer Frobenius methods vs the same spectral partner
              ("muon_sam_gfrob", "muon_sam"), ("fsam_gfrob_muon_stale", "fsam_ortho_muon_stale"),
              ("fsam_gfrob_muon_stale_momentum", "fsam_ortho_muon_stale_momentum")]
# (per-layer Frobenius reference, global Frobenius tested): only the scope of the Frobenius norm differs.
SCOPE_PAIRS = [("muon_sam_frob", "muon_sam_gfrob"), ("fsam_frob_muon_stale", "fsam_gfrob_muon_stale"),
               ("fsam_frob_muon_stale_momentum", "fsam_gfrob_muon_stale_momentum")]
ADAMW = "adam"

MODELS = {
    "nanogpt": dict(config="configs/nanogpt_fineweb.yaml", cid="nanogpt_fineweb", epoch=15,
                    project="Forgetting-NanoGPT", tokenizer="gpt2", pt_tokens=15 * 100 * 16 * 32 * SEQ),
    "pythia70m": dict(config="configs/pythia70m_pretrain_chinchilla.yaml", cid="pythia70m_pretrain_chinchilla",
                      epoch=14, project="Forgetting-Pythia70M", tokenizer="EleutherAI/pythia-70m",
                      pt_tokens=14 * 100 * 16 * 32 * SEQ),
}


def _stackmath(ex):
    return f"Question: {ex['Q']}\nAnswer: {ex['A']}"


def _chat(ex):
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ex["messages"])


DATASETS = {
    "codeparrot": dict(path="codeparrot/codeparrot-clean", kwargs={}, text=lambda ex: ex["content"],
                       domain="far", note="Python GitHub code; stands in for StarCoder-Python (gated)"),
    "stackmathqa": dict(path="math-ai/StackMathQA", kwargs={"name": "stackmathqa1600k"}, text=_stackmath,
                        domain="math", note="Math StackExchange Q&A"),
    "musicpile": dict(path="m-a-p/MusicPile", kwargs={}, text=lambda ex: ex["text"],
                      domain="far", note="Music knowledge, ABC notation and music dialogue"),
    "tulu3": dict(path="allenai/tulu-3-sft-mixture", kwargs={}, text=_chat,
                  domain="near", note="Instruction following (chat rendered as plain text)"),
}


def cache_path(model_key, dataset):
    return os.path.join(CACHE, f"{MODELS[model_key]['tokenizer'].replace('/', '_')}_{dataset}.pt")


def load_tokenizer(name):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name)
    tok.model_max_length = int(1e12)  # documents are packed ourselves; silence length warnings
    return tok


def protocol():
    return {
        "seq_len": SEQ, "ft_batch": FT_BATCH, "ft_steps": FT_STEPS, "ft_tokens": FT_STEPS * FT_BATCH * SEQ,
        "heldout_tokens": HELDOUT_WINDOWS * SEQ, "doc_cap_tokens": DOC_CAP, "lrs": LRS,
        "optimizer": "AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0)", "schedule": "linear warmup 10% -> cosine to 0",
        "grad_clip": 1.0, "precision": "fp32 master weights, bf16 autocast; evaluation in fp32",
        "loss": "full-token causal LM loss on packed 512-token windows (no separators, as in pretraining)",
        "data_seed": DATA_SEED, "base_optimizers": OPTIMIZERS, "family_base": FAMILY_BASE, "norm_pairs": NORM_PAIRS, "scope_pairs": SCOPE_PAIRS, "adamw_reference": ADAMW,
        "matched_rule": "pairwise App. C.4: tau = max(min FT loss of method, min FT loss of reference); "
                        "each reports the least forgetting among its runs with FT loss <= tau",
        "forgetting_eval": "FineWeb-Edu sample-10BT held-out 'eval' shard (013), 1,048,576 tokens (ptq/data.py)",
        "pretraining_tokens": {k: v["pt_tokens"] for k, v in MODELS.items()},
        "base_init": "continued pretraining of released gpt2 / EleutherAI/pythia-70m weights",
        "datasets": {k: {"path": v["path"], **v["kwargs"], "domain": v["domain"], "note": v["note"]} for k, v in DATASETS.items()},
    }


def _exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # HF streaming threads can hang the interpreter at shutdown


# ----------------------------------------------------------------------------- prepare

def cmd_prepare(a):
    """Stream the dataset once in a fixed shuffled order; the first documents fill the held-out split, the next
    ones the training split, independently per tokenizer. Splits are therefore disjoint at the document level."""
    from datasets import load_dataset
    spec = DATASETS[a.dataset]
    need = {"heldout": HELDOUT_WINDOWS * SEQ, "train": FT_STEPS * FT_BATCH * SEQ}
    toks = {mk: load_tokenizer(m["tokenizer"]) for mk, m in MODELS.items()}
    state = {mk: {"phase": "heldout", "heldout": [], "train": [], "docs": {"heldout": 0, "train": 0}} for mk in MODELS}
    ds = load_dataset(spec["path"], split="train", streaming=True, **spec["kwargs"]).shuffle(seed=DATA_SEED, buffer_size=10_000)
    seen = 0
    for ex in ds:
        text = spec["text"](ex)
        if not text or not text.strip():
            continue
        seen += 1
        for mk, st in state.items():
            if st["phase"] == "done":
                continue
            ph = st["phase"]
            st[ph].extend(toks[mk].encode(text)[:DOC_CAP])
            st["docs"][ph] += 1
            if len(st[ph]) >= need[ph]:
                st["phase"] = "train" if ph == "heldout" else "done"
        if all(st["phase"] == "done" for st in state.values()):
            break
    os.makedirs(CACHE, exist_ok=True)
    for mk, st in state.items():
        out = {ph: torch.tensor(st[ph][:need[ph]], dtype=torch.int32).view(-1, SEQ) for ph in need}
        out["meta"] = {"dataset": a.dataset, **{k: spec[k] for k in ("path", "kwargs", "domain")},
                       "tokenizer": MODELS[mk]["tokenizer"], "docs": st["docs"], "docs_streamed": seen,
                       "tokens": {ph: need[ph] for ph in need}, "doc_cap": DOC_CAP, "seed": DATA_SEED}
        torch.save(out, cache_path(mk, a.dataset))
        print(f"[{a.dataset}] {mk}: heldout {out['heldout'].shape} from {st['docs']['heldout']} docs, "
              f"train {out['train'].shape} from {st['docs']['train']} docs -> {cache_path(mk, a.dataset)}")
    _exit()


# ----------------------------------------------------------------------------- one sweep

@torch.no_grad()
def mean_loss(model, windows, device, bs=32):
    model.eval()
    nll, n = 0.0, 0
    for i in range(0, windows.shape[0], bs):
        ids = windows[i:i + bs].to(device, dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
        n += ids[:, 1:].numel()
    return nll / n


def lr_lambda(step):
    warm = max(1, int(round(WARMUP_FRAC * FT_STEPS)))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, FT_STEPS - warm)))


def finetune(model, base, train, order, lr, device):
    with torch.no_grad():
        for p, b in zip(model.parameters(), base):
            p.copy_(b)
    torch.manual_seed(DATA_SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()
    curve = []
    for step in range(FT_STEPS):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for j in range(0, FT_BATCH, FT_MICRO):
            ids = train[order[step, j:j + FT_MICRO]].to(device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            (loss * (FT_MICRO / FT_BATCH)).backward()
            step_loss += loss.item() * FT_MICRO / FT_BATCH
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        curve.append(step_loss)
    with torch.no_grad():
        dist = math.sqrt(sum(((p - b) ** 2).sum().item() for p, b in zip(model.parameters(), base)))
    return curve, dist


def upsert_csv(path, rows, key=("optimizer", "dataset", "lr")):
    fields = ["model", "optimizer", "dataset", "domain", "lr", "ft_loss", "pt_loss", "ft_loss_base", "pt_loss_base",
              "delta_pt", "delta_ft", "delta_norm", "base_norm", "final_train_loss", "status", "time_s"]
    new_keys = {tuple(str(r[k]) for k in key) for r in rows}
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        kept = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                kept = [r for r in csv.DictReader(f) if tuple(r[k] for k in key) not in new_keys]
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in kept + rows:
                w.writerow({k: r.get(k, "") for k in fields})
        os.replace(tmp, path)


def job_json(model_key, opt, dataset):
    return os.path.join(OUT, model_key, f"{opt}_{dataset}.json")


def cmd_run(a):
    from utils.helpers import build_model, device
    from ptq import data as pdata
    torch.backends.cuda.matmul.fp32_precision = "ieee"

    m, spec = MODELS[a.model], DATASETS[a.dataset]
    with open(os.path.join(ROOT, m["config"])) as f:
        cfg = yaml.safe_load(f)
    model, _, tok = build_model(cfg, device)
    ckpt = os.path.join(ROOT, "checkpoints", f"{m['cid']}_{a.optimizer}_seed42_epoch{m['epoch']}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.float()
    base = [p.detach().clone() for p in model.parameters()]  # tied weights appear once
    base_norm = math.sqrt(sum((b ** 2).sum().item() for b in base))

    data = torch.load(cache_path(a.model, a.dataset))
    train, heldout = data["train"], data["heldout"]
    order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(DATA_SEED)).view(FT_STEPS, FT_BATCH)
    pt_eval = pdata.lm_eval_sets(tok, SEQ)["fineweb"]

    t0 = time.time()
    pt_base, ft_base = mean_loss(model, pt_eval, device), mean_loss(model, heldout, device)
    common = dict(model=a.model, optimizer=a.optimizer, dataset=a.dataset, domain=spec["domain"],
                  ft_loss_base=ft_base, pt_loss_base=pt_base, base_norm=base_norm)
    rows = [dict(common, lr=0.0, ft_loss=ft_base, pt_loss=pt_base, delta_pt=0.0, delta_ft=0.0, delta_norm=0.0,
                 final_train_loss="", status="base", time_s=round(time.time() - t0, 1))]
    curves = {}
    print(f"[{a.model}|{a.optimizer}|{a.dataset}] base: pt_loss={pt_base:.4f} ft_loss={ft_base:.4f}", flush=True)
    for lr in LRS:
        t0 = time.time()
        curve, dist = finetune(model, base, train, order, lr, device)
        ok = all(math.isfinite(x) for x in curve)
        ft_l = mean_loss(model, heldout, device) if ok else float("nan")
        pt_l = mean_loss(model, pt_eval, device) if ok else float("nan")
        rows.append(dict(common, lr=lr, ft_loss=ft_l, pt_loss=pt_l, delta_pt=pt_l - pt_base, delta_ft=ft_l - ft_base,
                         delta_norm=dist, final_train_loss=curve[-1], status="ok" if ok else "diverged",
                         time_s=round(time.time() - t0, 1)))
        curves[f"{lr:g}"] = curve
        print(f"  lr={lr:.3g}: ft_loss={ft_l:.4f} pt_loss={pt_l:.4f} dPT={pt_l - pt_base:+.4f} "
              f"||dtheta||={dist:.3f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.join(OUT, a.model), exist_ok=True)
    upsert_csv(os.path.join(OUT, f"{a.model}.csv"), rows)
    with open(job_json(a.model, a.optimizer, a.dataset), "w") as f:
        json.dump({"protocol": protocol(), "checkpoint": os.path.relpath(ckpt, ROOT), "data_meta": data["meta"],
                   "rows": rows, "train_loss_curves": curves}, f, indent=1)

    import wandb  # reporting only, after results are on disk
    run = wandb.init(project=m["project"], name=f"{a.optimizer}_{a.dataset}", group=a.dataset, job_type="lr_sweep",
                     tags=[a.optimizer, a.dataset, spec["domain"]], config={**protocol(), "base_optimizer": a.optimizer,
                                                                           "dataset": a.dataset, "checkpoint": os.path.basename(ckpt)})
    wandb.define_metric("sweep/point")
    wandb.define_metric("sweep/*", step_metric="sweep/point")
    for i, r in enumerate(rows[1:]):
        wandb.log({"sweep/point": i, "sweep/lr": r["lr"], "sweep/log10_lr": math.log10(r["lr"]),
                   "sweep/ft_loss": r["ft_loss"], "sweep/pt_loss": r["pt_loss"],
                   "sweep/forgetting": r["delta_pt"], "sweep/delta_norm": r["delta_norm"]})
    cols = ["lr", "ft_loss", "pt_loss", "delta_pt", "delta_ft", "delta_norm", "status"]
    table = wandb.Table(columns=cols, data=[[r[c] if c != "status" else r[c] for c in cols] for r in rows])
    wandb.log({"sweep_table": table,
               "frontier": wandb.plot.scatter(table, "pt_loss", "ft_loss", title=f"{a.optimizer} on {a.dataset}"),
               "train_loss_curves": wandb.plot.line_series(xs=list(range(1, FT_STEPS + 1)), ys=list(curves.values()),
                                                           keys=[f"lr={k}" for k in curves], title="Fine-tuning loss",
                                                           xname="step")})
    run.summary.update({"base/pt_loss": pt_base, "base/ft_loss": ft_base, "base/param_norm": base_norm})
    wandb.finish()
    _exit()


# ----------------------------------------------------------------------------- launch on all GPUs

def cmd_launch(a):
    gpus = [g.strip() for g in a.gpus.split(",")]
    opts = a.optimizers.split(",") if a.optimizers else OPTIMIZERS
    jobs = [(mk, opt, ds) for mk in MODELS for ds in DATASETS for opt in opts
            if not os.path.exists(job_json(mk, opt, ds))]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    with open(os.path.join(OUT, "protocol.json"), "w") as f:
        json.dump(protocol(), f, indent=1)
    print(f"{len(jobs)} sweeps on GPUs {gpus}", flush=True)
    free, running, attempts = list(gpus), {}, {}
    while jobs or running:
        while free and jobs:
            gpu, job = free.pop(0), jobs.pop(0)
            mk, opt, ds = job
            log = open(os.path.join(OUT, "logs", f"{mk}_{opt}_{ds}.log"), "a")
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "run", "--model", mk,
                                  "--optimizer", opt, "--dataset", ds], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            running[p] = (gpu, job, time.time(), log)
            attempts[job] = attempts.get(job, 0) + 1
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: start {job} (attempt {attempts[job]})", flush=True)
        time.sleep(10)
        for p, (gpu, job, t0, log) in list(running.items()):
            timed_out = p.poll() is None and time.time() - t0 > a.timeout
            if p.poll() is None and not timed_out:
                continue
            if timed_out:
                p.kill()
                p.wait()
            log.close()
            del running[p]
            free.append(gpu)
            done = os.path.exists(job_json(*job))
            status = "done" if done else ("TIMEOUT" if timed_out else f"FAILED rc={p.returncode}")
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: {status} {job} ({time.time() - t0:.0f}s)", flush=True)
            if not done and attempts[job] < 2:
                jobs.append(job)
    print("all sweeps finished", flush=True)


# ----------------------------------------------------------------------------- analysis

def _load(model_key):
    with open(os.path.join(OUT, f"{model_key}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("lr", "ft_loss", "pt_loss", "ft_loss_base", "pt_loss_base", "delta_pt", "delta_norm"):
            r[k] = float(r[k])
    return rows


def pareto(points):
    """Points (pt_loss, ft_loss) not dominated by any other (both lower is better), sorted by pt_loss."""
    pts = sorted(points)
    front, best_ft = [], float("inf")
    for pt, ft in pts:
        if ft < best_ft:
            front.append((pt, ft))
            best_ft = ft
    return front


def best_at(runs, tau):
    """Least forgetting among the runs whose fine-tuning loss reaches tau."""
    return min((r for r in runs if r["ft_loss"] <= tau), key=lambda r: r["pt_loss"])


def matched_pair(rows, method, ref):
    """Paper App. C.4 applied to one (method, reference) pair: tau = max of the two minimum fine-tuning losses, so
    both reach it; each reports its least forgetting among runs with fine-tuning loss <= tau. Pairwise matching keeps
    one badly undertrained base model (e.g. SGD) from setting a loose threshold for every other comparison."""
    runs = {o: [r for r in rows if r["optimizer"] == o and r["status"] == "ok" and math.isfinite(r["ft_loss"])]
            for o in (method, ref)}
    if not runs[method] or not runs[ref]:
        return None
    tau = max(min(r["ft_loss"] for r in v) for v in runs.values())
    bm, br = best_at(runs[method], tau), best_at(runs[ref], tau)
    kappa = lambda r: 2 * r["delta_pt"] / r["delta_norm"] ** 2 if r["delta_norm"] > 0 else float("nan")
    return dict(tau=tau, lr=bm["lr"], ft_loss=bm["ft_loss"], pt_loss=bm["pt_loss"], pt_loss_base=bm["pt_loss_base"],
                delta_pt=bm["delta_pt"], delta_norm=bm["delta_norm"], sharpness=kappa(bm),
                ref_lr=br["lr"], ref_delta_pt=br["delta_pt"], ref_delta_norm=br["delta_norm"], ref_sharpness=kappa(br),
                reduction=1 - bm["delta_pt"] / br["delta_pt"] if br["delta_pt"] > 0 else float("nan"),
                lr_at_grid_edge=bm["lr"] in (LRS[0], LRS[-1]) or br["lr"] in (LRS[0], LRS[-1]))


FAMILIES = {"muon_family_sam": ["muon", "muon_sam", "muon_sam_frob", "muon_sam_gfrob", "randsam_muon", "fsam_muon", "fsam_ortho_muon", "adam"],
            "muon_family_stale": ["muon", "fsam_ortho_muon_stale", "fsam_frob_muon_stale", "fsam_gfrob_muon_stale",
                                  "fsam_ortho_muon_stale_momentum", "fsam_frob_muon_stale_momentum",
                                  "fsam_gfrob_muon_stale_momentum", "adam"],
            "sgd_family": ["sgd", "sam", "sam_ortho", "fsam", "fsam_ortho", "adam"]}
PAIR_COLS = ["model", "dataset", "domain", "frobenius", "spectral", "tau", "frobenius_lr", "spectral_lr",
             "frobenius_base_pt_loss", "spectral_base_pt_loss", "frobenius_delta_pt", "spectral_delta_pt",
             "spectral_less_forgetting", "frobenius_delta_norm", "spectral_delta_norm", "lr_at_grid_edge", "kind"]
SUMMARY_COLS = ["model", "dataset", "domain", "optimizer", "reference", "tau", "lr", "ft_loss", "pt_loss", "pt_loss_base",
                "delta_pt", "ref_lr", "ref_delta_pt", "reduction", "delta_norm", "ref_delta_norm", "sharpness",
                "ref_sharpness", "lr_at_grid_edge"]


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = {o: cmap(i) for i, o in enumerate(OPTIMIZERS)}
    adir = os.path.join(OUT, "analysis")
    os.makedirs(adir, exist_ok=True)
    for mk, m in MODELS.items():
        if not os.path.exists(os.path.join(OUT, f"{mk}.csv")):
            continue
        rows = _load(mk)
        dsets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
        present = [o for o in OPTIMIZERS if any(r["optimizer"] == o for r in rows)]

        summary = []
        for ds in dsets:
            drows = [r for r in rows if r["dataset"] == ds]
            for o in present:
                for ref in dict.fromkeys([FAMILY_BASE[o], ADAMW]):
                    if ref == o or ref not in present:
                        continue
                    res = matched_pair(drows, o, ref)
                    if res:
                        summary.append(dict(model=mk, dataset=ds, domain=DATASETS[ds]["domain"], optimizer=o,
                                            reference=ref, **res))
        with open(os.path.join(adir, f"{mk}_matched.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerows(summary)

        pair_rows = []
        for ds in dsets:
            drows = [r for r in rows if r["dataset"] == ds]
            base_pt = {r["optimizer"]: r["pt_loss"] for r in drows if r["status"] == "base"}
            for (frob, spec), kind in [(q, "norm") for q in NORM_PAIRS] + [(q, "scope") for q in SCOPE_PAIRS]:
                res = matched_pair(drows, spec, frob) if frob in present and spec in present else None
                if res:
                    pair_rows.append(dict(model=mk, dataset=ds, domain=DATASETS[ds]["domain"], frobenius=frob, spectral=spec,
                                          tau=res["tau"], frobenius_lr=res["ref_lr"], spectral_lr=res["lr"],
                                          frobenius_base_pt_loss=base_pt[frob], spectral_base_pt_loss=base_pt[spec],
                                          frobenius_delta_pt=res["ref_delta_pt"], spectral_delta_pt=res["delta_pt"],
                                          spectral_less_forgetting=res["reduction"],
                                          frobenius_delta_norm=res["ref_delta_norm"], spectral_delta_norm=res["delta_norm"],
                                          lr_at_grid_edge=res["lr_at_grid_edge"], kind=kind))
        with open(os.path.join(adir, f"{mk}_norm_pairs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PAIR_COLS)
            w.writeheader()
            w.writerows(pair_rows)

        images = {}
        if pair_rows:
            labels = [p for p in NORM_PAIRS + SCOPE_PAIRS if any((r["frobenius"], r["spectral"]) == p for r in pair_rows)]
            fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4.6))
            wdt = 0.8 / len(dsets)
            for i, ds in enumerate(dsets):
                vals = [next((100 * r["spectral_less_forgetting"] for r in pair_rows if (r["frobenius"], r["spectral"]) == l and r["dataset"] == ds), np.nan)
                        for l in labels]
                ax.bar(np.arange(len(labels)) + i * wdt - 0.4 + wdt / 2, vals, wdt, label=f"{ds} ({DATASETS[ds]['domain']})")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(labels)), [f"{l[1]}\nvs {l[0]}".replace("_", " ") for l in labels], fontsize=5)
            ax.set_ylabel("% less forgetting: 2nd-named method vs 1st-named reference")
            ax.set_title(f"{mk}: same method, different perturbation norm/scope (>0 = tested forgets less)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            images["spectral_vs_frobenius"] = os.path.join(adir, f"{mk}_spectral_vs_frobenius.png")
            fig.savefig(images["spectral_vs_frobenius"], dpi=160)
            plt.close(fig)

        for fam, members in FAMILIES.items():
            members = [o for o in members if o in present]
            if len(members) < 2:
                continue
            fig, axes = plt.subplots(1, len(dsets), figsize=(4.6 * len(dsets), 4.3), squeeze=False)
            for ax, ds in zip(axes[0], dsets):
                drows = [r for r in rows if r["dataset"] == ds]
                for o in members:
                    pts = [(r["pt_loss"], r["ft_loss"]) for r in drows if r["optimizer"] == o and r["status"] == "ok"]
                    base = [(r["pt_loss"], r["ft_loss"]) for r in drows if r["optimizer"] == o and r["status"] == "base"]
                    if not pts:
                        continue
                    ax.scatter(*zip(*pts), s=12, alpha=0.35, color=colors[o])
                    ax.plot(*zip(*pareto(pts + base)), marker="o", ms=3.5, lw=1.6, color=colors[o], label=o)
                    if base:
                        ax.scatter(*zip(*base), marker="*", s=90, color=colors[o], edgecolor="black", lw=0.5, zorder=5)
                ax.set_title(f"{ds} ({DATASETS[ds]['domain']})")
                ax.set_xlabel("FineWeb-Edu held-out loss (forgetting)")
                ax.set_ylabel(f"{ds} held-out loss (learning)")
                ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=7)
            fig.suptitle(f"{mk}: learning-forgetting frontier, {fam.replace('_', ' ')} (star = base model; lower-left is better)")
            fig.tight_layout()
            images[fam] = os.path.join(adir, f"{mk}_frontier_{fam}.png")
            fig.savefig(images[fam], dpi=160)
            plt.close(fig)

        for ref_kind in ("family base", "adamw"):
            sel = [s for s in summary if (s["reference"] == ADAMW) == (ref_kind == "adamw")]
            methods = [o for o in present if any(s["optimizer"] == o for s in sel)]
            if not methods:
                continue
            fig, ax = plt.subplots(figsize=(2.2 * len(dsets) + 3, 4))
            wdt = 0.8 / len(methods)
            for i, o in enumerate(methods):
                vals = [next((100 * s["reduction"] for s in sel if s["optimizer"] == o and s["dataset"] == ds), np.nan)
                        for ds in dsets]
                ax.bar(np.arange(len(dsets)) + i * wdt - 0.4 + wdt / 2, vals, wdt, color=colors[o], label=o)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(dsets)), [f"{d}\n({DATASETS[d]['domain']})" for d in dsets])
            ax.set_ylabel("% less forgetting at matched FT loss")
            ax.set_title(f"{mk}: forgetting reduction vs {'own base optimizer' if ref_kind == 'family base' else 'AdamW'}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            key = "reduction_vs_base" if ref_kind == "family base" else "reduction_vs_adamw"
            images[key] = os.path.join(adir, f"{mk}_{key}.png")
            fig.savefig(images[key], dpi=160)
            plt.close(fig)

        print(f"\n=== {mk}: same method, tested vs reference norm/scope (% less forgetting for the tested one) ===")
        for r in pair_rows:
            print(f"  {r['dataset']:<12} [{r['kind']}] {r['spectral']:<32} vs {r['frobenius']:<32} {100 * r['spectral_less_forgetting']:+7.1f}%"
                  f"{'  [grid edge]' if r['lr_at_grid_edge'] else ''}")
        print(f"\n=== {mk}: % less forgetting at matched fine-tuning loss (pairwise tau) ===")
        for s in summary:
            print(f"  {s['dataset']:<12} {s['optimizer']:<32} vs {s['reference']:<5} {100 * s['reduction']:+7.1f}%  "
                  f"dPT={s['delta_pt']:+.4f} (ref {s['ref_delta_pt']:+.4f})  ||dtheta||={s['delta_norm']:.2f} "
                  f"(ref {s['ref_delta_norm']:.2f})  lr={s['lr']:.3g}{'  [grid edge]' if s['lr_at_grid_edge'] else ''}")

        if a.no_wandb:
            continue
        import wandb
        for old in wandb.Api().runs(f"{wandb.Api().default_entity}/{m['project']}", filters={"display_name": "frontier_analysis"}):
            old.delete()  # a re-analysis supersedes the previous one
        wandb.init(project=m["project"], name="frontier_analysis", job_type="analysis", config=protocol())
        wandb.log({**{k: wandb.Image(v) for k, v in images.items()},
                   "matched_forgetting": wandb.Table(columns=SUMMARY_COLS, data=[[s[c] for c in SUMMARY_COLS] for s in summary]),
                   "norm_pairs": wandb.Table(columns=PAIR_COLS, data=[[str(r[c]) if c in ("dataset", "domain", "frobenius", "spectral", "model") else r[c] for c in PAIR_COLS] for r in pair_rows])})
        wandb.finish()
    _exit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare"); s.add_argument("--dataset", choices=list(DATASETS), required=True)
    s = sub.add_parser("run"); s.add_argument("--model", choices=list(MODELS), required=True)
    s.add_argument("--optimizer", choices=OPTIMIZERS, required=True); s.add_argument("--dataset", choices=list(DATASETS), required=True)
    s = sub.add_parser("launch"); s.add_argument("--gpus", required=True); s.add_argument("--timeout", type=int, default=5400)
    s.add_argument("--optimizers", default=None, help="Comma list (default: all)")
    s = sub.add_parser("analyze"); s.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "launch": cmd_launch, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
