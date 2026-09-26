"""Weight-only PTQ benchmark (RTN / GPTQ / AWQ) for one trained checkpoint.

Example:
  python run_ptq.py -c configs/nanogpt_fineweb.yaml -o adam -ckpt checkpoints/nanogpt_fineweb_adam_epoch15.pt

Results are upserted (by optimizer) into ptq_results/<config_id>.csv; a JSON with the full record
is written to ptq_results/<config_id>/<optimizer>.json.
"""
import os
import sys
import csv
import json
import copy
import math
import time
import argparse
import subprocess
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import build_model, device
from ptq.quant import apply_ptq, conv1d_to_linear, quantized_layers
from ptq import data as pdata

CSV_FIELDS = ["config_id", "optimizer", "checkpoint", "method", "wbits", "group_size", "calib_seed",
              "calib_nsamples", "wikitext2_ppl", "fineweb_ppl", "fineweb_acc", "test_acc", "test_loss",
              "quant_time_s", "status"]


def parse_settings(s):
    out = []
    for item in s.split(","):
        b, g = item.split(":")
        out.append((int(b), int(g)))
    return out


@torch.no_grad()
def eval_lm(model, windows, batch_size=16):
    nll, correct, count = 0.0, 0, 0
    for i in range(0, windows.shape[0], batch_size):
        ids = windows[i:i + batch_size].to(device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        shift_logits, shift_labels = logits[:, :-1], ids[:, 1:]
        nll += F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="sum").item()
        correct += (shift_logits.argmax(-1) == shift_labels).sum().item()
        count += shift_labels.numel()
    return math.exp(nll / count), 100.0 * correct / count


@torch.no_grad()
def eval_cnn(model, loader):
    loss, correct, count = 0.0, 0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x).float()
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    return 100.0 * correct / count, loss / count


def evaluate(model, task_type, eval_data):
    model.eval()
    if task_type == "image_classification":
        acc, loss = eval_cnn(model, eval_data)
        return {"test_acc": acc, "test_loss": loss}
    wt_ppl, _ = eval_lm(model, eval_data["wikitext2"])
    fw_ppl, fw_acc = eval_lm(model, eval_data["fineweb"])
    return {"wikitext2_ppl": wt_ppl, "fineweb_ppl": fw_ppl, "fineweb_acc": fw_acc}


def upsert_rows(csv_path, optimizer, rows):
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["optimizer"] != optimizer]
    tmp = csv_path + f".tmp{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in existing + rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    os.replace(tmp, csv_path)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--optimizer", "-o", required=True, help="Label of the optimizer that produced the checkpoint")
    ap.add_argument("--checkpoint", "-ckpt", required=True)
    ap.add_argument("--methods", default="rtn,gptq,awq")
    ap.add_argument("--settings", default="4:128,3:128,4:-1", help="Comma list of wbits:group_size (-1 = per-channel)")
    ap.add_argument("--seeds", default="0,1,2", help="Calibration seeds for GPTQ/AWQ (RTN is data-free)")
    ap.add_argument("--nsamples", type=int, default=None, help="Calibration size (default: 128 seqs LM, 1024 images CNN)")
    ap.add_argument("--seqlen", type=int, default=512, help="LM calibration/eval length (= training context)")
    ap.add_argument("--out-dir", default="ptq_results")
    args = ap.parse_args()

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    with open(args.config) as f:
        config = yaml.safe_load(f)
    exp = config.get("experiment", {})
    config_id = exp.get("experiment_id", os.path.splitext(os.path.basename(args.config))[0])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    settings = parse_settings(args.settings)
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(os.path.join(args.out_dir, config_id), exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{config_id}.csv")
    json_path = os.path.join(args.out_dir, config_id, f"{args.optimizer}.json")

    print("=" * 90)
    print(f"  PTQ BENCHMARK | config={config_id} | optimizer={args.optimizer}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  methods={methods} | settings(wbits:group)={settings} | calib seeds={seeds}")
    print("=" * 90, flush=True)

    model, task_type, tokenizer = build_model(config, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    del state
    conv1d_to_linear(model)
    model = model.float().eval()

    is_lm = task_type != "image_classification"
    nsamples = args.nsamples or (128 if is_lm else 1024)
    base_row = {"config_id": config_id, "optimizer": args.optimizer,
                "checkpoint": os.path.basename(args.checkpoint), "calib_nsamples": nsamples}
    rows = []

    if is_lm:
        eval_data = pdata.lm_eval_sets(tokenizer, args.seqlen)
        get_calib = lambda seed: pdata.lm_calibration(tokenizer, nsamples, args.seqlen, seed)
        calib_bs = 8
        print(f"  eval tokens: wikitext2={eval_data['wikitext2'].numel():,} fineweb={eval_data['fineweb'].numel():,}")
    else:
        eval_data = pdata.cifar_eval_loader(batch_size=500)
        get_calib = lambda seed: pdata.cifar_calibration(eval_data, nsamples, seed)
        calib_bs = 128

    finite = all(torch.isfinite(p).all() for p in model.parameters())
    if not finite:
        print("  [SKIP] checkpoint contains non-finite weights (diverged training)")
        rows.append({**base_row, "method": "fp", "status": "nonfinite_checkpoint"})
        upsert_rows(csv_path, args.optimizer, rows)
        return

    n_q = sum(l.weight.numel() for l in quantized_layers(model).values())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  quantized weights: {n_q:,} / {n_all:,} params ({100 * n_q / n_all:.1f}%)")

    fp = evaluate(model, task_type, eval_data)
    rows.append({**base_row, "method": "fp", "wbits": 16, "group_size": "", "calib_seed": "", "calib_nsamples": "",
                 "quant_time_s": 0.0, "status": "ok", **fp})
    print(f"  [FP     ] " + " | ".join(f"{k}={fmt(v)}" for k, v in fp.items()), flush=True)
    upsert_rows(csv_path, args.optimizer, rows)

    base = copy.deepcopy(model).cpu()
    del model

    for bits, gs in settings:
        for method in methods:
            for seed in ([None] if method == "rtn" else seeds):
                torch.manual_seed(0 if seed is None else seed)
                m = copy.deepcopy(base).to(device)
                calib = None if method == "rtn" else get_calib(seed)
                t0 = time.time()
                status = "ok"
                try:
                    apply_ptq(m, method, bits, gs, calib=calib, batch_size=calib_bs,
                              log=lambda s: print(s, flush=True))
                    torch.cuda.synchronize()
                    qt = time.time() - t0
                    res = evaluate(m, task_type, eval_data)
                except Exception as e:  # record and continue with the remaining settings
                    qt, res, status = time.time() - t0, {}, f"error: {type(e).__name__}: {e}"
                tag = f"W{bits}g{gs if gs > 0 else 'ch'} {method.upper():<4} seed={seed if seed is not None else '-'}"
                print(f"  [{tag}] " + (" | ".join(f"{k}={fmt(v)}" for k, v in res.items()) or status)
                      + f" | quant {qt:.1f}s", flush=True)
                rows.append({**base_row, "method": method, "wbits": bits, "group_size": gs,
                             "calib_seed": "" if seed is None else seed,
                             "calib_nsamples": "" if method == "rtn" else nsamples,
                             "quant_time_s": round(qt, 2), "status": status, **res})
                upsert_rows(csv_path, args.optimizer, rows)
                del m, calib
                torch.cuda.empty_cache()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    with open(json_path, "w") as f:
        json.dump({"config": config, "args": vars(args), "git_commit": commit, "torch": torch.__version__,
                   "quantized_params": n_q, "total_params": n_all, "rows": rows}, f, indent=2, default=str)
    print(f"  results -> {csv_path}, {json_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # HF streaming threads can crash the interpreter at shutdown
