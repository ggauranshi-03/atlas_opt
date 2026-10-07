"""Fine-tuning curves WITH validation accuracy for the 11 optimizers (v2 checkpoints, read-only).

For every (model, optimizer, pretraining seed) and every fine-tuning dataset, the exact fine-tuning recipe of tools/forgetting.py (v2) is re-run at ONE
learning rate (1.78e-4, a point of the v2 fine-tuning grid) and the held-out new-task loss and next-token accuracy are measured every 8 steps (and at
step 0). This script imports the repository code and writes only into paper_ft_curves/raw.
Correctness checks stored with every run: (a) the re-run training-loss curve is compared with the curve saved by the v2 forgetting run at the same
learning rate; (b) validation loss at step 0 / 64 is compared with the v2 CSV (ft_loss_base / ft_loss).

  python tools/ft_worker.py --groups nanogpt:adam:42,nanogpt:sgd:42,...      (CUDA_VISIBLE_DEVICES selects the GPU)
"""
import argparse, csv, json, math, os, sys, time
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
sys.path.insert(0, V2); sys.path.insert(0, V2 + "/tools")
os.chdir(V2)                      # the code uses paths relative to the repository root (configs, forgetting_cache); this process only READS there
import torch, yaml
import forgetting as F
from utils.helpers import build_model, device, amp_context
OUT = os.path.join(V2, "paper_ft_curves", "raw")
LR = 0.000178
EVAL_EVERY = 8
torch.backends.cuda.matmul.fp32_precision = "ieee"


@torch.no_grad()
def evaluate(model, windows, bs=32):
    model.eval()
    nll, correct, n = 0.0, 0, 0
    for i in range(0, windows.shape[0], bs):
        ids = windows[i:i + bs].to(device, dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        tgt = ids[:, 1:]
        nll += torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        correct += (logits[:, :-1].argmax(-1) == tgt).sum().item()
        n += tgt.numel()
    model.train()
    return nll / n, 100.0 * correct / n


def finetune(model, base, train, heldout, order):
    """Same as forgetting.finetune (AdamW, WSD-free cosine, clip 1.0) + periodic held-out evaluation."""
    with torch.no_grad():
        for p, b in zip(model.parameters(), base):
            p.copy_(b)
    torch.manual_seed(F.DATA_SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, F.lr_lambda)
    model.train()
    curve, vals = [], {0: evaluate(model, heldout)}
    for step in range(F.FT_STEPS):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for j in range(0, F.FT_BATCH, F.FT_MICRO):
            ids = train[order[step, j:j + F.FT_MICRO]].to(device, dtype=torch.long)
            with amp_context(model):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            (loss * (F.FT_MICRO / F.FT_BATCH)).backward()
            step_loss += loss.item() * F.FT_MICRO / F.FT_BATCH
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        curve.append(step_loss)
        if (step + 1) % EVAL_EVERY == 0:
            vals[step + 1] = evaluate(model, heldout)
    return curve, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    a = ap.parse_args()
    for g in a.groups.split(","):
        model_name, opt_name, seed = g.split(":")
        out = f"{OUT}/{model_name}_{opt_name}_s{seed}.json"
        if os.path.exists(out):
            continue
        t0 = time.time()
        mk = f"{model_name}_s{seed}"
        m = F.MODELS[mk]
        cfg = yaml.safe_load(open(os.path.join(V2, m["config"])))
        model, _, tok = build_model(cfg, device)
        ckpt = f"{V2}/checkpoints/{m['cid']}_{opt_name}_seed{seed}_epoch{m['epoch']}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model = model.float()
        base = [p.detach().clone() for p in model.parameters()]
        res = {}
        rows = [r for r in csv.DictReader(open(f"{V2}/forgetting_results/{mk}.csv")) if r["optimizer"] == opt_name]
        for ds in F.DATASETS:
            data = torch.load(F.cache_path(mk, ds))
            train, heldout = data["train"], data["heldout"]
            order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(F.DATA_SEED)).view(F.FT_STEPS, F.FT_BATCH)
            curve, vals = finetune(model, base, train, heldout, order)
            sj = json.load(open(f"{V2}/forgetting_results/{mk}/{opt_name}_{ds}.json"))
            sc = sj["train_loss_curves"]["0.000178"]
            r_lr = next(r for r in rows if r["dataset"] == ds and abs(float(r["lr"]) - LR) < 1e-9)
            res[ds] = dict(train_loss=curve, val_steps=sorted(vals), val_loss=[vals[s][0] for s in sorted(vals)], val_acc=[vals[s][1] for s in sorted(vals)],
                           check_train_curve_max_abs_diff=max(abs(x - y) for x, y in zip(curve, sc)),
                           check_val_base_diff=abs(vals[0][0] - float(r_lr["ft_loss_base"])), check_val_final_diff=abs(vals[F.FT_STEPS][0] - float(r_lr["ft_loss"])))
            print(f"[{g}|{ds}] val acc {vals[0][1]:.2f}->{vals[F.FT_STEPS][1]:.2f}  curve diff {res[ds]['check_train_curve_max_abs_diff']:.2e}  "
                  f"val-final diff {res[ds]['check_val_final_diff']:.2e}", flush=True)
        json.dump(dict(model=model_name, optimizer=opt_name, seed=int(seed), lr=LR, eval_every=EVAL_EVERY, results=res), open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"[{g}] done in {time.time() - t0:.0f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
