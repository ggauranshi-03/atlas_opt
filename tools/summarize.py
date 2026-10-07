"""Collect the v2 results (3 training seeds) into results/: training, INT4 PTQ and forgetting summaries (CSV + Markdown).

  python tools/summarize.py
"""
import os
import csv
import json
import math
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "results"
SEEDS = [42, 43, 44]
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon",
         "fsam_ortho_muon": "FP-SOMA", "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon",
         "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}
MODELS = {"nanogpt": ("nanogpt_v2", 15, True), "pythia70m": ("pythia70m_v2", 14, True), "cifar10": ("cifar10_v2", 7, False)}
FAMILY_BASE = {"adam": "adam", "sgd": "sgd", "sam": "sgd", "fsam": "sgd", "muon": "muon", "muon_sam": "muon",
               "fsam_ortho_muon": "muon", "fsam_ortho_muon_stale_momentum": "muon", "randsam_muon": "muon",
               "soma_prens5": "muon", "op_soma_postns5": "muon"}
DATASETS = ["codeparrot", "stackmathqa", "musicpile", "tulu3"]
WEAK = 1.10  # base model counted as weaker than AdamW's if its FineWeb perplexity is >10% higher


def read(path):
    return list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []


def ms(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return "n/a", float("nan"), float("nan"), 0
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return f"{fmt.format(m)} ± {fmt.format(s)}", m, s, len(xs)


def write(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))  # rows may carry different metric columns (CIFAR vs LM)
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    cols = [c for c in fields if not c.startswith("_")]
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    training, ptq_rows, forget_rows = [], [], []
    base_ppl = {}
    for mk, (cid, ep, lm) in MODELS.items():
        logs = read(f"logs/{cid}_logs.csv")
        ptq = [r for r in read(f"ptq_results/{cid}.csv") if r["status"] == "ok"]
        for o, name in NAMES.items():
            fin = [r for r in logs if r["optimizer"] == o and r["seed"] in map(str, SEEDS) and int(r["epoch"]) == ep]
            key = "val_loss" if lm else "val_accuracy"
            d = dict(model=mk, optimizer=name, n_seeds=len(fin),
                     final_train_loss=ms([float(r["train_loss"]) for r in fin], "{:.4f}")[0],
                     final_val_loss=ms([float(r["val_loss"]) for r in fin], "{:.4f}")[0])
            d["final_val_ppl" if lm else "final_val_acc"] = ms([float(r["val_perplexity" if lm else "val_accuracy"]) for r in fin])[0]
            training.append(d)
            fp = {int(r["seed"]): r for r in ptq if r["optimizer"] == o and r["method"] == "fp" and r["seed"].isdigit()}
            aw = {}
            for r in ptq:
                if r["optimizer"] == o and r["method"] == "awq" and r["seed"].isdigit():
                    aw.setdefault(int(r["seed"]), []).append(r)
            seeds = [s for s in SEEDS if s in fp and len(aw.get(s, [])) == 3]
            row = dict(model=mk, optimizer=name, n_seeds=len(seeds))
            metrics = [("fineweb_ppl", "FineWeb ppl"), ("wikitext2_ppl", "WikiText-2 ppl")] if lm else [("test_acc", "test acc %")]
            for k, label in metrics:
                f_ = [float(fp[s][k]) for s in seeds]
                a_ = [st.mean(float(r[k]) for r in aw[s]) for s in seeds]  # mean over the 3 calibration seeds
                dlt = [(a - f) / f * 100 if lm else a - f for f, a in zip(f_, a_)]
                row[f"{label} FP"] = ms(f_)[0]
                row[f"{label} INT4-AWQ"] = ms(a_)[0]
                row[f"{label} change ({'%' if lm else 'pts'})"] = ms(dlt, "{:+.2f}")[0]
                if k == "fineweb_ppl":
                    base_ppl[(mk, o)] = ms(f_)[1]
            ptq_rows.append(row)
    for mk in ("nanogpt", "pythia70m"):
        per_seed = {s: read(f"forgetting_results/analysis/{mk}_s{s}_matched.csv") for s in SEEDS}
        adam_ppl = base_ppl.get((mk, "adam"), float("nan"))
        for o, name in NAMES.items():
            for ref in dict.fromkeys([FAMILY_BASE[o], "adam"]):
                if ref == o:
                    continue
                for ds in DATASETS:
                    hits = [r for s in SEEDS for r in per_seed[s] if r["optimizer"] == o and r["reference"] == ref and r["dataset"] == ds]
                    if not hits:
                        continue
                    red = ms([100 * float(r["reduction"]) for r in hits], "{:+.1f}")
                    weak = base_ppl.get((mk, o), float("nan")) > WEAK * adam_ppl or base_ppl.get((mk, ref), float("nan")) > WEAK * adam_ppl
                    forget_rows.append(dict(
                        model=mk, optimizer=name, reference=NAMES[ref], dataset=ds, n_seeds=red[3],
                        less_forgetting_pct=red[0],
                        delta_pt=ms([float(r["delta_pt"]) for r in hits], "{:.4f}")[0],
                        ref_delta_pt=ms([float(r["ref_delta_pt"]) for r in hits], "{:.4f}")[0],
                        matched_ft_loss=ms([float(r["tau"]) for r in hits], "{:.4f}")[0],
                        lr_at_grid_edge=f"{sum(r['lr_at_grid_edge'] == 'True' for r in hits)}/{len(hits)}",
                        base_fineweb_ppl=f"{base_ppl.get((mk, o), float('nan')):.2f}",
                        weaker_base_than_adamw=weak))
    write("training_summary", training)
    write("ptq_summary", ptq_rows)
    write("forgetting_summary", forget_rows)
    tuned = {m: json.load(open("results/tuning/selected_hyperparameters.json"))[m]
             for m in MODELS} if os.path.exists("results/tuning/selected_hyperparameters.json") else {}
    with open(os.path.join(OUT, "hyperparameters.md"), "w") as f:
        for m, opts in tuned.items():
            f.write(f"## {m}\n| optimizer | " + " | ".join(["lr", "rho", "momentum", "weight_decay", "adam_lr"]) + " |\n|---|---|---|---|---|---|\n")
            for o, c in opts.items():
                f.write(f"| {NAMES.get(o, o)} | " + " | ".join(str(c.get(k, "")) for k in ["lr", "rho", "momentum", "weight_decay", "adam_lr"]) + " |\n")
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
