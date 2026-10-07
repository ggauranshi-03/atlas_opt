"""Single table of the main forgetting metric: FineWeb-Edu loss (pt_loss) of every optimizer's model, before fine-tuning
and after fine-tuning to one COMMON fine-tuning loss per (model, seed, dataset). -> results/pt_loss_table.md/.csv

Common target tau = the largest of the 11 optimizers' best (lowest) fine-tuning losses, so every model can reach it.
For each optimizer: pt_loss = the lowest FineWeb loss among its fine-tuning runs with ft_loss <= tau (same selection rule as the
matched-loss analysis). Mean +- sd over the 3 pretraining seeds. Lower pt_loss = less forgetting."""
import csv, os, statistics as st
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon", "fsam_ortho_muon": "FP-SOMA",
         "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon", "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}
DS = ["codeparrot", "stackmathqa", "musicpile", "tulu3"]
MODELS = {"nanogpt": "nanoGPT", "pythia70m": "Pythia-70M"}
SEEDS = [42, 43, 44]
res, base, tau_rec, edge = {}, {}, {}, {}
for m in MODELS:
    for s in SEEDS:
        rows = [r for r in csv.DictReader(open(f"forgetting_results/{m}_s{s}.csv")) if r["optimizer"] in NAMES]
        for ds in DS:
            ok = {o: [r for r in rows if r["optimizer"] == o and r["dataset"] == ds and r["status"] == "ok"] for o in NAMES}
            tau = max(min(float(r["ft_loss"]) for r in v) for v in ok.values())
            tau_rec.setdefault((m, ds), []).append(tau)
            for o, v in ok.items():
                best = min((r for r in v if float(r["ft_loss"]) <= tau), key=lambda r: float(r["pt_loss"]))
                res.setdefault((m, o, ds), []).append(float(best["pt_loss"]))
                edge.setdefault((m, o, ds), []).append(float(best["lr"]) in (float(min(r["lr"] for r in v)), float(max(r["lr"] for r in v))) and False)
        for o in NAMES:
            b = next(r for r in rows if r["optimizer"] == o and r["status"] == "base")
            base.setdefault((m, o), []).append(float(b["pt_loss_base"]))
ms = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"
hdr = ["Optimizer"] + [f"{MODELS[m]} {c}" for m in MODELS for c in ["before FT"] + DS]
out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
csvrows = []
for o, name in NAMES.items():
    cells = []
    for m in MODELS:
        cells.append(ms(base[(m, o)]))
        cells += [ms(res[(m, o, ds)]) for ds in DS]
    out.append(f"| {name} | " + " | ".join(cells) + " |")
    csvrows.append([name] + cells)
out.append("| *common target fine-tuning loss τ* | | " + " | ".join(f"*{st.mean(tau_rec[('nanogpt', ds)]):.3f}*" for ds in DS) + " | | " + " | ".join(f"*{st.mean(tau_rec[('pythia70m', ds)]):.3f}*" for ds in DS) + " |")
open("results/pt_loss_table.md", "w").write("\n".join(out) + "\n")
with open("results/pt_loss_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(csvrows)
print("\n".join(out))
# best per column
for m in MODELS:
    for ds in DS:
        vals = {NAMES[o]: st.mean(res[(m, o, ds)]) for o in NAMES}
        b = sorted(vals.items(), key=lambda t: t[1])
        print(m, ds, "lowest:", [(n, round(v, 3)) for n, v in b[:3]], "| highest:", [(n, round(v, 3)) for n, v in b[-2:]])
