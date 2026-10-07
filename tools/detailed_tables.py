"""Detailed per-model tables (training, INT4 PTQ, forgetting) from results/*.csv -> results/detailed_tables.md."""
import csv, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = lambda n: list(csv.DictReader(open(f"results/{n}.csv")))
tr, pq, fg = R("training_summary"), R("ptq_summary"), R("forgetting_summary")
ORDER = ["AdamW", "SGD", "Muon", "SAM", "FSAM", "SpecSAM-Muon", "FP-SOMA", "SOMA", "RandSAM-Muon", "SOMA-PreNS5", "OP-SOMA-PostNS5"]
BASE = {"AdamW": "AdamW", "SAM": "SGD", "FSAM": "SGD", "SGD": "SGD", **{o: "Muon" for o in ["Muon", "SpecSAM-Muon", "FP-SOMA", "SOMA", "RandSAM-Muon", "SOMA-PreNS5", "OP-SOMA-PostNS5"]}}
DS = ["codeparrot", "stackmathqa", "musicpile", "tulu3"]
MODEL = {"nanogpt": "nanoGPT (GPT-2 small)", "pythia70m": "Pythia-70M"}
get = lambda rows, **k: next((r for r in rows if all(r[a] == b for a, b in k.items())), None)
def cell(r):
    if r is None: return "n/a"
    v = r["less_forgetting_pct"].replace("+", "")
    return v + (f" ⚠edge {r['lr_at_grid_edge']}" if not r["lr_at_grid_edge"].startswith("0/") else "")
out = []
for m, title in MODEL.items():
    out += [f"## {title}", "", "### A. Pretraining and INT4-AWQ (mean ± sd over 3 seeds)", "",
            "| Optimizer | Val loss | Val ppl | FineWeb ppl FP | FineWeb ppl AWQ | Δ% | WikiText-2 ppl FP | WikiText-2 ppl AWQ | Δ% |", "|---|---|---|---|---|---|---|---|---|"]
    for o in ORDER:
        t, p = get(tr, model=m, optimizer=o), get(pq, model=m, optimizer=o)
        out.append(f"| {o} | {t['final_val_loss']} | {t['final_val_ppl']} | {p['FineWeb ppl FP']} | {p['FineWeb ppl INT4-AWQ']} | {p['FineWeb ppl change (%)']} | "
                   f"{p['WikiText-2 ppl FP']} | {p['WikiText-2 ppl INT4-AWQ']} | {p['WikiText-2 ppl change (%)']} |")
    for ref_name, refs in [("B. Forgetting vs the optimizer's own base (Muon for the Muon family, SGD for SAM/FSAM)", "own"), ("C. Forgetting vs AdamW", "AdamW")]:
        out += ["", f"### {ref_name}", "", "% less forgetting at matched fine-tuning loss, mean ± sd over 3 seeds; positive = less forgetting. ⚠edge k/3 = fine-tuning LR at the grid edge in k seeds.", "",
                "| Optimizer | " + " | ".join(DS) + " |", "|---|" + "---|" * 4]
        for o in ORDER:
            ref = BASE[o] if refs == "own" else "AdamW"
            if ref == o:
                out.append(f"| {o} | (reference) | | | |"); continue
            out.append(f"| {o} | " + " | ".join(cell(get(fg, model=m, optimizer=o, reference=ref, dataset=d)) for d in DS) + " |")
    out += ["", "### D. Absolute forgetting ΔPT at matched fine-tuning loss, vs AdamW (lower = less forgetting; method / AdamW)", "",
            "| Optimizer | " + " | ".join(DS) + " |", "|---|" + "---|" * 4]
    for o in ORDER[1:]:
        cs = []
        for d in DS:
            r = get(fg, model=m, optimizer=o, reference="AdamW", dataset=d)
            cs.append("n/a" if r is None else f"{r['delta_pt'].split(' ± ')[0]} / {r['ref_delta_pt'].split(' ± ')[0]}")
        out.append(f"| {o} | " + " | ".join(cs) + " |")
    out.append("")
open("results/detailed_tables.md", "w").write("\n".join(out))
print("\n".join(out))
