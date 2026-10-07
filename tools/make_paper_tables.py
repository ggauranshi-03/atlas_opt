"""Writes the two forgetting tables as self-contained LaTeX files (values hard-coded, no CSV / pgfplotstable needed).

  python tools/make_paper_tables.py   ->   paper_tables/table_forgetting_nanogpt.tex, paper_tables/table_forgetting_pythia70m.tex
Needs in the paper preamble: booktabs and graphicx (both are loaded by the usual ICML / NeurIPS / ICLR styles or are one line each).
Numbers: held-out FineWeb-Edu loss; L_after at one common fine-tuning-loss target per (model, seed, dataset); mean and sample std over 3 pretraining seeds.
"""
import csv, math, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SEEDS = [42, 43, 44]
DS = ["codeparrot", "stackmathqa", "musicpile", "tulu3"]
MODELS = {"nanogpt": "nanoGPT (GPT-2 small)", "pythia70m": "Pythia-70M"}
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon", "fsam_ortho_muon": "FP-SOMA",
         "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon", "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}


def compute(m):
    before = {o: [] for o in NAMES}
    after = {o: {d: [] for d in DS} for o in NAMES}
    taus = {d: [] for d in DS}
    for s in SEEDS:
        rows = [r for r in csv.DictReader(open(f"forgetting_results/{m}_s{s}.csv")) if r["optimizer"] in NAMES]
        for o in NAMES:
            before[o].append(float(next(r["pt_loss_base"] for r in rows if r["optimizer"] == o and r["status"] == "base")))
        for d in DS:
            ok = {o: [r for r in rows if r["optimizer"] == o and r["dataset"] == d and r["status"] == "ok"] for o in NAMES}
            tau = max(min(float(r["ft_loss"]) for r in v) for v in ok.values())
            taus[d].append(tau)
            for o, v in ok.items():
                after[o][d].append(min(float(r["pt_loss"]) for r in v if float(r["ft_loss"]) <= tau))
    k = st.mean(x for o in NAMES for x in before[o])
    res = {}
    for o in NAMES:
        avg_seed = [st.mean(after[o][d][i] for d in DS) for i in range(3)]
        adj_seed = [100 * (a - b) / (b + k) for a, b in zip(avg_seed, before[o])]
        res[o] = dict(before=(st.mean(before[o]), st.stdev(before[o])), ds={d: (st.mean(after[o][d]), st.stdev(after[o][d])) for d in DS},
                      avg=(st.mean(avg_seed), st.stdev(avg_seed)), adj=(st.mean(adj_seed), st.stdev(adj_seed)))
    adam_b = res["adam"]["before"][0]
    for o in NAMES:
        res[o]["dagger"] = math.exp(res[o]["before"][0] - adam_b) > 1.03
    return res, {d: st.mean(taus[d]) for d in DS}, k


def cell(x, sdv, nd=3, bold=False):
    v = f"{x:.{nd}f}"
    return (r"\textbf{" + v + "}" if bold else v) + r"{\scriptsize$\pm$" + f"{sdv:.{nd}f}" + "}"


def write_table(m, res, tau, k):
    low = lambda key, nd: min(round(res[o][key][0], nd) for o in NAMES)
    is_best = {d: {o: round(res[o]["ds"][d][0], 3) == min(round(res[x]["ds"][d][0], 3) for x in NAMES) for o in NAMES} for d in DS}
    is_best["avg"] = {o: round(res[o]["avg"][0], 3) == low("avg", 3) for o in NAMES}
    is_best["adj"] = {o: round(res[o]["adj"][0], 2) == low("adj", 2) for o in NAMES}
    body = []
    for o, n in NAMES.items():
        r = res[o]
        cells = [n, cell(*r["before"])]
        cells += [cell(*r["ds"][d], bold=is_best[d][o]) for d in DS]
        cells += [cell(*r["avg"], bold=is_best["avg"][o]), cell(*r["adj"], nd=2, bold=is_best["adj"][o])]
        body.append("    " + " & ".join(cells) + r" \\")
    name = {"nanogpt": "nanoGPT (GPT-2 small)", "pythia70m": "Pythia-70M"}[m]
    taus = ", ".join(f"{d} {tau[d]:.3f}" for d in DS)
    tex = r"""%% Forgetting table, %s. Values are hard-coded; needs \usepackage{booktabs} and \usepackage{graphicx}.
\begin{table*}[t]
  \centering
  \caption{Forgetting of %s models pretrained with different optimizers: held-out FineWeb-Edu loss before ($L_{\mathrm{before}}$) and after
  ($L_{\mathrm{after}}$) fine-tuning on four datasets (lower is better). $\Delta\%%_{\mathrm{adj}}$ is the smoothed relative increase of the average
  $L_{\mathrm{after}}$. Mean $\pm$ std over 3 seeds; bold: lowest in column.}
  \label{tab:forgetting-%s}
  \vspace{2pt}
  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%%
  \footnotesize\setlength{\tabcolsep}{4.5pt}
  \begin{tabular}{lccccccc}
    \toprule
    & & \multicolumn{4}{c}{$L_{\mathrm{after}}$ by fine-tuning dataset} & & \\
    \cmidrule(lr){3-6}
    Optimizer & $L_{\mathrm{before}}$ & codeparrot & stackmathqa & musicpile & tulu3 & Average & $\Delta\%%_{\mathrm{adj}}$ \\
    \midrule
%s
    \bottomrule
  \end{tabular}}
\end{table*}
""" % (name, name, m, "\n".join(body))
    os.makedirs("paper_tables", exist_ok=True)
    open(f"paper_tables/table_forgetting_{m}.tex", "w").write(tex)
    return tex


if __name__ == "__main__":
    for m in MODELS:
        res, tau, k = compute(m)
        write_table(m, res, tau, k)
    print("written:", sorted(os.listdir("paper_tables")))
