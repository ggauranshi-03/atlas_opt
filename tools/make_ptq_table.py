"""INT4 post-training quantization table (FineWeb-Edu only, both language models) as a self-contained LaTeX file with hard-coded values.

  python tools/make_ptq_table.py   ->   paper_tables/table_ptq_fineweb.tex   (artifacts/paper_tables/ if the repository uses the artifacts/ layout)
Needs in the paper preamble: booktabs and graphicx.
Numbers: held-out FineWeb-Edu perplexity of the full-precision (FP) model and of the 4-bit AWQ model (group size 128). Per training seed the INT4 value is the mean
over 3 calibration seeds and the change is 100 (INT4 - FP) / FP; the table shows mean and sample std (n-1) over the 3 training seeds (42, 43, 44).
"""
import csv, os, statistics as st
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
BASE = "artifacts" if os.path.isdir("artifacts/ptq_results") else "."
SEEDS = ["42", "43", "44"]
MODELS = [("nanogpt_v2", "nanoGPT (GPT-2 small)"), ("pythia70m_v2", "Pythia-70M")]
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon", "fsam_ortho_muon": "FP-SOMA",
         "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon", "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}


def stats(cid):
    rows = [r for r in csv.DictReader(open(f"{BASE}/ptq_results/{cid}.csv")) if r["status"] == "ok"]
    out = {}
    for o in NAMES:
        fp, q, ch = [], [], []
        for s in SEEDS:
            f = [float(r["fineweb_ppl"]) for r in rows if r["optimizer"] == o and r["seed"] == s and r["method"] == "fp"]
            a = [float(r["fineweb_ppl"]) for r in rows if r["optimizer"] == o and r["seed"] == s and r["method"] == "awq"]
            assert len(f) == 1 and len(a) == 3, (cid, o, s, len(f), len(a))
            fp.append(f[0]); q.append(st.mean(a)); ch.append(100 * (st.mean(a) - f[0]) / f[0])
        out[o] = {"fp": (st.mean(fp), st.stdev(fp)), "q": (st.mean(q), st.stdev(q)), "d": (st.mean(ch), st.stdev(ch))}
    return out


def cell(m, sdv, nd, bold, sign=False):
    v = f"{m:+.{nd}f}" if sign else f"{m:.{nd}f}"
    return (r"\textbf{" + v + "}" if bold else v) + r"{\scriptsize$\pm$" + f"{sdv:.{nd}f}" + "}"


def main():
    data = {cid: stats(cid) for cid, _ in MODELS}
    best = {}
    for cid, _ in MODELS:
        for k, nd in (("fp", 2), ("q", 2), ("d", 2)):
            lo = min(round(data[cid][o][k][0], nd) for o in NAMES)
            best[(cid, k)] = {o for o in NAMES if round(data[cid][o][k][0], nd) == lo}
    body = []
    for o, n in NAMES.items():
        cells = [n]
        for cid, _ in MODELS:
            s = data[cid][o]
            cells += [cell(*s["fp"], 2, o in best[(cid, "fp")]), cell(*s["q"], 2, o in best[(cid, "q")]), cell(*s["d"], 2, o in best[(cid, "d")], sign=True)]
        body.append("    " + " & ".join(cells) + r" \\")
    tex = r"""%% INT4 post-training quantization on FineWeb-Edu. Values are hard-coded; needs \usepackage{booktabs} and \usepackage{graphicx}.
\begin{table*}[t]
  \centering
  \caption{INT4 post-training quantization of models pretrained with different optimizers. Held-out FineWeb-Edu perplexity (lower is better) of the
  full-precision model (FP) and of the 4-bit AWQ model (INT4), and the relative change $\Delta\%%$ from FP to INT4. AWQ uses 4-bit weights with group size 128 for the
  linear layers of the transformer blocks; embeddings and output head stay in full precision. Mean $\pm$ std over 3 pretraining seeds (each INT4 value is the
  mean of 3 calibration seeds); bold: lowest in column.}
  \label{tab:ptq-fineweb}
  \vspace{2pt}
  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%%
  \footnotesize\setlength{\tabcolsep}{5pt}
  \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{3}{c}{%s} & \multicolumn{3}{c}{%s} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}
    Optimizer & FP & INT4 & $\Delta\%%$ & FP & INT4 & $\Delta\%%$ \\
    \midrule
%s
    \bottomrule
  \end{tabular}}
\end{table*}
""" % (MODELS[0][1], MODELS[1][1], "\n".join(body))
    os.makedirs(f"{BASE}/paper_tables", exist_ok=True)
    open(f"{BASE}/paper_tables/table_ptq_fineweb.tex", "w").write(tex)
    print("written", f"{BASE}/paper_tables/table_ptq_fineweb.tex")


if __name__ == "__main__":
    main()
