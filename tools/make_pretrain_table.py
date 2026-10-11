"""Pretraining results (the nine curves of the training-curves figure) as one self-contained LaTeX table with hard-coded values.

  python tools/make_pretrain_table.py   ->   paper_tables/table_pretraining.tex   (artifacts/paper_tables/ if the repository uses the artifacts/ layout)
Needs in the paper preamble: booktabs and graphicx.
Values: last epoch (nanoGPT 15, Pythia-70M 14, CIFAR-10 CNN 7), mean and sample std (n-1) over the 3 pretraining seeds 42, 43, 44.
Training loss = mean over the 100 steps of the epoch; validation accuracy (%) and perplexity on the fixed held-out validation sample;
CIFAR-10 perplexity = exp(validation cross-entropy), computed per seed.
"""
import csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
getcontext().prec = 40
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
BASE = "artifacts" if os.path.isdir("artifacts/logs") else "."
SEEDS = {"42", "43", "44"}
MODELS = [("nanogpt_v2", 15, "nanoGPT (GPT-2 small)", True), ("pythia70m_v2", 14, "Pythia-70M", True), ("cifar10_v2", 7, "CIFAR-10 CNN", False)]
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon", "fsam_ortho_muon": "FP-SOMA",
         "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon", "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}


def exact_mean(vals):
    if isinstance(vals[0], str):
        return sum(Fraction(v) for v in vals) / len(vals)
    return st.mean(vals)


def rnd(m, nd):
    """Round half up; exact for Fractions."""
    d = Decimal(m.numerator) / Decimal(m.denominator) if isinstance(m, Fraction) else Decimal(repr(m))
    return d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def stats(cid, epoch, lm):
    rows = [r for r in csv.DictReader(open(f"{BASE}/logs/{cid}_logs.csv")) if r["seed"] in SEEDS and int(r["epoch"]) == epoch]
    out = {}
    for o in NAMES:
        rs = [r for r in rows if r["optimizer"] == o]
        assert len(rs) == 3, (cid, o, len(rs))
        # the logs hold 4-decimal values; their mean is computed exactly (Fraction) so that ties such as 0.2685 round half-up deterministically
        tl = [r["train_loss"] for r in rs]
        acc = [r["val_accuracy"] for r in rs]
        ppl = [r["val_perplexity"] for r in rs] if lm else [math.exp(float(r["val_loss"])) for r in rs]
        out[o] = [(exact_mean(x), st.stdev([float(v) for v in x])) for x in (tl, acc, ppl)]
    return out


def cell(m, sdv, nd, bold):
    v = str(rnd(m, nd))
    return (r"\textbf{" + v + "}" if bold else v) + r"{\scriptsize$\pm$" + f"{sdv:.{nd}f}" + "}"


def main():
    data = {cid: stats(cid, ep, lm) for cid, ep, _, lm in MODELS}
    nd = {"nanogpt_v2": (3, 2, 2), "pythia70m_v2": (3, 2, 2), "cifar10_v2": (3, 2, 3)}    # decimals for loss, accuracy, perplexity
    best = {}
    for cid, _, _, _ in MODELS:
        for j, hi in ((1, True), (2, False)):                                              # bold only the validation columns
            vals = {o: rnd(data[cid][o][j][0], nd[cid][j]) for o in NAMES}
            tgt = max(vals.values()) if hi else min(vals.values())
            best[(cid, j)] = {o for o in NAMES if vals[o] == tgt}
    body = []
    for o, n in NAMES.items():
        cells = [n]
        for cid, _, _, _ in MODELS:
            for j in range(3):
                m, s = data[cid][o][j]
                cells.append(cell(m, s, nd[cid][j], o in best.get((cid, j), set())))
        body.append("    " + " & ".join(cells) + r" \\")
    head1 = " & ".join(r"\multicolumn{3}{c}{%s}" % name for _, _, name, _ in MODELS)
    tex = r"""%% Pretraining results (last epoch). Values are hard-coded; needs \usepackage{booktabs} and \usepackage{graphicx}.
\begin{table*}[t]
  \centering
  \caption{Pretraining results of the eleven optimizers after the last epoch (nanoGPT 15, Pythia-70M 14, CIFAR-10 CNN 7 epochs): training loss, validation
  accuracy (\%%) and validation perplexity (for the CNN, the exponential of the validation cross-entropy). The training loss of SOMA, RandSAM-Muon, SOMA-PreNS5 and
  OP-SOMA-PostNS5 is measured at the perturbed weights. Mean $\pm$ std over 3 seeds; bold: best validation value in column.}
  \label{tab:pretraining}
  \vspace{2pt}
  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%%
  \footnotesize\setlength{\tabcolsep}{4pt}
  \begin{tabular}{lccccccccc}
    \toprule
    & %s \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
    Optimizer & Train loss & Val acc (\%%) & Val ppl & Train loss & Val acc (\%%) & Val ppl & Train loss & Val acc (\%%) & Val ppl \\
    \midrule
%s
    \bottomrule
  \end{tabular}}
\end{table*}
""" % (head1, "\n".join(body))
    os.makedirs(f"{BASE}/paper_tables", exist_ok=True)
    open(f"{BASE}/paper_tables/table_pretraining.tex", "w").write(tex)
    print("written", f"{BASE}/paper_tables/table_pretraining.tex")


if __name__ == "__main__":
    main()
