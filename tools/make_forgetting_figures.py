"""Forgetting summary as (1) a compact table of the FineWeb loss before/after fine-tuning (Markdown + LaTeX) and (2) one TikZ/pgfplots figure per model.

Forgetting dPT = FineWeb-Edu loss after fine-tuning minus before fine-tuning (lower = less forgetting), measured at ONE COMMON
fine-tuning loss tau per (model, seed, dataset): tau = the largest of the 11 optimizers' best fine-tuning losses, so every model
reaches it. Each optimizer's value is its lowest FineWeb loss among runs with fine-tuning loss <= tau. Mean +- sd over 3 seeds.

  python tools/make_forgetting_figures.py   ->   paper_figures_v2/
All numbers are written to CSV; the .tex files only reference them (pgfplots / pgfplotstable read them at compile time).
"""
import csv
import math
import os
import shutil
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "paper_figures_v2"
SEEDS = [42, 43, 44]
DS = ["codeparrot", "stackmathqa", "musicpile", "tulu3"]
MODELS = {"nanogpt": "nanoGPT (GPT-2 small)", "pythia70m": "Pythia-70M"}
SHORT = {"nanogpt": "nanoGPT", "pythia70m": "Pythia-70M"}
NAMES = {"adam": "AdamW", "sgd": "SGD", "muon": "Muon", "sam": "SAM", "fsam": "FSAM", "muon_sam": "SpecSAM-Muon",
         "fsam_ortho_muon": "FP-SOMA", "fsam_ortho_muon_stale_momentum": "SOMA", "randsam_muon": "RandSAM-Muon",
         "soma_prens5": "SOMA-PreNS5", "op_soma_postns5": "OP-SOMA-PostNS5"}
WEAK_PPL_RATIO = 1.03   # starting FineWeb perplexity more than 3% above AdamW's -> clearly weaker starting point, flagged with a dagger
COLORS = {"codeparrot": "0072B2", "stackmathqa": "E69F00", "musicpile": "009E73", "tulu3": "CC79A7"}


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def collect():
    """dpt[m][o][ds] = per-seed list; base[m][o] = per-seed base FineWeb loss."""
    dpt = {m: {o: {d: [] for d in DS} for o in NAMES} for m in MODELS}
    aft = {m: {o: {d: [] for d in DS} for o in NAMES} for m in MODELS}   # FineWeb loss after fine-tuning (L_after)
    taus = {m: {d: [] for d in DS} for m in MODELS}                       # common fine-tuning-loss target per dataset
    base = {m: {o: [] for o in NAMES} for m in MODELS}
    for m in MODELS:
        for s in SEEDS:
            rows = [r for r in csv.DictReader(open(f"forgetting_results/{m}_s{s}.csv")) if r["optimizer"] in NAMES]
            for d in DS:
                ok = {o: [r for r in rows if r["optimizer"] == o and r["dataset"] == d and r["status"] == "ok"] for o in NAMES}
                tau = max(min(float(r["ft_loss"]) for r in v) for v in ok.values())
                taus[m][d].append(tau)
                for o, v in ok.items():
                    best = min((r for r in v if float(r["ft_loss"]) <= tau), key=lambda r: float(r["pt_loss"]))
                    dpt[m][o][d].append(float(best["pt_loss"]) - float(best["pt_loss_base"]))
                    aft[m][o][d].append(float(best["pt_loss"]))
            for o in NAMES:
                base[m][o].append(float(next(r for r in rows if r["optimizer"] == o and r["status"] == "base")["pt_loss_base"]))
    return dpt, base, aft, taus


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    for sub in ("data", "common"):
        shutil.rmtree(os.path.join(OUT, sub), ignore_errors=True)
    dpt, base, aft, taus = collect()
    table = {}   # m -> list of dict rows
    for m in MODELS:
        adam_base = st.mean(base[m]["adam"])
        rows = []
        for i, (o, name) in enumerate(NAMES.items()):
            per_ds = {d: dpt[m][o][d] for d in DS}
            avg_seed = [st.mean(per_ds[d][k] for d in DS) for k in range(len(SEEDS))]
            weak = math.exp(st.mean(base[m][o]) - adam_base) > WEAK_PPL_RATIO
            rows.append(dict(idx=i, name=name, internal=o, weak=weak, avg=st.mean(avg_seed), avg_sd=sd(avg_seed),
                             **{d: st.mean(per_ds[d]) for d in DS}, **{d + "_sd": sd(per_ds[d]) for d in DS}))
        best = min((r for r in rows if not r["weak"]), key=lambda r: r["avg"])["name"]
        for r in rows:
            r["label"] = r["name"] + (r"$^\dagger$" if r["weak"] else "")
            r["cell"] = f"{r['avg']:.3f} $\\pm$ {r['avg_sd']:.3f}"
            r["cell"] += r"$^\dagger$" if r["weak"] else ""
            r["cell_best"] = (r"\textbf{" + r["cell"] + "}") if r["name"] == best else r["cell"]
        table[m] = rows
        write_csv(f"{OUT}/data/forgetting_{m}.csv",
                  ["idx", "label", "optimizer"] + [x for d in DS for x in (d, d + "_sd")] + ["avg", "avg_sd", "weak_base"],
                  [[r["idx"], r["label"], r["name"]] + [f"{r[x]:.5f}" for d in DS for x in (d, d + "_sd")] + [f"{r['avg']:.5f}", f"{r['avg_sd']:.5f}", int(r["weak"])]
                   for r in rows])
    # ----- one table per model: L_before, L_after on every fine-tuning dataset, average L_after, smoothed % rise -----
    names = list(NAMES.values())
    ba, k_smooth = {}, {}
    for m in MODELS:
        ba[m] = {}
        for o in NAMES:
            b = base[m][o]
            a_seed = [st.mean(aft[m][o][d][k] for d in DS) for k in range(len(SEEDS))]
            ba[m][o] = dict(b=st.mean(b), b_sd=sd(b), a=st.mean(a_seed), a_sd=sd(a_seed), b_seed=b, a_seed=a_seed,
                            ds={d: (st.mean(aft[m][o][d]), sd(aft[m][o][d])) for d in DS})
        # additive smoothing: adj. % = 100 (L_after - L_before) / (L_before + k), k = MEAN of all L_before values of this model (11 optimizers x 3 seeds);
        # computed per seed, then mean +- sd over seeds
        k_smooth[m] = st.mean(x for o in NAMES for x in ba[m][o]["b_seed"])
        for o in NAMES:
            sm = [100.0 * (a - b0) / (b0 + k_smooth[m]) for a, b0 in zip(ba[m][o]["a_seed"], ba[m][o]["b_seed"])]
            ba[m][o]["sm"], ba[m][o]["sm_sd"] = st.mean(sm), sd(sm)
        weak = {r["internal"]: r["weak"] for r in table[m]}
        for o in NAMES:
            ba[m][o]["weak"] = weak[o]
        comp = [o for o in NAMES if not weak[o]]          # bold = lowest among comparable (non-dagger) models
        for d in DS:
            bd = min(comp, key=lambda o: ba[m][o]["ds"][d][0])
            for o in NAMES:
                ba[m][o].setdefault("best_ds", {})[d] = (o == bd)
        for key_, o_best in (("best", min(comp, key=lambda o: ba[m][o]["a"])), ("best_sm", min(comp, key=lambda o: ba[m][o]["sm"]))):
            for o in NAMES:
                ba[m][o][key_] = (o == o_best)
    cell = lambda x, sdv, nd=3: f"{x:.{nd}f} $\\pm$ {sdv:.{nd}f}"
    bold = lambda txt, on: (r"\textbf{" + txt + "}") if on else txt
    cols = ["optimizer", "before"] + DS + ["avg", "smooth"]
    md = []
    for m, long in MODELS.items():
        rows_csv = []
        md += [f"**{long}**", "", "| Optimizer | L_before | " + " | ".join(f"L_after {d}" for d in DS) + " | L_after average | adj. Δ% |", "|---|---|" + "---|" * (len(DS) + 2)]
        for o, n in NAMES.items():
            d_ = ba[m][o]
            tex = [n, cell(d_["b"], d_["b_sd"]) + (r"$^\dagger$" if d_["weak"] else "")]
            tex += [bold(cell(*d_["ds"][d]), d_["best_ds"][d]) for d in DS]
            tex += [bold(cell(d_["a"], d_["a_sd"]), d_["best"]), bold(cell(d_["sm"], d_["sm_sd"], 2), d_["best_sm"])]
            rows_csv.append(tex)
            mdc = lambda x, sdv, on, nd=3: (f"**{x:.{nd}f} ± {sdv:.{nd}f}**" if on else f"{x:.{nd}f} ± {sdv:.{nd}f}")
            md.append(f"| {n} | {d_['b']:.3f} ± {d_['b_sd']:.3f}{'†' if d_['weak'] else ''} | "
                      + " | ".join(mdc(*d_["ds"][d], d_["best_ds"][d]) for d in DS) + f" | {mdc(d_['a'], d_['a_sd'], d_['best'])} | {mdc(d_['sm'], d_['sm_sd'], d_['best_sm'], 2)} |")
        write_csv(f"{OUT}/data/forgetting_table_{m}.csv", cols, rows_csv)
        md += ["", "τ (common fine-tuning-loss target, mean over seeds): " + ", ".join(f"{d} {st.mean(taus[m][d]):.3f}" for d in DS) + f"; k = {k_smooth[m]:.3f}", ""]
    md += ["L = held-out FineWeb-Edu loss (lower is better): L_before after pretraining; L_after after fine-tuning on the named dataset, at the common fine-tuning-loss "
           "target τ of that dataset; mean ± sd over 3 seeds. Forgetting ΔPT = L_after − L_before. Bold = lowest among models with a comparable starting point. "
           "† = L_before clearly worse than AdamW's (perplexity >3% higher), so these rows are not comparable. adj. Δ% = additive-smoothed percentage rise of the "
           "average L_after, 100·(L_after − L_before)/(L_before + k), k = mean of all L_before values of the model; computed per seed, mean ± sd over seeds."]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/forgetting_table.md", "w").write("\n".join(md) + "\n")
    # ----- LaTeX -----
    pre = [r"% AUTO-GENERATED by tools/make_forgetting_figures.py",
           r"\usepackage{booktabs,pgfplots,pgfplotstable,xcolor}", r"\usepgfplotslibrary{statistics}", r"\pgfplotsset{compat=1.17}",
           r"\providecommand{\figroot}{paper_figures_v2}  % folder that contains data/",
           r"\providecommand{\fitwidth}[1]{\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{#1}}  % shrink to the text width only if wider", ""]
    for d, hexcol in COLORS.items():
        pre.append(r"\definecolor{c-%s}{HTML}{%s}" % (d, hexcol))
    pre.append(r"""
\pgfplotsset{
  dpt bars/.style={ybar, width=\textwidth, height=0.30\textwidth, bar width=2.6pt, ymin=0, enlarge x limits=0.045,
    ymajorgrids, grid style={gray!25}, xtick=data, x tick label style={rotate=35, anchor=east, font=\scriptsize},
    tick label style={font=\scriptsize}, label style={font=\footnotesize}, title style={font=\footnotesize, yshift=0.55cm},
    xlabel={Pretraining optimizer}, ylabel={Forgetting $\Delta$PT (FineWeb loss rise)}, scaled y ticks=false,
    legend style={font=\scriptsize, at={(0.5,1.02)}, anchor=south, legend columns=5, draw=none},
    error bars/y dir=both, error bars/y explicit, error bars/error bar style={line width=0.4pt}},
}
""")
    os.makedirs(f"{OUT}/common", exist_ok=True)
    open(f"{OUT}/common/preamble.tex", "w").write("\n".join(pre))
    shifts = ["-3.9pt", "-1.3pt", "1.3pt", "3.9pt"]          # four bars of width 2.6pt, centred on the tick; the diamond sits at 0pt
    macro = {"nanogpt": r"\forgetTabNanoGPT", "pythia70m": r"\forgetTabPythia"}
    for m, long in MODELS.items():
        f = f"\\figroot/data/forgetting_{m}.csv"
        t = [r"% AUTO-GENERATED: grouped bars = forgetting per fine-tuning dataset, diamond = average over datasets; error bars = sd over 3 seeds",
             r"% the table is loaded explicitly (comma separated) so that the optimizer names can be used as x tick labels",
             r"\pgfplotstableread[col sep=comma]{%s}%s" % (f, macro[m]),
             r"\begin{tikzpicture}", r"\begin{axis}[dpt bars,", f"  xticklabels from table={{{macro[m]}}}{{label}}, title={{{long}}}]"]
        for d, sh in zip(DS, shifts):
            t += [r"\addplot+[ybar, bar shift=%s, fill=c-%s, draw=none, error bars/.cd, y dir=both, y explicit] table[col sep=comma, x=idx, y=%s, y error=%s_sd]{%s};" % (sh, d, d, d, f),
                  r"\addlegendentry{%s}" % d]
        t += [r"\addplot+[only marks, bar shift=0pt, mark=diamond*, mark size=2.6pt, color=black, fill=black, error bars/.cd, y dir=both, y explicit] "
              r"table[col sep=comma, x=idx, y=avg, y error=avg_sd]{%s};" % f, r"\addlegendentry{average}",
              r"\end{axis}", r"\end{tikzpicture}", ""]
        open(f"{OUT}/forgetting_{m}.tex", "w").write("\n".join(t))
    open(f"{OUT}/figure_forgetting.tex", "w").write("\n".join(
        [r"% AUTO-GENERATED: one full-width figure per model (needs common/preamble.tex in the preamble)"] +
        [("\\begin{figure*}[t]\n  \\centering\n  \\fitwidth{\\input{\\figroot/forgetting_%s.tex}}\n"
          "  \\caption{Catastrophic forgetting of %s models pretrained with each optimizer, after fine-tuning on four datasets. Bars: rise "
          "$\\Delta$PT of the held-out FineWeb-Edu loss at a common fine-tuning loss (lower is better); diamond: average over datasets; "
          "error bars: standard deviation over 3 pretraining seeds. $^\\dagger$ starting model clearly weaker than AdamW's (not comparable).}\n"
          "  \\label{fig:forgetting-%s}\n\\end{figure*}\n") % (m, MODELS[m], m) for m in MODELS]))
    TABLE_TPL = r"""% AUTO-GENERATED: numbers read from data/forgetting_table_@M@.csv
\begin{table*}[t]
  \centering
  \small
  \caption{@LONG@: held-out FineWeb-Edu loss $L$ of models pretrained with each optimizer, before ($L_{\text{before}}$) and after ($L_{\text{after}}$) fine-tuning on each
  of four datasets (lower is better; forgetting $\Delta\mathrm{PT}=L_{\text{after}}-L_{\text{before}}$). $L_{\text{after}}$ is taken at a common fine-tuning loss per dataset
  (@TAU@); mean $\pm$ sd over 3 seeds. ``Average'' is the mean of the four $L_{\text{after}}$ values. $\Delta\%_{\text{adj}}=100\,(L_{\text{after}}-L_{\text{before}})/(L_{\text{before}}+k)$ with
  $k=@K@$ (mean of all $L_{\text{before}}$ of this model). Bold: lowest among comparable models. $^\dagger$: $L_{\text{before}}$ clearly worse than AdamW's (perplexity $>3\%$ higher), so these rows are not comparable.}
  \label{tab:forgetting-@M@}
  \fitwidth{\pgfplotstabletypeset[col sep=comma, string type, every head row/.style={before row=\toprule, after row=\midrule},
    every last row/.style={after row=\bottomrule},
    columns={optimizer,before,codeparrot,stackmathqa,musicpile,tulu3,avg,smooth},
    columns/optimizer/.style={column name=Optimizer, string type, column type=l},
    columns/before/.style={column name={$L_{\text{before}}$}, string type, column type=c},
    columns/codeparrot/.style={column name={$L_{\text{after}}$ codeparrot}, string type, column type=c},
    columns/stackmathqa/.style={column name={$L_{\text{after}}$ stackmathqa}, string type, column type=c},
    columns/musicpile/.style={column name={$L_{\text{after}}$ musicpile}, string type, column type=c},
    columns/tulu3/.style={column name={$L_{\text{after}}$ tulu3}, string type, column type=c},
    columns/avg/.style={column name={$L_{\text{after}}$ average}, string type, column type=c},
    columns/smooth/.style={column name={$\Delta\%_{\text{adj}}$}, string type, column type=c}]{\figroot/data/forgetting_table_@M@.csv}}
\end{table*}
"""
    for m, long in MODELS.items():
        tau_txt = ", ".join(f"{d}: {st.mean(taus[m][d]):.3f}" for d in DS)
        open(f"{OUT}/table_forgetting_{m}.tex", "w").write(
            TABLE_TPL.replace("@M@", m).replace("@LONG@", long).replace("@TAU@", "target $\\tau$ " + tau_txt).replace("@K@", f"{k_smooth[m]:.3f}"))
    open(f"{OUT}/table_forgetting.tex", "w").write("% AUTO-GENERATED: one table per model\n" + "".join(f"\\input{{\\figroot/table_forgetting_{m}.tex}}\n" for m in MODELS))

    open(f"{OUT}/main.tex", "w").write(r"""% Complete example document: compile main.tex with pdfLaTeX (Overleaf default). Put this file in the same folder as common/ and data/.
\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.75in,columnsep=0.25in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}

\newcommand{\figroot}{.}               % folder that contains common/ and data/ (here: the folder of main.tex)
\input{\figroot/common/preamble.tex}   % pgfplots, pgfplotstable, booktabs, xcolor + plot styles

\title{Catastrophic Forgetting of Models Pretrained with Different Optimizers}
\author{}
\date{}

\begin{document}
\maketitle

\section{Forgetting summary}
Table~\ref{tab:forgetting} compares the eleven optimizers by the held-out FineWeb-Edu loss before and after fine-tuning
on a new dataset (lower is better; the rise is the forgetting). Figures~\ref{fig:forgetting-nanogpt} and~\ref{fig:forgetting-pythia70m} show the
same quantity for each fine-tuning dataset separately.

\input{\figroot/table_forgetting.tex}

\input{\figroot/figure_forgetting.tex}   % two full-width figures, one per model

\end{document}
""")

    open(f"{OUT}/README.md", "w").write("""# paper_figures_v2: forgetting table + one plot per model (11 optimizers)

* `forgetting_table.md` / `table_forgetting.tex`: 11-row table with FineWeb-Edu loss before and after fine-tuning (L_before, L_after) for both models (mean ± sd over 3 seeds).
* `forgetting_nanogpt.tex`, `forgetting_pythia70m.tex`: one grouped-bar plot per model (4 datasets per optimizer, diamond = average, error bars = sd over seeds).
* `figure_forgetting.tex`: both plots as full-width `figure*` with captions.
* `data/*.csv`: every number; the .tex files only read them. `common/preamble.tex`: packages and styles.

Overleaf: upload this folder; preamble `\\newcommand{\\figroot}{paper_figures_v2}` + `\\input{\\figroot/common/preamble.tex}`
(needs pgfplots >= 1.17, pgfplotstable, booktabs); body `\\input{\\figroot/table_forgetting.tex}` and `\\input{\\figroot/figure_forgetting.tex}`.
Regenerate: `python tools/make_forgetting_figures.py`.
""")
    print("\n".join(md))


if __name__ == "__main__":
    main()
