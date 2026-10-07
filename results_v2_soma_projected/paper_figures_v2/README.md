# paper_figures_v2: forgetting table + one plot per model (11 optimizers)

* `forgetting_table.md` / `table_forgetting.tex`: 11-row table with FineWeb-Edu loss before and after fine-tuning (L_before, L_after) for both models (mean ± sd over 3 seeds).
* `forgetting_nanogpt.tex`, `forgetting_pythia70m.tex`: one grouped-bar plot per model (4 datasets per optimizer, diamond = average, error bars = sd over seeds).
* `figure_forgetting.tex`: both plots as full-width `figure*` with captions.
* `data/*.csv`: every number; the .tex files only read them. `common/preamble.tex`: packages and styles.

Overleaf: upload this folder; preamble `\newcommand{\figroot}{paper_figures_v2}` + `\input{\figroot/common/preamble.tex}`
(needs pgfplots >= 1.17, pgfplotstable, booktabs); body `\input{\figroot/table_forgetting.tex}` and `\input{\figroot/figure_forgetting.tex}`.
Regenerate: `python tools/make_forgetting_figures.py`.
