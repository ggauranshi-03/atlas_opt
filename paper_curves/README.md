# paper_curves: training curves of the 11 optimizers on 3 models (normal experiments)

* `plots/<model>_<metric>.tex`: 9 single-column plots (models: nanogpt, pythia70m, cifar10; metrics: train_loss, val_acc, val_ppl).
* `plots/grid_all.tex` + `figure_curves.tex`: all 9 as one 3x3 full-width figure with a shared legend.
* `data/<model>/<optimizer>.csv`: per-epoch mean and sd over 3 seeds (the .tex files only read these). `common/preamble.tex`: styles.
* Source: logs/*_v2_logs.csv. Regenerate: `python tools/make_curves.py`.
Overleaf: upload this folder; preamble `\newcommand{\curvesroot}{paper_curves}` + `\input{\curvesroot/common/preamble.tex}`; body `\input{\curvesroot/figure_curves.tex}`
or `\input{\curvesroot/plots/<model>_<metric>.tex}` inside a `figure`.
