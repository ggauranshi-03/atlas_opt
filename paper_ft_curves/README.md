# paper_ft_curves: fine-tuning curves (training loss, validation accuracy) of the 11 v2 optimizers

* `figure_ft_train_loss.tex`, `figure_ft_val_acc.tex`: 2 models x 4 datasets grids (each a full-width figure*, shared legend).
* `plots/grid_*.tex`: the TikZ grids. `data/<model>/<opt>_<dataset>_{train,val}.csv`: every plotted number (mean and sd over 3 seeds).
* Training loss = curves saved by the v2 forgetting run (LR 1.78e-4); validation accuracy = re-run of the same recipe with evaluation every 8 steps
  (`tools/ft_worker.py`, raw output in `raw/`). `data/reproduction_check.csv` compares the re-run with the saved v2 results.
Overleaf: upload the folder, set `\newcommand{\ftroot}{paper_ft_curves}` + `\input{\ftroot/common/preamble.tex}`, then `\input{\ftroot/figure_ft_val_acc.tex}`.
