# Forgetting with a fixed base model and different fine-tuning optimizers

The main forgetting experiment fixes the fine-tuning optimizer (AdamW) and varies the optimizer that **pretrained** the model.
This study does the opposite. For a base optimizer **B** in {AdamW, Muon, SOMA}:

* `L_before` = held-out FineWeb-Edu loss of the model pretrained with B (identical for every row of the table);
* for each of the 11 optimizers, that same model is fine-tuned on a new dataset **with that optimizer**; `L_after` = FineWeb-Edu loss afterwards.

So the only thing that changes between rows is the **fine-tuning optimizer**. 2 models (nanoGPT, Pythia-70M) x 3 bases = **6 tables**.
This folder is separate from the earlier experiments: own code, results, logs, tables and wandb projects. Nothing here writes to `artifacts/`.

## Protocol (same as the main forgetting experiment unless noted)
* Base models: the final v2 checkpoints (seeds 42, 43, 44) of the AdamW, Muon and SOMA (`fsam_ortho_muon_stale_momentum`) pretraining runs
  (read through symlinks in `base_checkpoints/`, nothing is copied).
* Datasets: codeparrot, stackmathqa, musicpile, tulu3; 64 steps, batch 64 x 512 tokens (2.1M tokens), 10% warmup + cosine, gradient clipping 1.0, fixed data order;
  held-out sets and the FineWeb-Edu evaluation sample are the same as before (loss in float32).
* Fine-tuning optimizers: AdamW, SGD, Muon, SAM, FSAM, SpecSAM-Muon, FP-SOMA, SOMA, RandSAM-Muon, SOMA-PreNS5, OP-SOMA-PostNS5, each with the hyper-parameters tuned for
  pretraining (momentum, weight decay, rho, lambda, sigma, ns_steps, auxiliary-AdamW ratio lr/25). Only the learning rate is swept. (The AdamW row therefore uses the tuned
  weight decay 0.1 and differs slightly from the main experiment, which fine-tuned with weight decay 0.)
* Learning-rate grid per (model, optimizer): `calibrate` first runs 19 learning rates (1e-8 .. 10) on the AdamW base (seed 42; codeparrot and tulu3) and takes the rate with the lowest
  mean held-out fine-tuning loss, `lr_best`; the study grid is 11 log-spaced rates from `lr_best/100` to `10 x lr_best` (`results/lr_grids.json`).
* Matched-loss rule: for each (model, base, seed, dataset) the target tau is the largest, over the 11 fine-tuning optimizers, of their lowest held-out fine-tuning loss;
  `L_after` of an optimizer is its lowest FineWeb-Edu loss among the runs that reach tau. Tables: mean +- sample std over the 3 base-model seeds; additive-smoothed
  rise `100 (L_after - L_before) / (L_before + k)`, k = mean of all `L_before` of the table.

## Layout
```
ft_study.py      all code (setup, selftest, calibrate, launch, analyze)
run_study.sh     unattended pipeline (setup -> calibrate -> sweeps -> tables)
results/         calibration/ and runs/<model>/<base>/<ft_optimizer>_s<seed>.json (every run: lr, ft_loss, pt_loss, status), lr_grids.json
analysis/        one CSV per table and all_tables.md
paper_tables/    the six LaTeX tables (hard-coded values)   table_ftopt_<model>_base_<base>.tex
logs/            runtime logs (not committed)
```
wandb: projects `FTOpt-Forgetting-NanoGPT` and `FTOpt-Forgetting-Pythia70M`, group `base-<B>`, run name `base-<B>__ft-<optimizer>__s<seed>`.

## Running
```bash
python ft_optimizer_study/ft_study.py setup && python ft_optimizer_study/ft_study.py selftest --gpu 0
setsid nohup bash ft_optimizer_study/run_study.sh > /dev/null 2>&1 < /dev/null &      # about 14-18 h on 7 shared GPUs
```
The self-test checks that fine-tuning with AdamW (weight decay 0) reproduces the main experiment's result and that all 11 optimizers run.
