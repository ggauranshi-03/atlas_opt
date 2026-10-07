# Sharpness-aware Muon variants: optimizer study

Code and results for comparing eleven optimizers on three deep learning models (**nanoGPT** (GPT-2 small) and **Pythia-70M** continued pretraining on FineWeb-Edu, and an **airbench-style CNN on CIFAR-10**), as well as on heavily corrupted **Synthetic Benchmarks**.

For each optimizer we measure:
- **Training quality** (loss and accuracy)
- **INT4 Post-Training Quantization (PTQ)** using AWQ
- **Catastrophic Forgetting** (for language models) after fine-tuning on four new datasets.
- **Robustness to Heavy-Tailed Noise** (synthetic ablations).

All main deep learning numbers are mean over three seeds (42, 43, 44).

| Name in the paper | Name in code / configs |
|---|---|
| AdamW | `adam` |
| SGD | `sgd` |
| Muon | `muon` |
| SAM | `sam` / `sam-sgd` |
| FSAM | `fsam` / `friendly-sam-sgd` |
| SpecSAM-Muon | `muon_sam` / `full-spectral-sam-muon` |
| FP-SOMA | `fsam_ortho_muon` / `spectral-friendly-sam-muon` |
| SOMA | `fsam_ortho_muon_stale_momentum` / `stale-momentum-friendly-spectral-sam-muon` |
| RandSAM-Muon | `randsam_muon` / `random-sam-muon` |
| S2SAM-Muon (Stale) | `stale-grad-sam-muon` (Synthetic Ablations) |
| S2SAM-Muon (O) (Stale) | `stale-grad-spectral-sam-muon` (Synthetic Ablations) |
| SOMA-PreNS5 | `soma_prens5` / `op-soma-prens5` |
| OP-SOMA-PostNS5 | `op_soma_postns5` / `op-soma-postns5` |

## Layout
- `optimizers/`: Core PyTorch optimizer implementations.
- `utils/`, `data/`, `ptq/`: Models, data loading, INT4 quantization routines (RTN / AWQ / GPTQ).
- `synthetic_ablations/`: **Synthetic Experiments & Ablations**. Contains isolated testbeds (Linear MSE and Nonlinear classification) with highly anisotropic features and heavy-tailed Pareto noise injection to test optimizer robustness to flat vs sharp minima. 
  - Uses `algorithms.py`, `objective_nn.py`, and `experiment.py` for execution.
- `main_experiment.py`: Main entry point for training one optimizer on one deep learning config (nanoGPT/Pythia/CNN).
- `run_ptq.py`: Quantizes a checkpoint and evaluates it.
- `tools/tune.py`: Learning-rate and rho tuning and the final 3-seed runs.
- `tools/experiments.py`: The GPU job queue.
- `tools/forgetting.py`: The catastrophic forgetting experiment sweeps.
- `tools/summarize.py`: Generates result tables.
- `configs/*_v2.yaml`: Base configurations; `configs/tuned/` contains configurations after tuning.
- `artifacts/`: Outputs of all experiments (results, tables, figures, CSV logs).

## Artifacts
```
artifacts/
  results/               summary tables (training, INT4, forgetting), detailed tables, tuning record
  ptq_results/           INT4 post-training-quantization results (CSV + per-run JSON)
  forgetting_results/    fine-tuning sweeps and the forgetting analysis
  logs/                  per-epoch training logs (CSV)
  paper_tables/          LaTeX tables with hard-coded values
  paper_figures_v2/      forgetting bar plots (TikZ + CSV)
  paper_curves/          training curves (TikZ + CSV)
  paper_ft_curves/       fine-tuning training-loss and accuracy curves (TikZ + CSV + raw re-run output)
```

## Setup
```bash
pip install -r requirements.txt     # install the PyTorch build that matches your CUDA version first
wandb login                         # or: export WANDB_MODE=disabled
```
Data and weights are downloaded from the Hugging Face Hub on first use: FineWeb-Edu (`sample-10BT`, streamed), WikiText-2, and the fine-tuning sets `codeparrot/codeparrot-clean`, `math-ai/StackMathQA`, `m-a-p/MusicPile`, `allenai/tulu-3-sft-mixture`; models `gpt2` and `EleutherAI/pythia-70m`.
The model loader refuses to start from anything but the released weights (an initial-perplexity check aborts the run otherwise).

## Reproducing the Results

### 1. Main Deep Learning Models
```bash
# 1. tune lr and rho, train 3 seeds of every optimizer on all three models (each run is followed by INT4-AWQ PTQ)
python tools/tune.py lr    --gpus 0,1,2,3
python tools/tune.py rho   --gpus 0,1,2,3
python tools/tune.py final --gpus 0,1,2,3          # or: bash run_v2.sh (runs all stages unattended)

# 2. Catastrophic Forgetting: prepare the fine-tuning data once, then sweep (17 learning rates x 4 datasets per model)
for d in codeparrot stackmathqa musicpile tulu3; do python tools/forgetting.py prepare --dataset $d; done
python tools/forgetting.py launch  --gpus 0,1,2,3
python tools/forgetting.py analyze

# 3. Tables and figures
python tools/summarize.py && python tools/detailed_tables.py && python tools/make_paper_tables.py
python tools/make_forgetting_figures.py && python tools/make_curves.py
bash tools/run_ft_curves.sh && python tools/make_ft_figures.py        # fine-tuning accuracy curves (re-runs the fine-tuning)
```

### 2. Synthetic Experiments & Ablations
To reproduce the geometry ablations (Frobenius vs Spectral) and freshness ablations (Fresh vs Stale) under heavy-tailed noise:
```bash
# Run the synthetic ablations
cd synthetic_ablations
python experiment.py configs/nn_two_matrix_anisotropic.yaml --gpus 0,1,2,3
python experiment.py configs/two_matrix_mse_anisotropic_only_spectral.yaml --gpus 0,1,2,3
```

## Method details worth knowing
- **Synthetic Experiments:** Evaluates optimizers on $L=2$ networks (`k=32` classes) with severe anisotropy (condition number 100) and heavy-tailed Pareto noise.
- **Deep Learning Training:** 100 optimizer steps per epoch, batch 16 x 32 accumulation x 512 tokens for the language models (15 epochs nanoGPT, 14 Pythia-70M), 7 epochs for CIFAR-10; warmup-stable-decay schedule. Muon is used for hidden matrices and a small AdamW for embeddings, the output head, biases and norms.
- **SAM Variations:** Fresh SAM-type optimizers compute both gradients of a step on the same minibatch (2 oracle calls). Stale variations (SOMA, S2SAM) reuse the previous step's gradient (1 oracle call).
- **OP-SOMA (Orthogonally Projected SOMA):** PreNS5 and PostNS5 force strictly orthogonal perturbation trajectories relative to momentum or previous updates.
- **Precision:** fp32 master weights; bf16 autocast for GPT-2 and the CNN; Pythia-70M runs in fp32 (its released weights degrade under bf16).
- **Tuning:** Learning rate of AdamW, SGD and Muon on a grid, then rho of the sharpness-aware methods. Selection by held-out validation loss on a short proxy run, seed 0.
- **Forgetting:** AdamW fine-tuning for 64 steps at 17 learning rates; loss on held-out FineWeb-Edu is compared at one common fine-tuning loss per dataset.

## Notes and limitations
- Three seeds only; one fine-tuning recipe; the fine-tuning sets are small (2.1M tokens).
- The `fine-tuning accuracy` figure re-runs the fine-tuning to measure accuracy; Pythia-70M re-runs are not bit-identical to the original sweep (final loss differs by up to 0.03; see `paper_ft_curves/data/reproduction_check.csv`).
