# Atlas Optimizer Experiments

Comparing **Atlas / Muon / AdamW / SGD** (and SAM variants) on nanoGPT, CIFAR-10 CNN, and Pythia-70M.

---

## 1. How to Run

Main entry point: **`main_experiment.py`**

```bash
python main_experiment.py --config <config.yaml> [--optimizer <opt_list>] [--epochs N] [--all]
```

| Flag | Alias | Description | Default |
| :--- | :--- | :--- | :--- |
| `--config` | `-c` | Path to a single config YAML in `configs/` | `configs/pythia70m_schatten_sweep.yaml` |
| `--optimizer` | `-o` | Optimizer(s) to run, comma-separated, or `all` | `all` |
| `--epochs` | `-e` | Override epochs from the config | value in config |
| `--all` / `--all-configs` | — | Run every `configs/*.yaml` | off |

Valid optimizer names: `atlas`, `atlas_raw`, `atlas_random`, `muon`, `muon_nesterov`, `muon_polyak`, `muon_sam`, `adam`, `sgd`.
`--optimizer all` runs `atlas, atlas_raw, atlas_random, muon, adam, sgd` (6 optimizers).

### Examples

Actual runs use `nohup` + background (`&`) so training survives a closed terminal, with
`CUDA_VISIBLE_DEVICES` picking the GPU and stdout/wandb console redirected to a log file:

```bash
# All 6 optimizers, one config
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/nanogpt_fineweb.yaml -o all -e 15 > run2_nanogpt.log 2>&1 &

# Single optimizer
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/cifar10_cnn.yaml -o atlas -e 7 > run1_cifar10_atlas.log 2>&1 &

# A few specific optimizers
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/pythia70m_pretrain_chinchilla.yaml -o atlas,muon,adam -e 14 > run3_pythia_subset.log 2>&1 &

# Muon-SAM only
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/pythia70m_pretrain_chinchilla.yaml -o muon_sam -e 14 > run3_muonsam_pythia.log 2>&1 &

# Run every config in configs/ with all optimizers
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py --all -o all > run_all.log 2>&1 &
```

### Reference: actual commands used for the 3 main runs

```bash
# run1 - CIFAR-10 CNN, all 6 optimizers, 7 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/cifar10_cnn.yaml -o all -e 7 > run1_cifar10.log 2>&1 &

# run1 - CIFAR-10 CNN, Muon-SAM only, 7 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/cifar10_cnn.yaml -o muon_sam -e 7 > run1_muonsam_cifar10.log 2>&1 &

# run2 - nanoGPT-FineWeb, all 6 optimizers, 15 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/nanogpt_fineweb.yaml -o all -e 15 > run2_nanogpt.log 2>&1 &

# run2 - nanoGPT-FineWeb, Muon-SAM only, 15 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/nanogpt_fineweb.yaml -o muon_sam -e 15 > run2_muonsam_nanogpt.log 2>&1 &

# run3 - Pythia-70M Chinchilla, all 6 optimizers, 14 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/pythia70m_pretrain_chinchilla.yaml -o all -e 14 > run3_pythia.log 2>&1 &

# run3 - Pythia-70M Chinchilla, Muon-SAM only, 14 epochs
CUDA_VISIBLE_DEVICES=1 nohup python main_experiment.py -c configs/pythia70m_pretrain_chinchilla.yaml -o muon_sam -e 14 > run3_muonsam_pythia.log 2>&1 &
```

Watch a running job with `tail -f run3_pythia.log`.

### Legacy driver

`exp1_atlas_baseline.py` supports the same flags (`--config`, `--optimizer`, `--epochs`, `--all`) but only
`atlas, muon, adam, sgd` (no `atlas_raw`/`atlas_random`/`muon_sam`). Used by `run_experiments.sh` to batch
through the `paper1_*` / `paper2_*` configs:

```bash
bash run_experiments.sh
```

---

## 2. Config Files (`configs/`)

| Config | Experiment |
| :--- | :--- |
| `nanogpt_fineweb.yaml` | nanoGPT-135M optimizer comparison (main) |
| `cifar10_cnn.yaml` | CIFAR-10 CNN optimizer comparison (main) |
| `pythia70m_pretrain_chinchilla.yaml` | Pythia-70M Chinchilla-optimal pretraining (main) |
| `nanogpt_ablations.yaml` | nanoGPT rank/steps/sketch/solver ablations |
| `cifar10_batch_scaling.yaml` | CIFAR-10 batch size + ablation sweeps |
| `pythia70m_schatten_sweep.yaml` | Pythia-70M Schatten-r + LR grid sweep |
| `pythia70m_noise_analysis.yaml` | Heavy-tailed gradient noise measurement |

---

## 3. Outputs

| Path | Contents |
| :--- | :--- |
| `logs/<config_id>_logs.csv` | Per-epoch metrics for a run (loss, acc, val loss/acc/perplexity, time) |
| `logs/<config_id>_plot.png` | Loss + metric comparison plot (epoch x-axis) |
| `logs/all_experiments_summary.csv` | One row per (config, optimizer) final result |
| `checkpoints/<optimizer>_epoch<N>.pt` | Model checkpoint after each epoch |
| `wandb/` | Weights & Biases run logs (offline unless `wandb login`) |
| `<runN>_*.log` (repo root) | Raw stdout/wandb console logs from manual/nohup runs |

---

## 4. Epoch → Step Conversion (for paper plots)

All experiments use a hardcoded `steps_per_epoch = 100` (`main_experiment.py:118`), i.e. **step = epoch × 100**.

`logs_to_steps.py` converts either the CSV logs in `logs/` or raw `.log` run files into step-indexed
CSVs + plots, saved to `final_logs/`.

```bash
# From raw .log run files (parses "[OPT] Epoch X/Y | ..." lines directly)
python logs_to_steps.py run1_cifar10.log run2_nanogpt.log run3_pythia.log

# From existing logs/*_logs.csv files (adds a 'step' column + step-axis plots)
python logs_to_steps.py
```

Output: `final_logs/<name>_logs.csv` (with `step` column) and `final_logs/<name>_steps_plot.png`.

---

## 5. Current Experiment Setup

| | nanoGPT-FineWeb | CIFAR-10 CNN | Pythia-70M Chinchilla |
| :--- | :--- | :--- | :--- |
| Config | `nanogpt_fineweb.yaml` | `cifar10_cnn.yaml` | `pythia70m_pretrain_chinchilla.yaml` |
| Model | GPT-2 arch, 12L/12H/768d | Airbench CNN | `EleutherAI/pythia-70m` |
| Params | ~135M | small (custom CNN) | ~70M |
| Dataset | `fineweb-edu` (sample-10BT) | `cifar10` | `fineweb-edu` (sample-10BT) |
| Seq len / image size | 2048 tokens | 32×32 | 2048 tokens |
| Batch size (micro × grad_acc) | 16 × 32 = 512 seqs/step | 500 × 1 | 16 × 32 = 512 seqs/step |
| Tokens/step | ~1.05M | — | ~1.05M |
| Total steps / epochs | 1500 steps / 15 epochs | 720 steps / 7 epochs | 1400 steps / 14 epochs |
| Total tokens | ~1.57B | — | 1.4B (Chinchilla-optimal, 20 tok/param) |
| LR schedule | WSD: 5% warmup, 20% cooldown | WSD: 5% warmup, 20% cooldown | WSD: 5% warmup, 20% cooldown |
| Parameter routing | hidden matrices → Muon/Atlas, embed+head → AdamW | conv weights → Muon/Atlas, norm/bias → SGD | hidden matrices → Muon/Atlas, embed+head → AdamW |

### Optimizer Hyperparameters

**nanoGPT-FineWeb** (`nanogpt_fineweb.yaml`)

| Optimizer | LR | Adam LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.0018 | — | betas (0.9, 0.95) | 0.01 | — | — |
| SGD (Nesterov) | 0.01 | — | 0.9 | 0.01 | — | — |
| Muon (Nesterov/Polyak) | 0.0325 | 0.003 | 0.9665 | 0.01 | — | — |
| Atlas | 0.015 | 0.003 | 0.9665 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.003 | 0.95 | 0.01 | 0.0015 | 0.01 |

**CIFAR-10 CNN** (`cifar10_cnn.yaml`)

| Optimizer | LR | Adam LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.001 | — | — | 0.001 | — | — |
| SGD (Nesterov) | 0.0001 | — | 0.9 | 0.001 | — | — |
| Muon (Nesterov/Polyak) | 0.03 | 0.001 | 0.9665 | 0.001 | — | — |
| Atlas | 0.02 | 0.001 | 0.9665 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.001 | 0.95 | 0.01 | 0.0015 | 0.01 |

**Pythia-70M Chinchilla** (`pythia70m_pretrain_chinchilla.yaml`)

| Optimizer | LR | Adam LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.0018 | — | betas (0.9, 0.95) | 0.01 | — | — |
| SGD (Euclidean, Schatten-2) | 0.015 | — | 0.95 | 0.1 | — | — |
| Muon (Schatten-∞) | 0.015 | 0.003 | 0.95 | 0.1 | — | — |
| Atlas | 0.015 | 0.003 | 0.95 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.003 | 0.95 | 0.01 | 0.0015 | 0.01 |

---

### Final Loss and Accuracy Results

#### Experiment 1: CIFAR-10 Image Classification
| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Atlas Random** | 0.5771 | 80.54% | 0.7683 | 74.91% |
| **Atlas Raw** | 0.5910 | 80.06% | 0.7336 | 75.47% |
| **Atlas** | 0.6427 | 78.52% | 0.7338 | 75.39% |
| **AdamW** | 1.0577 | 63.27% | 1.0548 | 62.87% |
| **Muon** | 0.5922 | 80.16% | 0.6242 | 78.50% |
| **Muon-SAM** | 0.9166 | 34.67% | 0.8812 | 69.37% |
| **SGD** | 2.2712 | 16.02% | 2.2474 | 17.56% |

#### Experiment 2: nanoGPT Continual Pre-Training
| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Atlas Random** | 3.9176 | 32.40% | 3.7416 | 34.15% |
| **Atlas Raw** | 3.9505 | 32.02% | 3.7754 | 33.81% |
| **Atlas** | 4.2237 | 29.02% | 4.0431 | 30.89% |
| **AdamW** | 4.5826 | 27.24% | 4.4446 | 28.50% |
| **Muon** | 3.9330 | 32.28% | 3.7586 | 34.04% |
| **Muon-SAM** | 4.7394 | 24.12% | 4.5769 | 25.40% |
| **SGD** | 8.2731 | 9.10% | 8.2703 | 9.26% |

#### Experiment 3: Pythia-70M Continual Pre-Training
| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Atlas Random** | 4.0635 | 31.47% | 4.0107 | 31.91% |
| **Atlas Raw** | 4.0775 | 31.29% | 4.0240 | 31.73% |
| **Atlas** | 4.2337 | 29.34% | 4.1612 | 30.06% |
| **AdamW** | 4.9518 | 23.79% | 4.9156 | 23.91% |
| **Muon** | 4.0690 | 31.48% | 4.0169 | 31.87% |
| **Muon-SAM** | 4.8280 | 23.48% | 4.7622 | 23.97% |
| **SGD** | 8.1014 | 6.38% | 8.0589 | 6.87% |

### Language Modeling Validation Perplexity Results

#### Experiment 2: nanoGPT Continual Pre-Training
| Optimizer | Final Validation Perplexity |
| :--- | :--- |
| **Atlas Random** | 42.17 |
| **Atlas Raw** | 43.61 |
| **Atlas** | 57.00 |
| **AdamW** | 85.17 |
| **Muon** | 42.89 |
| **Muon-SAM** | 97.21 |
| **SGD** | 3906.12 |

#### Experiment 3: Pythia-70M Continual Pre-Training
| Optimizer | Final Validation Perplexity |
| :--- | :--- |
| **Atlas Random** | 55.19 |
| **Atlas Raw** | 55.92 |
| **Atlas** | 64.15 |
| **AdamW** | 136.40 |
| **Muon** | 55.53 |
| **Muon-SAM** | 117.00 |
| **SGD** | 3161.81 |

*(Note: CIFAR-10 is an image classification task evaluated on Cross-Entropy Loss and Accuracy, and therefore does not have a Perplexity metric).*

## 6. Heavy-Tailed Matrix Synthetic Experiment

Reference: [arXiv:2508.04860](https://arxiv.org/pdf/2508.04860). Convex objective on a single
matrix parameter `W in R^{k x d}`:

```
F(W) = (1/(n*k)) * ||A W^T - B||_{1,entrywise} + (mu/2) * ||W||_F^2
G_t(W) = grad F(W) + Xi_t     # Xi_t entrywise two-sided Pareto(alpha) noise
```

Script: **`synthetic_heavy_tailed.py`**, config: `configs/synthetic_heavy_tailed.yaml`.

Compares 7 optimizers, each run on CPU (matrices are tiny — no GPU needed):

| Name | Maps to |
| :--- | :--- |
| `sgd` | Plain SGD baseline |
| `fsam` | Friendly-SAM (`optimizers/fsam.py`, new) |
| `muon` | `SingleDeviceMuon` |
| `muon_sam` | Existing `MuonSAM` wrapping `SingleDeviceMuon` |
| `atlas` | Ours — last orthogonalized momentum perturbation (main method) |
| `atlas_raw` | Ours — current-gradient perturbation (ablation) |
| `atlas_random` | Ours — random perturbation (ablation) |

Sweeps tail index `alpha in {1.2, 1.6, 2.0}`, an LR grid (and rho grid for SAM/Atlas methods) per
algorithm, and 5 seeds — all defined in the config. `--quick` runs a 1-seed/1-lr/100-iteration
smoke test first.

```bash
# Smoke test everything runs (~1 min, single seed/lr per algorithm)
python synthetic_heavy_tailed.py --quick

# Full sweep, all 7 algorithms x 3 alphas x full LR/rho grids x 5 seeds (long-running -> nohup it)
nohup python synthetic_heavy_tailed.py > synthetic_heavy_tailed.log 2>&1 &

# Just a subset of algorithms / alphas
python synthetic_heavy_tailed.py --algorithms sgd,fsam,muon,muon_sam --alphas 1.2,2.0

# Custom seeds or output directory
python synthetic_heavy_tailed.py --seeds 0,1,2 --out-dir logs/synthetic_heavy_tailed_v2
```

Outputs (default `logs/synthetic_heavy_tailed/`):

| File | Contents |
| :--- | :--- |
| `synthetic_heavy_tailed_raw.csv` | Every logged iteration for every (algorithm, alpha, lr, rho, seed) run: `F`, `gap` (`F(W_t)-F(W*)`), `dist_to_Wstar`, `gradnorm`, `sharpness_gap`, `diverged`, `wall_time_s` |
| `synthetic_heavy_tailed_summary.csv` | Per-hyperparameter-combo means/stds across seeds, plus divergence rate |
| `synthetic_heavy_tailed_best.csv` | Best (lr, rho) per (algorithm, alpha) by lowest mean final `F(W_t)` |
| `synthetic_heavy_tailed_alpha{1.2,1.6,2.0}_plot.png` | `F(W_t)` vs iteration (log-scale, mean ± std band) for all 7 algorithms at their best hyperparameters |

A run is flagged `diverged` if `W` becomes non-finite, or `F(W_t)` exceeds `divergence_multiplier x F(W_0)`
(20x the initial loss by default) — relative to the run's own starting loss, not a fixed absolute number.

### Logging to wandb

Pass `--wandb` to also push each algorithm's **best-hyperparameter curve** (mean across seeds) to wandb,
*after* the local sweep + best-hyperparameter selection above — not the raw grid. One wandb **project**
per alpha, one **run** per algorithm inside it, so every chart shows exactly one clean line per algorithm:

```bash
python synthetic_heavy_tailed.py --algorithms atlas,atlas_random,atlas_raw,fsam,muon_sam --wandb
```

This creates `synthetic-experiment-alpha1.2`, `-alpha1.6`, `-alpha2.0` (override the name with
`--wandb-project-template`), each with one run per algorithm named after it, logging the 7 metrics above
at each `log_every` step. Use `--algorithms` to control which optimizers appear in each project.

Tune `data`, `mu`, `R` (the `||W||_inf <= R` projection constraint), `iterations`, `divergence_multiplier`,
and each algorithm's `lr_grid`/`rho_grid` directly in `configs/synthetic_heavy_tailed.yaml`.

### Synthetic Experiment Results (Best Hyperparameters)

| Algorithm | Alpha | LR | Rho | Mean (W_t)$ | Mean Gap | Mean Dist to ^*$ | Mean Grad Norm | Mean Sharpness Gap | Diverged | Mean Wall Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Atlas | 1.2 | 0.005 | 0.01 | 0.10584 | 0.10275 | 11.590 | 0.01026 | -0.00008 | 0.0 | 2.74 |
| Atlas Random | 1.2 | 0.005 | 0.01 | 0.10584 | 0.10275 | 11.590 | 0.01026 | +0.00000 | 0.0 | 2.47 |
| Atlas Raw | 1.2 | 0.005 | 0.01 | 0.10584 | 0.10275 | 11.590 | 0.01026 | -0.00000 | 0.0 | 2.82 |
| Muon-SAM | 1.2 | 0.005 | 0.05 | 0.09769 | 0.09460 | 10.756 | 0.01015 | +0.00316 | 0.0 | 8.08 |
| Atlas | 1.6 | 0.005 | 0.01 | 0.10182 | 0.09873 | 11.188 | 0.01020 | -0.00006 | 0.0 | 2.95 |
| Atlas Random | 1.6 | 0.005 | 0.02 | 0.10182 | 0.09873 | 11.188 | 0.01020 | +0.00000 | 0.0 | 3.20 |
| Atlas Raw | 1.6 | 0.005 | 0.01 | 0.10182 | 0.09873 | 11.188 | 0.01020 | -0.00000 | 0.0 | 2.59 |
| FSAM | 1.6 | 0.003 | 0.02 | 1.29055 | 1.28747 | 96.301 | 0.01872 | +0.00000 | 0.0 | 3.04 |
| Muon-SAM | 1.6 | 0.005 | 0.05 | 0.09333 | 0.09024 | 10.317 | 0.01009 | +0.00313 | 0.0 | 7.22 |
| Atlas | 2.0 | 0.005 | 0.01 | 0.09804 | 0.09496 | 10.804 | 0.01015 | -0.00005 | 0.0 | 3.37 |
| Atlas Random | 2.0 | 0.005 | 0.01 | 0.09804 | 0.09496 | 10.804 | 0.01015 | +0.00000 | 0.0 | 2.45 |
| Atlas Raw | 2.0 | 0.005 | 0.02 | 0.09804 | 0.09496 | 10.804 | 0.01015 | -0.00000 | 0.0 | 2.88 |
| FSAM | 2.0 | 0.003 | 0.05 | 0.37283 | 0.36975 | 36.080 | 0.01285 | +0.00000 | 0.0 | 2.91 |
| Muon-SAM | 2.0 | 0.005 | 0.05 | 0.08964 | 0.08655 | 9.941 | 0.01004 | +0.00312 | 0.0 | 7.76 |

#### Analysis
- **Muon-SAM Convergence**: Across all noise regimes ($lpha=1.2, 1.6, 2.0$), Muon-SAM achieves the lowest mean final objective value ($) and strictly minimizes the distance to ^*$ better than all other optimizers. This verifies that applying orthogonalized perturbations to the matrix parameters is highly effective for heavy-tailed robustness.
- **Atlas Variants**: The tlas, tlas_raw, and tlas_random variants perform almost identically across all alphas. They converge reliably but hit a slightly higher loss floor than Muon-SAM (e.g., .1058$ vs .0977$ at $lpha=1.2$).
- **Computational Cost**: While Muon-SAM dominates in convergence, it is significantly more computationally expensive, requiring roughly 2.5x to 3x the wall time per iteration compared to Atlas ($\sim 7-8s$ vs $\sim 2-3s$). This tracks with the heavy cost of running a double forward pass with two full Newton-Schulz matrix orthogonalizations per step.
- **FSAM Failure**: FSAM fails to match the performance of the spectral-based geometry, returning massive distances to ^*$ (.3$ at $lpha=1.6$), highlighting the necessity of matrix-specific perturbations for matrix parameters.
- **Divergence Stability**: Despite the extremely heavy tails ($lpha=1.2$), the divergence rate remains .0$ for all recorded optimizers at their best hyperparameters, showing that the base scaling mechanisms successfully prevent numeric explosion.
