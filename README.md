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

| Optimizer | LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.0018 | betas (0.9, 0.95) | 0.01 | — | — |
| SGD (Nesterov) | 0.0001 | 0.9 | 0.01 | — | — |
| Muon (Nesterov/Polyak) | 0.0325 | 0.9665 | 0.01 | — | — |
| Atlas | 0.015 | 0.9665 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.95 | 0.01 | 0.0015 | 0.01 |

**CIFAR-10 CNN** (`cifar10_cnn.yaml`)

| Optimizer | LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.001 | — | 0.001 | — | — |
| SGD (Nesterov) | 0.0001 | 0.9 | 0.001 | — | — |
| Muon (Nesterov/Polyak) | 0.03 | 0.9665 | 0.001 | — | — |
| Atlas | 0.02 | 0.9665 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.95 | 0.01 | 0.0015 | 0.01 |

**Pythia-70M Chinchilla** (`pythia70m_pretrain_chinchilla.yaml`)

| Optimizer | LR | Momentum | Weight Decay | rho | rho_vector |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW | 0.0018 | betas (0.9, 0.95) | 0.01 | — | — |
| SGD (Euclidean, Schatten-2) | 0.015 | 0.95 | 0.1 | — | — |
| Muon (Schatten-∞) | 0.015 | 0.95 | 0.1 | — | — |
| Atlas | 0.015 | 0.95 | 0.0001 | 0.0015 | 0.01 |
| Muon-SAM | 0.015 | 0.95 | 0.01 | 0.0015 | 0.01 |
