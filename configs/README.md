# Paper Experiment Configurations

This directory contains the exact configuration files for the experiments performed in:
- **Paper 1:** *"Muon with Nesterov Momentum: Heavy-Tailed Noise and (Randomized) Inexact Polar Decomposition"* (Choudhury et al., 2026, arXiv:2605.06884)
- **Paper 2:** *"Free Heavy-Tailed Lunch for Muon: A Theoretical Justification of Empirical Success"* (Hübler et al., 2026, arXiv:2606.14560)

---

## Paper 1 Configurations (`paper1_exp*`)

| Config File | Model & Dataset | Section & Topic |
| :--- | :--- | :--- |
| [`paper1_exp1_nanogpt_fineweb.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper1_exp1_nanogpt_fineweb.yaml) | **nanoGPT-135M** (`gpt2`) on `fineweb-edu` | **Section 4.1, Table 2 & Fig. 1a:** Pretraining optimizer comparison (Full Muon Nesterov/Polyak, Randomized Muon Gaussian/Kaczmarz, AdamW, SGD). |
| [`paper1_exp2_nanogpt_ablations.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper1_exp2_nanogpt_ablations.yaml) | **nanoGPT-135M** on `fineweb-edu` | **Section 4.1, Figs. 1b, 2a, 2b, 3a, 3b:** Ablation sweeps across rank $s \in [100, 400]$, steps $q \in [1, 9]$, sketch type (Gaussian vs Kaczmarz), and polynomial solvers. |
| [`paper1_exp3_cifar10_cnn.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper1_exp3_cifar10_cnn.yaml) | **Airbench CNN** on `cifar10` | **Section 4.2, Table 2 & Fig. 1c:** Vision CNN optimizer comparison; 4D Conv filters trained with Muon/Atlas, normalization with SGD-Nesterov. |
| [`paper1_exp4_cifar10_batch_scaling.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper1_exp4_cifar10_batch_scaling.yaml) | **Airbench CNN** on `cifar10` | **Section 4.2, Figs. 1d, 2c, 2d, 4a, 4b:** Batch size scaling ($B \in [500, 4000]$), rank sweep ($s \in [16, 128]$), and polynomial step ablations. |

---

## Paper 2 Configurations (`paper2_exp*`)

| Config File | Model & Dataset | Section & Topic |
| :--- | :--- | :--- |
| [`paper2_exp1_pythia70m_schatten_sweep.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper2_exp1_pythia70m_schatten_sweep.yaml) | **Pythia-70M** on `fineweb-edu` | **Section 4, Table 1 & Fig. 4:** Schatten-$r$ sweep ($r \in [1, 8/7, 4/3, 1.5, 2, 3, 4, 8, \infty]$) and 7-point learning rate grid with WSD schedule. |
| [`paper2_exp2_pythia70m_noise_analysis.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper2_exp2_pythia70m_noise_analysis.yaml) | **Pythia-70M** on `fineweb-edu` | **Section 4, Figs. 1, 2, 3:** Heavy-tailed noise measurement ($\sigma_{S_1, p} / \sigma_{F, p}$ for $p=1.5$) across weight families at init and final checkpoint. |
| [`paper2_exp3_pythia70m_pretrain_chinchilla.yaml`](file:///mnt/disk1/slakshna/atlas_opt/configs/paper2_exp3_pythia70m_pretrain_chinchilla.yaml) | **Pythia-70M** on `fineweb-edu` | **Section 4 & Appendix F:** Full 1.4B token Chinchilla pretraining benchmark comparing Muon/Atlas (spectral $r=\infty$), Euclidean SGD ($r=2$), and AdamW. |

---

## Execution Guide

Pass any config to the training scripts using `--config`:

```bash
# Paper 1 - nanoGPT optimizer comparison
python exp1_atlas_baseline.py --config configs/paper1_exp1_nanogpt_fineweb.yaml

# Paper 1 - CIFAR-10 CNN benchmark
python exp1_atlas_baseline.py --config configs/paper1_exp3_cifar10_cnn.yaml

# Paper 2 - Pythia-70M Schatten-r sweep
python exp1_atlas_baseline.py --config configs/paper2_exp1_pythia70m_schatten_sweep.yaml

# Paper 2 - Chinchilla 1.4B token pretraining
python exp1_atlas_baseline.py --config configs/paper2_exp3_pythia70m_pretrain_chinchilla.yaml
```
