| Optimizer | nanoGPT L_before | nanoGPT L_after | Pythia-70M L_before | Pythia-70M L_after |
|---|---|---|---|---|
| AdamW | 3.142 ± 0.000 | **3.304 ± 0.019** | 3.528 ± 0.001 | **3.811 ± 0.011** |
| SGD | 3.185 ± 0.000† | 3.310 ± 0.001 | 3.631 ± 0.000† | 3.783 ± 0.009 |
| Muon | 3.144 ± 0.000 | 3.409 ± 0.020 | 3.515 ± 0.000 | 4.111 ± 0.013 |
| SAM | 3.185 ± 0.000† | 3.310 ± 0.000 | 3.633 ± 0.000† | 3.791 ± 0.001 |
| FSAM | 3.185 ± 0.000† | 3.310 ± 0.000 | 3.633 ± 0.000† | 3.785 ± 0.009 |
| SpecSAM-Muon | 3.147 ± 0.000 | 3.382 ± 0.009 | 3.516 ± 0.001 | 4.107 ± 0.006 |
| FP-SOMA | 3.146 ± 0.000 | 3.367 ± 0.022 | 3.516 ± 0.001 | 4.107 ± 0.004 |
| SOMA | 3.144 ± 0.001 | 3.397 ± 0.023 | 3.517 ± 0.000 | 4.107 ± 0.003 |
| RandSAM-Muon | 3.144 ± 0.001 | 3.419 ± 0.004 | 3.514 ± 0.000 | 4.104 ± 0.008 |
| SOMA-PreNS5 | 3.144 ± 0.001 | 3.388 ± 0.020 | 3.516 ± 0.000 | 4.111 ± 0.003 |
| OP-SOMA-PostNS5 | 3.144 ± 0.001 | 3.400 ± 0.021 | 3.515 ± 0.000 | 4.106 ± 0.007 |

L = held-out FineWeb-Edu loss (lower is better): L_before after pretraining, L_after after fine-tuning (average over 4 datasets, at a common fine-tuning loss per dataset), mean ± sd over 3 seeds. Forgetting ΔPT = L_after − L_before. Bold = lowest L_after among models with a comparable starting point. † = L_before clearly worse than AdamW's (perplexity >3% higher), so these rows are not comparable.
