# INT4 Post-Training Quantization (PTQ) — Methodology

This benchmark measures how well each optimizer's trained weights survive weight-only
**INT4** post-training quantization, using **AWQ** (Activation-aware Weight Quantization).

## Method: AWQ

Reference: Lin et al., *"AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration"*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
Official implementation: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq).

`ptq/quant.py` is an independent re-implementation of the published algorithm (the official
`llm-awq` / `AutoAWQ` packages require a compiled CUDA extension that was not available in this
environment), following the same three steps as the reference `auto_scale.py` / `auto_clip.py`:

1. **Activation-aware scale search** (`_search_scale`): for each linear layer (or group of layers
   sharing a predecessor, e.g. fused QKV), compute per-input-channel mean activation magnitude from
   calibration data, then grid-search 20 candidates `s = x_mean^ratio` (`ratio ∈ {0, 0.05, ..., 0.95}`)
   for the scale that minimizes the layer's output MSE after INT4 fake-quantization. The scale is
   folded into the preceding LayerNorm/activation (mathematically exact, verified below) and the
   inverse folded into the weight.
2. **Weight clipping search** (`_search_clip`): per output-channel-group, search a symmetric clipping
   threshold (up to 50% shrink, 20-step grid) that minimizes quantized-output MSE against calibration
   activations.
3. **Quantization**: per-group (group size 128) asymmetric min-max quantization to 4-bit
   (`maxq = 15`), including zero in the range (same convention as the GPTQ/AWQ reference code).

Weights are **fake-quantized**: quantize → dequantize back to FP32 in place. This is the same
protocol the AWQ and GPTQ papers themselves use to report accuracy — it isolates the accuracy
effect of the INT4 grid from unrelated kernel/packing engineering. It does not produce a packed
INT4 checkpoint or measure inference speedup/memory savings; only accuracy/perplexity degradation
under 4-bit quantization is measured here.

### What is/isn't quantized
- **LLMs (GPT-2 / GPT-NeoX / Pythia)**: every linear layer inside transformer blocks (attention
  projections, MLP up/down) is quantized. Embeddings and the LM head are kept at full precision —
  standard GPTQ/AWQ practice (see `get_arch()` in `ptq/quant.py`).
- **CNN (AirbenchCNN)**: the three hidden 3×3 conv layers are quantized; the stem conv and the final
  linear classifier are kept at full precision (standard practice of protecting first/last layers).

### Correctness checks performed
- **Scale-transform identity check**: applying the AWQ scale transform with quantization disabled
  (16-bit passthrough) reproduces the FP model's output/perplexity *exactly* — confirms the
  scale-folding math (`_apply_scale`, `ScaledActivation`, `ScaledInputConv`) is output-preserving,
  as it must be by construction.
- **Expected method ordering**: on real trained checkpoints (CIFAR CNN, Pythia-70M, GPT-2), AWQ and
  GPTQ both improve over RTN, and results are stable across calibration seeds (see smoke test logs).

### Known limitation
This is a faithful re-implementation of the published algorithm, not literally the official
`AutoAWQ`/`llm-awq` codebase (which wasn't installable in this environment without a CUDA
extension build). No bit-exact cross-check against the official library's output was performed.
For a submission where independent library confirmation is required, install `autoawq` and compare
its INT4 accuracy on one checkpoint against `ptq_results/*.csv`.

## Data

FineWeb-Edu `sample-10BT` has 14 parquet shards. `data/loaders.py:fineweb_splits` assigns them disjoint
roles: shards `000`–`010` train the models, `011` is training-time validation (used for hyperparameter
selection), `012` is PTQ calibration and `013` is PTQ/final evaluation. No evaluation token is ever
trained on, calibrated on, or used for model selection.

- **Calibration** (AWQ input statistics): 128 random windows of 512 tokens from shard `012`, drawn with
  calibration seeds 0, 1, 2; 1024 random CIFAR-10 training images (test-time transform) for the CNN.
- **Evaluation**: 2^20 tokens of shard `013` (in-distribution) + WikiText-2 test (standard
  out-of-distribution PPL benchmark, `"\n\n"`-joined non-overlapping windows) for LLMs; the official
  CIFAR-10 test set (10,000 images) for the CNN. CIFAR hyperparameters are selected on a separate
  5k split of the training set, never on the test set.

## Settings run

- **Method**: `awq` (RTN/GPTQ code paths exist in `ptq/quant.py` and via `run_ptq.py --methods`).
- **Bit-width**: **INT4** (`wbits=4`), group size 128.
- **Calibration seeds**: 0, 1, 2 per checkpoint; tables average over calibration seeds within each
  training seed, then report mean ± std over training seeds (`tools/summarize.py`).

## Outputs

- `ptq_results/<config_id>.csv` — one row per (optimizer, training seed, method, calibration seed):
  FP baseline + INT4-AWQ accuracy/PPL, quantization wall-clock time, status.
- `ptq_results/<config_id>/<optimizer>_seed<seed>.json` — full record (config, args, git commit, torch
  version, quantized/total param counts, all rows) for reproducibility.

## Files

| File | Purpose |
|---|---|
| `ptq/quant.py` | RTN / GPTQ / AWQ quantization algorithms, architecture specs |
| `ptq/data.py` | Calibration + evaluation data loading (cached under `ptq_cache/`) |
| `run_ptq.py` | CLI: quantize one checkpoint, evaluate FP vs INT4, write CSV/JSON |
| `tools/experiments.py` | LR/rho sweeps, selection, final multi-seed queue, GPU workers |
| `tools/summarize.py` | Paper tables, mean ± std over training seeds |
