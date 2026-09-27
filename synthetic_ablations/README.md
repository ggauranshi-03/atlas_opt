# Synthetic heavy-tailed ablations for SAM × Muon

A controlled testbed for comparing every SAM / Muon / SGD / Adam variant used in this project on
**matrix-valued parameters** under **heavy-tailed gradient noise**, for 1, 2, 3, … matrices.

It replaces `mnist-sam-muon-experiments/mnist/heavy_tailed_matrix.py`,
`heavy_tailed_two_matrix.py` and `atlas_opt/synthetic_heavy_tailed.py` with one objective family
whose optimum is known **exactly**, a noise model that is **structured like real gradient noise**,
and a protocol (tuning on disjoint seeds, many seeds, robust statistics, compute-matched views,
radius-matched views) that makes comparisons between methods meaningful.

---

## 1. Quick start

Run from the `atlas_opt/` directory.

```bash
# 1. tune the learning rate of each base optimizer (sgd, adam, muon, nsgdm, clip-sgd), per alpha
python -m synthetic_ablations.tune  -c configs/two_matrix.yaml --workers 14

# 2. full sweep: every variant x alpha x rho, 32 seeds each (resumable; re-run to continue)
python -m synthetic_ablations.sweep -c configs/two_matrix.yaml --workers 14

# 3. figures are produced by the sweep; to regenerate them
python -m synthetic_ablations.plots synthetic_ablations/results/two_matrix

# tests (includes equivalence checks against atlas_opt/optimizers)
python -m pytest synthetic_ablations/tests -q
```

Useful overrides: `--algorithms muon lazy-spectral-sam-muon full-spectral-sam-muon`,
`--alphas 1.1 2 inf`, `--rhos 0.05 0.2`, `--seeds 8 --steps 200` (smoke test),
`--rho-units matched_frobenius`, `--out <dir>`, `--fresh`.

Measured cost: one job (one variant, one alpha, one rho, 32 seeds, 1000 steps) takes 25–40 s on a
single CPU thread. A full `two_matrix` (or `three_matrix`) sweep is 954 jobs, about 35 min with
14 workers; `single_matrix` is 660 jobs; tuning is about 5 min.

---

## 2. Folder layout

| File | Contents |
|---|---|
| `objective.py` | problem construction, objective `F`, exact optimum `F*`, excess risk, Hessian `λ_max`, balancedness |
| `noise.py` | two-sided Pareto sampler and the two stochastic gradient oracles (`label`, `additive`) |
| `linalg.py` | batched Newton–Schulz, exact polar factor, spectral-ball projection, effective rank |
| `algorithms.py` | **the registry of all optimizer variants** and the unified step engine |
| `runner.py` | runs one (variant, α, ρ) for a batch of seeds; logging, divergence handling, aggregation |
| `experiment.py` | config loading, job lists, parallel execution, resumable CSV output |
| `tune.py` / `sweep.py` / `plots.py` | the three command-line entry points |
| `configs/` | presets: `single_matrix`, `two_matrix`, `three_matrix`, `two_matrix_generalization`, `single_matrix_additive` |
| `tests/` | exactness of `F*`, noise distribution and unbiasedness, orthogonalization, **equivalence with `atlas_opt/optimizers`** |
| `results/<name>/` | `finals.csv`, `curves.csv`, `summary_*.csv`, `tuned_lrs.json`, `figures/` (git-ignored) |

All seeds of a configuration are simulated **simultaneously** as a leading batch dimension, so 32
seeds cost little more than one.

---

## 3. Optimizers considered

Every method is a point in one design space:

> **SAM step** = (perturbation **source** `d`) × (perturbation **geometry**) × (**outer** optimizer)

**Sources:**
- `grad`: fresh clean gradient `g_t`.
- `friendly`: `g_t − σ m_t`, where `m_t = λ m_{t−1} + (1−λ) g_t`.
- `random`: a Gaussian matrix.
- `stale_update`: the previous Muon update `O_{t−1}`.
- `stale_grad`: the previous adversarial gradient `g̃_{t−1}`.
- `stale_momentum`: the Muon momentum `M_{t−1}`.
- `stale_friendly`: `g̃_{t−1} − σ m_{t−1}`.
- `stale_momentum_friendly`: `g̃_{t−1} − σ(1−β)v_{t−1}`.

**Geometries:**
- `frobenius`: `ρ d/‖d‖_F`, either per layer or with one global norm over all layers.
- `spectral`: `ρ·Ortho(d)`, an operator-norm ball of radius ρ.
- `raw`: `ρ d`, used by Atlas, whose `d` is already orthogonal.

The perturbed gradient `g̃_t = ∇f(W_t + ε_t; ξ_t)` uses the **same** stochastic sample `ξ_t` as the
clean gradient (the analogue of reusing the minibatch).

"Source" says where each method comes from: `both` = in `atlas_opt/optimizers` **and** the mnist
heavy-tailed scripts; `atlas_opt` / `mnist` = only one of them; `added` = new here (see notes).

| Name (`--algorithms`) | Outer | Perturbation | Oracle calls / step | `atlas_opt/optimizers` | mnist scripts | Source |
|---|---|---|---|---|---|---|
| `sgd` | SGD | – | 1 | `torch.optim.SGD` | `sgd` | both |
| `adam` | AdamW | – | 1 | `torch.optim.AdamW` | – | requested |
| `muon` | Muon | – | 1 | `muon.py` | `muon` | both |
| `nsgdm` | Normalized SGD-M | – | 1 | – | – | added |
| `clip-sgd` | Clip-SGD | – | 1 | – | – | added (paper baseline) |
| `sam-sgd` | SGD | grad · Frobenius | 2 | – | `sam-sgd` | mnist |
| `spectral-sam-sgd` | SGD | grad · spectral | 2 | – | – | added |
| `friendly-sam-sgd` | SGD | friendly · Frobenius | 2 | `fsam.py` | `friendly-sam-sgd` | both |
| `spectral-friendly-sam-sgd` | SGD | friendly · spectral | 2 | `fsam_ortho.py` | – | atlas_opt |
| `sam-muon` | Muon | grad · Frobenius | 2 | `muon_sam_frob.py` (per-layer) | – | atlas_opt |
| `full-spectral-sam-muon` | Muon | grad · spectral | 2 | `muon_sam.py` | `full-spectral-sam-muon` | both |
| `global-friendly-sam-muon` | Muon | friendly · Frobenius | 2 | `fsam_muon.py` (global) | `global-friendly-sam-muon` | both |
| `spectral-friendly-sam-muon` | Muon | friendly · spectral | 2 | `fsam_ortho_muon.py` | `spectral-friendly-sam-muon` | both |
| `random-sam-muon` | Muon | random · Frobenius | 1 | `atlas_random.py` (per-layer) | `random-sam-muon` | both |
| `random-spectral-sam-muon` | Muon | random · spectral | 1 | – | `random-spectral-sam-muon` | mnist |
| `lazy-spectral-sam-muon` | Muon | stale_update · raw (**Atlas**) | 1 (2 at t=0) | `atlas_baseline.py` | `lazy-spectral-sam-muon` | both |
| `stale-grad-sam-muon` | Muon | stale_grad · Frobenius | 1 (2 at t=0) | `atlas_raw_grad.py` (per-layer) | – | atlas_opt |
| `stale-momentum-sam-muon` | Muon | stale_momentum · Frobenius | 1 (2 at t=0) | `muon_sam_stale.py` (per-layer) | – | atlas_opt |
| `stale-friendly-spectral-sam-muon` | Muon | stale_friendly · spectral | 1 (2 at t=0) | – | `stale-spectral-friendly-sam-muon` | mnist |
| `stale-momentum-friendly-spectral-sam-muon` | Muon | stale_momentum_friendly · spectral | 1 (2 at t=0) | – | `stale-muon-momentum-spectral-sam-muon` (sign-fixed) | mnist |

Every Frobenius method is run in both `global` and `per-layer` modes when there are ≥ 2 matrices;
with one matrix the modes coincide and only one is run. So there are 27 variants for ≥ 2 matrices
and 20 for one.

**Outer updates** (all per matrix, `lr_t` from the schedule):
- **Muon:** `v ← βv + g`, `n = g + βv` (Nesterov), `W ← W − lr·√max(1, rows/cols)·NS5(n)`. This is identical to `atlas_baseline.py` and equivalent to the EMA form of `muon.py`, because the two differ by a constant factor that NS5 removes.
- **NSGD-M:** the same as Muon but with `NS5(n)` replaced by `√min(rows, cols)·n/‖n‖_F`. It has the same momentum, the same Nesterov step and the same update size; **only the geometry differs** (Frobenius instead of spectral normalization). It is the control that separates "Muon helps because it normalizes" from "Muon helps because of the spectral geometry".
- **Clip-SGD:** `g ← g·min(1, τ/‖g‖_F)` with a constant `τ = clip_factor·‖∇F(W_0)‖_F`, the robust baseline of the heavy-tailed literature.
- **SGD and AdamW:** standard; no momentum for SGD by default (paper setting).

**Discrepancies found in the two sources, and how they are resolved here:**
1. *Friendly-EMA ordering.* mnist `friendly-sam-sgd` builds `d_t` from `m_{t−1}` and then updates
   the EMA, while its own `global-friendly-sam-muon`, `atlas_opt/fsam*.py` and the F-SAM-Muon spec
   update `m_t` first. Everything here uses `m_t` first.
2. *Sign bug in mnist `stale-muon-momentum-spectral-sam-muon`.* It uses `d = g − σ v` with the
   **sum-form** Muon momentum `v ≈ g/(1−β) = 20 g`, so `d ≈ −19 g` and the "adversarial"
   perturbation points **downhill**. Here the momentum is rescaled to its EMA form `(1−β)v`, which
   turns the method into F-SAM whose EMA is Muon's own momentum (λ = β). No extra state is needed.
3. *Radius units.* A spectral perturbation `ρ·Ortho(d)` has Frobenius norm ≈ `ρ√rank` (≈ 5–8× larger
   for these shapes) than a Frobenius perturbation with the same nominal `ρ`. Plots against nominal
   ρ therefore compare perturbations of very different size; this is the likely reason the Atlas
   curve blows up with ρ in the earlier figures. See `rho_units` and the `gap_vs_perturbation_*`
   figures (§6).
4. *Random-direction bootstrap.* `atlas_random.py` evaluates a clean gradient at step 0 that it does
   not need; here random methods use one oracle call from the start, as in the mnist script.

`tests/test_equivalence_atlas_opt.py` checks the engine against the real classes:
- `AtlasOptimizer`, `AtlasOptimizerRaw`, `FSAM` and `FSAMOrtho`: whole trajectories must match (relative error < 2e-4 over 6 steps).
- `MuonSAM`, `MuonSAMFrob`, `FSAMMuon`, `FSAMOrthoMuon` and `MuonSAMStale`: the perturbations must match. These wrap the bf16 library Muon, so the perturbation is what distinguishes them.

---

## 4. Heavy-tailed noise

This follows Fatkhullin, Hübler & Lan (2025), *Can SGD Handle Heavy-Tailed Noise?*, Section 6. There, an ℓ1
objective is optimized with the oracle `∇f(x, ξ) = Aᵀ sign(Ax − b) + μx + ξ`, where the coordinates
`ξ_i = s_i u_i^{−1/α}` have `s_i ∼ Unif{±1}` and `u_i ∼ Unif(0, 1)`. For this two-sided Pareto law,
`P(|ξ_i| > t) = t^{−α}` for `t ≥ 1` and `E|ξ_i|^p = α/(α − p)`:

- moments of order `p < α` are finite and those of order `p ≥ α` are infinite, so the oracle
  satisfies the *p-th bounded (central) moment* assumption for every `p < α`;
- `α ∈ (1, 2]` means a finite mean but **infinite variance** (the regime of the paper);
- `α ≤ 1` means no mean, so the oracle is not unbiased. This is only allowed as a labelled stress
  test via `noise.allow_undefined_mean`;
- `α > 2` gives finite variance, and `α = inf` is used here for a Gaussian control.

Two changes make the α-sweep interpretable:

* **Median normalization.** `ζ = s·u^{−1/α} / 2^{1/α}` has median `|ζ| = 1` for every α, and the
  Gaussian control is normalized the same way. In the raw form, the typical noise size itself
  changes with α (the median is `2^{1/α}` and the support is `|ξ| ≥ 1`). After normalizing, **α
  changes only the tail**.
* **Heavy tails do not average out.** With a minibatch of `b` samples, a sum of `b` i.i.d.
  α-tailed terms scales like `b^{1/α}`, so the averaged noise shrinks like `b^{1/α − 1}` rather than
  `b^{−1/2}`. For example, at `α = 1.1` a batch of 64 reduces the noise only about 1.5×. This is
  the physical reason small α is hard, and it happens automatically in the `label` oracle below.

---

## 5. The objective: reduced-rank deep linear regression

### 5.1 Why the previous objectives are not appropriate for Muon

*`F(W) = (1/nk)‖A Wᵀ − B‖_{1,entrywise} + (μ/2)‖W‖²_F`, with i.i.d. entrywise noise
(`synthetic_heavy_tailed.py`, `heavy_tailed_matrix.py`):*

* **It is not a matrix problem.** The entrywise ℓ1 loss and the ridge term separate over the rows
  of `W`, so the problem is `k` independent vector regressions. Muon's premise is that the loss is
  sensitive to the *singular structure* of `W` (steepest descent in the spectral norm); an
  entrywise/row-separable loss has no such structure, so there is nothing for orthogonalization to
  exploit and nothing for spectral vs Frobenius perturbations to distinguish.
* **The signal-to-noise ratio is essentially zero.** The clean subgradient
  `sign(R)ᵀA/(nk)` has entries of size about `1/(k√(nd))` (≈ 4e-5 for the defaults n=1024, d=128,
  k=64), while the Pareto noise has entries of size ≥ 1. Every method is then a random walk clamped to a box, and the comparison
  measures clamping behavior.
* **The noise is isotropic and independent of the parameters.** Real gradient noise in a linear
  layer is a sum of per-example outer products `δ_i x_iᵀ`: it is low-rank, structured, and passes
  through the other layers via the chain rule.
* **Sharpness is degenerate.** The ℓ1 loss is piecewise linear, so its Hessian is 0 almost
  everywhere and "sharpness" (what SAM targets) is not a meaningful, varying quantity.

*The two-matrix MSE (`heavy_tailed_two_matrix.py`)* has the right model class (a product of
matrices), but:
- The data and target are isotropic, so the gradient spectrum is flat and Muon's advantage
  (equalizing an *ill-conditioned* gradient spectrum) cannot show.
- The labels are noiseless and `n ≫ d`, so "test MSE" ≈ train MSE, and a generalization benefit of
  SAM cannot appear.
- The optimum is known only in the realizable case.
- The noise scale is `1e-4` in absolute units, so the SNR is uncontrolled.

### 5.2 Definition

**Data.**
- The inputs are `x ∼ N(0, Σ)` in `ℝ^d`, where `Σ = Q diag(λ) Qᵀ` has a random orthogonal `Q`. Its eigenvalues decay geometrically with condition number `cond_x` and are rescaled so that `tr Σ = d`.
- The teacher is `P* = U diag(s) Vᵀ ∈ ℝ^{k×d}` of rank `teacher_rank`. Its singular values decay geometrically with condition number `cond_teacher`. `V` is random by default; `teacher_alignment: low_variance` puts the signal in the small-eigenvalue directions of Σ (the hard, ill-conditioned case).
- `P*` is scaled so that `E‖P*x‖²/k = tr(P*ΣP*ᵀ)/k = 1`, i.e. each output has unit variance.
- The labels are `y = P* x + σ_y ε` with `ε ∼ N(0, I_k)`, and there are `n` training samples `X ∈ ℝ^{n×d}`, `Y ∈ ℝ^{n×k}`.

**Model** (depth `L` = number of matrices):

```
P(W) = W_L W_{L−1} ⋯ W_1,
W_1 ∈ ℝ^{h×d},   W_ℓ ∈ ℝ^{h×h} (1 < ℓ < L),   W_L ∈ ℝ^{k×h}     (L = 1: W_1 ∈ ℝ^{k×d})
```

**Objective:**

```
F(W) = 1/(2nk) · ‖ X P(W)ᵀ − Y ‖²_F        (+ (μ/2)‖W_1‖²_F for L = 1)
```

This is multi-output least squares for `L = 1` and a deep linear network for `L ≥ 2`. `F(0) ≈ ½`
by construction.

**Gradient.** Write `G_P = ∂F/∂P = (1/nk)(P XᵀX − YᵀX)`. Then
`∂F/∂W_ℓ = (W_L ⋯ W_{ℓ+1})ᵀ G_P (W_{ℓ−1} ⋯ W_1)ᵀ`. `G_P = (1/k)(P − P_ols) Σ̂` is right-multiplied
by the empirical covariance, so **the gradient spectrum inherits the conditioning of Σ and of
`P*`**. This is exactly the regime where orthogonalized (spectral) updates differ from Euclidean
ones, and it is controlled by `cond_x` and `cond_teacher`.

### 5.3 Exact optimum `F*` (no numerical solver)

`F` depends on `W` only through `P`, and a depth-`L` product can represent every `P` with
`rank P ≤ r`, where `r = min(k, d)` for `L = 1` and `r = min(k, d, h)` for `L ≥ 2`. So
`F* = min_{rank P ≤ r} (1/2nk)‖Y − X Pᵀ‖²_F`. This is **reduced-rank regression** and has a closed
form.

Let `Ŷ = Π_X Y` be the least-squares fit, with `Π_X = X X⁺`. Every `X Pᵀ` lies in `col(X)`, so
Pythagoras gives `‖Y − XPᵀ‖² = ‖Y − Ŷ‖² + ‖Ŷ − XPᵀ‖²`. The matrices `XPᵀ` with `rank P ≤ r` are
exactly the rank-≤`r` matrices in `col(X)`. By Eckart–Young, the best one is the SVD truncation
`Ŷ_r`, which stays in `col(X)`. Therefore

```
F* = [ ‖Y − Ŷ‖²_F + Σ_{i>r} σ_i(Ŷ)² ] / (2nk),        P_opt = (X⁺ Ŷ_r)ᵀ.
```

For `L = 1` with `μ > 0` it is ridge regression: `(XᵀX + nkμ I) P_optᵀ = XᵀY`.

- In the realizable default (`σ_y = 0`, `h ≥ teacher_rank`), `F* = 0` exactly (≈ 1e-30 numerically).
- With label noise or a binding bottleneck, `F* > 0` and is still exact.
- The tests confirm that noiseless descent on the two-matrix model converges to this `F*`
  (relative error < 1e-4).

### 5.4 Why this is the right testbed for SAM × Muon

* **It is genuinely matrix-dependent.**
  - The loss couples all entries through `P = W_L ⋯ W_1`.
  - The curvature is anisotropic, set by `Σ` and by the spectrum of `P*`.
  - The per-example gradient noise is low-rank.

  These are the three properties that motivate Muon and spectral SAM.
* **Convex for one matrix, benign non-convex for more.**
  - For `L = 1`, `F` is a strongly convex quadratic (`n > d`) with a unique minimizer, which gives
    a clean optimization-under-noise test like the paper's.
  - For `L ≥ 2`, `F` has no spurious local minima: every local minimum is global (Kawaguchi 2016;
    Laurent & von Brecht 2018, since `h ≥ min(d, k)` in the defaults), but it has saddles such as
    `W = 0`.
* **For `L ≥ 2` the global minima have different sharpness.** Rescaling `W_{ℓ+1} → W_{ℓ+1}A⁻¹`,
  `W_ℓ → A W_ℓ` leaves `F` unchanged, so the minima form a manifold. On it, the Hessian's `λ_max`
  depends on how *balanced* the factors are, and the flattest minima are balanced (Mulayoff &
  Michaeli 2020). Two consequences:
  - Under gradient flow, `W_{ℓ+1}ᵀW_{ℓ+1} − W_ℓW_ℓᵀ` is **conserved** (Arora et al. 2018; Du et
    al. 2018). Any change in it therefore measures the optimizer's *implicit bias*.
  - SAM is known to drive factorized models toward balanced solutions (e.g. Li, Zhang & He,
    NeurIPS 2024).

  So the testbed measures not only *how fast* each method reaches `F*` but *which* minimum it
  selects, via the `lambda_max` and `balancedness` metrics.
* **Generalization is measured exactly.**
  - Since `x ∼ N(0, Σ)` with `Σ` known, the population excess risk
    `E(P) = E_x‖(P − P*)x‖²/(2k) = tr((P − P*)Σ(P − P*)ᵀ)/(2k)` is computed in closed form, with no
    test-set sampling error.
  - With `label_noise_std > 0` and small `n` (`two_matrix_generalization.yaml`), minimizing `F`
    overfits. Methods can then be compared on what SAM is actually for: the population risk of the
    solution found.

### 5.5 Stochastic oracles

Both oracles are exact gradients of a stochastic function `f(W; ξ)` with `E_ξ ∇f = ∇F`, and the two
SAM passes of a step share `ξ`.

* **`label` (default): backpropagated per-sample noise.** Given a minibatch `B` of size `b` and
  `E ∈ ℝ^{b×k}` with `E_ij = scale·ζ_ij`,

  ```
  f(W; ξ) = F_B(W) − 1/(bk) · ⟨E, X_B P(W)ᵀ⟩,      ∇_P f = ∇_P F_B − 1/(bk) · Σ_{i∈B} ε_i x_iᵀ.
  ```

  Properties of this noise:
  - It is a sum of `b` **rank-one** heavy-tailed terms, structurally like real per-example
    gradient noise in a linear layer.
  - It reaches every layer by the chain rule, so depth changes its structure consistently.
  - It is unbiased for `α > 1` and has finite `p`-th moments exactly for `p < α` (the `x_i` are
    Gaussian).
  - It combines with ordinary minibatch-sampling noise; `batch_size: null` removes the sampling
    noise.
  - `scale` is in label units (the signal per output is 1).
* **`additive`: the paper's model.** `f(W; ξ) = F_B(W) + Σ_ℓ ⟨Ξ_ℓ, W_ℓ⟩` with i.i.d. entries
  `(Ξ_ℓ)_ij = scale·τ_ℓ·ζ_ij`, where `τ_ℓ` is the RMS of the clean full-batch gradient of layer
  `ℓ` at initialization. So `scale` is a noise-to-signal ratio. With `batch_size: null` this is
  exactly the paper's oracle in matrix form (`single_matrix_additive.yaml`).

### 5.6 Constraint and initialization

- Like the paper's `ℓ∞` ball, the iterates are projected after every step onto
  `{‖W_ℓ‖₂ ≤ R}`, the spectral ball (the norm natural to Muon), by clipping singular values.
- `R = radius_factor · max(‖P_opt‖₂^{1/L}, ‖W_0‖₂)`. This guarantees that the balanced
  factorization of `P_opt` is feasible, so `F*` is also the constrained optimum.
- `constraint: entrywise | none` are available.
- The initialization `W_ℓ ∼ N(0, init_scale²/fan_in)` is the same for all seeds; seeds differ only
  in the noise stream, as in the paper.

---

## 6. Protocol and metrics

1. **Learning rates** (`tune.py`). Each base optimizer (`sgd`, `adam`, `muon`, `nsgdm`,
   `clip_sgd`) is tuned per α on a log grid, using 16 seeds disjoint from the evaluation seeds
   (`tuning.seed_offset`). The score is the median final gap, restricted to learning rates with a
   divergence rate ≤ 25%. **SAM variants inherit the learning rate of their outer optimizer**, and
   only ρ is swept: first tune the base optimizer, then fix its learning rate and tune ρ (the
   protocol of Zhong, Milsom & Murray 2026). A warning is printed if the best learning rate lies on
   the edge of the grid.
2. **Common random numbers.** For a given (α, seed), every variant sees the identical noise stream.
   Single-pass and two-pass methods consume one sample `ξ_t` per step, so differences between
   variants are not due to different noise draws.
3. **Robust statistics.**
   - Outcomes are themselves heavy-tailed, so the headline numbers are the **median** and IQR over
     32 seeds (means are reported too).
   - A diverged seed is recorded with gap `+∞`, which keeps the medians conservative (no survivor
     bias), and it contributes to `divergence_rate`.
   - Divergence means a non-finite iterate or `F > divergence_factor · F(W_0)`. The seed is frozen
     at its last finite iterate.
4. **Compute fairness.** Oracle calls are logged. Fresh SAM uses 2 calls per step, while Atlas and
   the other stale or random methods use 1. See `gap_vs_oracle_calls_*`.
5. **Radius fairness.**
   - `rho_units: nominal` uses ρ as each implementation defines it. `matched_frobenius` rescales
     every method's perturbation to total Frobenius norm exactly ρ, so only the *direction*
     differs.
   - The measured `‖ε‖_F` is always logged, and `gap_vs_perturbation_*` plots against it.
   - Run both settings before drawing conclusions about which direction is best.

**Logged metrics** (median / q25 / q75 / mean per logged step in `curves.csv`, per seed in
`finals.csv`):

| Metric | Meaning |
|---|---|
| `gap` | `F(W_t) − F*`, the primary metric (paper) |
| `excess_risk` | exact population excess risk of `P(W_t)` |
| `dist_to_opt` | `‖P(W_t) − P_opt‖_F` (invariant to the factorization) |
| `grad_norm` | clean full-gradient norm over all layers |
| `perturbation_frob`, `perturbation_op` | total Frobenius norm and max operator norm of `ε_t` |
| `sharpness_gap` | `F(W_t + ε_t) − F(W_t)`, the sharpness each method actually probes |
| `lambda_max` | top Hessian eigenvalue (power iteration on Hessian-vector products; every `sharpness_every` steps) |
| `balancedness` | `Σ_ℓ ‖W_{ℓ+1}ᵀW_{ℓ+1} − W_ℓW_ℓᵀ‖_F` (L ≥ 2) |
| `effective_rank` | entropy effective rank of `P` |
| `update_frob`, `oracle_calls`, `alive_fraction` | diagnostics |

**Figures** (`results/<name>/figures/`, each for the groups `muon` = Muon/NSGD-M outer, `sgd` =
SGD/Clip-SGD outer, and `all`):
- `gap_vs_rho_*`, `excess_risk_vs_rho_*`, `lambda_max_vs_rho_*` and `balancedness_vs_rho_*`: one
  panel per α. Baselines are drawn as dashed horizontal lines, the solid lines are medians, and the
  bands are IQRs. These are the successors of the attached plots.
- `gap_vs_perturbation_*`: the same, against the measured `‖ε‖_F`.
- `gap_vs_step_best_rho_*` and `gap_vs_oracle_calls_best_rho_*`: convergence curves at each
  variant's best ρ.
- `robustness_vs_alpha_*`: gap at the best ρ as a function of the tail index.

The tables are `summary_best_rho.csv` (per variant and α at the best ρ) and `summary_all.csv`.

---

## 7. Configuration reference

| Key | Default | Meaning |
|---|---|---|
| `problem.depth` | 1 / 2 / 3 | number of matrices `L` |
| `problem.d`, `k`, `hidden` | 64, 32, 32 | input dimension, output dimension, hidden width `h` |
| `problem.n_train` | 512 (128 in generalization) | number of training samples |
| `problem.teacher_rank` | 8 | rank of `P*` |
| `problem.cond_x`, `cond_teacher` | 100, 10 | condition numbers of `Σ` and of `P*` |
| `problem.teacher_alignment` | random | `random`, `high_variance` or `low_variance` |
| `problem.label_noise_std` | 0 (0.5) | Gaussian observation noise in the data |
| `problem.ridge` | 0 | `μ` (L = 1 only) |
| `problem.constraint`, `radius_factor` | spectral, 3 | projection set and its radius factor |
| `problem.init_scale` | 1.0 | initialization scale |
| `noise.model` | label | `label` or `additive` |
| `noise.scale` | 1.0 (0.1 additive) | median noise magnitude |
| `noise.batch_size` | 64 (null additive) | minibatch size; null = full batch |
| `noise.alphas` | 1.1 1.3 1.6 2 3 inf | tail indices; `inf` = Gaussian |
| `run.steps`, `seeds`, `log_every` | 1000, 32, 10 | horizon, seeds per configuration, logging interval |
| `run.schedule` | cosine | `constant`, `cosine` (to `min_lr_fraction`), or `power` (`(1+t)^−p`, `p` may be `"1/alpha"`) |
| `run.sharpness_every` | 250 | Hessian `λ_max` interval (0 = off) |
| `sam.rhos` | 0.01 … 1.0 | radii swept |
| `sam.rho_units` | nominal | `nominal` or `matched_frobenius` |
| `sam.frobenius_modes` | global, per-layer | Frobenius radius modes (L ≥ 2) |
| `sam.ortho` | ns5 | orthogonalization for spectral perturbations: `ns5` or `svd` (exact) |
| `sam.friendly_lambda`, `friendly_sigma` | 0.9, 1.0 | F-SAM `λ` and `σ` |
| `optim.muon_momentum` | 0.95 | `β` (Muon and NSGD-M) |
| `optim.sgd_momentum` | 0 | heavy-ball momentum for SGD (paper: 0) |
| `optim.adam_betas`, `adam_eps` | (0.9, 0.999), 1e-8 | AdamW |
| `optim.clip_factor` | 1.0 | Clip-SGD threshold = factor · ‖∇F(W_0)‖_F |
| `optim.weight_decay` | 0 | decoupled weight decay, all methods |
| `optim.lr.*` | – | fallback learning rates when there is no `tuned_lrs.json` |
| `tuning.lr_grid.*`, `seeds`, `seed_offset` | – | tuning grids (calibrated so optima are interior) and disjoint tuning seeds |

---

## 8. Calibration results (preliminary)

These come from calibrating the learning-rate grids: `tune.py` with 16 tuning seeds, 1000 steps,
the best learning rate for each method, and no SAM. They are **not** the final sweep, but they show
that the testbed separates methods in the intended way.

Median final gap `F − F*`:

| Preset | α | SGD | AdamW | Muon | NSGD-M | Clip-SGD |
|---|---|---|---|---|---|---|
| single_matrix (label noise) | 1.1 | 6.96 | 0.41 | 0.28 | 0.83 | 0.17 |
| | ∞ | 3.0e-3 | 3.3e-3 | 3.3e-3 | 3.2e-3 | 3.0e-3 |
| two_matrix (label noise) | 1.1 | 3.82 | 0.57 | 0.22 | 0.73 | 0.10 |
| | ∞ | 1.5e-3 | 1.6e-3 | 1.4e-3 | 1.7e-3 | 1.5e-3 |
| single_matrix_additive (paper oracle) | 1.1 | 1.30 | 0.029 | 0.058 | 0.21 | 0.038 |
| | ∞ | 5.7e-4 | 7.0e-4 | 8.1e-4 | 6.8e-4 | 5.7e-4 |

Early observations:
- **Light tails:** all methods are within about 20% of each other under label noise (about 40% under
  the additive oracle).
- **Heavy tails:** at `α = 1.1`, SGD collapses (gaps 17–40× larger than Muon or Clip-SGD).
- **Spectral geometry adds robustness beyond normalization:** Muon beats its Frobenius twin NSGD-M
  by about 3× under structured noise.
- **The noise model changes the ranking:** Adam wins under the paper's entrywise noise but loses to
  Muon under the structured, rank-one noise.

### 8.1 L1 Anisotropic Heavy-Tailed Results

The `two_matrix` preset was recently evaluated under a custom **L1 Entrywise Loss** objective using a heavy-tailed **Kronecker-structured Anisotropic Additive Noise** model (`cond_noise = 10.0`). The test was run for 500 steps across 8 seeds. 

Median final gap `F − F*` across optimizers:

| Optimizer | `α = 1.1` (Infinite Variance) | `α = ∞` (Gaussian Control) |
|---|---|---|
| SGD | 1.2358 | 0.0124 |
| SAM-SGD (global) | 1.2327 | 0.0132 |
| SAM-SGD (per-layer) | 1.2288 | 0.0135 |
| Muon | 0.5458 | 0.0621 |
| Spectral-Friendly SAM-Muon | **0.5428** | 0.0630 |
| Global-Friendly SAM-Muon (per-layer) | 0.5453 | 0.0624 |
| Random SAM-Muon (global) | 0.5458 | 0.0622 |
| Random-Spectral SAM-Muon | 0.5458 | 0.0625 |
| Full-Spectral SAM-Muon | 0.5460 | 0.0640 |
| Lazy-Spectral SAM-Muon | 0.5475 | 0.0677 |

![Muon Optimizers - Gap vs Perturbation](results/two_matrix/figures/gap_vs_perturbation_muon.png)

**Takeaways:**
- **SGD Collapse:** Under the extreme anisotropic noise (`α = 1.1`), SGD collapses entirely with a massive gap of 1.23+. 
- **Muon Dominance:** The base Muon optimizer consistently converges ~2.2x better than SGD under extreme heavy-tailed noise.
- **SAM Refining:** `Spectral-Friendly SAM-Muon` achieves the absolute lowest gap (0.5428), showing that spectral scouting is effective even under a highly warped gradient landscape.
- **Gaussian Control:** Under clean Gaussian noise (`α = ∞`), SGD achieves a tighter gap (0.0124), confirming that Muon's advantage here is purely driven by its robustness to the heavy-tailed anisotropy.

---

## 9. References

- Fatkhullin, Hübler & Lan (2025). *Can SGD Handle Heavy-Tailed Noise?* arXiv:2508.04860. The
  noise model and protocol are from Section 6.
- Zhong, Milsom & Murray (2026). *Sharpness-Aware Minimization and Muon: Robustness under the
  Spectral Norm.* arXiv:2607.26001. Source of SpecSAM, of the inner × outer design, and of the
  tuning protocol.
- Li et al. (2024). *Friendly Sharpness-Aware Minimization.* CVPR. Source of F-SAM.
- Jordan et al. (2024). *Muon.* Bernstein & Newhouse (2025). *Modular Duality in Deep Learning.*
- Cutkosky & Mehta (2021). Normalized SGD with momentum under heavy tails.
- Kawaguchi (2016); Laurent & von Brecht (2018). No spurious minima in deep linear networks.
  Arora et al. (2018); Du et al. (2018). Conservation of balancedness. Mulayoff & Michaeli (2020).
  Flat minima of linear networks are balanced.
- Izenman (1975). Reduced-rank regression.
