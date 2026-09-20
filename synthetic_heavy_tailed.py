"""
Heavy-Tailed Matrix Synthetic Experiment
Reference: https://arxiv.org/pdf/2508.04860

Objective (convex, single matrix parameter W in R^{k x d}):
    F(W) = (1/(n*k)) * || A W^T - B ||_{1,entrywise} + (mu/2) * ||W||_F^2

Deterministic subgradient:
    grad F(W) = (1/(n*k)) * sign(R)^T @ A + mu * W,   R = A W^T - B

Stochastic heavy-tailed oracle:
    G_t(W) = grad F(W) + Xi_t,   (Xi_t)_ij = s_ij * u_ij^{-1/alpha},  s_ij in {-1,+1}, u_ij ~ U(0,1)

Compares 7 optimizers: sgd, fsam, muon, muon_sam, atlas, atlas_raw, atlas_random
across a tail-index sweep alpha in {1.2, 1.6, 2.0}.
"""
import argparse
import csv
import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
import yaml

from muon import SingleDeviceMuon
from optimizers.atlas_baseline import AtlasOptimizer
from optimizers.atlas_raw_grad import AtlasOptimizerRaw
from optimizers.atlas_random import AtlasOptimizerRandom
from optimizers.fsam import FSAM
from optimizers.muon_sam import MuonSAM, newtonschulz5 as ns5_muonsam

ALGORITHMS = ["sgd", "fsam", "muon", "muon_sam", "atlas", "atlas_raw", "atlas_random"]
RHO_ALGORITHMS = {"fsam", "muon_sam", "atlas", "atlas_raw", "atlas_random"}

STYLE = {
    "sgd":          {"color": "#dc2626", "label": "SGD"},
    "fsam":         {"color": "#0891b2", "label": "F-SAM"},
    "muon":         {"color": "#059669", "label": "Muon"},
    "muon_sam":     {"color": "#7c3aed", "label": "Muon + SAM"},
    "atlas":        {"color": "#4f46e5", "label": "Ours (last ortho. momentum)"},
    "atlas_raw":    {"color": "#6366f1", "label": "Ours (gradient perturb.)"},
    "atlas_random": {"color": "#818cf8", "label": "Ours (random perturb.)"},
}


# ─────────────────────────────── Data & Objective ──────────────────────────────
def generate_data(n, d, k, seed, obs_noise_std=0.0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, generator=g) / (d ** 0.5)
    W_star = torch.randn(k, d, generator=g) / (d ** 0.5)
    B = A @ W_star.T
    if obs_noise_std > 0:
        B = B + obs_noise_std * torch.randn(n, k, generator=g)
    return A, B, W_star


def deterministic_loss(W, A, B, mu, n, k):
    R = A @ W.T - B
    return (R.abs().sum() / (n * k) + 0.5 * mu * (W ** 2).sum()).item()


def deterministic_grad(W, A, B, mu, n, k):
    R = A @ W.T - B
    return torch.sign(R).T @ A / (n * k) + mu * W


def pareto_noise(shape, alpha, generator):
    u = torch.rand(shape, generator=generator).clamp_min(1e-12)
    s = (torch.randint(0, 2, shape, generator=generator).float() * 2 - 1)
    return s * u.pow(-1.0 / alpha)


def make_closure(W, A, B, mu, n, k, alpha, generator):
    def closure():
        g = deterministic_grad(W.data, A, B, mu, n, k)
        noise = pareto_noise(g.shape, alpha, generator)
        W.grad = (g + noise).clone()
        return deterministic_loss(W.data, A, B, mu, n, k)
    return closure


# ─────────────────────────────── Optimizer Factory ──────────────────────────────
def build_optimizer(name, W, hp):
    lr = hp["lr"]
    if name == "sgd":
        return torch.optim.SGD([W], lr=lr, momentum=0.0, weight_decay=0.0)
    if name == "fsam":
        return FSAM([W], lr=lr, rho=hp["rho"], lam=hp.get("lam", 0.9), sigma=hp.get("sigma", 1.0))
    if name == "muon":
        return SingleDeviceMuon([W], lr=lr, weight_decay=0.0, momentum=hp.get("momentum", 0.95))
    if name == "muon_sam":
        base = SingleDeviceMuon([W], lr=lr, weight_decay=0.0, momentum=hp.get("momentum", 0.95))
        return MuonSAM(base, rho=hp["rho"], rho_vector=hp["rho"], ns_steps=5)
    if name in ("atlas", "atlas_raw", "atlas_random"):
        group = dict(params=[W], use_muon=True, lr=lr, weight_decay=0.0, rho=hp["rho"],
                     rho_vector=hp["rho"], momentum=hp.get("momentum", 0.9665),
                     nesterov=True, ns_steps=5, adam_lr=lr)
        cls = {"atlas": AtlasOptimizer, "atlas_raw": AtlasOptimizerRaw, "atlas_random": AtlasOptimizerRandom}[name]
        return cls([group])
    raise ValueError(f"Unknown algorithm: {name}")


# ─────────────────────────────── Sharpness Gap Diagnostic ──────────────────────────────
def sharpness_gap(name, optimizer, W, A, B, mu, n, k, diag_generator):
    """
    F(W_t + eps_t) - F(W_t) using each method's own (deterministic) perturbation direction.
    Uses a dedicated `diag_generator`, independent of the training-noise RNG, so that how
    often this diagnostic is logged never changes the actual noise sequence seen by training.
    """
    if name not in RHO_ALGORITHMS:
        return float("nan")
    state = optimizer.state[W]
    F0 = deterministic_loss(W.data, A, B, mu, n, k)

    if name == "fsam":
        if "m" not in state:
            return float("nan")
        rho = optimizer.param_groups[0]["rho"]
        g_t = deterministic_grad(W.data, A, B, mu, n, k)
        d_t = g_t - optimizer.param_groups[0]["sigma"] * state["m"]
        direction = d_t / (d_t.norm() + 1e-12)
    elif name == "muon_sam":
        rho = optimizer.param_groups[0]["rho"]
        g_t = deterministic_grad(W.data, A, B, mu, n, k)
        direction = ns5_muonsam(g_t.reshape(g_t.size(0), -1)).reshape(g_t.shape)
    elif name == "atlas":
        if "O_stale" not in state:
            return float("nan")
        rho = optimizer.param_groups[0]["rho"]
        direction = state["O_stale"]
    elif name == "atlas_raw":
        if "g_stale" not in state:
            return float("nan")
        rho = optimizer.param_groups[0]["rho"]
        d = state["g_stale"]
        direction = d / (d.norm() + 1e-12)
    elif name == "atlas_random":
        rho = optimizer.param_groups[0]["rho"]
        r = torch.randn(W.shape, generator=diag_generator)
        direction = r / (r.norm() + 1e-12)
    else:
        return float("nan")

    F1 = deterministic_loss(W.data + rho * direction, A, B, mu, n, k)
    return F1 - F0


# ─────────────────────────────── Single Run ──────────────────────────────
def run_single(name, alpha, lr, rho, seed, cfg):
    dcfg = cfg["data"]
    n, d, k, mu, R = dcfg["n"], dcfg["d"], dcfg["k"], dcfg["mu"], dcfg["R"]
    iterations = cfg["sweep"]["iterations"]
    log_every = cfg["sweep"]["log_every"]
    # Divergence is judged relative to the loss at initialization (F0), not a fixed absolute
    # number -- this objective's scale depends on the data/mu, so an absolute threshold either
    # never fires (too loose) or fires immediately (too tight).
    divergence_multiplier = float(cfg["sweep"].get("divergence_multiplier", 20.0))

    A, B, W_star = generate_data(n, d, k, seed=dcfg["seed"], obs_noise_std=dcfg.get("obs_noise_std", 0.0))
    F_star = deterministic_loss(W_star, A, B, mu, n, k)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    diag_generator = torch.Generator().manual_seed(seed + 1_000_000)  # independent of training noise
    W = torch.nn.Parameter(torch.zeros(k, d))
    F0 = deterministic_loss(W.data, A, B, mu, n, k)
    divergence_threshold = divergence_multiplier * F0

    hp = dict(cfg["algorithms"][name])
    hp["lr"] = lr
    if rho is not None:
        hp["rho"] = rho
    optimizer = build_optimizer(name, W, hp)

    rows = []
    diverged = False
    t_start = time.time()
    for t in range(1, iterations + 1):
        closure = make_closure(W, A, B, mu, n, k, alpha, generator)
        optimizer.step(closure)
        with torch.no_grad():
            W.data.clamp_(-R, R)

        if not torch.isfinite(W.data).all():
            diverged = True

        if t % log_every == 0 or t == iterations or diverged:
            F_t = deterministic_loss(W.data, A, B, mu, n, k) if not diverged else float("nan")
            if diverged or (F_t is not None and F_t > divergence_threshold):
                diverged = True
            gap = F_t - F_star if not diverged else float("nan")
            dist = (W.data - W_star).norm().item() if not diverged else float("nan")
            gradnorm = deterministic_grad(W.data, A, B, mu, n, k).norm().item() if not diverged else float("nan")
            sgap = sharpness_gap(name, optimizer, W, A, B, mu, n, k, diag_generator) if not diverged else float("nan")
            rows.append({
                "algorithm": name, "alpha": alpha, "lr": lr, "rho": rho if rho is not None else "",
                "seed": seed, "iter": t, "F": F_t, "gap": gap, "dist_to_Wstar": dist,
                "gradnorm": gradnorm, "sharpness_gap": sgap, "diverged": int(diverged),
                "wall_time_s": time.time() - t_start,
            })
        if diverged:
            break

    return rows


# ─────────────────────────────── Sweep Driver ──────────────────────────────
def hp_grid(name, acfg):
    lrs = acfg["lr_grid"]
    if name in RHO_ALGORITHMS:
        return list(itertools.product(lrs, acfg["rho_grid"]))
    return [(lr, None) for lr in lrs]


def run_sweep(cfg, algorithms, alphas, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "synthetic_heavy_tailed_raw.csv")
    fieldnames = ["algorithm", "alpha", "lr", "rho", "seed", "iter", "F", "gap",
                  "dist_to_Wstar", "gradnorm", "sharpness_gap", "diverged", "wall_time_s"]

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name in algorithms:
            acfg = cfg["algorithms"][name]
            grid = hp_grid(name, acfg)
            for alpha in alphas:
                for lr, rho in grid:
                    for seed in seeds:
                        print(f"[{name:<13}] alpha={alpha} lr={lr} rho={rho} seed={seed}")
                        rows = run_single(name, alpha, lr, rho, seed, cfg)
                        writer.writerows(rows)
                        f.flush()

    print(f"\nRaw per-iteration log saved to {raw_path}")
    return raw_path


# ─────────────────────────────── wandb: best-hyperparameter curves only ──────────────────────────────
# One wandb project per alpha, one run per algorithm (its best lr/rho, mean across seeds) --
# gives exactly one clean line per algorithm per chart, per the paper's 7 required metrics.
WANDB_METRICS = ["F", "gap", "dist_to_Wstar", "gradnorm", "diverged", "sharpness_gap", "wall_time_s"]


def push_best_to_wandb(df, best, wandb_project_template):
    for _, row in best.iterrows():
        algo, alpha = row["algorithm"], row["alpha"]
        sub = df[(df["algorithm"] == algo) & (df["alpha"] == alpha) & (df["hp_key"] == row["hp_key"])]
        if sub.empty:
            continue
        curve = sub.groupby("iter")[WANDB_METRICS].mean().reset_index()

        wandb.init(
            project=wandb_project_template.format(alpha=alpha),
            name=algo,
            reinit=True,
            config={"algorithm": algo, "alpha": alpha, "lr": row["lr"], "rho": row["rho"],
                    "n_seeds": int(row["n_seeds"])},
        )
        for _, r in curve.iterrows():
            wandb.log({m: r[m] for m in WANDB_METRICS}, step=int(r["iter"]))
        wandb.finish()
        print(f"  wandb: logged {algo} (alpha={alpha}, lr={row['lr']}, rho={row['rho']}) "
              f"to project {wandb_project_template.format(alpha=alpha)}")


# ─────────────────────────────── Aggregation & Plotting ──────────────────────────────
def aggregate_and_plot(raw_path, out_dir, wandb_project_template=None):
    import pandas as pd

    df = pd.read_csv(raw_path)
    df["hp_key"] = df["lr"].astype(str) + "|" + df["rho"].astype(str)
    final = df.sort_values("iter").groupby(["algorithm", "alpha", "lr", "rho", "seed"]).tail(1)

    summary = final.groupby(["algorithm", "alpha", "lr", "rho"]).agg(
        mean_final_F=("F", "mean"), std_final_F=("F", "std"),
        mean_gap=("gap", "mean"), mean_dist=("dist_to_Wstar", "mean"),
        mean_gradnorm=("gradnorm", "mean"), mean_sharpness_gap=("sharpness_gap", "mean"),
        divergence_rate=("diverged", "mean"), mean_wall_time_s=("wall_time_s", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    summary_path = os.path.join(out_dir, "synthetic_heavy_tailed_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Per-hyperparameter summary saved to {summary_path}")

    # Best hyperparameters per (algorithm, alpha): lowest mean final F among runs
    # that didn't mostly diverge.
    candidates = summary[summary["divergence_rate"] < 0.5]
    best_rows = []
    for (algo, alpha), grp in candidates.groupby(["algorithm", "alpha"]):
        best_rows.append(grp.loc[grp["mean_final_F"].idxmin()])
    if not best_rows:
        print("No non-diverged runs found; skipping best-hyperparameter selection and plots.")
        return
    best = pd.DataFrame(best_rows)
    best["hp_key"] = best["lr"].astype(str) + "|" + best["rho"].astype(str)
    best_path = os.path.join(out_dir, "synthetic_heavy_tailed_best.csv")
    best.drop(columns=["hp_key"]).to_csv(best_path, index=False)
    print(f"Best hyperparameters per (algorithm, alpha) saved to {best_path}")

    if wandb_project_template:
        push_best_to_wandb(df, best, wandb_project_template)

    # Convergence plots: F(W_t) vs iteration, mean +/- std across seeds, at best hyperparams.
    for alpha in sorted(best["alpha"].unique()):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        for _, row in best[best["alpha"] == alpha].iterrows():
            algo = row["algorithm"]
            mask = (df["algorithm"] == algo) & (df["alpha"] == alpha) & (df["hp_key"] == row["hp_key"])
            sub = df[mask]
            if sub.empty:
                continue
            stats = sub.groupby("iter")["F"].agg(["mean", "std"]).reset_index()
            style = STYLE.get(algo, {"color": "gray", "label": algo})
            ax.plot(stats["iter"], stats["mean"], color=style["color"], label=style["label"], linewidth=2)
            ax.fill_between(stats["iter"], stats["mean"] - stats["std"].fillna(0),
                             stats["mean"] + stats["std"].fillna(0), color=style["color"], alpha=0.15)

        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("F(W_t)")
        ax.set_title(f"Heavy-Tailed Synthetic: alpha = {alpha}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"synthetic_heavy_tailed_alpha{alpha}_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot: {plot_path}")


# ─────────────────────────────── CLI ──────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default="configs/synthetic_heavy_tailed.yaml")
    parser.add_argument("--algorithms", "-a", default="all",
                         help="Comma-separated subset of: " + ",".join(ALGORITHMS) + " (default: all)")
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha values, overrides config")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, overrides config")
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke test: 1 seed, 1 lr per algorithm, 100 iterations")
    parser.add_argument("--out-dir", default="logs/synthetic_heavy_tailed")
    parser.add_argument("--wandb", action="store_true",
                         help="After the local sweep, push each algorithm's best-hyperparameter curve "
                              "(mean across seeds) to wandb -- one project per alpha, one run per algorithm.")
    parser.add_argument("--wandb-project-template", default="synthetic-experiment-alpha{alpha}",
                         help="wandb project name per alpha; '{alpha}' is substituted, e.g. "
                              "synthetic-experiment-alpha1.2")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algorithms = ALGORITHMS if args.algorithms == "all" else [a.strip() for a in args.algorithms.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else cfg["sweep"]["alphas"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["sweep"]["seeds"]

    if args.quick:
        cfg["sweep"]["iterations"] = min(cfg["sweep"]["iterations"], 100)
        cfg["sweep"]["log_every"] = 10
        seeds = seeds[:1]
        for name in algorithms:
            cfg["algorithms"][name]["lr_grid"] = cfg["algorithms"][name]["lr_grid"][:1]
            if "rho_grid" in cfg["algorithms"][name]:
                cfg["algorithms"][name]["rho_grid"] = cfg["algorithms"][name]["rho_grid"][:1]

    wandb_project_template = args.wandb_project_template if args.wandb else None

    print(f"\n>>> Synthetic Heavy-Tailed Sweep: algorithms={algorithms} alphas={alphas} seeds={seeds} "
          f"wandb={'on, best-only (' + args.wandb_project_template + ')' if args.wandb else 'off'} <<<\n")
    raw_path = run_sweep(cfg, algorithms, alphas, seeds, args.out_dir)
    aggregate_and_plot(raw_path, args.out_dir, wandb_project_template=wandb_project_template)


if __name__ == "__main__":
    main()
