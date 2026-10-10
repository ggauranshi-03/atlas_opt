"""cos(perturbation, grad/momentum) deliverable, parameterized by central-tendency statistic.

Supports --stat {median,mean,mode}, reading a measure_alignment finals.csv and writing:
  cos_pert_grad_by_rho{_STAT}{_isotropic}.csv / cos_pert_momentum_by_rho{_STAT}{_isotropic}.csv
  cos_pert_grad_vs_rho{_STAT}{_isotropic}.png / cos_pert_momentum_vs_rho{_STAT}{_isotropic}.png
  vector_angle_grad{_STAT}{_isotropic}.png / vector_angle_momentum{_STAT}{_isotropic}.png

`median` is left unsuffixed (matches the files already published and embedded in the README) --
only pass --stat mean / --stat mode to produce the new, explicitly-suffixed variants. Never
re-run --stat median against the already-published filenames from this script unless you intend
to replace them; this script does not check for an existing file before writing.

Mode is the half-sample mode (Bickel & Fruehwirth 2006): with only 8 seeds, a density/KDE-based
mode would be bandwidth-dependent and arbitrary; the half-sample mode is deterministic, needs no
tuning parameter, and is the standard small-n robust mode estimator.
"""
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT_NAMES = {
    "stale-momentum-friendly-spectral-sam-muon": "SOMA",
    "spectral-friendly-sam-muon": "FP-SOMA",
    "full-spectral-sam-muon": "SpecSAM-Muon",
    "random-sam-muon": "RandSAM-Muon",
    "friendly-sam-sgd": "FSAM",
    "sam-sgd": "SAM",
    "op-soma-prens5": "OP-SOMA PreNS5",
    "op-soma-postns5": "OP-SOMA PostNS5",
    "sam-adam": "SAM+AdamW",
    "friendly-sam-adam": "FSAM+AdamW",
    "lazy-spectral-sam-muon": "Lazy-SOMA",
    "msam": "MSAM",
    "msoma": "MSOMA",
}

PALETTE = {
    "FP-SOMA": "#C74A4A",
    "FSAM": "#4AC7AA",
    "FSAM+AdamW": "#D08D4C",
    "Lazy-SOMA": "#4A70C7",
    "MSAM": "#B2A740",
    "MSOMA": "#D04AB9",
    "OP-SOMA PostNS5": "#4AAB64",
    "OP-SOMA PreNS5": "#A54568",
    "RandSAM-Muon": "#4AA5C7",
    "SAM": "#C74A83",
    "SAM+AdamW": "#714CD0",
    "SOMA": "#AD713C",
    "SpecSAM-Muon": "#6471A8",
}
ORDER = sorted(PALETTE)


def half_sample_mode(values):
    """Bickel & Fruehwirth (2006) half-sample mode: recursively keep the densest half."""
    values = sorted(values)
    n = len(values)
    if n <= 2:
        return sum(values) / n
    half = math.ceil(n / 2)
    best_start, best_range = 0, float("inf")
    for i in range(n - half + 1):
        rng = values[i + half - 1] - values[i]
        if rng < best_range:
            best_range = rng
            best_start = i
    return half_sample_mode(values[best_start:best_start + half])


STAT_FUNCS = {
    "median": lambda v: float(np.median(v)),
    "mean": lambda v: float(np.mean(v)),
    "mode": lambda v: half_sample_mode(list(v)),
}


def build_csv(finals, value_col, stat):
    stat_fn = STAT_FUNCS[stat]
    finals = finals[finals["algorithm"].isin(SHORT_NAMES)].copy()
    # Algorithms with both frobenius_modes (global, per-layer) would otherwise be double-counted
    # (16 "seeds" instead of 8) when grouped by (algorithm, alpha, rho) alone -- keep only the
    # global mode (and the empty-string mode used by algorithms with no split), one line per
    # algorithm, matching the convention already used for the published median CSVs.
    finals = finals[finals["frobenius_mode"] != "per-layer"]
    finals["short"] = finals["algorithm"].map(SHORT_NAMES)
    rows = []
    for (short, alpha, rho), g in finals.groupby(["short", "alpha", "rho"]):
        vals = g[value_col].dropna()
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        center = stat_fn(vals.to_numpy())
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        center_clamped = min(1.0, max(-1.0, center))
        rows.append({
            "algorithm": short, "alpha": alpha, "rho": rho,
            f"cos_theta_{stat}": center,
            "cos_theta_q25": q25, "cos_theta_q75": q75,
            "theta_degrees": math.degrees(math.acos(center_clamped)),
            "n_seeds": int(vals.shape[0]),
        })
    out = pd.DataFrame(rows)
    out["algorithm"] = pd.Categorical(out["algorithm"], categories=ORDER, ordered=True)
    return out.sort_values(["algorithm", "alpha", "rho"]).reset_index(drop=True)


def plot_line(table, stat, title, ylabel, path):
    center_col = f"cos_theta_{stat}"
    alphas = sorted(table["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.0), sharey=True)
    for axis, alpha in zip(axes, alphas):
        sub = table[table["alpha"] == alpha]
        for algo in ORDER:
            rows = sub[sub["algorithm"] == algo].sort_values("rho")
            if rows.empty:
                continue
            axis.plot(rows["rho"], rows[center_col], color=PALETTE[algo], linewidth=1.3, label=algo)
            axis.fill_between(rows["rho"], rows["cos_theta_q25"], rows["cos_theta_q75"],
                               color=PALETTE[algo], alpha=0.08, linewidth=0)
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(f"alpha = {alpha:g}")
        axis.set_xlabel("rho")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    by_label = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            by_label.setdefault(label, handle)
    fig.suptitle(title)
    fig.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_vector_angle(table, stat, title, path, rho_fixed=0.2):
    center_col = f"cos_theta_{stat}"
    alphas = sorted(table["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.3 * len(alphas), 4.6),
                              subplot_kw=dict(projection="polar"))
    if len(alphas) == 1:
        axes = [axes]
    for axis, alpha in zip(axes, alphas):
        sub = table[(table["alpha"] == alpha) & np.isclose(table["rho"], rho_fixed)]
        axis.plot([0, 0], [0, 1], color="black", linewidth=2.5, zorder=5)
        axis.annotate("", xy=(0, 1), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2.5))
        for algo in ORDER:
            row = sub[sub["algorithm"] == algo]
            if row.empty:
                continue
            row = row.iloc[0]
            center = min(1.0, max(-1.0, row[center_col]))
            q25 = min(1.0, max(-1.0, row["cos_theta_q25"]))
            q75 = min(1.0, max(-1.0, row["cos_theta_q75"]))
            theta = math.acos(center)
            theta_lo = math.acos(q75)
            theta_hi = math.acos(q25)
            axis.annotate("", xy=(theta, 1), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=PALETTE[algo], linewidth=1.6))
            wedge = np.linspace(theta_lo, theta_hi, 20)
            axis.fill_between(wedge, 0, 1, color=PALETTE[algo], alpha=0.08, linewidth=0)
        axis.set_thetamin(-10)
        axis.set_thetamax(190)
        axis.set_theta_zero_location("E")
        axis.set_rticks([])
        axis.set_title(f"alpha = {alpha:g}  (rho={rho_fixed:g})", pad=16)
    stat_label = {"median": "median", "mean": "mean", "mode": "half-sample mode"}[stat]
    fig.suptitle(title + f"\n(black = reference direction at 0 deg; colored arrows at arccos({stat_label} cos theta))")
    by_label = {algo: plt.Line2D([0], [0], color=PALETTE[algo], linewidth=1.6) for algo in ORDER}
    fig.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("finals_path")
    p.add_argument("out_dir")
    p.add_argument("--stat", choices=["median", "mean", "mode"], default="median")
    p.add_argument("--noise-label", choices=["anisotropic", "isotropic"], required=True)
    args = p.parse_args()

    finals = pd.read_csv(args.finals_path)
    stat = args.stat
    suffix = "" if stat == "median" else f"_{stat}"
    iso_suffix = "_isotropic" if args.noise_label == "isotropic" else ""
    noise_title = args.noise_label

    grad_table = build_csv(finals, "final_cos_pert_grad", stat)
    mom_table = build_csv(finals, "final_cos_pert_momentum", stat)

    grad_table.to_csv(f"{args.out_dir}/cos_pert_grad_by_rho{suffix}{iso_suffix}.csv", index=False)
    mom_table.to_csv(f"{args.out_dir}/cos_pert_momentum_by_rho{suffix}{iso_suffix}.csv", index=False)

    plot_line(grad_table, stat, f"cos(perturbation, gradient) vs rho -- {noise_title} noise ({stat})",
              "cos(theta)", f"{args.out_dir}/cos_pert_grad_vs_rho{suffix}{iso_suffix}.png")
    plot_line(mom_table, stat, f"cos(perturbation, momentum) vs rho -- {noise_title} noise ({stat})",
              "cos(theta)", f"{args.out_dir}/cos_pert_momentum_vs_rho{suffix}{iso_suffix}.png")

    plot_vector_angle(grad_table, stat, f"Perturbation vs gradient angle -- {noise_title} noise",
                       f"{args.out_dir}/vector_angle_grad{suffix}{iso_suffix}.png")
    plot_vector_angle(mom_table, stat, f"Perturbation vs momentum angle -- {noise_title} noise",
                       f"{args.out_dir}/vector_angle_momentum{suffix}{iso_suffix}.png")

    print(f"done [{stat}, {args.noise_label}]:", grad_table.shape, mom_table.shape)


if __name__ == "__main__":
    main()
