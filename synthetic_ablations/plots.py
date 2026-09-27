"""Figures and summary tables from a results directory.

    python -m synthetic_ablations.plots results/single_matrix
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = {
    "muon": lambda t: (t["outer"].isin(["muon", "nsgdm"])),
    "sgd": lambda t: (t["outer"].isin(["sgd", "clip_sgd"])),
    "all": lambda t: t["outer"].notna(),
}
METRIC_LABELS = {
    "gap": "Final objective gap  F(W_T) - F*",
    "excess_risk": "Final population excess risk",
    "lambda_max": "Final Hessian lambda_max",
    "balancedness": "Final balancedness",
    "effective_rank": "Final effective rank of P",
}


def _alpha_title(alpha):
    return "alpha = inf (Gaussian)" if math.isinf(alpha) else f"alpha = {alpha:g}"


def final_table(finals, metric):
    column = f"final_{metric}"
    keys = ["variant", "label", "algorithm", "outer", "origin", "alpha", "rho"]
    grouped = finals.groupby(keys, dropna=False)
    # Diverged seeds carry +inf for gap-type metrics, so the median is conservative (no survivor bias).
    table = grouped[column].agg(median=lambda s: np.nanmedian(s), q25=lambda s: np.nanquantile(s, 0.25),
                                q75=lambda s: np.nanquantile(s, 0.75)).reset_index()
    table["mean_finite"] = grouped[column].apply(lambda s: s[np.isfinite(s)].mean()).values
    table["divergence_rate"] = grouped["diverged"].mean().values
    table["perturbation"] = grouped["final_perturbation_frob"].median().values
    table["oracle_calls"] = grouped["oracle_calls"].max().values
    return table


def _colors(variants):
    cmap = plt.get_cmap("tab20")
    ordered = sorted(variants)
    return {v: cmap(i % 20) for i, v in enumerate(ordered)}


def _panels(alphas):
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.6 * len(alphas), 4.2), squeeze=False)
    return fig, axes[0]


def plot_vs_rho(table, metric, group, path, x="rho"):
    subset = table[GROUPS[group](table)]
    if subset.empty:
        return
    alphas = sorted(subset["alpha"].unique())
    colors = _colors(subset["variant"].unique())
    fig, axes = _panels(alphas)
    for axis, alpha in zip(axes, alphas):
        at = subset[subset["alpha"] == alpha]
        sam = at[at["rho"].notna()]
        for variant, rows in sam.groupby("variant"):
            rows = rows.sort_values(x)
            finite = np.isfinite(rows["median"])
            axis.plot(rows[x][finite], rows["median"][finite], marker="o", markersize=3,
                      color=colors[variant], label=rows["label"].iloc[0])
            axis.fill_between(rows[x][finite], rows["q25"][finite], rows["q75"][finite],
                              color=colors[variant], alpha=0.12, linewidth=0)
        for _, row in at[at["rho"].isna()].iterrows():
            if np.isfinite(row["median"]):
                axis.axhline(row["median"], color=colors[row["variant"]], linestyle="--", linewidth=1.2,
                             label=f"{row['label']} (no SAM)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(_alpha_title(alpha))
        axis.set_xlabel("rho (nominal)" if x == "rho" else "measured ||eps||_F (median)")
        axis.set_ylabel(METRIC_LABELS.get(metric, metric))
        axis.grid(alpha=0.3, which="both")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def best_rho(table):
    sam = table[table["rho"].notna()]
    best = sam.loc[sam.groupby(["variant", "alpha"])["median"].idxmin().dropna()]
    return pd.concat([best, table[table["rho"].isna()]], ignore_index=True)


def plot_curves(curves, best, group, path, x="step", metric="gap"):
    best = best[GROUPS[group](best)]
    if best.empty:
        return
    alphas = sorted(best["alpha"].unique())
    colors = _colors(best["variant"].unique())
    fig, axes = _panels(alphas)
    for axis, alpha in zip(axes, alphas):
        for _, row in best[best["alpha"] == alpha].iterrows():
            mask = (curves["variant"] == row["variant"]) & (curves["alpha"] == alpha)
            mask &= curves["rho"].isna() if pd.isna(row["rho"]) else np.isclose(curves["rho"], row["rho"])
            rows = curves[mask].sort_values("step")
            if rows.empty:
                continue
            rho = "" if pd.isna(row["rho"]) else f" (rho={row['rho']:g})"
            linestyle = "--" if pd.isna(row["rho"]) else "-"
            axis.plot(rows[x], rows[f"{metric}_median"], color=colors[row["variant"]], linestyle=linestyle,
                      label=row["label"] + rho)
            axis.fill_between(rows[x], rows[f"{metric}_q25"], rows[f"{metric}_q75"],
                              color=colors[row["variant"]], alpha=0.1, linewidth=0)
        axis.set_yscale("log")
        axis.set_title(_alpha_title(alpha))
        axis.set_xlabel("iteration" if x == "step" else "oracle (gradient) calls")
        axis.set_ylabel(f"{metric} (median, IQR band)")
        axis.grid(alpha=0.3, which="both")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(best, group, path, metric="gap"):
    best = best[GROUPS[group](best)]
    if best.empty:
        return
    colors = _colors(best["variant"].unique())
    alphas = sorted(best["alpha"].unique())
    positions = {a: i for i, a in enumerate(alphas)}
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for variant, rows in best.groupby("variant"):
        rows = rows.sort_values("alpha")
        xs = [positions[a] for a in rows["alpha"]]
        axis.plot(xs, rows["median"], marker="o", color=colors[variant],
                  linestyle="--" if rows["rho"].isna().all() else "-", label=rows["label"].iloc[0])
    axis.set_xticks(range(len(alphas)))
    axis.set_xticklabels(["inf" if math.isinf(a) else f"{a:g}" for a in alphas])
    axis.set_xlabel("tail index alpha (smaller = heavier tail)")
    axis.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} at best rho")
    axis.set_yscale("log")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_all(out):
    out = Path(out)
    finals_path, curves_path = out / "finals.csv", out / "curves.csv"
    if not finals_path.exists():
        print(f"No finals.csv in {out}")
        return
    finals = pd.read_csv(finals_path)
    curves = pd.read_csv(curves_path)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    depth = int(finals["depth"].iloc[0])
    #metrics = ["gap", "excess_risk"] + (["lambda_max", "balancedness"] if depth >= 2 else ["lambda_max"])

    metrics = ["gap"]  # User requested only gap plots, skips lambda_max to avoid NaN crash

    gap_table = final_table(finals, "gap")
    best = best_rho(gap_table)
    best.to_csv(out / "summary_best_rho.csv", index=False)
    gap_table.to_csv(out / "summary_all.csv", index=False)
    for group in GROUPS:
        for metric in metrics:
            table = gap_table if metric == "gap" else final_table(finals, metric)
            plot_vs_rho(table, metric, group, figures / f"{metric}_vs_rho_{group}.png")
        plot_vs_rho(gap_table, "gap", group, figures / f"gap_vs_perturbation_{group}.png", x="perturbation")
        # plot_curves(curves, best, group, figures / f"gap_vs_step_best_rho_{group}.png")
        # plot_curves(curves, best, group, figures / f"gap_vs_oracle_calls_best_rho_{group}.png", x="oracle_calls")
        # plot_robustness(best, group, figures / f"robustness_vs_alpha_{group}.png")
    print(f"Figures written to {figures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    make_all(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
