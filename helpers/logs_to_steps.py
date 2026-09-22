"""
Add a 'step' column (step = epoch * STEPS_PER_EPOCH) to every experiment CSV in logs/
and re-plot train loss / the run's secondary metric against steps instead of epochs.

STEPS_PER_EPOCH=100 matches the hardcoded value in main_experiment.py (steps_per_epoch = 100,
see main_experiment.py:118), used for every experiment (CIFAR-10, nanoGPT, Pythia-70M) in this repo.
"""
import argparse
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

STEPS_PER_EPOCH = 100
OUTPUT_DIR = "final_logs"

# Candidate column names for the "secondary" metric plotted alongside loss.
METRIC_CANDIDATES = ["val_perplexity", "Perplexity", "val_accuracy", "Val Acc (%)"]

# Matches lines like:
#   [ATLAS       ] Epoch  1/7 | Train Loss: 2.5011 | Train Acc: 8.74% | Val Loss: 2.3798 | Val Acc: 10.52% | Time: 22.5s
EPOCH_LINE_RE = re.compile(
    r"\[(?P<optimizer>[A-Za-z0-9_]+)\s*\]\s*Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s*\|"
    r"\s*Train Loss:\s*(?P<train_loss>[-\d.]+)"
    r"(?:\s*\|\s*Train Acc:\s*(?P<train_accuracy>[-\d.]+)%)?"
    r"\s*\|\s*Val Loss:\s*(?P<val_loss>[-\d.]+)"
    r"\s*\|\s*Val Acc(?: \(%\))?:\s*(?P<val_accuracy>[-\d.]+)%?"
    r"\s*\|\s*Time:\s*(?P<time_s>[-\d.]+)s"
)


def parse_raw_log(log_path):
    """Parse a raw training .log file's '[OPT] Epoch X/Y | ...' lines into a DataFrame."""
    rows = []
    with open(log_path) as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "epoch": int(d["epoch"]),
                "optimizer": d["optimizer"].lower(),
                "train_loss": float(d["train_loss"]),
                "train_accuracy": float(d["train_accuracy"]) if d["train_accuracy"] else None,
                "val_loss": float(d["val_loss"]),
                "val_accuracy": float(d["val_accuracy"]),
                "time_s": float(d["time_s"]),
            })
    return pd.DataFrame(rows)


def convert_raw_log(log_path):
    """Parse a raw .log file, write a *_logs.csv (with 'step'), and plot vs steps."""
    df = parse_raw_log(log_path)
    if df.empty:
        print(f"Skipping {log_path} (no epoch lines found)")
        return
    df["step"] = df["epoch"] * STEPS_PER_EPOCH

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(log_path))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{base}_logs.csv")
    df.to_csv(csv_path, index=False)
    print(f"Parsed {log_path} -> {csv_path}")

    convert_and_plot(csv_path)


def find_metric_column(df):
    for col in METRIC_CANDIDATES:
        if col in df.columns:
            return col
    return None


def convert_and_plot(csv_path):
    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        print(f"Skipping {csv_path} (no 'epoch' column)")
        return

    if "step" not in df.columns:
        df["step"] = df["epoch"] * STEPS_PER_EPOCH
        df.to_csv(csv_path, index=False)
        print(f"Added 'step' column to {csv_path}")

    metric_col = find_metric_column(df)
    if metric_col is None or "optimizer" not in df.columns:
        print(f"  (no plot generated for {csv_path}: missing optimizer/metric columns)")
        return

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    for opt in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == opt].sort_values("step")
        ax[0].plot(opt_data["step"], opt_data["train_loss"], label=str(opt).upper(), marker="o", markersize=4)
        ax[1].plot(opt_data["step"], opt_data[metric_col], label=str(opt).upper(), marker="s", markersize=4)

    ax[0].set_title("Train Loss vs Step")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Train Loss")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.6)

    ax[1].set_title(f"{metric_col} vs Step")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel(metric_col)
    ax[1].legend()
    ax[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(csv_path).replace("_logs.csv", "_steps_plot.png")
    plot_path = os.path.join(OUTPUT_DIR, base)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_files", nargs="*",
        help="Optional raw .log file(s) to parse (e.g. run1_cifar10.log run2_nanogpt.log). "
             "If omitted, converts every logs/*_logs.csv instead.",
    )
    args = parser.parse_args()

    if args.log_files:
        for log_file in args.log_files:
            convert_raw_log(log_file)
        return

    csv_files = sorted(glob.glob("logs/*_logs.csv"))
    if not csv_files:
        print("No CSV logs found in the logs/ directory.")
        return
    for csv_path in csv_files:
        convert_and_plot(csv_path)


if __name__ == "__main__":
    main()
