import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    logs = {
        "Atlas (Exp 1 - Baseline)": "logs/atlas_atlas_exp1_logs.csv",
        "Atlas (Exp 2 - Raw Grad)": "logs/atlas_atlas_exp2_logs.csv",
        "Atlas (Exp 3 - Random)": "logs/atlas_atlas_exp3_logs.csv",
        "AdamW": "logs/adam_atlas_exp1_logs.csv",
        "SGD": "logs/sgd_atlas_exp1_logs.csv",
        "Muon": "logs/muon_atlas_exp1_logs.csv",
    }

    # Prepare data
    results = {}
    for label, path in logs.items():
        if os.path.exists(path):
            results[label] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found. Skipping {label}.")

    if not results:
        print("No log files found in logs/ directory. Please run the experiments first.")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Loss
    for label, df in results.items():
        axes[0].plot(df["epoch"], df["train_loss"], marker='o', label=label)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Validation Accuracy
    for label, df in results.items():
        axes[1].plot(df["epoch"], df["val_acc"], marker='s', label=label)
    axes[1].set_title("Validation Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = "comparison_all_experiments.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved successfully to {out_file}!")

if __name__ == "__main__":
    main()
