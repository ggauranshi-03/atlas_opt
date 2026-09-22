import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def convert_and_replot():
    # Find all generated CSV logs
    csv_files = glob.glob("logs/*_logs.csv")
    
    if not csv_files:
        print("No CSV logs found in the logs/ directory.")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # We know that 1 epoch = 100 steps
        if "step" not in df.columns:
            df["step"] = df["epoch"] * 100
            
            # Save the updated CSV with the new step column
            df.to_csv(csv_file, index=False)
            print(f"Updated {csv_file} to include 'step' column.")
        
        # Create a new plot using steps!
        fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
        
        optimizers = df["optimizer"].unique()
        for opt in optimizers:
            opt_data = df[df["optimizer"] == opt]
            
            # Plot Loss vs Steps
            ax[0].plot(opt_data["step"], opt_data["train_loss"], label=opt.upper(), marker='o', markersize=4)
            
            # Plot Metric vs Steps
            if "val_perplexity" in df.columns and opt_data["val_perplexity"].iloc[0] > 0:
                ax[1].plot(opt_data["step"], opt_data["val_perplexity"], label=opt.upper(), marker='s', markersize=4)
                ax[1].set_title("Validation Perplexity vs Step")
                ax[1].set_ylabel("Perplexity")
            else:
                ax[1].plot(opt_data["step"], opt_data["val_accuracy"], label=opt.upper(), marker='s', markersize=4)
                ax[1].set_title("Validation Accuracy vs Step")
                ax[1].set_ylabel("Accuracy (%)")

        # Styling
        ax[0].set_title("Train Loss vs Step")
        ax[0].set_xlabel("Steps")
        ax[0].set_ylabel("Train Loss")
        ax[0].legend()
        ax[0].grid(True, linestyle="--", alpha=0.6)

        ax[1].set_xlabel("Steps")
        ax[1].legend()
        ax[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        
        # Save the new plot
        plot_name = csv_file.replace("_logs.csv", "_steps_plot.png")
        plt.savefig(plot_name)
        print(f"Generated new plot: {plot_name}\n")

if __name__ == "__main__":
    convert_and_replot()
