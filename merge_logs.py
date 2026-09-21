import os
import pandas as pd
import matplotlib.pyplot as plt

def merge_and_plot():
    os.chdir('final_logs')
    
    # 1. Merge CIFAR-10
    if os.path.exists('run1_cifar10_logs.csv') and os.path.exists('run1_muonsam_cifar10_logs.csv'):
        df_main = pd.read_csv('run1_cifar10_logs.csv')
        df_muon = pd.read_csv('run1_muonsam_cifar10_logs.csv')
        # Remove any existing muon_sam just in case
        df_main = df_main[df_main['optimizer'] != 'muon_sam']
        df_merged = pd.concat([df_main, df_muon], ignore_index=True)
        df_merged.to_csv('run1_cifar10_logs.csv', index=False)
        print("Merged CIFAR-10")

    # 2. Merge NanoGPT
    if os.path.exists('run2_nanogpt_logs.csv'):
        df_main = pd.read_csv('run2_nanogpt_logs.csv')
        
        # Remove old SGD (matches sgd or sgd_euclidean)
        df_main = df_main[~df_main['optimizer'].str.contains('sgd', na=False)]
        
        # Add new SGD fixed
        if os.path.exists('run2_sgd_fixed_logs.csv'):
            df_sgd = pd.read_csv('run2_sgd_fixed_logs.csv')
            df_main = pd.concat([df_main, df_sgd], ignore_index=True)
            
        # Add muon_sam
        if os.path.exists('run2_muonsam_nanogpt_logs.csv'):
            df_muon = pd.read_csv('run2_muonsam_nanogpt_logs.csv')
            df_main = df_main[df_main['optimizer'] != 'muon_sam']
            df_main = pd.concat([df_main, df_muon], ignore_index=True)
            
        df_main.to_csv('run2_nanogpt_logs.csv', index=False)
        print("Merged NanoGPT")

    # 3. Merge Pythia
    if os.path.exists('run3_pythia_logs.csv'):
        df_main = pd.read_csv('run3_pythia_logs.csv')
        
        # Remove old SGD
        df_main = df_main[~df_main['optimizer'].str.contains('sgd', na=False)]
        
        # Add new SGD fixed
        if os.path.exists('run3_sgd_fixed_logs.csv'):
            df_sgd = pd.read_csv('run3_sgd_fixed_logs.csv')
            df_main = pd.concat([df_main, df_sgd], ignore_index=True)
            
        # Add muon_sam
        if os.path.exists('run3_muonsam_pythia_logs.csv'):
            df_muon = pd.read_csv('run3_muonsam_pythia_logs.csv')
            df_main = df_main[df_main['optimizer'] != 'muon_sam']
            df_main = pd.concat([df_main, df_muon], ignore_index=True)
            
        df_main.to_csv('run3_pythia_logs.csv', index=False)
        print("Merged Pythia")

    # Now replot them!
    main_files = ['run1_cifar10_logs.csv', 'run2_nanogpt_logs.csv', 'run3_pythia_logs.csv']
    for csv_file in main_files:
        if not os.path.exists(csv_file): continue
        df = pd.read_csv(csv_file)
        
        if "step" not in df.columns:
            df["step"] = df["epoch"] * 100
            df.to_csv(csv_file, index=False)
            
        fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
        optimizers = df["optimizer"].unique()
        for opt in optimizers:
            opt_data = df[df["optimizer"] == opt]
            ax[0].plot(opt_data["step"], opt_data["train_loss"], label=opt.upper(), marker='o', markersize=4)
            
            if "val_perplexity" in df.columns and (opt_data["val_perplexity"] > 0).any():
                ax[1].plot(opt_data["step"], opt_data["val_perplexity"], label=opt.upper(), marker='s', markersize=4)
                ax[1].set_title("Validation Perplexity vs Step")
                ax[1].set_ylabel("Perplexity")
            else:
                ax[1].plot(opt_data["step"], opt_data["val_accuracy"], label=opt.upper(), marker='s', markersize=4)
                ax[1].set_title("Validation Accuracy vs Step")
                ax[1].set_ylabel("Accuracy (%)")

        ax[0].set_title("Train Loss vs Step")
        ax[0].set_xlabel("Steps")
        ax[0].set_ylabel("Train Loss")
        ax[0].legend()
        ax[0].grid(True, linestyle="--", alpha=0.6)

        ax[1].set_xlabel("Steps")
        ax[1].legend()
        ax[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plot_name = csv_file.replace("_logs.csv", "_steps_plot.png")
        plt.savefig(plot_name)
        print(f"Generated new plot: {plot_name}")

if __name__ == '__main__':
    merge_and_plot()
