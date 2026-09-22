import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from docx import Document

def update_final_csvs():
    # NanoGPT
    final_nano = pd.read_csv('final_logs/run2_nanogpt_logs.csv')
    final_nano = final_nano[~final_nano['optimizer'].isin(['muon', 'muon_sam', 'MUON', 'MUON_SAM'])]
    new_nano = pd.read_csv('logs/nanogpt_fineweb_logs.csv')
    new_nano = new_nano[new_nano['optimizer'].isin(['muon', 'muon_sam'])]
    updated_nano = pd.concat([final_nano, new_nano], ignore_index=True)
    updated_nano.to_csv('final_logs/run2_nanogpt_logs.csv', index=False)
    
    # Pythia
    final_pythia = pd.read_csv('final_logs/run3_pythia_logs.csv')
    final_pythia = final_pythia[~final_pythia['optimizer'].isin(['muon', 'muon_sam', 'MUON', 'MUON_SAM'])]
    new_pythia = pd.read_csv('logs/pythia70m_pretrain_chinchilla_logs.csv')
    new_pythia = new_pythia[new_pythia['optimizer'].isin(['muon', 'muon_sam'])]
    updated_pythia = pd.concat([final_pythia, new_pythia], ignore_index=True)
    updated_pythia.to_csv('final_logs/run3_pythia_logs.csv', index=False)

def plot_metrics(csv_path, plot_path_prefix, title, total_steps):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    # tab10 colors matching user's image
    colors = {
        'atlas': 'tab:blue', 
        'atlas_raw': 'tab:orange', 
        'atlas_random': 'tab:green', 
        'muon': 'tab:red', 
        'adam': 'tab:purple', 
        'sgd': 'tab:brown', 
        'muon_sam': 'tab:pink'
    }
    
    # 1. PPL Plot
    if 'cifar' not in plot_path_prefix.lower():
        plt.figure(figsize=(8, 5))
        for opt in df['optimizer'].unique():
            opt_df = df[df['optimizer'] == opt]
            steps = opt_df['step']
            val_loss = opt_df['val_loss']
            val_perplexity = np.exp(val_loss)
            color = colors.get(opt.lower(), 'black')
            
            # Using circle markers 'o', markersize 4
            plt.plot(steps, val_perplexity, label=opt.upper(), color=color, linewidth=1.5, marker='o', markersize=4)
        
        plt.title('Validation Perplexity vs Step')
        plt.xlabel('Steps')
        plt.ylabel('Validation Perplexity (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(frameon=True, fancybox=False, edgecolor='lightgray')
        plt.tight_layout()
        plt.savefig(f'{plot_path_prefix}_perplexity_plot.png', dpi=300)
        plt.close()
    
    # 2. Loss & Acc Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for opt in df['optimizer'].unique():
        opt_df = df[df['optimizer'] == opt]
        steps = opt_df['step']
        train_loss = opt_df['train_loss']
        val_acc = opt_df['val_accuracy']
        color = colors.get(opt.lower(), 'black')
        
        # Circle markers for Loss
        ax[0].plot(steps, train_loss, label=opt.upper(), color=color, linewidth=1.5, marker='o', markersize=4)
        # Square markers for Accuracy
        ax[1].plot(steps, val_acc, label=opt.upper(), color=color, linewidth=1.5, marker='s', markersize=4)
    
    ax[0].set_title('Train Loss vs Step')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Train Loss')
    # Linear scale as requested in image
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].legend(frameon=True, fancybox=False, edgecolor='lightgray')
    
    ax[1].set_title('Validation Accuracy vs Step')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].legend(frameon=True, fancybox=False, edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig(f'{plot_path_prefix}_plot.png', dpi=300)
    plt.close()

def generate_markdown_tables():
    runs = [
        ('Experiment 1: CIFAR-10 Image Classification', 'final_logs/run1_cifar10_logs.csv'),
        ('Experiment 2: nanoGPT Continual Pre-Training', 'final_logs/run2_nanogpt_logs.csv'),
        ('Experiment 3: Pythia-70M Continual Pre-Training', 'final_logs/run3_pythia_logs.csv')
    ]
    
    md_loss = "### Final Loss and Accuracy Results\n\n"
    md_ppl = "### Language Modeling Validation Perplexity Results\n\n"
    
    for title, csv_path in runs:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        
        # Loss/Acc table
        md_loss += f"#### {title}\n"
        md_loss += "| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |\n"
        md_loss += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        # PPL table
        if 'cifar' not in title.lower():
            md_ppl += f"#### {title}\n"
            md_ppl += "| Optimizer | Final Validation Perplexity |\n"
            md_ppl += "| :--- | :--- |\n"
            
        for opt in ['atlas_random', 'atlas_raw', 'atlas', 'adam', 'muon', 'muon_sam', 'sgd']:
            opt_lower = opt.lower()
            opt_df = df[df['optimizer'].str.lower() == opt_lower]
            if opt_df.empty: continue
            
            final_row = opt_df.iloc[-1]
            t_loss = f"{final_row['train_loss']:.4f}"
            t_acc = f"{final_row['train_accuracy']:.2f}%"
            v_loss = f"{final_row['val_loss']:.4f}"
            v_acc = f"{final_row['val_accuracy']:.2f}%"
            
            # Reformat names for display
            display_name = opt.upper()
            if display_name == 'MUON_SAM': display_name = 'Muon-SAM'
            elif display_name == 'ADAM': display_name = 'AdamW'
            elif display_name == 'SGD': display_name = 'SGD'
            elif display_name == 'MUON': display_name = 'Muon'
            elif display_name == 'ATLAS': display_name = 'Atlas'
            elif display_name == 'ATLAS_RAW': display_name = 'Atlas Raw'
            elif display_name == 'ATLAS_RANDOM': display_name = 'Atlas Random'
            
            md_loss += f"| **{display_name}** | {t_loss} | {t_acc} | {v_loss} | {v_acc} |\n"
            
            if 'cifar' not in title.lower():
                ppl = f"{np.exp(final_row['val_loss']):.2f}"
                md_ppl += f"| **{display_name}** | {ppl} |\n"
                
        md_loss += "\n"
        if 'cifar' not in title.lower():
            md_ppl += "\n"
            
    md_ppl += "*(Note: CIFAR-10 is an image classification task evaluated on Cross-Entropy Loss and Accuracy, and therefore does not have a Perplexity metric).*\n\n"
    
    return md_loss, md_ppl

def update_readme():
    md_loss, md_ppl = generate_markdown_tables()
    
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace Loss/Acc tables
    content = re.sub(r'### Final Loss and Accuracy Results.*?(?=### Language Modeling Validation Perplexity Results)', md_loss, content, flags=re.DOTALL)
    
    # Replace PPL tables
    content = re.sub(r'### Language Modeling Validation Perplexity Results.*?(?=## 6\. Heavy-Tailed)', md_ppl, content, flags=re.DOTALL)
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    # update_final_csvs()
    # print("CSV updated.")
    
    plot_metrics('final_logs/run2_nanogpt_logs.csv', 'final_logs/run2_nanogpt', 'NanoGPT-135M', 1500)
    plot_metrics('final_logs/run3_pythia_logs.csv', 'final_logs/run3_pythia', 'Pythia-70M', 1400)
    print("Plots generated.")
    
    # update_readme()
    # print("README updated.")
