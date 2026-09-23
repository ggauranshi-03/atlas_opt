import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def parse_log_text(log_path, steps_per_epoch=100):
    if not os.path.exists(log_path): return pd.DataFrame()
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Epoch' in line and '| Train Loss:' in line:
                try:
                    parts = line.split(']')
                    opt_raw = parts[0].replace('[', '').strip().lower()
                    
                    rest = parts[1]
                    epoch_str = rest.split('Epoch')[1].split('/')[0].strip()
                    epoch = int(epoch_str)
                    
                    t_loss = float(re.search(r'Train Loss:\s*([\d.]+)', rest).group(1))
                    t_acc = float(re.search(r'Train Acc:\s*([\d.]+)', rest).group(1))
                    v_loss = float(re.search(r'Val Loss:\s*([\d.]+)', rest).group(1))
                    v_acc = float(re.search(r'Val Acc:\s*([\d.]+)', rest).group(1))
                    
                    step = epoch * steps_per_epoch
                    
                    data.append({
                        'optimizer': opt_raw,
                        'epoch': epoch,
                        'step': step,
                        'train_loss': t_loss,
                        'train_accuracy': t_acc,
                        'val_loss': v_loss,
                        'val_accuracy': v_acc
                    })
                except Exception as e:
                    pass
    return pd.DataFrame(data)

def combine_and_save_csv(base_csv_path, new_dfs):
    if not os.path.exists(base_csv_path): return
    base_df = pd.read_csv(base_csv_path)
    
    for new_df in new_dfs:
        if new_df.empty: continue
        optimizers_to_replace = new_df['optimizer'].unique()
        base_df = base_df[~base_df['optimizer'].isin(optimizers_to_replace)]
        base_df = pd.concat([base_df, new_df], ignore_index=True)
        
    base_df.to_csv(base_csv_path, index=False)
    return base_df

def plot_metrics(csv_path, plot_path_prefix, title):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    colors = {
        'atlas': 'tab:blue', 
        'atlas_raw': 'tab:orange', 
        'atlas_random': 'tab:green', 
        'muon': 'tab:red', 
        'adam': 'tab:purple', 
        'sgd': 'tab:brown', 
        'muon_sam': 'tab:pink',
        'muon_sam_frob': 'tab:olive',
        'muon_sam_stale': 'tab:cyan'
    }
    
    # 1. PPL Plot (if not cifar)
    if 'cifar' not in plot_path_prefix.lower():
        plt.figure(figsize=(10, 6))
        for opt in df['optimizer'].unique():
            opt_df = df[df['optimizer'] == opt]
            color = colors.get(opt.lower(), 'black')
            plt.plot(opt_df['step'], np.exp(opt_df['val_loss']), label=opt.upper(), color=color, linewidth=1.5, marker='o', markersize=4)
        
        plt.title(f'{title} - Validation Perplexity')
        plt.xlabel('Steps')
        plt.ylabel('Validation Perplexity (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(frameon=True, fancybox=False, edgecolor='lightgray', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{plot_path_prefix}_perplexity_plot.png', dpi=300)
        plt.close()
    
    # 2. Loss & Acc Plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for opt in df['optimizer'].unique():
        opt_df = df[df['optimizer'] == opt]
        color = colors.get(opt.lower(), 'black')
        ax[0].plot(opt_df['step'], opt_df['train_loss'], label=opt.upper(), color=color, linewidth=1.5, marker='o', markersize=4)
        ax[1].plot(opt_df['step'], opt_df['val_accuracy'], label=opt.upper(), color=color, linewidth=1.5, marker='s', markersize=4)
    
    ax[0].set_title(f'{title} - Train Loss')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Train Loss')
    ax[0].grid(True, linestyle='--', alpha=0.5)
    
    ax[1].set_title(f'{title} - Validation Accuracy')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].legend(frameon=True, fancybox=False, edgecolor='lightgray', bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
        
        md_loss += f"#### {title}\n"
        md_loss += "| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |\n"
        md_loss += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        if 'cifar' not in title.lower():
            md_ppl += f"#### {title}\n"
            md_ppl += "| Optimizer | Final Validation Perplexity |\n"
            md_ppl += "| :--- | :--- |\n"
            
        opts_order = ['atlas_random', 'atlas_raw', 'atlas', 'adam', 'muon', 'muon_sam', 'muon_sam_frob', 'muon_sam_stale', 'sgd']
        for opt in opts_order:
            opt_df = df[df['optimizer'].str.lower() == opt]
            if opt_df.empty: continue
            
            final_row = opt_df.iloc[-1]
            t_loss = f"{final_row['train_loss']:.4f}"
            t_acc = f"{final_row['train_accuracy']:.2f}%"
            v_loss = f"{final_row['val_loss']:.4f}"
            v_acc = f"{final_row['val_accuracy']:.2f}%"
            
            display_name = opt.upper()
            if display_name == 'MUON_SAM': display_name = 'Muon-SAM'
            elif display_name == 'MUON_SAM_FROB': display_name = 'Muon-SAM (Frob)'
            elif display_name == 'MUON_SAM_STALE': display_name = 'Muon-SAM (Stale)'
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
        
    content = re.sub(r'### Final Loss and Accuracy Results.*?(?=### Language Modeling Validation Perplexity Results)', md_loss, content, flags=re.DOTALL)
    content = re.sub(r'### Language Modeling Validation Perplexity Results.*?(?=## 6\. Heavy-Tailed)', md_ppl, content, flags=re.DOTALL)
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    # 1. CIFAR10
    df_frob_stale_1 = parse_log_text('run1_frob_stale_cifar10.log', 100)
    df_sgd_1 = parse_log_text('final_logs/run1_sgd_new.log', 100)
    combine_and_save_csv('final_logs/run1_cifar10_logs.csv', [df_frob_stale_1, df_sgd_1])
    plot_metrics('final_logs/run1_cifar10_logs.csv', 'final_logs/run1_cifar10', 'Experiment 1: CIFAR-10')

    # 2. NanoGPT
    df_frob_stale_2 = parse_log_text('run2_frob_stale_nanogpt.log', 100)
    df_sgd_2 = parse_log_text('final_logs/run2_sgd_new.log', 100)
    combine_and_save_csv('final_logs/run2_nanogpt_logs.csv', [df_frob_stale_2, df_sgd_2])
    plot_metrics('final_logs/run2_nanogpt_logs.csv', 'final_logs/run2_nanogpt', 'Experiment 2: NanoGPT')

    # 3. Pythia
    df_frob_stale_3 = parse_log_text('run3_frob_stale_pythia.log', 100)
    df_sgd_3 = parse_log_text('final_logs/run3_sgd_new.log', 100)
    combine_and_save_csv('final_logs/run3_pythia_logs.csv', [df_frob_stale_3, df_sgd_3])
    plot_metrics('final_logs/run3_pythia_logs.csv', 'final_logs/run3_pythia', 'Experiment 3: Pythia')

    update_readme()
    print("Done processing logs, generating plots, and updating README!")
