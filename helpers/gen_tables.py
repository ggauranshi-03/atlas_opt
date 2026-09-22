import pandas as pd
import os

results_md = '### Final Loss and Accuracy Results\n\n'

runs = [
    ('Experiment 1: CIFAR-10 Image Classification', 'final_logs/run1_cifar10_logs.csv'),
    ('Experiment 2: nanoGPT Continual Pre-Training', 'final_logs/run2_nanogpt_logs.csv'),
    ('Experiment 3: Pythia-70M Continual Pre-Training', 'final_logs/run3_pythia_logs.csv')
]

for title, csv_path in runs:
    if not os.path.exists(csv_path): continue
    results_md += f'#### {title}\n'
    results_md += '| Optimizer | Final Train Loss | Final Train Acc | Final Val Loss | Final Val Acc |\n'
    results_md += '| :--- | :--- | :--- | :--- | :--- |\n'
    
    df = pd.read_csv(csv_path)
    for opt in df['optimizer'].unique():
        opt_df = df[df['optimizer'] == opt]
        final_row = opt_df.iloc[-1]
        
        t_loss = f"{final_row['train_loss']:.4f}"
        t_acc = f"{final_row['train_accuracy']:.2f}%"
        v_loss = f"{final_row['val_loss']:.4f}"
        v_acc = f"{final_row['val_accuracy']:.2f}%"
        
        opt_name = opt.upper()
        if opt_name == 'MUON_SAM': opt_name = 'Muon-SAM'
        elif opt_name == 'ADAM': opt_name = 'AdamW'
        elif opt_name == 'SGD': opt_name = 'SGD (Fixed)'
        elif opt_name == 'MUON': opt_name = 'Muon'
        elif opt_name == 'ATLAS': opt_name = 'Atlas'
        elif opt_name == 'ATLAS_RAW': opt_name = 'Atlas Raw'
        elif opt_name == 'ATLAS_RANDOM': opt_name = 'Atlas Random'
        
        results_md += f'| **{opt_name}** | {t_loss} | {t_acc} | {v_loss} | {v_acc} |\n'
    results_md += '\n'

with open('loss_acc_tables.txt', 'w') as f:
    f.write(results_md)
