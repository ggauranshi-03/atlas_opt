import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches

def extract_best_results(csv_path):
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Get the final row for each optimizer
    results = []
    optimizers = df['optimizer'].unique()
    for opt in optimizers:
        opt_df = df[df['optimizer'] == opt]
        final_row = opt_df.iloc[-1]
        
        # Round the values for display
        results.append({
            'Optimizer': opt.upper(),
            'Train Loss': f"{final_row['train_loss']:.4f}",
            'Train Acc (%)': f"{final_row['train_accuracy']:.2f}%",
            'Val Loss': f"{final_row['val_loss']:.4f}",
            'Val Acc (%)': f"{final_row['val_accuracy']:.2f}%"
        })
    return results

def add_experiment_section(doc, title, metadata, results):
    doc.add_heading(title, level=2)
    
    # Add metadata
    p = doc.add_paragraph()
    for key, value in metadata.items():
        p.add_run(f"{key}: ").bold = True
        p.add_run(f"{value}\n")
        
    if not results:
        doc.add_paragraph("No data found for this experiment.")
        return
        
    # Add table
    table = doc.add_table(rows=1, cols=5)
    hdr_cells = table.rows[0].cells
    headers = ['Optimizer', 'Final Train Loss', 'Final Train Acc', 'Final Val Loss', 'Final Val Acc']
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        hdr_cells[i].paragraphs[0].runs[0].bold = True
        
    for res in results:
        row_cells = table.add_row().cells
        row_cells[0].text = res['Optimizer']
        row_cells[1].text = res['Train Loss']
        row_cells[2].text = res['Train Acc (%)']
        row_cells[3].text = res['Val Loss']
        row_cells[4].text = res['Val Acc (%)']
        
    doc.add_paragraph() # Add space

def main():
    doc_path = r"G:\Gauranshi\IIITD-Research\Atlas_opt\Opt-exp-res\ATLAS OPTIMIZER EXPERIMENTS_060926.docx"
    
    doc = Document(doc_path)
    doc.add_page_break()
    doc.add_heading("Final Experiment Results (All Optimizers)", level=1)
    
    # Experiment 1
    res1 = extract_best_results("final_logs/run1_cifar10_logs.csv")
    meta1 = {
        "Model Name": "Airbench CNN",
        "Size": "Small Custom CNN",
        "Dataset": "CIFAR-10 (32x32 images)",
        "Batch Size": "500 (micro: 500, grad_acc: 1)",
        "Steps": "720 steps (7 epochs)"
    }
    add_experiment_section(doc, "Experiment 1: CIFAR-10 Image Classification", meta1, res1)
    
    # Experiment 2
    res2 = extract_best_results("final_logs/run2_nanogpt_logs.csv")
    meta2 = {
        "Model Name": "nanoGPT-135M",
        "Size": "135M parameters (12L / 12H / 768d)",
        "Dataset": "fineweb-edu (sample-10BT), 2048 tokens/seq",
        "Batch Size": "512 seqs/step (micro: 16, grad_acc: 32)",
        "Steps": "1500 steps (15 epochs)"
    }
    add_experiment_section(doc, "Experiment 2: nanoGPT Continual Pre-Training", meta2, res2)
    
    # Experiment 3
    res3 = extract_best_results("final_logs/run3_pythia_logs.csv")
    meta3 = {
        "Model Name": "Pythia-70M (EleutherAI)",
        "Size": "70M parameters",
        "Dataset": "fineweb-edu (sample-10BT), 2048 tokens/seq",
        "Batch Size": "512 seqs/step (micro: 16, grad_acc: 32)",
        "Steps": "1400 steps (14 epochs)"
    }
    add_experiment_section(doc, "Experiment 3: Pythia-70M Continual Pre-Training", meta3, res3)
    
    doc.save(doc_path)
    print("Successfully updated the Word document.")

if __name__ == '__main__':
    main()
