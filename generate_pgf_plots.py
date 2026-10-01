import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from synthetic_ablations import plots
import matplotlib

import matplotlib.backends.backend_pgf as pgf
# Monkey patch pgf escaping to prevent issues with LaTeX characters like \rho, \alpha
pgf.common_texification = lambda x: x
if hasattr(pgf, '_tex_escape'):
    pgf._tex_escape = lambda x: x

def dummy_get_text_width_height_descent(text, fontprop, ismath):
    # Use realistic point sizes: width ~4.5pt per char, height 8pt.
    return len(text) * 4.5, 8.0, 2.0
    
pgf.get_text_width_height_descent = dummy_get_text_width_height_descent
pgf.RendererPgf.get_text_width_height_descent = lambda self, text, prop, ismath: dummy_get_text_width_height_descent(text, prop, ismath)

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'font.size': 7.0,
    'axes.labelsize': 8.0,
    'axes.titlesize': 9.0,
    'legend.fontsize': 7.0,
    'figure.subplot.wspace': 0.1,  
    'figure.subplot.bottom': 0.20,
    'figure.subplot.left': 0.08,    
    'figure.subplot.right': 0.98,
    'figure.subplot.top': 0.82
})

def make_pgf_plots(frob_folder, spec_folder, adam_folder, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    frob_path = Path('synthetic_ablations/results') / frob_folder / 'finals.csv'
    df_frob = pd.read_csv(frob_path)
    df_frob = df_frob[(df_frob['frobenius_mode'].isna()) | (df_frob['frobenius_mode'] == 'global')].copy()
    
    spec_path = Path('synthetic_ablations/results') / spec_folder / 'finals.csv'
    df_spec = pd.read_csv(spec_path)
    
    adam_path = Path('synthetic_ablations/results') / adam_folder / 'finals.csv'
    df_adam = pd.read_csv(adam_path)
    
    # Use LITERAL strings for the final legend instead of macros
    map_frob = {
        'muon': 'Muon',
        'sam-sgd': 'SAM',
        'friendly-sam-sgd': 'FSAM',
        'random-sam-muon': 'RandSAM-Muon'
    }
    
    map_spec = {
        'full-spectral-sam-muon': 'SpecSAM-Muon',
        'spectral-friendly-sam-muon': 'FP-SOMA',
        'stale-momentum-friendly-spectral-sam-muon': 'SOMA'
    }
    
    map_adam = {
        'adam': 'AdamW'
    }
    
    df_frob_filtered = df_frob[df_frob['algorithm'].isin(map_frob.keys())].copy()
    df_frob_filtered['variant'] = df_frob_filtered['algorithm'].map(map_frob)
    df_frob_filtered['label'] = df_frob_filtered['variant']
    
    df_spec_filtered = df_spec[df_spec['algorithm'].isin(map_spec.keys())].copy()
    df_spec_filtered['variant'] = df_spec_filtered['algorithm'].map(map_spec)
    df_spec_filtered['label'] = df_spec_filtered['variant']
    
    df_adam_filtered = df_adam[df_adam['algorithm'].isin(map_adam.keys())].copy()
    df_adam_filtered['variant'] = df_adam_filtered['algorithm'].map(map_adam)
    df_adam_filtered['label'] = df_adam_filtered['variant']
    
    df_combined = pd.concat([df_frob_filtered, df_spec_filtered, df_adam_filtered], ignore_index=True)
    
    allowed_alphas = [1.1, 1.6, 2.0, 3.0]
    df_combined['alpha_num'] = pd.to_numeric(df_combined['alpha'], errors='coerce')
    df_combined = df_combined[df_combined['alpha_num'].isin(allowed_alphas)].copy()
    
    if 'nn' in frob_folder:
        metrics = ['val_loss', 'val_acc']
    else:
        metrics = ['gap']
        
    for metric in metrics:
        table = plots.final_table(df_combined, metric)
        
        subset = table
        if subset.empty:
            continue
            
        alphas = sorted(subset['alpha'].unique(), key=lambda x: float(x))
        
        # New Distinct Colors
        c_map = {
            'AdamW': "#999999",       # Gray
            'Muon': "#000000",        # Black
            'SAM': "#A65628",         # Brown
            'FSAM': "#377EB8",        # Blue
            'RandSAM-Muon': "#4DAF4A",# Green
            'SpecSAM-Muon': "#984EA3",# Purple
            'FP-SOMA': "#FF7F00",     # Bright Orange
            'SOMA': "#E41A1C"         # Bright Red
        }
        
        fig, axes = plt.subplots(1, len(alphas), figsize=(6.5, 2.3), sharey=True)
        if len(alphas) == 1:
            axes = [axes]
            
        for i, (axis, alpha) in enumerate(zip(axes, alphas)):
            at = subset[subset['alpha'] == alpha]
            sam = at[at['rho'].notna()]
            for variant, rows in sam.groupby('variant'):
                rows = rows.sort_values('rho')
                finite = np.isfinite(rows['median'])
                
                # Use dashed lines for non-SAM baselines
                ls = '--' if variant in ['Muon', 'AdamW'] else '-'
                axis.plot(rows['rho'][finite], rows['median'][finite], marker='o', markersize=1.2,
                          color=c_map[variant], label=rows['label'].iloc[0], linestyle=ls, linewidth=1.2)
                axis.fill_between(rows['rho'][finite], rows['q25'][finite], rows['q75'][finite],
                                  color=c_map[variant], alpha=0.15, linewidth=0)
                                  
            for _, row in at[at['rho'].isna()].iterrows():
                if np.isfinite(row['median']):
                    ls = '--'
                    axis.axhline(row['median'], color=c_map[row['variant']], linestyle=ls, linewidth=1.2,
                                 label=row['label'])
            
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_title(r'$\alpha=' + str(alpha) + '$')
            
            # Explicitly force labelpad to 2
            axis.set_xlabel(r'$\rho$', labelpad=2)
            
            # Grid
            axis.grid(True, which='both', color='0.85', alpha=0.5, linestyle='--')
            
            if i == 0:
                if metric == 'val_acc':
                    axis.set_ylabel('val accuracy', labelpad=4)
                elif metric == 'train_acc':
                    axis.set_ylabel('train accuracy', labelpad=4)
                elif metric == 'val_loss':
                    axis.set_ylabel('train loss', labelpad=4)
                else:
                    axis.set_ylabel('gap', labelpad=4)
                    
        handles, labels = axes[-1].get_legend_handles_labels()
        
        order_map = {
            'SOMA': 0,
            'FP-SOMA': 1,
            'SpecSAM-Muon': 2,
            'RandSAM-Muon': 3,
            'FSAM': 4,
            'SAM': 5,
            'Muon': 6,
            'AdamW': 7
        }
        hl = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 999))
        if hl:
            handles, labels = zip(*hl)
            
        # Only add legend for the FIRST metric in the list (so we get 1 legend per figure in LaTeX)
        add_legend = False
        if metric == 'val_acc' or metric == 'gap':
            add_legend = True
            
        if add_legend:
            # FIX: Use ncol=4 so it fits within the page bounds!
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.90), ncol=4, frameon=True)
            plt.subplots_adjust(top=0.75)
        else:
            plt.subplots_adjust(top=0.88)
            
        fig.savefig(out_dir / f'{metric}_vs_rho.pgf', bbox_inches='tight')
        
        import shutil
        shutil.copyfile(out_dir / f'{metric}_vs_rho.pgf', out_dir / f'{metric}_vs_rho.tikz')
        
        plt.close(fig)

base_out = Path(r'G:\Gauranshi\IIITD-Research\Atlas_opt\Sharpenss_aware_Muon\tikz_for_heavytailed')

make_pgf_plots(
    frob_folder='nn_anisotropic_frob_run',
    spec_folder='nn_two_matrix_anisotropic',
    adam_folder='adam_nn_anisotropic',
    out_dir=base_out / 'experiment_1'
)
make_pgf_plots(
    frob_folder='nn_isotropic_frob_run',
    spec_folder='nn_two_matrix_isotropic',
    adam_folder='adam_nn_isotropic',
    out_dir=base_out / 'appendex_isotropic'
)
make_pgf_plots(
    frob_folder='mse_anisotropic_frob_run',
    spec_folder='two_matrix_mse_anisotropic',
    adam_folder='adam_mse_anisotropic',
    out_dir=base_out / 'experiment_2'
)
print('Done!')
