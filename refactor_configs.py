import yaml
import copy
import glob

def refactor_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
        
    if 'optimizers' not in data:
        return
        
    opts = data['optimizers']
    new_opts = {}
    
    # Extract ADAM
    if 'adamw_baseline' in opts:
        new_opts['adam'] = opts['adamw_baseline']
        new_opts['adam']['optimizer'] = 'adam'
    elif 'adamw' in opts:
        new_opts['adam'] = opts['adamw']
        new_opts['adam']['optimizer'] = 'adam'
    elif 'adam' in opts:
        new_opts['adam'] = opts['adam']
        
    # Extract SGD
    if 'sgd_nesterov_baseline' in opts:
        new_opts['sgd'] = opts['sgd_nesterov_baseline']
        new_opts['sgd']['optimizer'] = 'sgd'
    elif 'sgd_euclidean' in opts:
        new_opts['sgd'] = opts['sgd_euclidean']
        new_opts['sgd']['optimizer'] = 'sgd'
    elif 'sgd' in opts:
        new_opts['sgd'] = opts['sgd']
        
    # Extract MUON
    if 'muon_spectral' in opts:
        new_opts['muon'] = opts['muon_spectral']
        new_opts['muon']['optimizer'] = 'muon'
    elif 'muon_nesterov' in opts:
        new_opts['muon'] = opts['muon_nesterov']
        new_opts['muon']['optimizer'] = 'muon'
    elif 'muon' in opts:
        new_opts['muon'] = opts['muon']
        
    # Extract MUON_SAM
    if 'muon_sam' in opts:
        new_opts['muon_sam'] = opts['muon_sam']
        
    # Extract ATLAS
    atlas_base = None
    if 'atlas_exp1' in opts:
        atlas_base = opts['atlas_exp1']
        atlas_base['optimizer'] = 'atlas'
    elif 'atlas' in opts:
        atlas_base = opts['atlas']
        
    if atlas_base:
        new_opts['atlas'] = copy.deepcopy(atlas_base)
        
        # Create ATLAS_RAW
        new_opts['atlas_raw'] = copy.deepcopy(atlas_base)
        new_opts['atlas_raw']['optimizer'] = 'atlas_raw'
        
        # Create ATLAS_RANDOM
        new_opts['atlas_random'] = copy.deepcopy(atlas_base)
        new_opts['atlas_random']['optimizer'] = 'atlas_random'
        
    data['optimizers'] = new_opts
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

for f in ['cifar10_cnn.yaml', 'nanogpt_fineweb.yaml', 'pythia70m_pretrain_chinchilla.yaml']:
    refactor_yaml(f'configs/{f}')
