import os, re

files = [
    ("exp1_atlas_baseline.py", "atlas_exp1"),
    ("exp2_raw_gradient.py", "atlas_exp2"),
    ("exp3_random_matrix.py", "atlas_exp3"),
]

for file_path, exp_name in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Imports
    if "import yaml" not in content:
        content = content.replace("import os\nimport csv", "import os\nimport csv\nimport yaml")

    # 2. Config reading
    config_block = """# ─────────────────────────────── Config ───────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

MODEL_NAME     = CONFIG["experiment"]["model_name"]
DATASET_NAME   = CONFIG["experiment"]["dataset_name"]
DATASET_CONFIG = CONFIG["experiment"]["dataset_config"]
NUM_LABELS     = CONFIG["experiment"]["num_labels"]
MAX_LEN        = CONFIG["experiment"]["max_len"]
BATCH_SIZE     = CONFIG["experiment"]["batch_size"]
GRAD_ACC       = CONFIG["experiment"]["grad_acc"]
TRAIN_SUBSET   = CONFIG["experiment"]["train_subset"]
"""
    content = re.sub(r"# ─────────────────────────────── Config ───────────────────────────────────────.*?# ─────────────────────────────── Data ─────────────────────────────────────────", 
                     config_block + "\n# ─────────────────────────────── Data ─────────────────────────────────────────", 
                     content, flags=re.DOTALL)

    # 3. Remove Optuna objective & run_tuning functions entirely
    content = re.sub(r"# ─────────────────────────────── Hyperparameter tuning ────────────────────────.*?# ─────────────────────────────── Full experiment ──────────────────────────────",
                     "# ─────────────────────────────── Full experiment ──────────────────────────────", 
                     content, flags=re.DOTALL)

    # 4. Modify main() function loop and best config
    main_tuning_block = r"    # ── 1. Hyperparameter Tuning ──.*?# ── 2. Final Training ──"
    
    if exp_name == "atlas_exp1":
        new_tuning_block = """    # ── 1. Setup Hyperparameters ──
    best = {
        "atlas": CONFIG["hyperparameters"]["atlas_exp1"],
        "adam": CONFIG["hyperparameters"]["adam"],
        "sgd": CONFIG["hyperparameters"]["sgd"],
        "muon": CONFIG["hyperparameters"]["muon"]
    }

    # ── 2. Final Training ──"""
        optimizers_list = "OPTIMIZERS_TO_TEST = [\"atlas\", \"adam\", \"sgd\", \"muon\"]"
    elif exp_name == "atlas_exp2":
        new_tuning_block = """    # ── 1. Setup Hyperparameters ──
    best = {
        "atlas": CONFIG["hyperparameters"]["atlas_exp2"]
    }

    # ── 2. Final Training ──"""
        optimizers_list = "OPTIMIZERS_TO_TEST = [\"atlas\"]"
    elif exp_name == "atlas_exp3":
        new_tuning_block = """    # ── 1. Setup Hyperparameters ──
    best = {
        "atlas": CONFIG["hyperparameters"]["atlas_exp3"]
    }

    # ── 2. Final Training ──"""
        optimizers_list = "OPTIMIZERS_TO_TEST = [\"atlas\"]"

    content = re.sub(main_tuning_block, new_tuning_block, content, flags=re.DOTALL)

    # 5. Fix OPTIMIZERS_TO_TEST
    content = re.sub(r"OPTIMIZERS_TO_TEST\s*=\s*\[.*?\]", optimizers_list, content)

    # 6. Fix WANDB_PROJECT and TRAIN_EPOCHS
    content = re.sub(r"TRAIN_EPOCHS\s*=\s*\d+", "TRAIN_EPOCHS = CONFIG[\"experiment\"][\"train_epochs\"]", content)
    content = re.sub(r"WANDB_PROJECT\s*=\s*\".*?\"", "WANDB_PROJECT = CONFIG[\"experiment\"][\"wandb_project\"]", content)
    content = re.sub(r"TUNING_TRIALS\s*=\s*\d+\n", "", content)

    # 7. Make the CSV logs output to logs/
    content = content.replace("csv_file = f\"{optimizer_name}_logs.csv\"", f"os.makedirs(\"logs\", exist_ok=True)\n        csv_file = f\"logs/{{optimizer_name}}_{exp_name}_logs.csv\"")
    content = content.replace("wandb_project=\"Hybrid-LLM-ZFS\"", "wandb_project=\"Atlas-Experiments\"")
    
    # 8. Remove the stray optuna import if it exists at the top
    content = content.replace("import optuna\n", "")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
print("done")
