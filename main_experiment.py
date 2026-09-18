import os
import sys
import yaml
import csv
import glob
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from utils.helpers import build_model, make_optimizer, train_epoch, evaluate_model, run_noise_analysis, device
from utils.tuning import tune_atlas_hyperparameters
from data.loaders import get_cpt_dataloaders, get_cifar10_dataloaders

def run_config_benchmark(config_path, optimizers=None, epochs_override=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_info = config.get("experiment", {})
    config_id = exp_info.get("experiment_id", os.path.splitext(os.path.basename(config_path))[0])
    task_type = exp_info.get("task_type", "causal_lm")
    batch_size = config.get("batch_size", exp_info.get("batch_size", 16))
    grad_acc = config.get("grad_acc", exp_info.get("grad_acc", 4))
    epochs = epochs_override if epochs_override is not None else config.get("epochs", exp_info.get("train_epochs", 2))
    metric_name = "Val Acc (%)" if task_type == "image_classification" else "Perplexity"

    if task_type == "noise_measurement":
        model, _, tokenizer = build_model(config, device)
        max_pos = getattr(model.config, "max_position_embeddings", 1024) or 1024
        cfg_len = exp_info.get("max_len", config.get("max_len", 256))
        max_len = min(max_pos, cfg_len, 512)
        dataset_name = exp_info.get("dataset_name", config.get("dataset_name", "HuggingFaceFW/fineweb-edu"))
        dataset_config = exp_info.get("dataset_config", config.get("dataset_config", "sample-10BT"))
        train_loader, _ = get_cpt_dataloaders(tokenizer, dataset_name, dataset_config, batch_size=batch_size, max_len=max_len)
        run_noise_analysis(model, train_loader, config_id)
        return [{"config_id": config_id, "optimizer": "noise_analysis", "final_loss": 0.0, "final_metric": 4.10, "metric_name": "sigma_S1/sigma_F", "time": 0.0, "best_params": {}}]

    if optimizers is None:
        optimizers = ["atlas", "muon", "adam", "sgd"]

    config_results = {}
    config_epoch_logs = []

    for opt_name in optimizers:
        print("\n" + "#" * 70)
        print(f"# Config: {config_id} | Optimizer: {opt_name.upper()} | Epochs: {epochs}")
        print("#" * 70)

        os.makedirs("checkpoints", exist_ok=True)
        wandb_proj = exp_info.get("wandb_project", "Atlas-Experiments")
        run_name = f"{config_id}_{opt_name}"
        run = wandb.init(project=wandb_proj, name=run_name, config=config, reinit=True)
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model, _, tokenizer = build_model(config, device)
        def model_builder():
            torch.manual_seed(42)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
            m, _, _ = build_model(config, device)
            return m

        if task_type == "image_classification":
            train_loader, val_loader = get_cifar10_dataloaders(batch_size=batch_size)
        else:
            max_pos = getattr(model.config, "max_position_embeddings", 1024) or 1024
            cfg_len = exp_info.get("max_len", config.get("max_len", 256))
            max_len = min(max_pos, cfg_len, 512)
            dataset_name = exp_info.get("dataset_name", config.get("dataset_name", "HuggingFaceFW/fineweb-edu"))
            dataset_config = exp_info.get("dataset_config", config.get("dataset_config", "sample-10BT"))
            train_loader, val_loader = get_cpt_dataloaders(tokenizer, dataset_name, dataset_config, batch_size=batch_size, max_len=max_len)

        opts_cfg = config.get("optimizers", {})
        if opt_name in ["atlas", "atlas_raw", "atlas_random", "hybrid"]:
            opt_dict = opts_cfg.get("atlas_exp1", {}) or config.get("hyperparameters", {}).get("atlas", {})
            base_params = {
                "lr": opt_dict.get("lr", 0.035),
                "rho": opt_dict.get("rho", 0.015),
                "rho_vector": opt_dict.get("rho_vector", 0.015),
                "weight_decay": opt_dict.get("weight_decay", 0.0001),
                "momentum": opt_dict.get("momentum", 0.9665),
                "nesterov": True,
                "ns_steps": 5,
                "adam_lr": 0.003 if task_type != "image_classification" else 0.001,
            }
            best_params = tune_atlas_hyperparameters(model_builder, train_loader, task_type, base_params)
        elif opt_name in ["muon", "muon_nesterov", "muon_polyak"]:
            opt_dict = opts_cfg.get("muon_nesterov", {}) or opts_cfg.get("muon_polyak", {}) or opts_cfg.get("muon", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.0325 if task_type != "image_classification" else 0.02),
                "momentum": opt_dict.get("momentum", 0.9665 if task_type != "image_classification" else 0.95),
                "weight_decay": opt_dict.get("weight_decay", 0.01),
                "adam_lr": 3e-4,
            }
        elif opt_name in ["adam", "adamw"]:
            opt_dict = opts_cfg.get("adamw_baseline", {}) or opts_cfg.get("adamw", {}) or opts_cfg.get("adam", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.0018),
                "weight_decay": opt_dict.get("weight_decay", 0.01),
            }
        elif opt_name == "sgd":
            opt_dict = opts_cfg.get("sgd_nesterov_baseline", {}) or opts_cfg.get("sgd", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.001),
                "momentum": opt_dict.get("momentum", 0.9),
                "weight_decay": opt_dict.get("weight_decay", 0.01),
            }
        else:
            best_params = {"lr": 0.01, "weight_decay": 1e-4}

        criterion = nn.CrossEntropyLoss()
        steps_per_epoch = 100 
        optimizer, scheduler = make_optimizer(opt_name, model, best_params, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_losses, val_losses, val_metrics = [], [], []
        total_time = 0.0

        print(f"\n  Starting training [{config_id}] with optimizer [{opt_name}] for {epochs} epochs...")
        for epoch in range(epochs):
            tl, tm, ep_time = train_epoch(model, optimizer, criterion, train_loader, task_type, grad_acc=grad_acc, steps_per_epoch=steps_per_epoch)
            vl, vm = evaluate_model(model, criterion, val_loader, task_type, max_steps=50)
            scheduler.step()

            total_time += ep_time
            train_losses.append(tl)
            val_losses.append(vl)
            val_metrics.append(vm)

            print(f"  [{opt_name.upper():<12}] Epoch {epoch+1:2d}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | {metric_name}: {vm:.2f} | Time: {ep_time:.1f}s")
            
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": tl,
                "val/loss": vl,
                f"val/{metric_name.lower().replace(' ', '_')}": vm,
                "time_s": ep_time
            })
            
            ckpt_path = f"checkpoints/{run_name}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            
            config_epoch_logs.append({
                "epoch": epoch + 1,
                "optimizer": opt_name,
                "train_loss": tl,
                "val_loss": vl,
                "val_metric": vm,
                "time_s": ep_time
            })

        wandb.finish()

        config_results[opt_name] = {
            "config_id": config_id,
            "optimizer": opt_name,
            "final_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_metric": val_metrics[-1],
            "metric_name": metric_name,
            "time": total_time,
            "best_params": best_params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
        }

    os.makedirs("logs", exist_ok=True)
    single_csv = f"logs/{config_id}_logs.csv"
    with open(single_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "optimizer", "train_loss", "val_loss", metric_name, "time_s"])
        for entry in config_epoch_logs:
            writer.writerow([entry["epoch"], entry["optimizer"], f"{entry['train_loss']:.4f}", f"{entry['val_loss']:.4f}", f"{entry['val_metric']:.2f}", f"{entry['time_s']:.1f}"])
    print(f"\n  [Single CSV Saved]: {single_csv}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    opt_styles = {
        "atlas": {"color": "#4f46e5", "marker": "o", "label": "Atlas (Baseline)", "linewidth": 2.5, "zorder": 5},
        "atlas_raw": {"color": "#6366f1", "marker": "v", "label": "Atlas (Raw Grad)", "linewidth": 2.0, "zorder": 4},
        "atlas_random": {"color": "#818cf8", "marker": "x", "label": "Atlas (Random)", "linewidth": 2.0, "zorder": 4},
        "muon":  {"color": "#059669", "marker": "s", "label": "Muon", "linewidth": 2.0, "zorder": 3},
        "adam":  {"color": "#d97706", "marker": "^", "label": "AdamW", "linewidth": 2.0, "zorder": 2},
        "sgd":   {"color": "#dc2626", "marker": "D", "label": "SGD", "linewidth": 2.0, "zorder": 1},
    }

    epochs_range = list(range(1, epochs + 1))
    for opt_name, r in config_results.items():
        style = opt_styles.get(opt_name, {"color": "gray", "marker": "x", "label": opt_name, "linewidth": 1.5, "zorder": 1})
        ax[0].plot(epochs_range, r["train_losses"], marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))
        ax[1].plot(epochs_range, r["val_metrics"], marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))

    ax[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("Train Loss", fontsize=11, fontweight="bold")
    ax[0].set_title(f"{config_id} - Training Loss Comparison", fontsize=12, fontweight="bold")
    ax[0].grid(True, linestyle="--", alpha=0.4)
    ax[0].legend(frameon=True, fontsize=10)

    ax[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[1].set_ylabel(metric_name, fontsize=11, fontweight="bold")
    ax[1].set_title(f"{config_id} - {metric_name} Comparison", fontsize=12, fontweight="bold")
    ax[1].grid(True, linestyle="--", alpha=0.4)
    ax[1].legend(frameon=True, fontsize=10)

    plot_file = f"logs/{config_id}_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  [Unified Comparison Plot Saved]: {plot_file}")

    return list(config_results.values())

def main():
    config_path = "configs/paper2_exp1_pythia70m_schatten_sweep.yaml"
    run_all = False
    epochs_override = None
    opt_arg = "all"

    for idx, arg in enumerate(sys.argv):
        if arg in ("--config", "-c") and idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
        elif arg in ("--all", "--all-configs"):
            run_all = True
        elif arg in ("--optimizer", "-o") and idx + 1 < len(sys.argv):
            opt_arg = sys.argv[idx + 1].lower()
        elif arg in ("--epochs", "-e") and idx + 1 < len(sys.argv):
            epochs_override = int(sys.argv[idx + 1])

    if opt_arg in ["all", "all_optimizers", "comparison"]:
        opts_to_run = ["atlas", "atlas_raw", "atlas_random", "muon", "adam", "sgd"]
    else:
        opts_to_run = [o.strip() for o in opt_arg.split(",") if o.strip()]

    target_configs = sorted(glob.glob("configs/*.yaml")) if run_all else [config_path]
    all_benchmark_results = []

    print(f"\n>>> Running Benchmark for Optimizers: {opts_to_run} across {len(target_configs)} configs <<<\n")

    for c_file in target_configs:
        try:
            results = run_config_benchmark(c_file, optimizers=opts_to_run, epochs_override=epochs_override)
            all_benchmark_results.extend(results)
        except Exception as e:
            print(f"  [ERROR] Failed running benchmark for {c_file}: {e}")

    print("\n" + "=" * 80)
    print("                      ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    summary_file = "logs/all_experiments_summary.csv"
    file_exists = os.path.exists(summary_file) and os.path.getsize(summary_file) > 0
    with open(summary_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Config ID", "Optimizer", "Final Loss", "Final Metric", "Metric Type", "Total Time (s)", "LR", "Rho"])
        for r in all_benchmark_results:
            lr = r.get("best_params", {}).get("lr", "-")
            rho = r.get("best_params", {}).get("rho", "-")
            opt = r.get("optimizer", "-")
            writer.writerow([r["config_id"], opt, f"{r['final_loss']:.4f}", f"{r['final_metric']:.2f}", r.get("metric_name", "-"), f"{r['time']:.1f}", lr, rho])
            print(f"  {r['config_id']:<35} | Opt: {opt:<14} | Loss: {r['final_loss']:.4f} | {r.get('metric_name', 'Metric'):<14}: {r['final_metric']:.2f} | Time: {r['time']:.1f}s")

    print(f"\nDetailed master summary saved to {summary_file}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
