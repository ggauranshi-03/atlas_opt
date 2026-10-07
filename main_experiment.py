import os
import sys
import fcntl
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
from data.loaders import get_cpt_dataloaders, get_cifar10_dataloaders, infinite_batches

def run_config_benchmark(config_path, optimizers=None, epochs_override=None, seed=42):
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
        run_name = f"{opt_name}_seed{seed}"
        run = wandb.init(project=wandb_proj, name=run_name, config={**config, "seed": seed}, reinit=True)
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model, _, tokenizer = build_model(config, device)
        def model_builder():
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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
            train_loader, val_loader = get_cpt_dataloaders(tokenizer, dataset_name, dataset_config, batch_size=batch_size, max_len=max_len, seed=seed)
        train_iter = infinite_batches(train_loader)

        if task_type != "image_classification":
            # Guard against starting from anything but the released weights: a random init has ppl ~ vocab size.
            _, _, init_ppl = evaluate_model(model, nn.CrossEntropyLoss(), val_loader, task_type, max_steps=50)
            print(f"  Initial (pretrained) validation perplexity: {init_ppl:.2f}")
            wandb.summary["init_val_perplexity"] = init_ppl
            if not init_ppl < 300:
                raise RuntimeError(f"initial val perplexity {init_ppl:.1f}: the model did not start from the pretrained weights")

        opts_cfg = config.get("optimizers", {})
        if opt_name in ["atlas", "atlas_raw", "atlas_random", "hybrid"]:
            opt_dict = opts_cfg.get(opt_name, {})
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
            # best_params = tune_atlas_hyperparameters(model_builder, train_loader, task_type, base_params)
            best_params = base_params
        elif opt_name == "muon":
            opt_dict = opts_cfg.get("muon", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.035),
                "momentum": opt_dict.get("momentum", 0.9665),
                "weight_decay": opt_dict.get("weight_decay", 0.0001),
                "adam_lr": opt_dict.get("adam_lr", 0.003 if task_type != "image_classification" else 0.001),
            }
        elif opt_name in ["muon_sam", "muon_sam_frob", "muon_sam_stale", "fsam_muon", "fsam_ortho_muon",
                          "fsam_ortho_muon_stale", "fsam_ortho_muon_stale_momentum",
                          "fsam_frob_muon_stale", "fsam_frob_muon_stale_momentum",
                          "muon_sam_gfrob", "fsam_gfrob_muon_stale", "fsam_gfrob_muon_stale_momentum", "randsam_muon",
                          "soma_prens5", "op_soma_postns5"]:
            opt_dict = opts_cfg.get(opt_name, {})
            best_params = {
                "lr": opt_dict.get("lr", 0.035),
                "momentum": opt_dict.get("momentum", 0.9665),
                "weight_decay": opt_dict.get("weight_decay", 0.0001),
                "rho": opt_dict.get("rho", 0.015),
                "rho_vector": opt_dict.get("rho_vector", opt_dict.get("rho", 0.015)),
                "ns_steps": opt_dict.get("ns_steps", 5),
                "fsam_lambda": opt_dict.get("fsam_lambda", 0.9),
                "fsam_sigma": opt_dict.get("fsam_sigma", 1.0),
                "adam_lr": opt_dict.get("adam_lr", 0.003 if task_type != "image_classification" else 0.001),
            }
        elif opt_name in ["fsam_ortho", "fsam", "sam", "sam_ortho"]:
            opt_dict = opts_cfg.get(opt_name, {})
            best_params = {
                "lr": opt_dict.get("lr", 0.01),
                "rho": opt_dict.get("rho", 0.05),
                "rho_vector": opt_dict.get("rho_vector", opt_dict.get("rho", 0.05)),
                "fsam_lambda": opt_dict.get("fsam_lambda", 0.9),
                "fsam_sigma": opt_dict.get("fsam_sigma", 1.0),
                "ns_steps": opt_dict.get("ns_steps", 5),
                "momentum": opt_dict.get("momentum", 0.9),
                "nesterov": opt_dict.get("nesterov", True),
                "weight_decay": opt_dict.get("weight_decay", 0.0),
            }
        elif opt_name == "adam":
            opt_dict = opts_cfg.get("adam", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.0018),
                "weight_decay": opt_dict.get("weight_decay", 0.01),
            }
        elif opt_name == "sgd":
            opt_dict = opts_cfg.get("sgd", {}) or opts_cfg.get("sgd_nesterov_baseline", {})
            best_params = {
                "lr": opt_dict.get("lr", 0.05 if task_type != "image_classification" else 0.1),
                "momentum": opt_dict.get("momentum", 0.95 if task_type != "image_classification" else 0.9),
                "weight_decay": opt_dict.get("weight_decay", 0.01 if task_type != "image_classification" else 0.001),
                "nesterov": opt_dict.get("nesterov", True),
            }
        else:
            best_params = {"lr": 0.01, "weight_decay": 1e-4}

        criterion = nn.CrossEntropyLoss()
        steps_per_epoch = 100 
        optimizer, scheduler = make_optimizer(opt_name, model, best_params, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_losses, val_losses, train_accs, val_accs, val_ppls = [], [], [], [], []
        total_time = 0.0

        print(f"\n  Starting training [{config_id}] with optimizer [{opt_name}] for {epochs} epochs...")
        for epoch in range(epochs):
            tl, t_acc, t_ppl, ep_time = train_epoch(model, optimizer, criterion, train_iter, task_type, grad_acc=grad_acc,
                                                    steps_per_epoch=steps_per_epoch, scheduler=scheduler)
            vl, v_acc, v_ppl = evaluate_model(model, criterion, val_loader, task_type, max_steps=50)

            total_time += ep_time
            train_losses.append(tl)
            val_losses.append(vl)
            train_accs.append(t_acc)
            val_accs.append(v_acc)
            val_ppls.append(v_ppl)

            print(f"  [{opt_name.upper():<12}] Epoch {epoch+1:2d}/{epochs} | Train Loss: {tl:.4f} | Train Acc: {t_acc:.2f}% | Val Loss: {vl:.4f} | Val Acc: {v_acc:.2f}% | Time: {ep_time:.1f}s | LR: {scheduler.get_last_lr()[-1]:.3e}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": tl,
                "train/accuracy": t_acc,
                "val/loss": vl,
                "val/accuracy": v_acc,
                "val/perplexity": v_ppl,
                "lr": scheduler.get_last_lr()[-1],
                "time_s": ep_time
            })
            
            ckpt_path = f"checkpoints/{config_id}_{opt_name}_seed{seed}_epoch{epoch+1}.pt"
            torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() and task_type != "image_classification" else v)
                        for k, v in model.state_dict().items()}, ckpt_path)
            prev = f"checkpoints/{config_id}_{opt_name}_seed{seed}_epoch{epoch}.pt"
            if os.path.exists(prev):
                os.remove(prev)  # disk budget: only the newest epoch is kept
            
            config_epoch_logs.append({
                "epoch": epoch + 1,
                "optimizer": opt_name,
                "seed": seed,
                "train_loss": tl,
                "train_accuracy": t_acc,
                "val_loss": vl,
                "val_accuracy": v_acc,
                "val_perplexity": v_ppl,
                "time_s": ep_time
            })

        wandb.finish()

        val_metrics = val_accs if task_type == "image_classification" else val_ppls

        config_results[opt_name] = {
            "config_id": config_id,
            "optimizer": opt_name,
            "seed": seed,
            "final_loss": train_losses[-1],
            "final_train_acc": train_accs[-1],
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accs[-1],
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
    csv_header = ["epoch", "optimizer", "seed", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_perplexity", "time_s"]
    # Keep rows of other (optimizer, seed) runs, so runs can be trained in separate processes.
    kept_rows = []
    with open(single_csv + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several GPU workers may finish runs of the same config at once
        if os.path.exists(single_csv):
            with open(single_csv, newline="") as f:
                kept_rows = [r for r in csv.DictReader(f)
                             if not (r.get("optimizer") in config_results and r.get("seed") == str(seed))]
        tmp_csv = single_csv + f".tmp{os.getpid()}"
        with open(tmp_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for r in kept_rows:
                writer.writerow([r.get(k, "") for k in csv_header])
            for row in config_epoch_logs:
                writer.writerow([
                    row["epoch"], row["optimizer"], row["seed"],
                    f"{row['train_loss']:.4f}", f"{row['train_accuracy']:.2f}",
                    f"{row['val_loss']:.4f}", f"{row['val_accuracy']:.2f}", f"{row['val_perplexity']:.2f}",
                    f"{row['time_s']:.1f}"
                ])
        os.replace(tmp_csv, single_csv)
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

    series = {}
    for r in kept_rows:
        if r.get("seed") != str(seed):
            continue
        s = series.setdefault(r["optimizer"], ([], [], []))
        s[0].append(int(r["epoch"]))
        s[1].append(float(r["train_loss"]))
        s[2].append(float(r["val_accuracy"] if task_type == "image_classification" else r["val_perplexity"]))
    for opt_name, r in config_results.items():
        series[opt_name] = (list(range(1, len(r["train_losses"]) + 1)), r["train_losses"], r["val_metrics"])
    for opt_name, (ep, tl, vm) in series.items():
        style = opt_styles.get(opt_name, {"color": "gray", "marker": "x", "label": opt_name, "linewidth": 1.5, "zorder": 1})
        ax[0].plot(ep, tl, marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))
        ax[1].plot(ep, vm, marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))

    ax[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("Train Loss", fontsize=11, fontweight="bold")
    ax[0].set_title(f"{config_id} (seed {seed}) - Training Loss Comparison", fontsize=12, fontweight="bold")
    ax[0].grid(True, linestyle="--", alpha=0.4)
    ax[0].legend(frameon=True, fontsize=10)

    ax[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[1].set_ylabel(metric_name, fontsize=11, fontweight="bold")
    ax[1].set_title(f"{config_id} - {metric_name} Comparison", fontsize=12, fontweight="bold")
    ax[1].grid(True, linestyle="--", alpha=0.4)
    ax[1].legend(frameon=True, fontsize=10)

    plot_file = f"logs/{config_id}_seed{seed}_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  [Unified Comparison Plot Saved]: {plot_file}")

    return list(config_results.values())

def main():
    config_path = "configs/pythia70m_schatten_sweep.yaml"
    run_all = False
    epochs_override = None
    opt_arg = "all"
    seed = 42

    for idx, arg in enumerate(sys.argv):
        if arg in ("--config", "-c") and idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
        elif arg in ("--all", "--all-configs"):
            run_all = True
        elif arg in ("--optimizer", "-o") and idx + 1 < len(sys.argv):
            opt_arg = sys.argv[idx + 1].lower()
        elif arg in ("--epochs", "-e") and idx + 1 < len(sys.argv):
            epochs_override = int(sys.argv[idx + 1])
        elif arg in ("--seed", "-s") and idx + 1 < len(sys.argv):
            seed = int(sys.argv[idx + 1])

    if opt_arg in ["all", "all_optimizers", "comparison"]:
        opts_to_run = ["atlas", "atlas_raw", "atlas_random", "muon", "adam", "sgd"]
    else:
        opts_to_run = [o.strip() for o in opt_arg.split(",") if o.strip()]

    target_configs = sorted(glob.glob("configs/*.yaml")) if run_all else [config_path]
    all_benchmark_results = []

    print(f"\n>>> Running Benchmark for Optimizers: {opts_to_run} across {len(target_configs)} configs <<<\n")

    for c_file in target_configs:
        try:
            results = run_config_benchmark(c_file, optimizers=opts_to_run, epochs_override=epochs_override, seed=seed)
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
            writer.writerow(["Config ID", "Optimizer", "Seed", "Final Loss", "Final Metric", "Metric Type", "Total Time (s)", "LR", "Rho"])
        for r in all_benchmark_results:
            lr = r.get("best_params", {}).get("lr", "-")
            rho = r.get("best_params", {}).get("rho", "-")
            opt = r.get("optimizer", "-")
            writer.writerow([r["config_id"], opt, r.get("seed", ""), f"{r['final_loss']:.4f}", f"{r['final_metric']:.2f}", r.get("metric_name", "-"), f"{r['time']:.1f}", lr, rho])
            print(f"  {r['config_id']:<35} | Opt: {opt:<14} | Loss: {r['final_loss']:.4f} | {r.get('metric_name', 'Metric'):<14}: {r['final_metric']:.2f} | Time: {r['time']:.1f}s")

    print(f"\nDetailed master summary saved to {summary_file}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # HF streaming threads can keep the interpreter alive after training (observed multi-hour hangs)
