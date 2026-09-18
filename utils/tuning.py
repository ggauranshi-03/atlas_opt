import torch
import torch.nn as nn
import math
from utils.helpers import make_optimizer, device

def tune_atlas_hyperparameters(build_model_fn, train_loader, task_type, base_params, num_trials=4):
    print("\n" + "=" * 60)
    print("  [Atlas Hyperparameter Tuning] Finding Optimal (lr, rho, adam_lr)")
    print("=" * 60)
    
    if task_type == "image_classification":
        candidates = [
            {"lr": 0.035, "rho": 0.006, "weight_decay": 0.0001, "momentum": 0.95, "adam_lr": 0.001},
            {"lr": 0.040, "rho": 0.008, "weight_decay": 0.0001, "momentum": 0.95, "adam_lr": 0.001},
            {"lr": 0.030, "rho": 0.005, "weight_decay": 0.0001, "momentum": 0.95, "adam_lr": 0.001},
            {"lr": 0.045, "rho": 0.008, "weight_decay": 0.0001, "momentum": 0.95, "adam_lr": 0.001},
        ][:num_trials]
    else:
        candidates = [
            {"lr": 0.035, "rho": 0.008, "weight_decay": 0.0001, "momentum": 0.9665, "adam_lr": 0.003},
            {"lr": 0.038, "rho": 0.006, "weight_decay": 0.0001, "momentum": 0.9665, "adam_lr": 0.003},
            {"lr": 0.0325, "rho": 0.005, "weight_decay": 0.0001, "momentum": 0.9665, "adam_lr": 0.003},
            {"lr": 0.040, "rho": 0.008, "weight_decay": 0.0001, "momentum": 0.9665, "adam_lr": 0.003},
        ][:num_trials]
    
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_params = dict(base_params)

    for idx, cand in enumerate(candidates):
        trial_params = {**base_params, **cand}
        model = build_model_fn()
        model.train()
        opt, sched = make_optimizer("atlas", model, trial_params, epochs=1, steps_per_epoch=20)
        
        last_loss = float("inf")
        steps = 0
        
        for b_idx, batch in enumerate(train_loader):
            if steps >= 16: break
            
            def closure():
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if task_type == "image_classification":
                        out = model(batch["image"].to(device))
                        loss = criterion(out, batch["label"].to(device))
                    else:
                        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                        loss = out.loss
                loss.backward()
                return loss

            loss = opt.step(closure)
            sched.step()
            last_loss = loss.item() if hasattr(loss, 'item') else float(loss)
            steps += 1
            
        print(f"  Trial {idx+1}/{len(candidates)}: lr={cand['lr']:<6} rho={cand['rho']:<6} adam_lr={cand.get('adam_lr', 0.002)} -> End Loss: {last_loss:.4f}")
        
        if last_loss < best_loss and not math.isnan(last_loss):
            best_loss = last_loss
            best_params = trial_params

    print(f"\n  [Optimal Atlas Config Selected]: lr={best_params['lr']}, rho={best_params['rho']}, adam_lr={best_params.get('adam_lr', 0.002)}")
    print("=" * 60 + "\n")
    return best_params
