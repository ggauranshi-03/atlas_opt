import torch
import torch.nn as nn
import time
import math
import os
import csv
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from optimizers.atlas_baseline import AtlasOptimizer
from optimizers.atlas_raw_grad import AtlasOptimizerRaw
from optimizers.atlas_random import AtlasOptimizerRandom
from utils.models import AirbenchCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(config_dict, device):
    exp = config_dict.get("experiment", {})
    task_type = exp.get("task_type", "causal_lm")
    model_name = exp.get("model_name", config_dict.get("model_name", "gpt2"))
    
    if task_type == "image_classification" or model_name == "airbench_cnn":
        print(f"  Instantiating Airbench CNN (num_classes=10)...")
        model = AirbenchCNN(num_classes=exp.get("num_classes", 10)).to(device)
        return model, "image_classification", None

    print(f"  Loading CPT model architecture: {model_name} in bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    model = model.to(device)
    return model, task_type, tokenizer

def make_optimizer(name, model, params, epochs=5, steps_per_epoch=100):
    lr = params.get("lr", 0.02)
    wd = params.get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    warmup = max(1, steps_per_epoch // 5)
    
    if name == "atlas":
        opt = AtlasOptimizer(model, lr=lr, weight_decay=wd, 
                             rho=params.get("rho", 0.015), 
                             rho_vector=params.get("rho_vector", 0.015),
                             momentum=params.get("momentum", 0.9665),
                             adam_lr=params.get("adam_lr", 0.002))
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
    elif name == "atlas_raw":
        opt = AtlasOptimizerRaw(model, lr=lr, weight_decay=wd, 
                                rho=params.get("rho", 0.015), 
                                rho_vector=params.get("rho_vector", 0.015),
                                momentum=params.get("momentum", 0.9665),
                                adam_lr=params.get("adam_lr", 0.002))
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
    elif name == "atlas_random":
        opt = AtlasOptimizerRandom(model, lr=lr, weight_decay=wd, 
                                   rho=params.get("rho", 0.015), 
                                   rho_vector=params.get("rho_vector", 0.015),
                                   momentum=params.get("momentum", 0.9665),
                                   adam_lr=params.get("adam_lr", 0.002))
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
    elif name in ["muon", "muon_nesterov", "muon_polyak"]:
        try:
            from muon import SingleDeviceMuonWithAuxAdam
            muon_params, adam_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if p.ndim >= 2 and "embed" not in n and "wte" not in n and "wpe" not in n:
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            adam_groups = [dict(params=adam_params, lr=params.get("adam_lr", 3e-4), betas=(0.9, 0.95), eps=1e-10, weight_decay=wd, use_muon=False)]
            muon_group = dict(params=muon_params, lr=lr, momentum=params.get("momentum", 0.95), weight_decay=wd, use_muon=True)
            opt = SingleDeviceMuonWithAuxAdam([*adam_groups, muon_group])
        except Exception:
            opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
    elif name == "adam":
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd, eps=1e-7)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
    elif name == "sgd":
        opt = torch.optim.SGD(trainable_params, lr=lr, momentum=params.get("momentum", 0.9), weight_decay=wd)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched

    raise ValueError(f"Unknown optimizer: {name}")

def train_epoch(model, optimizer, criterion, dataloader, task_type, grad_acc=1, steps_per_epoch=100):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    micro_batches = []
    t0 = time.time()
    
    is_atlas = isinstance(optimizer, (AtlasOptimizer, AtlasOptimizerRaw, AtlasOptimizerRandom))
    steps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if steps >= steps_per_epoch: break
        
        micro_batches.append(batch)
        if len(micro_batches) == grad_acc:
            curr_acc = len(micro_batches)
            
            if is_atlas:
                def closure():
                    nonlocal correct
                    optimizer.zero_grad()
                    c_loss = torch.tensor(0.0, device=device)
                    for mb in micro_batches:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            if task_type == "image_classification":
                                imgs = mb["image"].to(device)
                                targets = mb["label"].to(device)
                                logits = model(imgs)
                                loss = criterion(logits, targets) / curr_acc
                                preds = logits.argmax(dim=-1)
                                correct += preds.eq(targets).sum().item()
                            else:
                                ids = mb["input_ids"].to(device)
                                lbls = mb["labels"].to(device)
                                out = model(input_ids=ids, labels=lbls)
                                loss = out.loss / curr_acc
                        loss.backward()
                        c_loss = c_loss + loss.detach()
                    return c_loss
                
                loss_step = optimizer.step(closure)
                loss_val = float(loss_step) if isinstance(loss_step, (int, float)) else loss_step.item()
            else:
                optimizer.zero_grad()
                loss_val = 0.0
                for mb in micro_batches:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if task_type == "image_classification":
                            imgs = mb["image"].to(device)
                            targets = mb["label"].to(device)
                            logits = model(imgs)
                            loss = criterion(logits, targets) / curr_acc
                            preds = logits.argmax(dim=-1)
                            correct += preds.eq(targets).sum().item()
                        else:
                            ids = mb["input_ids"].to(device)
                            lbls = mb["labels"].to(device)
                            out = model(input_ids=ids, labels=lbls)
                            loss = out.loss / curr_acc
                    loss.backward()
                    loss_val += loss.item()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                for mb in micro_batches:
                    if task_type == "image_classification":
                        total += mb["label"].size(0)
                    else:
                        total += mb["input_ids"].size(0)
            total_loss += loss_val
            micro_batches.clear()
            steps += 1

    epoch_time = time.time() - t0
    num_updates = max(1, steps)
    mean_loss = total_loss / num_updates
    acc_val = (100.0 * correct / max(1, total)) if task_type == "image_classification" else math.exp(min(10.0, mean_loss))
    return mean_loss, acc_val, epoch_time

def evaluate_model(model, criterion, dataloader, task_type, max_steps=50):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if steps >= max_steps: break
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if task_type == "image_classification":
                    imgs = batch["image"].to(device)
                    targets = batch["label"].to(device)
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                    preds = logits.argmax(dim=-1)
                    correct += preds.eq(targets).sum().item()
                    total += targets.size(0)
                else:
                    ids = batch["input_ids"].to(device)
                    lbls = batch["labels"].to(device)
                    out = model(input_ids=ids, labels=lbls)
                    loss = out.loss
                    total += ids.size(0)
            total_loss += loss.item()
            steps += 1

    mean_loss = total_loss / max(1, steps)
    metric = (100.0 * correct / max(1, total)) if task_type == "image_classification" else math.exp(min(10.0, mean_loss))
    return mean_loss, metric

def run_noise_analysis(model, dataloader, config_id):
    print(f"\n[Running Heavy-Tailed Noise Analysis for {config_id}]")
    model.train()
    grad_samples = []
    p_exponent = 1.5
    
    for idx, batch in enumerate(dataloader):
        if idx >= 20: break
        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ids = batch["input_ids"].to(device)
            lbls = batch["labels"].to(device)
            out = model(input_ids=ids, labels=lbls)
            out.loss.backward()
        
        for n, p in model.named_parameters():
            if "dense" in n or "proj" in n or "mlp" in n:
                if p.grad is not None and p.ndim == 2:
                    grad_samples.append(p.grad.float().detach())
                    break

    if len(grad_samples) > 1:
        mean_g = torch.stack(grad_samples).mean(dim=0)
        frob_noise = [(g - mean_g).norm(p="fro").pow(p_exponent).item() for g in grad_samples]
        s1_noise   = [(g - mean_g).svd().S.sum().pow(p_exponent).item() for g in grad_samples]
        
        sigma_f = (sum(frob_noise) / len(frob_noise)) ** (1.0 / p_exponent)
        sigma_s1 = (sum(s1_noise) / len(s1_noise)) ** (1.0 / p_exponent)
        ratio = sigma_s1 / max(1e-8, sigma_f)
        print(f"  sigma_S1,p: {sigma_s1:.4f} | sigma_F,p: {sigma_f:.4f} | Ratio sigma_S1/sigma_F: {ratio:.2f}")
        
        os.makedirs("logs", exist_ok=True)
        csv_file = f"logs/{config_id}_logs.csv"
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "sigma_s1", "sigma_f", "noise_ratio"])
            writer.writerow([1, sigma_s1, sigma_f, ratio])
        print(f"  Noise analysis saved to {csv_file}")
    return
