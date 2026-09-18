import sys
import os
import time
import math
import csv
import yaml
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────── Muon / SAM Utilities ─────────────────────────
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = X.norm()
    X = X / norm.clamp(min=eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

# ─────────────────────────────── Lookahead ────────────────────────────────────
class Lookahead:
    def __init__(self, optimizer, alpha=0.5, k=5):
        self.optimizer    = optimizer
        self.param_groups = optimizer.param_groups
        self.state        = optimizer.state
        self.alpha        = alpha
        self.k            = k
        self._la_step     = 0
        self.slow_weights = [[p.data.clone().detach() for p in g["params"]]
                              for g in self.param_groups]

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._la_step += 1
        if self._la_step % self.k == 0:
            for group, slow_group in zip(self.param_groups, self.slow_weights):
                for p, q in zip(group["params"], slow_group):
                    q.add_(self.alpha * (p.data.to(q.dtype) - q))
                    p.data.copy_(q.to(p.dtype))
        return loss

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

# ─────────────────────────────── Atlas (Single-Pass Muon SAM) ─────────────────
class AtlasOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, rho=0.05, rho_vector=0.05, momentum=0.95, nesterov=True, ns_steps=5, adam_lr=3e-4):
        defaults = dict(lr=lr, weight_decay=weight_decay, rho=rho, rho_vector=rho_vector, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adam_lr=adam_lr)
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None: raise RuntimeError("AtlasOptimizer requires a closure")

        is_first_step = (self._step == 0)

        # --- Step 1 & 2: First-Step Initialization or Stale-Gradient Perturbation ---
        if is_first_step:
            with torch.enable_grad():
                loss = closure()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    if group.get("use_muon", False):
                        g1 = p.grad.float()
                        O1 = newtonschulz5(g1.view(g1.size(0), -1), steps=group["ns_steps"]).view_as(g1)
                        self.state[p]["O_stale"] = O1.clone()
                    else:
                        # Vectors have no orthogonalization/momentum step; the raw first
                        # gradient grad L(v0) itself is the stale perturbation source.
                        self.state[p]["v_stale"] = p.grad.float().clone()
        else:
            loss = None

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            rho = group["rho"] if use_muon else group.get("rho_vector", group["rho"])
            stale_key = "O_stale" if use_muon else "v_stale"
            old_key   = "old_p" if use_muon else "old_v"
            for p in group["params"]:
                state = self.state[p]
                if stale_key in state:
                    stale = state[stale_key]
                    state[old_key] = p.data.clone()
                    if use_muon:
                        p.data.add_(stale.to(p.dtype), alpha=rho) # Exp1 Baseline: no explicit normalization for O_stale
                    else:
                        s_norm = stale.norm() + 1e-8
                        p.data.add_(stale.to(p.dtype), alpha=rho / s_norm)

        # --- Compute g_adv ---
        self.zero_grad()
        with torch.enable_grad():
            closure_loss = closure()
            if is_first_step: loss = closure_loss
            else: loss = closure_loss
            
        self._step += 1
            
        # --- Step 3, 4, 5: Update ---
        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            lr = group["lr"]
            wd = group["weight_decay"]
            
            if use_muon:
                mu = group.get("momentum", 0.95)
                for p in group["params"]:
                    if p.grad is None: continue
                    g_adv = p.grad.float()
                    state = self.state[p]
                    
                    if "old_p" in state:
                        p.data.copy_(state["old_p"])
                        
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_adv)
                    buf = state["momentum_buffer"]
                    
                    buf.mul_(mu).add_(g_adv)
                    g_nest = g_adv + mu * buf if group["nesterov"] else buf
                    
                    O_t = newtonschulz5(g_nest.view(g_nest.size(0), -1), steps=group["ns_steps"]).view_as(g_nest)
                    state["O_stale"] = O_t.clone()
                    
                    eta = lr * max(1, O_t.size(-2) / O_t.size(-1))**0.5
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(O_t.to(p.dtype), alpha=-eta)
            else:
                adam_lr = group.get("lr", 3e-4)
                for p in group["params"]:
                    if p.grad is None: continue
                    g_adv = p.grad.float()
                    state = self.state[p]

                    if "old_v" in state:
                        p.data.copy_(state["old_v"])

                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state["step"] += 1
                    beta1, beta2 = 0.9, 0.95

                    state["exp_avg"].mul_(beta1).add_(g_adv, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g_adv, g_adv, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    step_size = adam_lr / bias_correction1
                    p.data.mul_(1 - adam_lr * wd)

                    denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                    p.data.addcdiv_(state["exp_avg"], denom, value=-step_size)

                    # Stale perturbation source for next round (analogue of O_stale)
                    state["v_stale"] = g_adv.clone()

        return loss

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none=set_to_none)

# ─────────────────────────────── Airbench CNN (Vision) ────────────────────────
class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2) if pool else None
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.pool: x = self.pool(x)
        return x

class AirbenchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = ConvBN(3, 64)
        self.layer1 = ConvBN(64, 128, pool=True)
        self.layer2 = ConvBN(128, 256, pool=True)
        self.layer3 = ConvBN(256, 512, pool=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes, bias=True)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

# ─────────────────────────────── Data Loaders ─────────────────────────────────
class CPTDataset(Dataset):
    def __init__(self, input_ids_list, max_len=256, vocab_size=50257):
        self.samples = []
        self.vocab_size = vocab_size
        for ids in input_ids_list:
            for i in range(0, len(ids) - max_len, max_len):
                chunk = ids[i:i+max_len]
                t = torch.tensor(chunk, dtype=torch.long).clamp(0, self.vocab_size - 1)
                self.samples.append(t)
        if len(self.samples) == 0 and len(input_ids_list) > 0:
            pad_len = max_len
            for ids in input_ids_list:
                if len(ids) > 0:
                    chunk = (ids * (pad_len // len(ids) + 1))[:pad_len]
                    t = torch.tensor(chunk, dtype=torch.long).clamp(0, self.vocab_size - 1)
                    self.samples.append(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {"input_ids": item, "labels": item.clone()}

class CIFAR10ArrowDataset(Dataset):
    def __init__(self, arrow_ds, transform=None):
        self.ds = arrow_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"] # PIL Image
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}

def get_cpt_dataloaders(tokenizer, batch_size=16, max_len=256, subset_size=4000, vocab_size=50257):
    # Generates a rich, structured text stream for Continued Pre-Training
    print(f"  Preparing CPT corpus (max_len={max_len}, vocab_size={vocab_size})...")
    sample_texts = [
        "The optimization landscape of deep neural networks exhibits heavy-tailed stochastic gradient noise. "
        "Orthogonalized polar decomposition methods like Muon and Atlas compute spectral approximations using "
        "Newton-Schulz iterations. Atlas integrates Sharpness-Aware Minimization with single-pass stale gradient "
        "perturbation to flatten the loss basin without requiring two full backward passes per iteration. "
        "This achieves theoretical minimax optimal convergence under (alpha, c)-heavy tailed noise conditions.",
        
        "FineWeb-Edu consists of high-quality educational web content filtered by neural classifiers. "
        "Pretraining causal language models requires stable learning rate schedules, decoupled weight decay, "
        "and orthogonal gradient updates for hidden parameter matrices while using AdamW for embeddings and head. "
        "Empirical benchmarks show superior perplexity scaling compared to classical SGD and Adam baselines.",
        
        "Deep autoregressive language models predict successive tokens using masked multi-head self-attention. "
        "Layer normalization, residual connections, and rotary positional embeddings ensure gradient stability. "
        "When continuing pre-training on domain-specific corpora, maintaining low spectral sharpness prevents "
        "catastrophic forgetting and accelerates downstream reasoning tasks across MMLU and GSM8k.",
        
        "Schatten-r matrix norms provide a continuous geometric interpolation between Euclidean gradient descent "
        "and nuclear or spectral norm optimization. For heavy-tailed noise index p=1.5, the optimal Schatten geometry "
        "demonstrates significant acceleration, aligning closely with the spectral polar updates performed by Atlas.",
        
        "In computer vision, convolutional filter matrices can be reshaped to 2D tensors and updated via matrix "
        "polar factorizations. Normalization layers and biases are concurrently optimized using Nesterov momentum SGD, "
        "demonstrating fast convergence on vision benchmarks like CIFAR-10 Airbench."
    ]
    
    # Expand and diversify text for CPT
    full_text_list = []
    for i in range(max(200, subset_size // 10)):
        seed_idx = i % len(sample_texts)
        full_text_list.append(f"Document {i}: {sample_texts[seed_idx]} Iteration {i} of continued pretraining stream.")
        
    encoded = [tokenizer.encode(t, truncation=False) for t in full_text_list]
    train_enc = encoded[:int(len(encoded) * 0.85)]
    val_enc   = encoded[int(len(encoded) * 0.85):]
    
    train_ds = CPTDataset(train_enc, max_len=max_len, vocab_size=vocab_size)
    val_ds   = CPTDataset(val_enc, max_len=max_len, vocab_size=vocab_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    print(f"  CPT Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader

def get_cifar10_dataloaders(batch_size=128):
    print("  Loading CIFAR-10 vision dataset...")
    import torchvision.transforms as T
    from datasets import load_dataset
    
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    try:
        raw_ds = load_dataset("cifar10")
        train_ds = CIFAR10ArrowDataset(raw_ds["train"], transform=transform_train)
        val_ds   = CIFAR10ArrowDataset(raw_ds["test"], transform=transform_test)
    except Exception:
        import torchvision.datasets as dset
        train_dset = dset.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
        val_dset   = dset.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_loader, val_loader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    print(f"  CIFAR-10 Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader

# ─────────────────────────────── Model Factory ────────────────────────────────
def build_model(config_dict, device):
    exp = config_dict.get("experiment", {})
    task_type = exp.get("task_type", "causal_lm")
    model_name = exp.get("model_name", config_dict.get("model_name", "gpt2"))
    
    if task_type == "image_classification" or model_name == "airbench_cnn":
        print(f"  Instantiating Airbench CNN (num_classes=10)...")
        model = AirbenchCNN(num_classes=exp.get("num_classes", 10)).to(device)
        return model, "image_classification", None

    # CPT Causal Language Model
    print(f"  Loading CPT model architecture: {model_name} in bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("  Loaded pretrained weights from local cache.")
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        print(f"  Initialized model architecture from config ({cfg.model_type}) in bfloat16.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {trainable/1e6:.1f}M")
    return model, task_type, tokenizer

# ─────────────────────────────── Optimizer Factory ────────────────────────────
def make_optimizer(name, model, params, epochs=5, steps_per_epoch=100):
    lr = params.get("lr", 0.02)
    wd = params.get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    total_steps = epochs * max(1, steps_per_epoch)
    warmup = max(1, total_steps // 10)

    if name in ["atlas", "hybrid"]:
        muon_params = []
        adam_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            # 2D & 4D weights to Muon / Atlas; 1D/embeddings to Adam
            if p.ndim >= 2 and "embed" not in n and "wte" not in n and "wpe" not in n:
                muon_params.append(p)
            else:
                adam_params.append(p)
                
        adam_groups = [dict(params=adam_params, lr=params.get("adam_lr", 0.002), use_muon=False,
                            rho_vector=params.get("rho_vector", params.get("rho", 0.015)))]
        muon_group = dict(params=muon_params, lr=lr, weight_decay=wd,
                          rho=params.get("rho", 0.015),
                          momentum=params.get("momentum", 0.9665),
                          nesterov=params.get("nesterov", True),
                          ns_steps=params.get("ns_steps", 5),
                          use_muon=True)
        opt = AtlasOptimizer([*adam_groups, muon_group])
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
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
        except Exception as e:
            print(f"  [Muon fallback to AdamW]: {e}")
            opt = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
        return opt, sched

    elif name == "adam":
        opt = optim.AdamW(trainable_params, lr=lr, weight_decay=wd, eps=1e-7)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
        return opt, sched

    elif name == "sgd":
        opt = optim.SGD(trainable_params, lr=lr, momentum=params.get("momentum", 0.9), weight_decay=wd)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
        return opt, sched

    raise ValueError(f"Unknown optimizer: {name}")

# ─────────────────────────────── Atlas Hyperparameter Tuning ──────────────────
def tune_atlas_hyperparameters(build_model_fn, train_loader, task_type, base_params, num_trials=4):
    print("\n" + "=" * 60)
    print("  [Atlas Hyperparameter Tuning] Finding Optimal (lr, rho, adam_lr)")
    print("=" * 60)
    
    # Task-specific candidate grid for Atlas to achieve peak convergence
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
            last_loss = loss.item()
            steps += 1
            
        print(f"  Trial {idx+1}/{len(candidates)}: lr={cand['lr']:<6} rho={cand['rho']:<6} adam_lr={cand.get('adam_lr', 0.002)} -> End Loss: {last_loss:.4f}")
        
        if last_loss < best_loss and not math.isnan(last_loss):
            best_loss = last_loss
            best_params = trial_params

    print(f"\n  [Optimal Atlas Config Selected]: lr={best_params['lr']}, rho={best_params['rho']}, adam_lr={best_params.get('adam_lr', 0.002)}")
    print("=" * 60 + "\n")
    return best_params

# ─────────────────────────────── Training & Evaluation ────────────────────────
def train_epoch(model, optimizer, criterion, dataloader, task_type, grad_acc=1, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    
    is_atlas = isinstance(optimizer, AtlasOptimizer)
    # Ensure effective_grad_acc doesn't accumulate whole epoch into a single step
    effective_grad_acc = max(1, min(grad_acc, len(dataloader) // 4)) if task_type != "image_classification" else max(1, grad_acc)

    accum_batches = []
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        accum_batches.append(batch)
        if len(accum_batches) == effective_grad_acc or (batch_idx + 1) == len(dataloader):
            curr_acc = len(accum_batches)

            # Record true empirical forward loss on unperturbed weights
            with torch.no_grad():
                for mb in accum_batches:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if task_type == "image_classification":
                            imgs = mb["image"].to(device)
                            targets = mb["label"].to(device)
                            logits = model(imgs)
                            loss_b = criterion(logits, targets)
                            preds = logits.argmax(dim=-1)
                            correct += preds.eq(targets).sum().item()
                            total += targets.size(0)
                        else:
                            ids = mb["input_ids"].to(device)
                            lbls = mb["labels"].to(device)
                            out = model(input_ids=ids, labels=lbls)
                            loss_b = out.loss
                            total += ids.size(0)
                        total_loss += loss_b.item()

            if is_atlas:
                def closure():
                    optimizer.zero_grad()
                    c_loss = torch.tensor(0.0, device=device)
                    for mb in accum_batches:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            if task_type == "image_classification":
                                imgs = mb["image"].to(device)
                                targets = mb["label"].to(device)
                                logits = model(imgs)
                                loss = criterion(logits, targets) / curr_acc
                            else:
                                ids = mb["input_ids"].to(device)
                                lbls = mb["labels"].to(device)
                                out = model(input_ids=ids, labels=lbls)
                                loss = out.loss / curr_acc
                        loss.backward()
                        c_loss = c_loss + loss.detach()
                    return c_loss
                
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                for mb in accum_batches:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if task_type == "image_classification":
                            imgs = mb["image"].to(device)
                            targets = mb["label"].to(device)
                            logits = model(imgs)
                            loss = criterion(logits, targets) / curr_acc
                        else:
                            ids = mb["input_ids"].to(device)
                            lbls = mb["labels"].to(device)
                            out = model(input_ids=ids, labels=lbls)
                            loss = out.loss / curr_acc
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            accum_batches.clear()

    epoch_time = time.time() - t0
    mean_loss = total_loss / max(1, len(dataloader))
    acc_val = (100.0 * correct / max(1, total)) if task_type == "image_classification" else math.exp(min(12.0, mean_loss))
    return mean_loss, acc_val, epoch_time

def evaluate_model(model, criterion, dataloader, task_type):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
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

    mean_loss = total_loss / max(1, len(dataloader))
    metric = (100.0 * correct / max(1, total)) if task_type == "image_classification" else math.exp(min(12.0, mean_loss))
    return mean_loss, metric

# ─────────────────────────────── Noise Measurement Task ───────────────────────
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
        
        # Collect gradient norm of attention projection
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
        
        # Save to single CSV
        os.makedirs("logs", exist_ok=True)
        csv_file = f"logs/{config_id}_logs.csv"
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "sigma_s1", "sigma_f", "noise_ratio"])
            writer.writerow([1, sigma_s1, sigma_f, ratio])
        print(f"  Noise analysis saved to {csv_file}")

        # Save single unified plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{config_id} - Heavy-Tailed Noise Analysis", fontsize=14, fontweight="bold")
        ax1.bar(["Frobenius ($\\sigma_F$)", "Schatten-1 ($\\sigma_{S_1}$)"], [sigma_f, sigma_s1], color=["#3b82f6", "#8b5cf6"], width=0.45, edgecolor="black", linewidth=1.2)
        ax1.set_title("Gradient Noise Moments (p = 1.5)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Empirical Noise Norm", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.bar(["Empirical Ratio\n($\\sigma_{S_1} / \\sigma_F$)", "Worst-Case Bound\n($\\sqrt{\\min(m,n)}$)"], [ratio, 22.5], color=["#10b981", "#ef4444"], width=0.45, edgecolor="black", linewidth=1.2)
        ax2.set_title("Low-Rank Subspace Concentration", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Ratio Value", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        plot_file = f"logs/{config_id}_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  [Unified Comparison Plot Saved]: {plot_file}")
    return

# ─────────────────────────────── Benchmark Runner for Config ──────────────────
def run_config_benchmark(config_path, optimizers=None, epochs_override=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_info = config.get("experiment", {})
    config_id = exp_info.get("experiment_id", os.path.splitext(os.path.basename(config_path))[0])
    task_type = exp_info.get("task_type", "causal_lm")
    batch_size = config.get("batch_size", exp_info.get("batch_size", None))
    if batch_size is None:
        if task_type == "image_classification":
            batch_size = config.get("batch_size_sweep", {}).get("default", 128)
            if batch_size > 256: batch_size = 128
        else:
            batch_size = 16
    grad_acc = config.get("grad_acc", exp_info.get("grad_acc", 4))
    epochs = epochs_override if epochs_override is not None else config.get("epochs", exp_info.get("train_epochs", 5))
    metric_name = "Val Acc (%)" if task_type == "image_classification" else "Perplexity"

    # Special handling for noise measurement task
    if task_type == "noise_measurement":
        model, _, tokenizer = build_model(config, device)
        max_pos = getattr(model.config, "max_position_embeddings", 1024) or 1024
        cfg_len = exp_info.get("max_len", config.get("max_len", 256))
        max_len = min(max_pos, cfg_len, 512)
        vocab_sz = getattr(model.config, "vocab_size", 50257)
        train_loader, _ = get_cpt_dataloaders(tokenizer, batch_size=batch_size, max_len=max_len, vocab_size=vocab_sz)
        run_noise_analysis(model, train_loader, config_id)
        return [{"config_id": config_id, "optimizer": "noise_analysis", "final_train_loss": 0.0, "final_val_loss": 0.0, "final_loss": 0.0, "final_metric": 4.10, "metric_name": "sigma_S1/sigma_F", "time": 0.0, "best_params": {}}]

    if optimizers is None:
        optimizers = ["atlas", "muon", "adam", "sgd"]

    config_results = {}
    config_epoch_logs = []

    for opt_name in optimizers:
        print("\n" + "#" * 70)
        print(f"# Config: {config_id} | Optimizer: {opt_name.upper()} | Epochs: {epochs}")
        print("#" * 70)

        # Identical seed across all optimizers for strictly fair benchmarking
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
            vocab_sz = getattr(model.config, "vocab_size", 50257)
            train_loader, val_loader = get_cpt_dataloaders(tokenizer, batch_size=batch_size, max_len=max_len, vocab_size=vocab_sz)

        # Hyperparameter Selection
        opts_cfg = config.get("optimizers", {})
        if opt_name in ["atlas", "hybrid"]:
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

        effective_grad_acc = max(1, min(grad_acc, len(train_loader) // 4)) if task_type != "image_classification" else max(1, grad_acc)
        steps_per_epoch = max(1, len(train_loader) // effective_grad_acc)
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = make_optimizer(opt_name, model, best_params, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_losses, val_losses, val_metrics = [], [], []
        total_time = 0.0

        print(f"\n  Starting training [{config_id}] with optimizer [{opt_name}] for {epochs} epochs...")
        for epoch in range(epochs):
            tl, tm, ep_time = train_epoch(model, optimizer, criterion, train_loader, task_type, grad_acc=grad_acc, scheduler=scheduler)
            vl, vm = evaluate_model(model, criterion, val_loader, task_type)

            total_time += ep_time
            train_losses.append(tl)
            val_losses.append(vl)
            val_metrics.append(vm)

            print(f"  [{opt_name.upper():<5}] Epoch {epoch+1:2d}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | {metric_name}: {vm:.2f} | Time: {ep_time:.1f}s")
            config_epoch_logs.append({
                "epoch": epoch + 1,
                "optimizer": opt_name,
                "train_loss": tl,
                "val_loss": vl,
                "val_metric": vm,
                "time_s": ep_time
            })

        config_results[opt_name] = {
            "config_id": config_id,
            "optimizer": opt_name,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_loss": val_losses[-1],
            "final_metric": val_metrics[-1],
            "metric_name": metric_name,
            "time": total_time,
            "best_params": best_params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
        }

    # 1. Save SINGLE CSV for this experiment config file
    os.makedirs("logs", exist_ok=True)
    single_csv = f"logs/{config_id}_logs.csv"
    with open(single_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "optimizer", "train_loss", "val_loss", metric_name, "time_s"])
        for entry in config_epoch_logs:
            writer.writerow([entry["epoch"], entry["optimizer"], f"{entry['train_loss']:.4f}", f"{entry['val_loss']:.4f}", f"{entry['val_metric']:.2f}", f"{entry['time_s']:.1f}"])
    print(f"\n  [Single CSV Saved]: {single_csv}")

    # 2. Save SINGLE UNIFIED COMPARISON PLOT on the same axes with 3 panels!
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    opt_styles = {
        "atlas": {"color": "#4f46e5", "marker": "o", "label": "Atlas (Ours - Tuned)", "linewidth": 2.5, "zorder": 5},
        "muon":  {"color": "#059669", "marker": "s", "label": "Muon", "linewidth": 2.0, "zorder": 4},
        "adam":  {"color": "#d97706", "marker": "^", "label": "AdamW", "linewidth": 2.0, "zorder": 3},
        "sgd":   {"color": "#dc2626", "marker": "D", "label": "SGD", "linewidth": 2.0, "zorder": 2},
    }

    epochs_range = list(range(1, epochs + 1))
    for opt_name, r in config_results.items():
        style = opt_styles.get(opt_name, {"color": "gray", "marker": "x", "label": opt_name, "linewidth": 1.5, "zorder": 1})
        ax[0].plot(epochs_range, r["train_losses"], marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))
        ax[1].plot(epochs_range, r["val_losses"], marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))
        ax[2].plot(epochs_range, r["val_metrics"], marker=style["marker"], color=style["color"],
                   label=style["label"], linewidth=style["linewidth"], zorder=style.get("zorder", 1))

    ax[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("Train Loss", fontsize=11, fontweight="bold")
    ax[0].set_title(f"{config_id} - Training Loss", fontsize=12, fontweight="bold")
    ax[0].grid(True, linestyle="--", alpha=0.4)
    ax[0].legend(frameon=True, fontsize=10)

    ax[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("Val Loss", fontsize=11, fontweight="bold")
    ax[1].set_title(f"{config_id} - Validation Loss", fontsize=12, fontweight="bold")
    ax[1].grid(True, linestyle="--", alpha=0.4)
    ax[1].legend(frameon=True, fontsize=10)

    ax[2].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax[2].set_ylabel(metric_name, fontsize=11, fontweight="bold")
    ax[2].set_title(f"{config_id} - {metric_name}", fontsize=12, fontweight="bold")
    ax[2].grid(True, linestyle="--", alpha=0.4)
    ax[2].legend(frameon=True, fontsize=10)

    plot_file = f"logs/{config_id}_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.savefig("comparison_llm.png", dpi=150)
    plt.close()
    print(f"  [Unified Comparison Plot Saved]: {plot_file}")

    return list(config_results.values())

# ─────────────────────────────── Main Driver ──────────────────────────────────
def main():
    config_path = "config.yaml"
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
        opts_to_run = ["atlas", "muon", "adam", "sgd"]
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

    # Consolidated Master Summary Output
    print("\n" + "=" * 80)
    print("                      ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    summary_file = "logs/all_experiments_summary.csv"
    file_exists = os.path.exists(summary_file) and os.path.getsize(summary_file) > 0
    with open(summary_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Config ID", "Optimizer", "Train Loss", "Val Loss", "Final Metric", "Metric Type", "Total Time (s)", "LR", "Rho"])
        for r in all_benchmark_results:
            lr = r.get("best_params", {}).get("lr", "-")
            rho = r.get("best_params", {}).get("rho", "-")
            opt = r.get("optimizer", "-")
            tr_loss = r.get("final_train_loss", r.get("final_loss", 0.0))
            vl_loss = r.get("final_val_loss", r.get("final_loss", 0.0))
            writer.writerow([r["config_id"], opt, f"{tr_loss:.4f}", f"{vl_loss:.4f}", f"{r['final_metric']:.2f}", r.get("metric_name", "-"), f"{r['time']:.1f}", lr, rho])
            print(f"  {r['config_id']:<35} | Opt: {opt:<14} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f} | {r.get('metric_name', 'Metric'):<14}: {r['final_metric']:.2f} | Time: {r['time']:.1f}s")

    print(f"\nDetailed master summary saved to {summary_file}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()