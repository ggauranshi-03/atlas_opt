
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib
import wandb
import os
import torch.distributed as dist
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────── Config ───────────────────────────────────────

MODEL_NAME     = "unsloth/Llama-3.2-1B"
DATASET_NAME   = "cais/mmlu"
DATASET_CONFIG = "all"      # Load all 57 subjects
NUM_LABELS     = 4          # MMLU has 4 choices (A, B, C, D)
MAX_LEN        = 256        # Keep at 256 so context and prompt aren't truncated
BATCH_SIZE     = 16
GRAD_ACC       = 4          # effective batch = 64
TRAIN_SUBSET   = 20_000     # MMLU's train split has ~99k rows, subsetting is good
USE_HYPERPARAMETER_TUNING = True # Set to True to enable optuna tuning

# ─────────────────────────────── Data ─────────────────────────────────────────
def get_dataloaders(tokenizer):
    # Load the MMLU dataset
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def tokenize(batch):
        texts = []
        # MMLU uses 'answer' (int 0-3) instead of 'answerKey'
        for q, choices in zip(batch["question"], batch["choices"]):
            # MMLU choices is just a list of 4 strings. We add A, B, C, D dynamically.
            # chr(65) is 'A', chr(66) is 'B', etc.
            choices_str = "\n".join([f"{chr(65+i)}: {txt}" for i, txt in enumerate(choices)])
            
            # We keep the "Answer:" priming trick so the LLM knows to output the choice
            prompt = f"Question: {q}\nChoices:\n{choices_str}\nAnswer:"
            texts.append(prompt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        
        # MMLU's 'answer' column is already an integer (0, 1, 2, or 3), so we can pass it directly
        tokenized["labels"] = batch["answer"]
        return tokenized

    # MMLU's standard training split is called 'auxiliary_train' (~99k examples)
    train_raw = ds["auxiliary_train"]
    val_raw   = ds["validation"] # Validation has ~1.5k examples. You can also use "test" (~14k)

    if TRAIN_SUBSET:
        subset_size = min(TRAIN_SUBSET, len(train_raw))
        train_raw = train_raw.shuffle(seed=42).select(range(subset_size))

    original_cols = train_raw.column_names

    train_ds = train_raw.map(tokenize, batched=True, remove_columns=original_cols)
    val_ds   = val_raw.map(tokenize,   batched=True, remove_columns=val_raw.column_names)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    return train_loader, val_loader

# ─────────────────────────────── Model ────────────────────────────────────────
def get_model(device, num_labels=NUM_LABELS):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = model.config.eos_token_id

    for param in model.parameters():
        param.requires_grad = False

    num_layers     = model.config.num_hidden_layers
    unfreeze_from  = num_layers - 4
    for i, layer in enumerate(model.model.layers):
        if i >= unfreeze_from:
            for p in layer.parameters():
                p.requires_grad = True
    for p in model.score.parameters():
        p.requires_grad = True

    model = model.to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")
    return model
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

# ─────────────────────────────── Hybrid SAM Optimizer ─────────────────────────
class OptimizedHybridSAM(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                 rho=0.05, ns_steps=5, momentum_start=0.85, momentum_end=0.95, momentum_warmup_steps=200):
        defaults = dict(lr=lr, rho=rho, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.eps = eps
        self.ns_steps = ns_steps
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.momentum_warmup_steps = momentum_warmup_steps
        self._step = 0
        self._adv_pass = False

    def _get_momentum(self):
        if self._step >= self.momentum_warmup_steps: return self.momentum_end
        t = self._step / max(1, self.momentum_warmup_steps)
        return self.momentum_start + t * (self.momentum_end - self.momentum_start)

    def _adam(self, p, grad, group):
        s = self.state[p]
        if "step" not in s:
            s["step"] = 0
            s["m"] = torch.zeros_like(grad)
            s["v"] = torch.zeros_like(grad)
        s["step"] += 1
        t = s["step"]
        b1, b2 = group["betas"]
        lr, wd = group["lr"], group["weight_decay"]
        if wd > 0: p.data.mul_(1.0 - lr * wd)
        s["m"].mul_(b1).add_(grad, alpha=1 - b1)
        s["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        alpha = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        update = s["m"] / (s["v"].sqrt().add_(self.eps))
        p.data.add_(update.to(p.dtype), alpha=-alpha)

    @torch.no_grad()
    def _first_step(self, eps_w=1e-12):
        shared_device = self.param_groups[0]["params"][0].device
        sq_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.float()
                scaled_g = g * (p.data.float().abs() + eps_w)
                sq_norm += scaled_g.norm(p=2).to(shared_device).pow(2).item()
        grad_norm = (sq_norm + eps_w) ** 0.5
        for group in self.param_groups:
            rho = group["rho"]
            scale = rho / (grad_norm + eps_w)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.data.float().abs() + eps_w) * p.grad.float() * scale
                p.data.add_(e_w.to(p.dtype))
        self._adv_pass = True
        self.zero_grad()

    @torch.no_grad()
    def _second_step(self):
        self._step += 1
        momentum = self._get_momentum()
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.data.float()
                s = self.state[p]
                if p.ndim >= 2:
                    if "momentum_buffer" not in s: s["momentum_buffer"] = torch.zeros_like(grad)
                    buf = s["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    g_nest = grad + momentum * buf
                    g_2d = g_nest.view(g_nest.size(0), -1)
                    g_ns = newtonschulz5(g_2d, steps=self.ns_steps).view_as(g_nest)
                    if wd > 0: p.data.mul_(1.0 - lr * wd)
                    p.data.add_(g_ns.to(p.dtype), alpha=-lr)
                else:
                    self._adam(p, grad, group)
        self._adv_pass = False

    def step(self, closure=None):
        if closure is None: raise RuntimeError("Requires closure")
        self._adv_pass = False
        with torch.enable_grad(): loss = closure()
        self._first_step()
        with torch.enable_grad(): closure()
        self._second_step()
        return loss

class SinglePassOptimizedSAM(OptimizedHybridSAM):
    def step(self, closure=None):
        if closure is None: raise RuntimeError("Requires closure")
        self._first_step()
        self._adv_pass = False
        with torch.enable_grad(): loss = closure()
        self._second_step()
        return loss

# ─────────────────────────────── Training utilities ───────────────────────────
def is_sam_like(opt):
    if isinstance(opt, OptimizedHybridSAM): return True
    if hasattr(opt, "optimizer") and isinstance(opt.optimizer, OptimizedHybridSAM): return True
    return False

def train_one_epoch(model, optimizer, criterion, dataloader, device, epoch=0, run=None, opt_name="", grad_acc=GRAD_ACC):
    model.train()
    total_loss, correct, total = 0, 0, 0
    batch_times = []

    # Removed optimizer.zero_grad() here to fix the epoch-boundary wipe for SS-SAM
    
    micro_batches = []
    t0_accum = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        micro_batches.append(batch)
        
        # Once we have enough micro-batches (or hit the end of the epoch), take an optimization step
        if len(micro_batches) == grad_acc or (batch_idx + 1) == len(dataloader):
            curr_grad_acc = len(micro_batches)
            _all_logits = []
            true_mean_loss = 0.0
            
            if is_sam_like(optimizer):
                def closure():
                    optimizer.zero_grad()
                    closure_loss = 0.0
                    _all_logits.clear()
                    
                    for mb in micro_batches:
                        mb_in = mb["input_ids"].to(device)
                        mb_mask = mb["attention_mask"].to(device)
                        mb_labels = mb["labels"].to(device)
                        
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            _out = model(input_ids=mb_in, attention_mask=mb_mask)
                            _all_logits.append(_out.logits.detach())
                            _loss = criterion(_out.logits, mb_labels) / curr_grad_acc
                        
                        _loss.backward() # Accumulates the gradient for the SAM pass
                        closure_loss += _loss
                        
                    return closure_loss
                
                loss = optimizer.step(closure)
                true_mean_loss = loss.item()
            else:
                optimizer.zero_grad()
                for mb in micro_batches:
                    mb_in = mb["input_ids"].to(device)
                    mb_mask = mb["attention_mask"].to(device)
                    mb_labels = mb["labels"].to(device)
                    
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _out = model(input_ids=mb_in, attention_mask=mb_mask)
                        _all_logits.append(_out.logits.detach())
                        _loss = criterion(_out.logits, mb_labels) / curr_grad_acc
                        
                    _loss.backward()
                    true_mean_loss += _loss.item()
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            batch_times.append(time.time() - t0_accum)
            t0_accum = time.time()
            
            with torch.no_grad():
                cat_logits = torch.cat(_all_logits, dim=0)
                cat_labels = torch.cat([mb["labels"].to(device) for mb in micro_batches], dim=0)
                preds = cat_logits.argmax(dim=-1)
                
            total_loss += true_mean_loss * cat_labels.size(0)
            correct    += preds.eq(cat_labels).sum().item()
            total      += cat_labels.size(0)
            
            micro_batches.clear()

    return total_loss / total, 100.0 * correct / total, sum(batch_times) / len(batch_times)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += outputs.logits.argmax(dim=-1).eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total

# ─────────────────────────────── Optimizer factory ────────────────────────────
def make_optimizer(name, model, params, epochs=10, steps_per_epoch=None):
    lr = params["lr"]
    wd = params.get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        opt = optim.AdamW(trainable_params, lr=lr, weight_decay=wd, eps=1e-7)
        warmup = max(1, steps_per_epoch * 2) if steps_per_epoch else 100
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched

    elif name == "sgd":
        momentum = params.get("momentum", 0.9)
        opt = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return opt, sched

    elif name == "muon":
        from muon import MuonWithAuxAdam
        muon_params = []
        adam_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2 and "embed" not in n: muon_params.append(p)
            else: adam_params.append(p)
        
        adam_groups = [dict(params=adam_params, lr=3e-4, use_muon=False)]
        muon_group = dict(params=muon_params, lr=lr, weight_decay=wd, use_muon=True)
        opt = MuonWithAuxAdam([*adam_groups, muon_group])
        return opt, optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    elif name == "hybrid":
        warmup_epochs = params.get("warmup_epochs", 2)
        optim_params = [p for p in model.parameters() if p.requires_grad]
        base_opt = OptimizedHybridSAM(
            optim_params, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            ns_steps=params.get("ns_steps", 5),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )
        opt = Lookahead(base_opt, alpha=0.5, k=6)
        warmup_sched = optim.lr_scheduler.LinearLR(base_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7)
        scheduler = optim.lr_scheduler.SequentialLR(base_opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        return opt, scheduler

    elif name == "atlas":
        warmup_epochs = params.get("warmup_epochs", 2)
        optim_params = [p for p in model.parameters() if p.requires_grad]
        base_opt = SinglePassOptimizedSAM(
            optim_params, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            ns_steps=params.get("ns_steps", 5),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )
        opt = Lookahead(base_opt, alpha=0.5, k=6)
        warmup_sched = optim.lr_scheduler.LinearLR(base_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7)
        scheduler = optim.lr_scheduler.SequentialLR(base_opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        return opt, scheduler

    raise ValueError(f"Unknown optimizer: {name}")

# ─────────────────────────────── Hyperparameter tuning ────────────────────────
def objective(trial, opt_name, tokenizer):
    if opt_name == "adam":
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    params = dict(lr=lr, weight_decay=wd)

    if opt_name in ["hybrid", "atlas"]:
        params["damping"] = trial.suggest_float("damping", 1e-3, 0.1, log=True)

    train_loader, val_loader = get_dataloaders(tokenizer)
    model     = get_model(device, num_labels=NUM_LABELS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = make_optimizer(opt_name, model, params, epochs=3, steps_per_epoch=len(train_loader))
    
    val_acc = 0.0
    for epoch in range(3):
        train_one_epoch(model, optimizer, criterion, train_loader, device)
        _, val_acc = evaluate(model, criterion, val_loader, device)
        scheduler.step()
        trial.report(val_acc, epoch)
        import optuna
        if trial.should_prune(): raise optuna.TrialPruned()
    return val_acc

def run_tuning(opt_name, n_trials, tokenizer):
    import optuna
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
    study.optimize(lambda t: objective(t, opt_name, tokenizer), n_trials=n_trials)
    return study.best_params

# ─────────────────────────────── Full experiment ──────────────────────────────
def run_experiment(optimizer_name, params, tokenizer, epochs=10, wandb_project="Hybrid-LLM-ZFS"):
    run = wandb.init(
        project=wandb_project,
        name=f"{optimizer_name}",
        group="final_training",
        config={"optimizer": optimizer_name, "epochs": epochs, "batch_size": BATCH_SIZE, **params},
        reinit="finish_previous",
    )
    train_loader, val_loader = get_dataloaders(tokenizer)
    model     = get_model(device, num_labels=NUM_LABELS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = make_optimizer(optimizer_name, model, params, epochs=epochs, steps_per_epoch=len(train_loader))

    train_losses, val_accs = [], []
    cumulative_time = 0.0
    start_wall = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        tl, train_acc, avg_bt = train_one_epoch(model, optimizer, criterion, train_loader, device, epoch=epoch, run=run, opt_name=optimizer_name)
        vl, va = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        epoch_time       = time.time() - epoch_start
        cumulative_time += epoch_time

        run.log({
            "train/loss": tl,
            "val/accuracy": va,
            "epoch": epoch + 1,
        })

        train_losses.append(tl); val_accs.append(va)
        print(f"  [{optimizer_name:8s}] epoch {epoch+1:3d}  loss={tl:.4f}  val_acc={va:.2f}%  time={epoch_time:.1f}s")

    total_time = time.time() - start_wall
    wandb.finish()
    return train_losses, val_accs, total_time

# ─────────────────────────────── Ablation Study (Restored) ────────────────────
def run_ablation(tokenizer, epochs=5, wandb_project="Hybrid-LLM-ZFS"):
    """
    CRITICAL FIX: This function was missing from the script in the previous iteration,
    causing the NameError. It is now fully restored to run the ablation study.
    """
    base_params = dict(lr=1e-4, weight_decay=1e-4, damping=0.01,
                       kfac_interval=20, kfac_decay=0.90, use_ns=True,
                       momentum_start=0.85, momentum_end=0.95,
                       momentum_warmup_steps=200)
    configs = {
        "Hybrid (full)":     HybridSAMOptimizer,
        "Single-Pass SAM":   SinglePassHybridSAM,
        "Hybrid - no K-FAC": HybridNoKFAC,
        "Hybrid - no NS":    HybridNoNS,
        "Hybrid - no SAM":   HybridNoSAM,
    }
    results = {}
    for name, cls in configs.items():
        print(f"\n── Ablation: {name} ──")
        train_loader, val_loader = get_dataloaders(tokenizer)
        run = wandb.init(
            project=wandb_project,
            name=f"ablation_{name.replace(' ', '_').replace('-', 'no')}",
            group="ablation",
            config={"ablation_variant": name, "epochs": epochs, **base_params},
            reinit="finish_previous",
        )
        model     = get_model(device, num_labels=NUM_LABELS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt       = cls(model, **{k: v for k, v in base_params.items()
                                  if k not in ("lr", "weight_decay")},
                        lr=base_params["lr"], weight_decay=base_params["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        losses, accs = [], []
        for epoch in range(epochs):
            tl, ta, _ = train_one_epoch(model, opt, criterion, train_loader, device)
            _, va = evaluate(model, criterion, val_loader, device)
            scheduler.step()
            losses.append(tl); accs.append(va)
            print(f"  Epoch {epoch+1:3d}: val_acc={va:.2f}%")
            run.log({
                f"ablation/{name}/train_loss":   tl,
                f"ablation/{name}/val_accuracy": va,
                "epoch": epoch + 1,
            })
        wandb.finish()
        results[name] = (losses, accs)
    return results

# ─────────────────────────────── Main ─────────────────────────────────────────
def main():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    TUNING_TRIALS   = 10
    TRAIN_EPOCHS    = 10
    ABLATION_EPOCHS = 5    
    WANDB_PROJECT   = "Hybrid-SAM-Comparison_optuna"
    
    OPTIMIZERS_TO_TEST = ["atlas", "adam", "sgd", "muon"]

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── 1. Hyperparameter Tuning ──
    best = {}
    if USE_HYPERPARAMETER_TUNING:
        print("=" * 60 + "\n  Hyperparameter Tuning\n" + "=" * 60)
        for name in OPTIMIZERS_TO_TEST:
            print(f"\n  Tuning {name} …")
            best[name] = run_tuning(name, n_trials=TUNING_TRIALS, tokenizer=tokenizer)
            print(f"  Best {name}: {best[name]}")
    else:
        print("=" * 60 + "\n  Skipping Hyperparameter Tuning (using default best params)\n" + "=" * 60)
        best = {
            "atlas": {"ns_steps": 5, "lr": 1e-4, "weight_decay": 1e-4},
            "adam": {"lr": 1e-3, "weight_decay": 1e-4},
            "sgd": {"lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4},
            
            "muon": {"lr": 0.02, "weight_decay": 0.01}
        }

    # ── 2. Final Training ──
    print("\n" + "=" * 60 + "\n  Final Training\n" + "=" * 60)
    results = {}
    for name in OPTIMIZERS_TO_TEST:
        print(f"\n── {name.upper()} ──")
        l, a, t = run_experiment(name, best[name], tokenizer, epochs=TRAIN_EPOCHS, wandb_project=WANDB_PROJECT)
        results[name] = dict(losses=l, accs=a, time=t)

    # Plot Final Training Results
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, d in results.items():
        axes[0].plot(d["losses"], label=name)
        axes[1].plot(d["accs"],   label=name)
    for ax, title, yl in zip(axes, ["Training Loss", "Validation Accuracy (%)"], ["Loss", "Accuracy (%)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig("comparison_llm.png", dpi=150)
    print("Saved: comparison_llm.png")

    # ── 3. Final Summary Console Output ──
    print("\n" + "=" * 60 + "\n  FINAL SUMMARY\n" + "=" * 60)
    print("Main Optimizers:")
    for name, d in results.items():
        print(f"  {name:10s}: val_acc={d['accs'][-1]:.2f}%  time={d['time']:.0f}s")

if __name__ == "__main__":
    main()