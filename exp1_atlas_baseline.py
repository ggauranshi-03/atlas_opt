
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib
import wandb
import os
import csv
import yaml
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

# ─────────────────────────────── Training utilities ───────────────────────────
def is_sam_like(opt):
    if isinstance(opt, AtlasOptimizer): return True
    if hasattr(opt, "optimizer") and isinstance(opt.optimizer, AtlasOptimizer): return True
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
        warmup = max(1, steps_per_epoch * 2) if steps_per_epoch else 100
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
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
        warmup = max(1, steps_per_epoch * 2) if steps_per_epoch else 100
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched

    elif name in ["hybrid", "atlas"]:
        muon_params = []
        adam_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2 and "embed" not in n: muon_params.append(p)
            else: adam_params.append(p)
            
        adam_groups = [dict(params=adam_params, lr=3e-4, use_muon=False,
                             rho_vector=params.get("rho_vector", params.get("rho", 0.05)))]
        muon_group = dict(params=muon_params, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05), use_muon=True)
        opt = AtlasOptimizer([*adam_groups, muon_group])
        
        warmup = max(1, steps_per_epoch * 2) if steps_per_epoch else 100
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched

    raise ValueError(f"Unknown optimizer: {name}")

# ─────────────────────────────── Full experiment ──────────────────────────────
def run_experiment(optimizer_name, params, tokenizer, epochs=10, wandb_project="Atlas-Experiments"):
    run = wandb.init(
        project=wandb_project,
        name=f"{optimizer_name}_exp1_baseline",
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
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': tl,
            'val_acc': va,
        }
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{optimizer_name}_epoch_{epoch+1}.pt")
        
        # Log to CSV
        os.makedirs("logs", exist_ok=True)
        csv_file = f"logs/{optimizer_name}_atlas_exp1_logs.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_acc', 'time_s'])
            writer.writerow([epoch + 1, tl, va, epoch_time])

    total_time = time.time() - start_wall
    wandb.finish()
    return train_losses, val_accs, total_time

# ─────────────────────────────── Main ─────────────────────────────────────────
def main():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        TRAIN_EPOCHS = CONFIG["experiment"]["train_epochs"]
    WANDB_PROJECT = CONFIG["experiment"]["wandb_project"]
    
    OPTIMIZERS_TO_TEST = ["atlas", "adam", "sgd", "muon"]

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── 1. Setup Hyperparameters ──
    best = {
        "atlas": CONFIG["hyperparameters"]["atlas_exp1"],
        "adam": CONFIG["hyperparameters"]["adam"],
        "sgd": CONFIG["hyperparameters"]["sgd"],
        "muon": CONFIG["hyperparameters"]["muon"]
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