
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib
import wandb
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

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
USE_HYPERPARAMETER_TUNING = False # Set to True to enable optuna tuning

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
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0, pin_memory=True)
    
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

# ─────────────────────────────── Muon Utilities ───────────────────────────────
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

class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, model, lr=0.02, wd=0.01, adam_lr=3e-4):
        muon_params, adam_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2: muon_params.append(p)
            else: adam_params.append(p)
        defaults = dict(lr=lr, adam_lr=adam_lr, weight_decay=wd)
        super().__init__([
            {"params": muon_params, "use_muon": True,  "lr": lr,      "weight_decay": wd},
            {"params": adam_params, "use_muon": False, "lr": adam_lr, "weight_decay": wd},
        ], defaults)
        self.momentum = 0.95
        self.ns_steps = 5

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            use_muon, lr, wd = group["use_muon"], group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                grad  = p.grad.float()
                state = self.state[p]
                if use_muon:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(self.momentum).add_(grad)
                    g   = grad + self.momentum * buf
                    g2d = g.view(g.size(0), -1) if g.ndim >= 2 else g
                    O   = newtonschulz5(g2d, self.ns_steps).view_as(g) if g.ndim >= 2 else g
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(O.to(p.dtype), alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(grad)
                        state["v"] = torch.zeros_like(grad)
                    state["step"] += 1
                    b1, b2 = 0.9, 0.95
                    state["m"].mul_(b1).add_(grad, alpha=1 - b1)
                    state["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    m_hat = state["m"] / (1 - b1 ** state["step"])
                    v_hat = state["v"] / (1 - b2 ** state["step"])
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(m_hat.to(p.dtype), v_hat.sqrt().to(p.dtype) + 1e-8, value=-lr)

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

# ─────────────────────────────── Sophia-G (Fixed with Bias Correction) ────────
class SophiaG(torch.optim.Optimizer):
    """
    Sophia-G: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.
    [FIXED FOR A*]: Includes rigorous early-step bias correction to prevent rho-clipping stall.
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1, eps=1e-12):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super(SophiaG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # CRITICAL FIX: Bias correction. Without this, early steps divide by ~0 and stall.
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                m_hat = exp_avg / bias_correction1
                h_hat = exp_avg_sq / bias_correction2

                step_size = group['lr']
                rho = group['rho']
                
                h = torch.maximum(h_hat, torch.tensor(group['eps']).to(h_hat.device))
                update = torch.clamp(m_hat / h, min=-rho, max=rho)
                p.data.add_(update.to(p.dtype), alpha=-step_size)

        return loss

# ─────────────────────────────── Shampoo ──────────────────────────────────────
class Shampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, eps=1e-4, update_freq=100, max_precond_dim=2048):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, update_freq=update_freq, max_precond_dim=max_precond_dim)
        super(Shampoo, self).__init__(params, defaults)

    def compute_power_svd(self, mat, power=-0.25):
        mat_f32 = mat.float() 
        try:
            U, S, Vh = torch.linalg.svd(mat_f32, full_matrices=False)
            S_pow = torch.diag(S.clamp(min=1e-8) ** power)
            res = U @ S_pow @ Vh
            return res.to(mat.dtype)
        except RuntimeError:
            return torch.eye(mat.size(0), device=mat.device, dtype=mat.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.float()
                state = self.state[p]

                use_shampoo = (grad.ndim >= 2 and 
                               grad.size(0) <= group['max_precond_dim'] and 
                               grad.size(1) <= group['max_precond_dim'])

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(grad)
                    if use_shampoo:
                        state['L'] = torch.eye(grad.size(0), device=grad.device) * group['eps']
                        state['R'] = torch.eye(grad.size(1), device=grad.device) * group['eps']
                        state['inv_L'] = torch.eye(grad.size(0), device=grad.device)
                        state['inv_R'] = torch.eye(grad.size(1), device=grad.device)

                state['step'] += 1
                if group['weight_decay'] > 0: grad.add_(p.data, alpha=group['weight_decay'])
                state['momentum'].mul_(group['momentum']).add_(grad)
                g_m = state['momentum']

                if use_shampoo:
                    state['L'].add_(g_m @ g_m.T)
                    state['R'].add_(g_m.T @ g_m)
                    if state['step'] % group['update_freq'] == 0 or state['step'] == 1:
                        state['inv_L'] = self.compute_power_svd(state['L'], -0.25)
                        state['inv_R'] = self.compute_power_svd(state['R'], -0.25)
                    preconditioned_grad = state['inv_L'] @ g_m @ state['inv_R']
                else:
                    preconditioned_grad = g_m

                p.data.add_(preconditioned_grad.to(p.dtype), alpha=-group['lr'])
        return loss

# ─────────────────────────────── Hybrid SAM Optimizer ─────────────────────────
class HybridSAMOptimizer(torch.optim.Optimizer):
    def __init__(self, model, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                 rho=0.05, damping=1e-2, kfac_interval=20, kfac_decay=0.90,
                 max_kfac_dim=512, use_ns=True, ns_steps=5,
                 momentum_start=0.85, momentum_end=0.95, momentum_warmup_steps=200):

        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, rho=rho, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.model                 = model
        self.eps                   = eps
        self.damping               = damping
        self.kfac_interval         = kfac_interval
        self.kfac_decay            = kfac_decay
        self.max_kfac_dim          = max_kfac_dim
        self.use_ns                = use_ns
        self.ns_steps              = ns_steps
        self.momentum_start        = momentum_start
        self.momentum_end          = momentum_end
        self.momentum_warmup_steps = momentum_warmup_steps

        self._step     = 0
        self._adv_pass = False
        self._m_A, self._m_G = {}, {}
        self._kA,  self._kG  = {}, {}
        self._Ai,  self._Gi  = {}, {}
        self._mods           = {}
        self._register_hooks()

        self._kfac_ids = set()
        for m in self._mods.values():
            self._kfac_ids.add(id(m.weight))
            if m.bias is not None:
                self._kfac_ids.add(id(m.bias))

    def _get_momentum(self):
        if self._step >= self.momentum_warmup_steps:
            return self.momentum_end
        t = self._step / max(1, self.momentum_warmup_steps)
        return self.momentum_start + t * (self.momentum_end - self.momentum_start)

    def _register_hooks(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                mid = id(m)
                self._mods[mid] = m
                m.register_forward_hook(self._fwd(mid))
                m.register_full_backward_hook(self._bwd(mid))

    def _fwd(self, mid):
        def h(mod, inp, out):
            if not self._adv_pass:
                a = inp[0].detach().float()
                self._m_A[mid] = a.mean(dim=1) if a.ndim == 3 else a
        return h
    
    def _bwd(self, mid):
        def h(mod, gin, gout):
            if not self._adv_pass:
                g = gout[0].detach().float()
                self._m_G[mid] = g.mean(dim=1) if g.ndim == 3 else g
        return h

    def _update_kfac(self):
        for mid, m in self._mods.items():
            A_raw = self._m_A.get(mid)
            G_raw = self._m_G.get(mid)
            if A_raw is None or G_raw is None: continue
            A = A_raw.view(-1, A_raw.size(-1))
            G = G_raw.view(-1, G_raw.size(-1))
            if m.bias is not None:
                A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
            n  = A.size(0)
            Af = (A.t() @ A) / n
            Gf = (G.t() @ G) / n
            if Af.size(0) > self.max_kfac_dim or Gf.size(0) > self.max_kfac_dim:
                continue
            d = self.kfac_decay
            if mid not in self._kA:
                self._kA[mid] = Af.clone()
                self._kG[mid] = Gf.clone()
            else:
                self._kA[mid].mul_(d).add_(Af, alpha=1 - d)
                self._kG[mid].mul_(d).add_(Gf, alpha=1 - d)
        for mid in list(self._kA.keys()):
            A, G = self._kA[mid], self._kG[mid]
            trA  = torch.trace(A).clamp(min=1e-8).item()
            trG  = torch.trace(G).clamp(min=1e-8).item()
            pi   = max(0.01, min(((trA * G.size(0)) / (trG * A.size(0) + 1e-8)) ** 0.5, 100.0))
            damp = self.damping
            dA   = (damp ** 0.5) * pi
            dG   = (damp ** 0.5) / pi
            try:
                IA = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
                IG = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
                self._Ai[mid] = torch.linalg.solve(A + dA * IA, IA)
                self._Gi[mid] = torch.linalg.solve(G + dG * IG, IG)
            except RuntimeError:
                pass

    def _precond(self, mid, m, gw, gb):
        if mid not in self._Ai: return gw, gb
        Ai, Gi    = self._Ai[mid], self._Gi[mid]
        orig_norm = gw.norm(p=2)
        if gb is not None:
            C      = torch.cat([gw, gb.unsqueeze(1)], dim=1)
            P      = Gi @ C @ Ai
            pw, pb = P[:, :-1], P[:, -1]
        else:
            pw = Gi @ gw @ Ai
            pb = None
        new_norm = pw.norm(p=2)
        if new_norm > 1e-8 and orig_norm > 1e-8:
            pw = pw * (orig_norm / new_norm)
        return pw, pb

    def _apply_ns(self, gw, momentum):
        shape     = gw.shape
        g_2d      = gw.view(shape[0], -1)
        g_ns      = newtonschulz5(g_2d, self.ns_steps).view_as(gw)
        orig_norm = gw.norm(p=2)
        ns_norm   = g_ns.norm(p=2)
        if ns_norm > 1e-8 and orig_norm > 1e-8:
            g_ns = g_ns * (orig_norm / ns_norm)
        return g_ns

    def _adam(self, p, grad, max_grad_norm=1.0):
        s = self.state[p]
        if "step" not in s:
            s["step"] = 0
            s["m"]    = torch.zeros_like(grad)
            s["v"]    = torch.zeros_like(grad)
        g_norm = grad.norm(p=2)
        if max_grad_norm > 0 and g_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (g_norm + 1e-8))
        s["step"] += 1
        t      = s["step"]
        group  = self.param_groups[0]
        b1, b2 = group["betas"]
        lr     = group["lr"]
        wd     = group["weight_decay"]
        if wd > 0: p.data.mul_(1.0 - lr * wd)
        s["m"].mul_(b1).add_(grad, alpha=1 - b1)
        s["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        alpha  = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        update = s["m"] / (s["v"].sqrt().add_(self.eps))
        p.data.add_(update.to(p.dtype) * (-alpha))

    @torch.no_grad()
    def _first_step(self, eps_w=1e-12):
        shared_device = self.param_groups[0]["params"][0].device
        sq_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g        = p.grad.float()
                scaled_g = g * (p.data.float().abs() + eps_w)
                sq_norm += scaled_g.norm(p=2).to(shared_device).pow(2).item()
        grad_norm = (sq_norm + eps_w) ** 0.5
        for group in self.param_groups:
            rho   = group["rho"]
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
        if self._step % self.kfac_interval == 0:
            self._update_kfac()
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
        for mid, m in self._mods.items():
            w, b = m.weight, m.bias
            if w.grad is None: continue
            grad = w.grad.data.float()
            gb   = b.grad.data.float() if (b is not None and b.grad is not None) else None
            s    = self.state[w]
            if "momentum_buffer" not in s:
                s["momentum_buffer"] = torch.zeros_like(grad)
            buf = s["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            g_nest     = grad + momentum * buf
            gw_k, gb_k = self._precond(mid, m, g_nest, gb)
            if self.use_ns and gw_k.ndim >= 2:
                gw_k = self._apply_ns(gw_k, momentum)
            self._adam(w, gw_k)
            if b is not None and gb_k is not None:
                self._adam(b, gb_k)
        for p in self.param_groups[0]["params"]:
            if id(p) in self._kfac_ids or p.grad is None: continue
            self._adam(p, p.grad.data.float())
        self._adv_pass = False

    def step(self, closure=None):
        if closure is None: raise RuntimeError("HybridSAMOptimizer requires a closure")
        self._adv_pass = False
        with torch.enable_grad(): loss = closure()
        self._first_step()
        with torch.enable_grad(): closure()
        self._second_step()
        return loss

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none=set_to_none)

class SinglePassHybridSAM(HybridSAMOptimizer):
    def step(self, closure=None):
        if closure is None: raise RuntimeError("SinglePassHybridSAM requires a closure")
        
        if getattr(self, "_is_perturbed", False) is False:
            self._is_perturbed = False

        if not self._is_perturbed:
            self._adv_pass = False
            with torch.enable_grad(): loss = closure()
            self._first_step()
            self._is_perturbed = True
            return loss
        else:
            self._adv_pass = False 
            with torch.enable_grad(): loss = closure()
            self._second_step()
            self._first_step()
            return loss

    def restore_base_weights(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])

# ── Ablation variants ─────────────────────────────────────────────────────────
class HybridNoKFAC(HybridSAMOptimizer):
    def _precond(self, mid, m, gw, gb): return gw, gb

class HybridNoNS(HybridSAMOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["use_ns"] = False
        super().__init__(*args, **kwargs)

class HybridNoSAM(HybridSAMOptimizer):
    def step(self, closure=None):
        if closure is None: raise RuntimeError("HybridNoSAM requires a closure")
        self._adv_pass = False
        with torch.enable_grad(): loss = closure()
        self._step += 1
        momentum = self._get_momentum()
        if self._step % self.kfac_interval == 0: self._update_kfac()
        for mid, m in self._mods.items():
            w, b = m.weight, m.bias
            if w.grad is None: continue
            grad = w.grad.data.float()
            gb   = b.grad.data.float() if (b is not None and b.grad is not None) else None
            s    = self.state[w]
            if "momentum_buffer" not in s: s["momentum_buffer"] = torch.zeros_like(grad)
            buf = s["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            g_nest     = grad + momentum * buf
            gw_k, gb_k = self._precond(mid, m, g_nest, gb)
            if self.use_ns and gw_k.ndim >= 2: gw_k = self._apply_ns(gw_k, momentum)
            self._adam(w, gw_k)
            if b is not None and gb_k is not None: self._adam(b, gb_k)
        for p in self.param_groups[0]["params"]:
            if id(p) in self._kfac_ids or p.grad is None: continue
            self._adam(p, p.grad.data.float())
        return loss

# ─────────────────────────────── Training utilities ───────────────────────────
def is_sam_like(opt):
    if isinstance(opt, HybridSAMOptimizer): return True
    if hasattr(opt, "optimizer") and isinstance(opt.optimizer, HybridSAMOptimizer): return True
    return False

def train_one_epoch(model, optimizer, criterion, dataloader, device, epoch=0, run=None, opt_name="", grad_acc=GRAD_ACC):
    model.train()
    total_loss, correct, total = 0, 0, 0
    batch_times = []

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{opt_name}]")
    for batch_idx, batch in enumerate(pbar):
        t0             = time.time()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        _logits = [None]
        if is_sam_like(optimizer):
            def closure():
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _out  = model(input_ids=input_ids, attention_mask=attention_mask)
                    _logits[0] = _out.logits.detach()
                    _loss = criterion(_out.logits, labels)
                _loss.backward()
                return _loss
            loss = optimizer.step(closure)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(_out.logits, labels) / grad_acc
            _logits[0] = _out.logits.detach()
            loss.backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == len(dataloader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            loss = loss * grad_acc

        batch_times.append(time.time() - t0)
        with torch.no_grad():
            preds = _logits[0].argmax(dim=-1)
        
        current_loss = loss.item()
        total_loss += current_loss * labels.size(0)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)
        
        current_acc = 100.0 * correct / total
        current_ppl = math.exp(current_loss) if current_loss < 20 else float('inf')
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.1f}%", "ppl": f"{current_ppl:.1f}"})

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

    elif name == "muon":
        opt = MuonWithAuxAdam(model, lr=lr, wd=wd)
        return opt, optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
    elif name == "sophia":
        rho = params.get("rho", 0.04)
        opt = SophiaG(trainable_params, lr=lr, weight_decay=wd, rho=rho)
        warmup = max(1, steps_per_epoch * 2) if steps_per_epoch else 100
        sched  = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=epochs * (steps_per_epoch or 100))
        return opt, sched
        
    elif name == "shampoo":
        momentum = params.get("momentum", 0.9)
        opt = Shampoo(trainable_params, lr=lr, momentum=momentum, weight_decay=wd, update_freq=100)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return opt, sched

    elif name == "hybrid":
        warmup_epochs = params.get("warmup_epochs", 2)
        base_opt = HybridSAMOptimizer(
            model, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            damping=params.get("damping", 1e-2), kfac_interval=params.get("kfac_interval", 20),
            kfac_decay=params.get("kfac_decay", 0.90), use_ns=params.get("use_ns", True),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )
        opt = Lookahead(base_opt, alpha=0.5, k=6)
        warmup_sched = optim.lr_scheduler.LinearLR(base_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7)
        scheduler = optim.lr_scheduler.SequentialLR(base_opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        return opt, scheduler

    elif name == "single_pass_hybrid":
        warmup_epochs = params.get("warmup_epochs", 2)
        base_opt = SinglePassHybridSAM(
            model, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            damping=params.get("damping", 1e-2), kfac_interval=params.get("kfac_interval", 20),
            kfac_decay=params.get("kfac_decay", 0.90), use_ns=params.get("use_ns", True),
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
    if opt_name in ["adam", "sophia"]:
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    params = dict(lr=lr, weight_decay=wd)

    if opt_name == "sophia":
        params["rho"] = trial.suggest_float("rho", 0.01, 0.1)
    if opt_name == "shampoo":
        params["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
    if opt_name in ["hybrid", "single_pass_hybrid"]:
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
            f"{optimizer_name}/train/loss": tl,
            f"{optimizer_name}/val/accuracy": va,
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
    TUNING_TRIALS   = 10
    TRAIN_EPOCHS    = 3
    ABLATION_EPOCHS = 5    
    WANDB_PROJECT   = "Hybrid-SAM-Comparison"
    
    OPTIMIZERS_TO_TEST = ["single_pass_hybrid", "hybrid"]

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
            "hybrid": {"damping": 0.01, "lr": 1e-4, "weight_decay": 1e-4},
            "single_pass_hybrid": {"damping": 0.01, "lr": 1e-4, "weight_decay": 1e-4}
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

    # ── 3. Ablation Study ──
    print("\n" + "=" * 60 + "\n  Ablation Study (Hybrid SAM)\n" + "=" * 60)
    ablation_results = run_ablation(tokenizer, epochs=ABLATION_EPOCHS, wandb_project=WANDB_PROJECT)

    # Plot Ablation Results
    fig_ab, axes_ab = plt.subplots(1, 2, figsize=(13, 5))
    for name, (losses, accs) in ablation_results.items():
        axes_ab[0].plot(losses, label=name)
        axes_ab[1].plot(accs, label=name)
    
    axes_ab[0].set_title("Ablation: Training Loss")
    axes_ab[0].set_xlabel("Epoch")
    axes_ab[0].set_ylabel("Loss")
    axes_ab[0].legend()

    axes_ab[1].set_title("Ablation: Validation Accuracy (%)")
    axes_ab[1].set_xlabel("Epoch")
    axes_ab[1].set_ylabel("Accuracy (%)")
    axes_ab[1].legend()

    plt.tight_layout()
    plt.savefig("ablation_comparison.png", dpi=150)
    print("Saved: ablation_comparison.png")

    # ── 4. Final Summary Console Output ──
    print("\n" + "=" * 60 + "\n  FINAL SUMMARY\n" + "=" * 60)
    print("Main Optimizers:")
    for name, d in results.items():
        print(f"  {name:10s}: val_acc={d['accs'][-1]:.2f}%  time={d['time']:.0f}s")
        
    print("\nHybrid Ablations:")
    for name, (losses, accs) in ablation_results.items():
        print(f"  {name:18s}: val_acc={accs[-1]:.2f}%")

if __name__ == "__main__":
    main()