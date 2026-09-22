import sys

# Read fresh copy
with open("atlas_gauranshi.py", "r") as f:
    lines = f.readlines()

new_optimizer = """class OptimizedHybridSAM(torch.optim.Optimizer):
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

"""

# Find indices
opt_start, opt_end = -1, -1
ablation_start, ablation_end = -1, -1

for i, line in enumerate(lines):
    if line.startswith("class HybridSAMOptimizer(torch.optim.Optimizer):"):
        opt_start = i
    if line.startswith("# ─────────────────────────────── Training utilities ───────────────────────────"):
        opt_end = i
    if line.startswith("# ─────────────────────────────── Ablation Study"):
        ablation_start = i
    if line.startswith("# ─────────────────────────────── Main"):
        ablation_end = i

print(f"Opt: {opt_start} to {opt_end}, Ablation: {ablation_start} to {ablation_end}")

if opt_start != -1 and opt_end != -1:
    lines = lines[:opt_start] + [new_optimizer] + lines[opt_end:]
else:
    print("Failed to find opt block")
    sys.exit(1)

content = "".join(lines)

# Fix is_sam_like
content = content.replace("isinstance(opt, HybridSAMOptimizer)", "isinstance(opt, OptimizedHybridSAM)")
content = content.replace("isinstance(opt.optimizer, HybridSAMOptimizer)", "isinstance(opt.optimizer, OptimizedHybridSAM)")

# Fix make_optimizer
old_atlas = '''    elif name == "atlas":
        warmup_epochs = params.get("warmup_epochs", 2)
        base_opt = SinglePassHybridSAM(
            model, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            damping=params.get("damping", 1e-2), kfac_interval=params.get("kfac_interval", 20),
            kfac_decay=params.get("kfac_decay", 0.90), use_ns=params.get("use_ns", True),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )'''

new_atlas = '''    elif name == "atlas":
        warmup_epochs = params.get("warmup_epochs", 2)
        optim_params = [p for p in model.parameters() if p.requires_grad]
        base_opt = SinglePassOptimizedSAM(
            optim_params, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            ns_steps=params.get("ns_steps", 5),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )'''
content = content.replace(old_atlas, new_atlas)

old_hybrid = '''    elif name == "hybrid":
        warmup_epochs = params.get("warmup_epochs", 2)
        base_opt = HybridSAMOptimizer(
            model, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            damping=params.get("damping", 1e-2), kfac_interval=params.get("kfac_interval", 20),
            kfac_decay=params.get("kfac_decay", 0.90), use_ns=params.get("use_ns", True),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )'''

new_hybrid = '''    elif name == "hybrid":
        warmup_epochs = params.get("warmup_epochs", 2)
        optim_params = [p for p in model.parameters() if p.requires_grad]
        base_opt = OptimizedHybridSAM(
            optim_params, lr=lr, weight_decay=wd, rho=params.get("rho", 0.05),
            ns_steps=params.get("ns_steps", 5),
            momentum_start=params.get("momentum_start", 0.85), momentum_end=params.get("momentum_end", 0.95),
            momentum_warmup_steps=params.get("momentum_warmup_steps", 200),
        )'''
content = content.replace(old_hybrid, new_hybrid)

# Update default hyperparams
old_hp = '"atlas": {"damping": 0.01, "lr": 1e-4, "weight_decay": 1e-4},'
new_hp = '"atlas": {"ns_steps": 5, "lr": 1e-4, "weight_decay": 1e-4},'
content = content.replace(old_hp, new_hp)

with open("atlas_optimized_sam.py", "w") as f:
    f.write(content)

print("Modification complete.")
