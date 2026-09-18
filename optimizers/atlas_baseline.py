import torch
import math

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

class AtlasOptimizer(torch.optim.Optimizer):
    """
    Baseline Atlas Optimizer: Uses Newton-Schulz orthogonalization for perturbation 
    and update directions on 2D parameters. 1D parameters use AdamW.
    """
    def __init__(self, model_or_params, lr=0.02, weight_decay=0.01, rho=0.015, rho_vector=0.015, momentum=0.9665, nesterov=True, ns_steps=5, adam_lr=3e-4):
        if isinstance(model_or_params, torch.nn.Module):
            muon_params = []
            adam_params = []
            for n, p in model_or_params.named_parameters():
                if not p.requires_grad: continue
                if p.ndim >= 2 and "embed" not in n and "wte" not in n and "wpe" not in n:
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            params = [
                dict(params=adam_params, lr=adam_lr, use_muon=False, rho_vector=rho_vector, weight_decay=weight_decay),
                dict(params=muon_params, lr=lr, weight_decay=weight_decay, rho=rho, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, use_muon=True)
            ]
        else:
            params = model_or_params

        defaults = dict(lr=lr, weight_decay=weight_decay, rho=rho, rho_vector=rho_vector, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adam_lr=adam_lr)
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None: raise RuntimeError("AtlasOptimizer requires a closure")
        is_first_step = (self._step == 0)

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
                        p.data.add_(stale.to(p.dtype), alpha=rho) # No normalization for baseline
                    else:
                        s_norm = stale.norm() + 1e-8
                        p.data.add_(stale.to(p.dtype), alpha=rho / s_norm)

        self.zero_grad()
        with torch.enable_grad():
            closure_loss = closure()
            if is_first_step: loss = closure_loss
            else: loss = closure_loss
            
        self._step += 1
            
        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            lr = group.get("lr", 0.02)
            wd = group.get("weight_decay", 1e-4)
            
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
                    g_nest = g_adv + mu * buf if group.get("nesterov", True) else buf
                    
                    O_t = newtonschulz5(g_nest.view(g_nest.size(0), -1), steps=group.get("ns_steps", 5)).view_as(g_nest)
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

                    state["v_stale"] = g_adv.clone()

        return loss
