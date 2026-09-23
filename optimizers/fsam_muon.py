import torch
from torch.optim.optimizer import Optimizer

class FSAMMuon(Optimizer):
    """
    Global Friendly SAM-Muon (FSAM-Muon).
    
    A double-pass sharpness-aware optimizer that uses a Friendly-SAM perturbation 
    direction (gradient minus EMA of past gradients) and normalizes the perturbation 
    globally across the entire model (both 1D and 2D parameters) using a single 
    Euclidean norm.
    """
    def __init__(self, base_optimizer, rho=0.0015, fsam_lambda=0.9, fsam_sigma=1.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 <= fsam_lambda <= 1.0, f"Invalid fsam_lambda, should be in [0, 1]: {fsam_lambda}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho, "fsam_lambda": fsam_lambda, "fsam_sigma": fsam_sigma}
        super(FSAMMuon, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        global_norm_sq = 0.0
        
        # 1. Update Friendly-EMA and compute Friendly Direction d[p]
        for group in self.param_groups:
            fsam_lambda = group["fsam_lambda"]
            fsam_sigma = group["fsam_sigma"]
            
            for p in group["params"]:
                if p.grad is None: continue
                
                state = self.state[p]
                if "friendly_ema" not in state:
                    state["friendly_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # g_t
                g = p.grad.float()
                
                # m_t = \lambda m_{t-1} + (1-\lambda) g_t
                state["friendly_ema"].mul_(fsam_lambda).add_(g, alpha=1 - fsam_lambda)
                
                # d_t = g_t - \sigma m_t
                d = g - fsam_sigma * state["friendly_ema"]
                state["friendly_dir"] = d
                
                # Add to global norm (Euclidean)
                global_norm_sq += d.norm(p=2).item() ** 2

        global_norm = (global_norm_sq ** 0.5) + 1e-12

        # 2. Apply global perturbation
        for group in self.param_groups:
            rho = group["rho"]
            scale = rho / global_norm
            
            for p in group["params"]:
                if p.grad is None: continue
                
                state = self.state[p]
                state["old_p"] = p.data.clone()
                
                d = state["friendly_dir"]
                p.add_(d.to(p.dtype), alpha=scale)

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Restore parameters back to original state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])
                    
        # Apply base optimizer update using the gradients calculated at the perturbed weights
        self.base_optimizer.step()
        
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAM-Muon requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        # 1. Take gradient at w
        loss = closure()

        # 2. Update EMA, compute Friendly-SAM direction, and perturb (climbs to w + e)
        self.first_step(zero_grad=True)

        # 3. Take gradient at perturbed weights w + e
        closure()

        # 4. Restore original weights w, and apply base optimizer step
        self.second_step()

        return loss
