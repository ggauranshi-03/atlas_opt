import torch
from torch.optim.optimizer import Optimizer


def newtonschulz5(G, steps=5, eps=1e-7):
    """
    Quintic Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Produces an orthogonal polar factor with unit spectral norm ||O||_2 = 1.
    """
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


class FSAMOrthoMuon(Optimizer):
    """
    Orthogonal Friendly SAM-Muon (FSAM-Ortho-Muon / Ortho-FSAM-Muon).
    
    A double-pass sharpness-aware optimizer combining Friendly-SAM perturbation
    with spectral orthogonalization:
      1. Tracks the batch-invariant ("friendly") component via gradient EMA m_t.
      2. Computes the friendly direction d_t = g_t - sigma * m_t.
      3. For 2D / Conv weight matrices (Muon block): orthogonalizes d_t via
         quintic Newton-Schulz iteration (unit spectral norm ||O_t||_2 = 1)
         and perturbs by rho * O_t.
      4. For 1D parameters (biases, normalization / AdamW block): applies
         Euclidean normalization scaled by rho_vector.
      5. Takes outer step via wrapped base optimizer (Muon + AdamW).
    """
    def __init__(self, base_optimizer, rho=0.0015, rho_vector=None, ns_steps=5,
                 fsam_lambda=0.9, fsam_sigma=1.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        if rho_vector is None:
            rho_vector = rho
        assert rho_vector >= 0.0, f"Invalid rho_vector, should be non-negative: {rho_vector}"
        assert 0.0 <= fsam_lambda <= 1.0, f"Invalid fsam_lambda, should be in [0, 1]: {fsam_lambda}"

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {
            "rho": rho,
            "rho_vector": rho_vector,
            "ns_steps": ns_steps,
            "fsam_lambda": fsam_lambda,
            "fsam_sigma": fsam_sigma,
        }
        super(FSAMOrthoMuon, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm_1d_sq = 0.0
        
        # 1. Update Friendly-EMA and compute Friendly Direction d[p]
        for group in self.param_groups:
            fsam_lambda = group["fsam_lambda"]
            fsam_sigma = group["fsam_sigma"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if "friendly_ema" not in state:
                    state["friendly_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                g = p.grad.float()
                
                # m_t = \lambda m_{t-1} + (1-\lambda) g_t
                state["friendly_ema"].mul_(fsam_lambda).add_(g, alpha=1 - fsam_lambda)
                
                # d_t = g_t - \sigma m_t
                d = g - fsam_sigma * state["friendly_ema"]
                state["friendly_dir"] = d
                
                if p.ndim < 2:
                    grad_norm_1d_sq += d.norm(p=2).item() ** 2

        grad_norm_1d = (grad_norm_1d_sq ** 0.5) + 1e-12

        # 2. Apply orthogonal perturbation to 2D matrices, Euclidean to 1D vectors
        for group in self.param_groups:
            rho = group["rho"]
            rho_vector = group.get("rho_vector", rho)
            ns_steps = group.get("ns_steps", 5)
            scale_1d = rho_vector / grad_norm_1d
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                state["old_p"] = p.data.clone()
                d = state["friendly_dir"]
                
                if p.ndim >= 2:
                    # Matrix/Conv parameters: Orthogonalization via Newton-Schulz
                    d_2d = d.float()
                    orig_shape = d_2d.shape
                    d_2d = d_2d.reshape(d_2d.size(0), -1)
                    O_t = newtonschulz5(d_2d, steps=ns_steps).reshape(orig_shape)
                    p.add_(O_t.to(p.dtype), alpha=rho)
                else:
                    # 1D vector parameters: Euclidean Geometry
                    p.add_(d.to(p.dtype), alpha=scale_1d)

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Restore parameters back to original state w_t
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])
                    
        # Apply base optimizer update using gradients computed at perturbed weights
        self.base_optimizer.step()
        
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAMOrthoMuon requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        # 1. Take gradient at w_t
        loss = closure()

        # 2. Update EMA, compute Friendly-SAM direction, and orthogonally perturb to w_t + e_t
        self.first_step(zero_grad=True)

        # 3. Take gradient at perturbed weights w_t + e_t
        closure()

        # 4. Restore original weights w_t, and apply base optimizer step
        self.second_step()

        return loss

