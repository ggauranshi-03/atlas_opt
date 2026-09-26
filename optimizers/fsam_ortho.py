import torch


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


class FSAMOrtho(torch.optim.Optimizer):
    """
    Orthogonal Friendly-SAM (F-SAM-Ortho / Ortho-FSAM), Algorithm 1 with Spectral Geometry.

    m_t tracks the batch-invariant ("friendly") component of the gradient via an EMA.
    The adversarial perturbation direction d_t = g_t - sigma * m_t is orthogonalized
    via quintic Newton-Schulz iteration for 2D matrix parameters (spectral norm = 1)
    and scaled by rho. For 1D vector parameters, Euclidean normalization is used.
    The outer update is gradient descent on g'_t = grad L(w_t + eps_t).
    """
    def __init__(self, params, lr=0.01, rho=0.05, rho_vector=None, lam=0.9, sigma=1.0, ns_steps=5):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        if rho_vector is None:
            rho_vector = rho
        assert rho_vector >= 0.0, f"Invalid rho_vector, should be non-negative: {rho_vector}"
        defaults = dict(lr=lr, rho=rho, rho_vector=rho_vector, lam=lam, sigma=sigma, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perturb(self):
        grad_norm_1d_sq = 0.0
        
        # Pass 1: Update EMA and compute friendly directions d_t
        for group in self.param_groups:
            lam, sigma = group["lam"], group["sigma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_t = p.grad.detach()

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                state["m"].mul_(lam).add_(g_t, alpha=1 - lam)

                state["d_t"] = g_t - sigma * state["m"]
                if p.ndim < 2:
                    grad_norm_1d_sq += state["d_t"].square().sum().item()
                
        grad_norm_1d = (grad_norm_1d_sq ** 0.5) + 1e-12
        
        # Pass 2: Apply perturbation (orthogonalization for 2D, Euclidean for 1D)
        for group in self.param_groups:
            rho = group["rho"]
            rho_vector = group.get("rho_vector", rho)
            ns_steps = group.get("ns_steps", 5)
            scale_1d = rho_vector / grad_norm_1d

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_t = state.pop("d_t")
                
                if p.ndim >= 2:
                    d_2d = d_t.float()
                    orig_shape = d_2d.shape
                    d_2d = d_2d.reshape(d_2d.size(0), -1)
                    eps_t = rho * newtonschulz5(d_2d, steps=ns_steps).reshape(orig_shape).to(p.dtype)
                else:
                    eps_t = scale_1d * d_t

                state["old_p"] = p.data.clone()
                p.add_(eps_t)

    @torch.no_grad()
    def _descend(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])
                p.add_(p.grad, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAMOrtho requires a closure that sets p.grad"
        closure_ = torch.enable_grad()(closure)

        # Line 4: g_t = grad L_B(w_t)
        loss = closure_()
        # Lines 5-7: friendly EMA update + orthogonal perturbation
        self._perturb()
        # Lines 8-9: g'_t = grad L_B(w_t + eps_t)
        closure_()
        # Lines 10-11: gradient-descent update, then restore w_t before applying it
        self._descend()

        return loss

