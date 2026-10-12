import torch


class FSAMAdam(torch.optim.Optimizer):
    """
    FSAM+AdamW: Friendly-SAM (F-SAM) with a plain AdamW outer step instead of SGD.
    Same friendly perturbation as optimizers/fsam.py (d_t = g_t - sigma * m_t, m_t an EMA of the
    gradient, one global L2 (Frobenius) norm over every parameter -- no spectral geometry), but the
    outer update is delegated to a wrapped torch.optim.AdamW instance, matching how the Muon-family
    wrappers delegate to their own base optimizer.
    """
    def __init__(self, base_optimizer, rho=0.05, fsam_lambda=0.9, fsam_sigma=1.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 <= fsam_lambda <= 1.0, f"Invalid fsam_lambda, should be in [0, 1]: {fsam_lambda}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho, "fsam_lambda": fsam_lambda, "fsam_sigma": fsam_sigma}
        super(FSAMAdam, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        global_d_norm_sq = 0.0

        # Pass 1: update the friendly EMA, compute d_t = g_t - sigma*m_t, and its global norm.
        for group in self.param_groups:
            lam, sigma = group["fsam_lambda"], group["fsam_sigma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_t = p.grad.detach()
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                state["m"].mul_(lam).add_(g_t, alpha=1 - lam)
                state["d_t"] = g_t - sigma * state["m"]
                global_d_norm_sq += state["d_t"].square().sum().item()
        global_d_norm = (global_d_norm_sq ** 0.5) + 1e-12

        # Pass 2: perturb using the global norm.
        for group in self.param_groups:
            rho = group["rho"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_t = state.pop("d_t")
                self.state[p]["old_p"] = p.data.clone()
                p.add_(d_t, alpha=rho / global_d_norm)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state.pop("old_p"))  # back to w from w + e(w)
        self.base_optimizer.step()  # AdamW update from the gradient at w + e(w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAMAdam requires a closure that sets p.grad"
        closure = torch.enable_grad()(closure)

        loss = closure()              # g_t = grad L(w_t)
        self.first_step(zero_grad=True)   # update EMA, w_t -> w_t + rho * d_t / ||d_t||
        closure()                     # g'_t = grad L(w_t + eps_t)
        self.second_step()            # restore w_t, then AdamW.step() on g'_t

        return loss
