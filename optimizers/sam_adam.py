import torch


class SAMAdam(torch.optim.Optimizer):
    """
    SAM+AdamW: Sharpness-Aware Minimization (Foret et al., 2020) with a plain AdamW outer step
    instead of SGD -- tests whether SAM's benefit survives swapping the outer optimizer to an
    adaptive one. Same perturbation as optimizers/sam.py (raw gradient, one global L2 (Frobenius)
    norm over every parameter -- no spectral/Newton-Schulz geometry, unlike the Muon-wrapping SAM
    variants), but the outer update is delegated to a wrapped torch.optim.AdamW instance instead of
    being hand-rolled, matching how the Muon-family wrappers (MuonSAM, FSAMOrthoMuon, ...) delegate
    to their own base optimizer.
    """
    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho}
        super(SAMAdam, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        global_g_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                global_g_norm_sq += p.grad.detach().square().sum().item()
        global_g_norm = (global_g_norm_sq ** 0.5) + 1e-12

        for group in self.param_groups:
            rho = group["rho"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad, alpha=rho / global_g_norm)

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
        assert closure is not None, "SAMAdam requires a closure that sets p.grad"
        closure = torch.enable_grad()(closure)

        loss = closure()              # g_t = grad L(w_t)
        self.first_step(zero_grad=True)   # w_t -> w_t + rho * g_t / ||g_t||
        closure()                     # g'_t = grad L(w_t + eps_t)
        self.second_step()            # restore w_t, then AdamW.step() on g'_t

        return loss
