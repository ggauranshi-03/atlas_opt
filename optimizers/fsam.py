import torch


class FSAM(torch.optim.Optimizer):
    """
    Friendly-SAM (F-SAM), Algorithm 1.

    m_t tracks the batch-invariant ("friendly") component of the gradient via an EMA.
    The adversarial perturbation direction d_t = g_t - sigma * m_t keeps only the
    batch-variant ("unfriendly") component, so the climb targets sharpness caused by
    stochasticity rather than the mean gradient itself. The outer update is plain
    gradient descent on g'_t = grad L(w_t + eps_t) (no momentum in the descent step).
    """
    def __init__(self, params, lr=0.01, rho=0.05, lam=0.9, sigma=1.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(lr=lr, rho=rho, lam=lam, sigma=sigma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perturb(self):
        for group in self.param_groups:
            lam, sigma, rho = group["lam"], group["sigma"], group["rho"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_t = p.grad.detach().clone()

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                state["m"].mul_(lam).add_(g_t, alpha=1 - lam)

                d_t = g_t - sigma * state["m"]
                d_norm = d_t.norm() + 1e-12
                eps_t = rho * d_t / d_norm

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
        assert closure is not None, "F-SAM requires a closure that sets p.grad"
        closure_ = torch.enable_grad()(closure)

        # Line 4: g_t = grad L_B(w_t)
        loss = closure_()
        # Lines 5-7: momentum update + adversarial perturbation
        self._perturb()
        # Lines 8-9: g'_t = grad L_B(w_t + eps_t)
        closure_()
        # Lines 10-11: gradient-descent update, then restore w_t before applying it
        self._descend()

        return loss
