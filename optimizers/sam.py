import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al., 2020).

    The adversarial perturbation direction is the raw gradient itself, climbing to
    w_t + rho * g_t / ||g_t||_2 (one global L2 norm over every parameter). The outer
    update is an SGD (momentum) step on g'_t = grad L(w_t + eps_t).
    """
    def __init__(self, params, lr=0.01, rho=0.05, momentum=0.9, nesterov=True, weight_decay=0.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(lr=lr, rho=rho, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perturb(self):
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
                state = self.state[p]
                eps_t = rho * p.grad.detach() / global_g_norm

                state["old_p"] = p.data.clone()
                p.add_(eps_t)

    @torch.no_grad()
    def _descend(self):
        # Base step = torch.optim.SGD semantics (coupled weight decay, dampening 0).
        for group in self.param_groups:
            lr, mu, wd, nesterov = group["lr"], group["momentum"], group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])
                g = p.grad.add(p, alpha=wd) if wd != 0 else p.grad
                if mu != 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    else:
                        state["momentum_buffer"].mul_(mu).add_(g)
                    buf = state["momentum_buffer"]
                    g = g.add(buf, alpha=mu) if nesterov else buf
                p.add_(g, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure that sets p.grad"
        closure_ = torch.enable_grad()(closure)

        # g_t = grad L_B(w_t)
        loss = closure_()
        # Adversarial perturbation to w_t + eps_t
        self._perturb()
        # g'_t = grad L_B(w_t + eps_t)
        closure_()
        # Gradient-descent update, then restore w_t before applying it
        self._descend()

        return loss
