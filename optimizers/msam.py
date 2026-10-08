import torch


class MSAM(torch.optim.Optimizer):
    """
    Momentum-SAM (Becker et al., 2024, arXiv:2401.12033), efficient variant.

    The perturbation direction is the optimizer's own momentum buffer, not a fresh gradient:
    climbing to w_t - rho * v_t / ||v_t||_2 (one global L2 norm over every parameter, v_t
    accumulated as plain heavy-ball momentum). Because the SAME gradient evaluated at that
    perturbed point both updates the momentum buffer and performs the outer SGD step, this
    costs one oracle call per step (after a one-time clean-gradient bootstrap, since v_0 = 0
    leaves nothing to perturb by).

    Paper's pseudocode persists the perturbed weights w~_t as the stored parameter state
    across steps:
        w~_t     = w_t     - rho * v_t     / ||v_t||
        g        = grad L(w~_t)
        v_{t+1}  = mu * v_t + g
        w_{t+1}  = w_t      - lr * v_{t+1}
        w~_{t+1} = w_{t+1}  - rho * v_{t+1} / ||v_{t+1}||
    Implemented here with the perturb/restore transient inside a single step() call instead
    (as in SAM above), so p.data is always the true, unperturbed weight between calls --
    mathematically identical, since only the momentum buffer needs to persist, and it matches
    every other optimizer in this file (no separate "remove perturbation before eval" step
    needed elsewhere in the training loop).
    """
    def __init__(self, params, lr=0.01, rho=0.05, momentum=0.9, weight_decay=0.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(lr=lr, rho=rho, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perturb(self):
        """w -= rho * v / ||v|| (global norm). No-op wherever no momentum buffer exists yet
        (v_0 = 0, the one-time bootstrap)."""
        global_v_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                buf = self.state[p].get("momentum_buffer")
                if buf is not None:
                    global_v_norm_sq += buf.square().sum().item()
        if global_v_norm_sq == 0.0:
            return
        global_v_norm = global_v_norm_sq ** 0.5

        for group in self.param_groups:
            rho = group["rho"]
            for p in group["params"]:
                buf = self.state[p].get("momentum_buffer")
                if buf is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(buf, alpha=-rho / global_v_norm)

    @torch.no_grad()
    def _restore(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state.pop("old_p"))

    @torch.no_grad()
    def _descend(self):
        # v_{t+1} = mu*v_t + g; w_{t+1} = w_t - lr*v_{t+1} (decoupled weight decay).
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.detach()
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                else:
                    state["momentum_buffer"].mul_(mu).add_(g)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(state["momentum_buffer"], alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "MSAM requires a closure that sets p.grad"
        closure_ = torch.enable_grad()(closure)

        # w~_t = w_t - rho * v_t / ||v_t||
        self._perturb()
        # g_MSAM = grad L(w~_t)  -- the one oracle call per step
        loss = closure_()
        # back to the true w_t
        self._restore()
        # v_{t+1} = mu*v_t + g_MSAM; w_{t+1} = w_t - lr*v_{t+1}
        self._descend()

        return loss
