import torch

from optimizers.fsam_ortho import newtonschulz5


class SAMOrtho(torch.optim.Optimizer):
    """
    Spectral counterpart of SAM: the raw gradient is orthogonalized (Newton-Schulz, rho * O_t) for 2D/conv
    matrices and Euclidean-normalized (rho_vector) for the 1D block, instead of one global L2 norm. The outer update
    is SAM's SGD-momentum step, so this differs from SAM only in the perturbation, as FSAMOrtho does from FSAM.
    """
    def __init__(self, params, lr=0.01, rho=0.05, rho_vector=None, ns_steps=5, momentum=0.9, nesterov=True,
                 weight_decay=0.0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        if rho_vector is None:
            rho_vector = rho
        assert rho_vector >= 0.0, f"Invalid rho_vector, should be non-negative: {rho_vector}"
        defaults = dict(lr=lr, rho=rho, rho_vector=rho_vector, ns_steps=ns_steps, momentum=momentum,
                        nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perturb(self):
        norm_1d_sq = sum(p.grad.detach().square().sum().item()
                         for group in self.param_groups for p in group["params"]
                         if p.grad is not None and p.ndim < 2)
        norm_1d = (norm_1d_sq ** 0.5) + 1e-12

        for group in self.param_groups:
            rho, rho_vector, ns_steps = group["rho"], group["rho_vector"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if p.ndim >= 2:
                    g_2d = g.float().reshape(g.size(0), -1)
                    eps_t = rho * newtonschulz5(g_2d, steps=ns_steps).reshape(g.shape).to(p.dtype)
                else:
                    eps_t = (rho_vector / norm_1d) * g

                self.state[p]["old_p"] = p.data.clone()
                p.add_(eps_t)

    @torch.no_grad()
    def _descend(self):
        # Base step = torch.optim.SGD semantics (coupled weight decay, dampening 0), identical to SAM.
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
        assert closure is not None, "SAMOrtho requires a closure that sets p.grad"
        closure_ = torch.enable_grad()(closure)

        loss = closure_()
        self._perturb()
        closure_()
        self._descend()

        return loss
