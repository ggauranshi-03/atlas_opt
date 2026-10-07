import torch


class MuonSAMGFrob(torch.optim.Optimizer):
    """
    Global-Frobenius counterpart of MuonSAMFrob: eps = rho * g / ||g|| with one Euclidean norm over every parameter
    (matrices and vectors together), instead of a norm per layer. With fsam_sigma = 0 this is FSAMMuon exactly.
    """
    def __init__(self, base_optimizer, rho=0.0015):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho}
        super(MuonSAMGFrob, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        norm_sq = sum(p.grad.float().square().sum().item()
                      for group in self.param_groups for p in group["params"] if p.grad is not None)
        global_norm = (norm_sq ** 0.5) + 1e-12

        for group in self.param_groups:
            scale = group["rho"] / global_norm
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_((p.grad.float() * scale).to(p.dtype))

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        # 1. Take gradient at w
        loss = closure()

        # 2. Add adversarial perturbation (climbs to w + e(w))
        self.first_step(zero_grad=True)

        # 3. Take gradient at w + e(w)
        closure()

        # 4. Step base optimizer using new gradients, and revert w + e(w) back to w
        self.second_step()

        return loss
