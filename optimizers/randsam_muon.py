import torch


class RandSAMMuon(torch.optim.Optimizer):
    """
    RandSAM-Muon: random-direction sharpness-aware perturbation + Muon base.

    The weights are moved to w + eps with eps = rho * u / ||u||_F, where u is a fresh Gaussian draw over *all*
    parameters (one global Frobenius norm, same scope as MuonSAMGFrob). The gradient is taken at the perturbed point
    and applied to w by the base optimizer. Unlike SAM the direction does not use the gradient, so only ONE
    forward/backward per step is needed (the first gradient of SAM is not required). This isolates how much of
    SAM's effect comes from the perturbation being *adversarial* rather than from perturbing at all.
    The reported training loss is therefore the loss at w + eps.
    """
    def __init__(self, base_optimizer, rho=0.01, seed=0):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho}
        super(RandSAMMuon, self).__init__(self.param_groups, self.defaults)
        self._seed = seed
        self._gens = {}

    def _gen(self, device):
        if device not in self._gens:   # private generator: does not disturb data/dropout randomness
            g = torch.Generator(device=device)
            g.manual_seed(self._seed)
            self._gens[device] = g
        return self._gens[device]

    @torch.no_grad()
    def perturb(self):
        draws, norm_sq = [], 0.0
        for group in self.param_groups:
            for p in group["params"]:
                u = torch.randn(p.shape, generator=self._gen(p.device), device=p.device, dtype=torch.float32)
                draws.append((group, p, u))
                norm_sq += u.square().sum().item()
        global_norm = (norm_sq ** 0.5) + 1e-12
        for group, p, u in draws:
            self.state[p]["old_p"] = p.data.clone()
            p.add_((u * (group["rho"] / global_norm)).to(p.dtype))

    @torch.no_grad()
    def restore_and_update(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state.pop("old_p"))  # back to w from w + eps
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "RandSAM-Muon requires a closure"
        closure = torch.enable_grad()(closure)
        self.perturb()
        loss = closure()          # gradient at w + eps (random direction)
        self.restore_and_update()
        return loss
