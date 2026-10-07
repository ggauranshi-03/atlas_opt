import torch
from torch.optim.optimizer import Optimizer


class FSAMFrobMuonStaleMomentum(Optimizer):
    """Frobenius counterpart of FSAMOrthoMuonStaleMomentum: identical, except the 2D/conv perturbation is rho * d / ||d||_F per layer instead of rho * NewtonSchulz(d)."""
    def __init__(self, base_optimizer, rho=0.0015, rho_vector=None,
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
            "fsam_lambda": fsam_lambda,
            "fsam_sigma": fsam_sigma,
        }
        super(FSAMFrobMuonStaleMomentum, self).__init__(self.param_groups, self.defaults)

        self.is_round_1 = True

    def state_dict(self):
        sd = super().state_dict()
        sd["is_round_1"] = self.is_round_1
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.is_round_1 = state_dict.pop("is_round_1", True)
        super().load_state_dict(state_dict)

    def _momentum_term(self, group, p):
        base_state = self.base_optimizer.state.get(p, {})
        if group.get("use_muon", False):
            assert "momentum_buffer" in base_state, (
                "FSAMFrobMuonStaleMomentum: no momentum_buffer found for a use_muon=True "
                "parameter on a stale (round >= 2) perturbation. Round 1 must run first.")
            beta = group.get("momentum", 0.95)
            return (1.0 - beta) * base_state["momentum_buffer"].float()
        assert "exp_avg" in base_state, (
            "FSAMFrobMuonStaleMomentum: no exp_avg found for a use_muon=False parameter "
            "on a stale (round >= 2) perturbation. Round 1 must run first.")
        beta1 = group.get("betas", (0.9, 0.999))[0]
        return (1.0 - beta1) * base_state["exp_avg"].float()

    @torch.no_grad()
    def perturb_weights(self, use_stale):
        grad_norm_1d_sq = 0.0

        # Pass 1: build the friendly direction d[p] for every parameter.
        for group in self.param_groups:
            fsam_sigma = group["fsam_sigma"]
            fsam_lambda = group["fsam_lambda"]
            for p in group["params"]:
                state = self.state[p]

                if use_stale:
                    assert "g_stale" in state, (
                        "FSAMFrobMuonStaleMomentum: no stale gradient found for a stale "
                        "(round >= 2) perturbation. Round 1 must run first to seed it.")
                    d = state["g_stale"] - fsam_sigma * self._momentum_term(group, p)
                else:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    d = g - fsam_sigma * (1.0 - fsam_lambda) * g
                state["friendly_dir"] = d

                if p.ndim < 2:
                    grad_norm_1d_sq += d.norm(p=2).item() ** 2

        grad_norm_1d = (grad_norm_1d_sq ** 0.5) + 1e-12

        # Pass 2: Frobenius perturbation for 2D/conv matrices, Euclidean for 1D vectors.
        for group in self.param_groups:
            rho = group["rho"]
            rho_vector = group.get("rho_vector", rho)
            scale_1d = rho_vector / grad_norm_1d

            for p in group["params"]:
                state = self.state[p]
                if "friendly_dir" not in state:
                    continue
                state["old_p"] = p.data.clone()
                d = state.pop("friendly_dir")

                if p.ndim >= 2:
                    p.add_((d / (d.norm(p="fro") + 1e-12)).to(p.dtype), alpha=rho)
                else:
                    p.add_(d.to(p.dtype), alpha=scale_1d)

        self.zero_grad()

    @torch.no_grad()
    def restore_and_update(self):
        # Save g~_t (this round's adversarial gradient) before base_optimizer.step()
        # updates its own momentum state using it (and may mutate p.grad in-place).
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if p.grad is not None:
                    state["g_stale"] = p.grad.float().clone()

                if "old_p" in state:
                    p.data.copy_(state["old_p"])

        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "FSAMFrobMuonStaleMomentum requires closure, but it was not provided")
        closure = torch.enable_grad()(closure)

        if self.is_round_1:
            # 1. Take gradient at w_0
            loss = closure()
            # 2. Perturb using the closed-form bootstrap direction
            self.perturb_weights(use_stale=False)
            # 3. Take gradient at w_adv (gives g~_0)
            closure()
            # 4. Restore, update weights (this seeds base_optimizer's momentum state)
            self.restore_and_update()

            self.is_round_1 = False
            return loss
        else:
            # 1. Perturb using the stale momentum-friendly direction
            self.perturb_weights(use_stale=True)
            # 2. First and ONLY forward/backward (gives g~_t)
            loss_adv = closure()
            # 3. Restore, update weights, refresh g_stale
            self.restore_and_update()

            return loss_adv
