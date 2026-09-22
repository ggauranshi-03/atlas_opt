import torch

class MuonSAMStale(torch.optim.Optimizer):
    """
    SAM+Muon with only last momentum re-used (not orthogonalized).
    Single-pass SAM:
    - Round 1: Double pass. Standard SAM perturbation. Saves updated momentum.
    - Round 2+: Single pass. Perturbs using stale momentum (2D) and stale grad (1D).

    NOTE on logged loss/accuracy: round 1 reports the clean loss at w_0 (from the
    first, unperturbed forward pass). From round 2 onward there is only one
    forward/backward pass per step, taken at the perturbed weights w_adv, so the
    loss and accuracy returned by step() are adversarial-point metrics, not clean
    ones. This matches AtlasOptimizer's convention, but is NOT directly comparable
    to MuonSAM/MuonSAMFrob, which always report the clean loss at w_t. See
    `self.reports_adversarial_metrics` below.
    """
    def __init__(self, base_optimizer, rho=0.0015, rho_vector=0.01):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert rho_vector >= 0.0, f"Invalid rho_vector, should be non-negative: {rho_vector}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho, "rho_vector": rho_vector}
        super(MuonSAMStale, self).__init__(self.param_groups, self.defaults)

        self.is_round_1 = True
        # True whenever the most recently returned loss/accuracy were measured at
        # the adversarially perturbed weights rather than at the clean w_t.
        self.reports_adversarial_metrics = False

    def state_dict(self):
        sd = super().state_dict()
        sd["is_round_1"] = self.is_round_1
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.is_round_1 = state_dict.pop("is_round_1", True)
        super().load_state_dict(state_dict)

    def _grad_norm_1d(self, use_stale=False):
        # Only the non-Muon (vector/Adam) groups contribute to this Euclidean block,
        # matched to how the outer optimizer actually routes each parameter.
        shared_device = self.param_groups[0]["params"][0].device
        tensors = []
        for group in self.param_groups:
            if group.get("use_muon", False):
                continue
            for p in group["params"]:
                if use_stale:
                    if "v_stale" in self.state[p]:
                        tensors.append(self.state[p]["v_stale"].norm(p=2).to(shared_device))
                else:
                    if p.grad is not None:
                        tensors.append(p.grad.norm(p=2).to(shared_device))

        if not tensors:
            return torch.tensor(1.0, device=shared_device)
        return torch.norm(torch.stack(tensors), p=2) + 1e-12

    @torch.no_grad()
    def perturb_weights(self, use_stale=False):
        # 1D Global Grad Norm
        norm_1d = self._grad_norm_1d(use_stale=use_stale)

        for group in self.param_groups:
            rho, rho_vector = group["rho"], group["rho_vector"]
            use_muon = group.get("use_muon", False)
            scale_1d = rho_vector / norm_1d

            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()

                if use_muon:
                    # 2D (Matrices): perturbation direction keyed on the group this
                    # optimizer actually routes to Muon, not on p.ndim, so a param
                    # that is 2D but NOT in the Muon group never lands here.
                    if use_stale:
                        base_state = self.base_optimizer.state.get(p, {})
                        assert "momentum_buffer" in base_state, (
                            "MuonSAMStale: no momentum_buffer found for a use_muon=True "
                            "parameter on a stale (round >= 2) perturbation. Round 1 "
                            "must run first to seed the base optimizer's momentum state."
                        )
                        direction = base_state["momentum_buffer"].float()
                    else:
                        if p.grad is None: continue
                        direction = p.grad.float()

                    frob_norm = direction.norm(p='fro') + 1e-12
                    P_t = direction / frob_norm
                    p.add_(P_t.to(p.dtype), alpha=rho)
                else:
                    # 1D (Vectors / AdamW)
                    if use_stale:
                        if "v_stale" in self.state[p]:
                            direction = self.state[p]["v_stale"]
                        else:
                            direction = p.grad if p.grad is not None else torch.zeros_like(p)
                    else:
                        if p.grad is None: continue
                        direction = p.grad

                    p.add_(direction * scale_1d.to(p.dtype))

        self.zero_grad()

    @torch.no_grad()
    def restore_and_update(self):
        # Save v_stale (g_adv) for non-Muon parameters before we call
        # base_optimizer.step(), because step() might modify or consume gradients
        # in some implementations.
        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            for p in group["params"]:
                if not use_muon and p.grad is not None:
                    self.state[p]["v_stale"] = p.grad.clone()

                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])  # Restore w_t

        # Update weights (this uses g_adv since we didn't zero_grad after the last closure)
        # For 2D, this automatically adds g_adv to momentum buffer M_t
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        closure = torch.enable_grad()(closure)

        if self.is_round_1:
            # 1. Take gradient at w_0
            loss = closure()
            self.reports_adversarial_metrics = False
            # 2. Perturb using standard grad
            self.perturb_weights(use_stale=False)
            # 3. Take gradient at w_adv (gives g_adv)
            closure()
            # 4. Restore, update weights, and save M_t / v_stale
            self.restore_and_update()

            self.is_round_1 = False
            return loss
        else:
            # 1. Perturb using stale momentum/grad
            self.perturb_weights(use_stale=True)
            # 2. First and ONLY forward/backward (gives g_adv) -- measured at w_adv
            loss_adv = closure()
            self.reports_adversarial_metrics = True
            # 3. Restore, update weights, and save M_t / v_stale
            self.restore_and_update()

            return loss_adv
