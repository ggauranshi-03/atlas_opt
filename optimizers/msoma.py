import torch


class MSOMA(torch.optim.Optimizer):
    """
    MSOMA: Momentum-SAM (Becker et al., 2024, arXiv:2401.12033) perturbation + Muon base optimizer.

    This is optimizers/msam.py with the outer SGD step replaced by the wrapped base optimizer (Muon on hidden matrices,
    auxiliary AdamW on embeddings / 1-D parameters, i.e. utils.helpers._muon_with_aux_adam), in the same wrapper
    structure as RandSAMMuon.

    As in MSAM, the perturbation direction is the optimizer's own momentum, not a gradient, so each step needs ONE
    forward/backward pass (after a one-time bootstrap in which no momentum exists yet and the gradient is taken at w):

        w~_t   = w_t - rho * v_t / ||v_t||_2      (ONE global L2 norm over every parameter)
        g_t    = grad L(w~_t)                     (the single oracle call; closure() leaves it in p.grad)
        w_{t+1} = BaseOptimizer(w_t, g_t)          (Muon update from g_t, applied at the unperturbed w_t)

    v_t is read from the base optimizer's state, exactly as SOMA's stale-momentum variant does: Muon's momentum_buffer
    for use_muon=True parameters and AdamW's exp_avg for the others. The buffer is updated by the base optimizer with
    the gradient taken at the perturbed point, so (as in MSAM) the same gradient drives both the momentum and the update.
    Muon's buffer is an EMA of gradients (m <- beta*m + (1-beta)*g) whereas MSAM's is heavy-ball (v <- mu*v + g); the two
    differ only by a constant factor 1-beta, which the normalisation by ||v|| removes.

    p.data is the true, unperturbed weight between step() calls. The reported loss is the loss at the perturbed point w~_t
    (0 perturbation in the first step).
    """
    def __init__(self, base_optimizer, rho=0.01):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho}
        super(MSOMA, self).__init__(self.param_groups, self.defaults)

    def _momentum(self, group, p):
        base_state = self.base_optimizer.state.get(p, {})
        return base_state.get("momentum_buffer" if group.get("use_muon", False) else "exp_avg")

    @torch.no_grad()
    def perturb(self):
        """w -= rho * v / ||v|| with one global L2 norm. No-op while no momentum exists (one-time bootstrap)."""
        bufs = []
        for group in self.param_groups:
            for p in group["params"]:
                buf = self._momentum(group, p)
                if buf is not None:
                    bufs.append((group, p, buf))
        if not bufs:
            return
        global_norm = torch.stack([buf.float().norm() for _, _, buf in bufs]).norm().item()
        if global_norm == 0.0:
            return
        for group, p, buf in bufs:
            self.state[p]["old_p"] = p.data.clone()
            p.add_(buf.to(p.dtype), alpha=-group["rho"] / global_norm)

    @torch.no_grad()
    def restore_and_update(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state.pop("old_p"))   # back to w_t from w~_t
        self.base_optimizer.step()                     # Muon / AdamW update from the gradient at w~_t

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "MSOMA requires a closure that sets p.grad"
        closure = torch.enable_grad()(closure)
        self.perturb()
        loss = closure()          # gradient at w~_t (at w_0 in the first step)
        self.restore_and_update()
        return loss
