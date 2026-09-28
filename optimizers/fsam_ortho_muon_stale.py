import torch
from torch.optim.optimizer import Optimizer


def newtonschulz5(G, steps=5, eps=1e-7):
    """
    Quintic Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Produces an orthogonal polar factor with unit spectral norm ||O||_2 = 1.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = X.norm()
    X = X / norm.clamp(min=eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class FSAMOrthoMuonStale(Optimizer):
    """
    Stale Friendly Spectral SAM-Muon.

    Single-pass Friendly-SAM: unlike FSAMOrthoMuon (which recomputes a fresh clean
    gradient and a fresh friendly direction every step, at the cost of two forward/
    backward passes), this variant perturbs using the *previous* round's adversarial
    gradient g~_{t-1}, corrected by a friendly EMA built entirely from the g~ stream
    (never from a clean gradient):

        d_t   = g~_{t-1} - sigma * m_{t-1},   m_t = lam * m_{t-1} + (1 - lam) * g~_t

    Round 1 has no g~_{-1} yet, so it falls back to a closed-form bootstrap equivalent
    to "as if m_{-1} = 0": d_0 = g_0 * (1 - sigma * (1 - lam)), using one clean gradient
    (hence 2 forward/backward passes at t=0, 1 thereafter). Geometry: spectral (Newton-
    Schulz) for 2D/conv (Muon) parameters, Euclidean for 1D (AdamW) parameters.
    """
    def __init__(self, base_optimizer, rho=0.0015, rho_vector=None, ns_steps=5,
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
            "ns_steps": ns_steps,
            "fsam_lambda": fsam_lambda,
            "fsam_sigma": fsam_sigma,
        }
        super(FSAMOrthoMuonStale, self).__init__(self.param_groups, self.defaults)

        self.is_round_1 = True

    def state_dict(self):
        sd = super().state_dict()
        sd["is_round_1"] = self.is_round_1
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.is_round_1 = state_dict.pop("is_round_1", True)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def perturb_weights(self, use_stale):
        grad_norm_1d_sq = 0.0

        # Pass 1: build the friendly direction d[p] for every parameter.
        for group in self.param_groups:
            fsam_sigma = group["fsam_sigma"]
            fsam_lambda = group["fsam_lambda"]
            for p in group["params"]:
                state = self.state[p]
                if "friendly_ema" not in state:
                    state["friendly_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if use_stale:
                    assert "g_stale" in state, (
                        "FSAMOrthoMuonStale: no stale gradient found for a stale (round >= 2) "
                        "perturbation. Round 1 must run first to seed it.")
                    d = state["g_stale"] - fsam_sigma * state["friendly_ema"]
                else:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    d = g - fsam_sigma * (1.0 - fsam_lambda) * g
                state["friendly_dir"] = d

                if p.ndim < 2:
                    grad_norm_1d_sq += d.norm(p=2).item() ** 2

        grad_norm_1d = (grad_norm_1d_sq ** 0.5) + 1e-12

        # Pass 2: orthogonal perturbation for 2D/conv matrices, Euclidean for 1D vectors.
        for group in self.param_groups:
            rho = group["rho"]
            rho_vector = group.get("rho_vector", rho)
            ns_steps = group.get("ns_steps", 5)
            scale_1d = rho_vector / grad_norm_1d

            for p in group["params"]:
                state = self.state[p]
                if "friendly_dir" not in state:
                    continue
                state["old_p"] = p.data.clone()
                d = state.pop("friendly_dir")

                if p.ndim >= 2:
                    d_2d = d.float()
                    orig_shape = d_2d.shape
                    d_2d = d_2d.reshape(d_2d.size(0), -1)
                    O_t = newtonschulz5(d_2d, steps=ns_steps).reshape(orig_shape)
                    p.add_(O_t.to(p.dtype), alpha=rho)
                else:
                    p.add_(d.to(p.dtype), alpha=scale_1d)

        self.zero_grad()

    @torch.no_grad()
    def restore_and_update(self):
        # Save g~_t (this round's adversarial gradient) and update the friendly EMA
        # with it *before* calling base_optimizer.step(), which may mutate p.grad.
        for group in self.param_groups:
            fsam_lambda = group["fsam_lambda"]
            for p in group["params"]:
                state = self.state[p]
                if p.grad is not None:
                    g_tilde = p.grad.float()
                    state["g_stale"] = g_tilde.clone()
                    state["friendly_ema"].mul_(fsam_lambda).add_(g_tilde, alpha=1.0 - fsam_lambda)

                if "old_p" in state:
                    p.data.copy_(state["old_p"])

        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAMOrthoMuonStale requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        if self.is_round_1:
            # 1. Take gradient at w_0
            loss = closure()
            # 2. Perturb using the closed-form bootstrap direction
            self.perturb_weights(use_stale=False)
            # 3. Take gradient at w_adv (gives g~_0)
            closure()
            # 4. Restore, update weights, seed g_stale / friendly_ema for round 2
            self.restore_and_update()

            self.is_round_1 = False
            return loss
        else:
            # 1. Perturb using the stale friendly direction
            self.perturb_weights(use_stale=True)
            # 2. First and ONLY forward/backward (gives g~_t)
            loss_adv = closure()
            # 3. Restore, update weights, refresh g_stale / friendly_ema
            self.restore_and_update()

            return loss_adv
