"""Orthogonally Projected SOMA: SOMA-PreNS5 and OP-SOMA-PostNS5.

Notation (per weight matrix i, Frobenius inner product):
    P_perp(A, X) = X - <X, A>_F / ||A||_F^2 * A          (returns X if A = 0)
    M      Muon's RAW momentum buffer, M <- beta * M + G   (its EMA-scaled version is (1 - beta) * M)
    NS5    the quintic Newton-Schulz orthogonalisation used by Muon (5 iterations)
    a_i    Muon's aspect-ratio correction sqrt(max(1, rows / cols))

Step t (both optimizers):
    if t == 1:  G = grad at W;  D = G - sigma * (1 - lambda) * G;  E_i = rho * NS5(D_i)          [fresh bootstrap]
    else:       PreNS5 : D_i = P_perp(M_i, G_prev_i);                 E_i = rho * NS5(D_i)
                PostNS5: D_i = G_prev_i - sigma * (1 - beta) * M_i;  Z_i = rho * NS5(D_i);
                         Q_i = P_perp(O_prev_i, Z_i);  E_i = rho * Q_i / ||Q_i||_op  (E_i = 0 if Q_i = 0)
    G_adv = grad at W + E (same minibatch)
    M_i = beta * M_i + G_adv_i;  N_i = G_adv_i + beta * M_i;  O_i = NS5(N_i);  W_i = W_i - eta * a_i * O_i
    weight constraint (decoupled weight decay W_i <- W_i * (1 - eta * wd), applied after the update)
    G_prev = G_adv;  O_prev = O
Two gradient evaluations at t = 1, one per step afterwards.

Scope: "matrix i" ranges over the matrices updated by Muon (the use_muon=True group: hidden 2-D weights; conv
filters are viewed as 2-D exactly as in Muon). The remaining parameters (embeddings, tied LM head, biases, norms)
have no Muon momentum M, so the algorithm defines no perturbation for them: E = 0, and they are updated by the
auxiliary AdamW (same code as the Muon baseline) with the single gradient G_adv evaluated at W + E.

Numerics (exact arithmetic is unchanged; these only control rounding):
  * Update path = Muon's own code (optimizers/muon.py: muon_update, bf16 NS5). It keeps the EMA buffer
    m = (1 - beta) * M, and since NS5 is scale-invariant, NS5(G + beta * M) = NS5((1 - beta) * G + beta * m): the update
    is the one above, and with rho = 0 both optimizers are bit-identical to the Muon baseline.
  * The raw buffer M = beta * M + G_adv used by the perturbation is kept separately in float64 (state "M"); rebuilding
    it as m / (1 - beta) from the float32 EMA buffer would add ~1e-7 relative rounding, which NS5 renormalises to a
    full-size direction when the exact D is 0 (t = 2: M = G_prev exactly, so PreNS5 gives D = 0 exactly).
  * Perturbation path (D, NS5(D), P_perp, ||.||_op, and O_prev = NS5(N_prev) used for the projection) in float64.
    Reason: at t = 2, M = G_prev, so for PostNS5 D = (1 - sigma (1 - beta)) G_1 and N_1 = (1 + beta) G_1 are parallel and
    Q = 0 exactly; with bf16 NS5 the computed Q is rounding noise (2-8% of ||Z|| on GPT-2), which the normalisation
    E = rho Q / ||Q||_op would blow up to a full-radius perturbation in a meaningless direction.
  * "Q_i = 0" is decided as ||Q_i||_F <= Q_ZERO_RTOL * ||Z_i||_F (float64 rounding is ~1e-16; genuine residuals on
    GPT-2 are 0.6-0.9 of ||Z_i||_F).
"""
import torch

from optimizers.muon import zeropower_via_newtonschulz5, muon_update, adam_update

NS_STEPS = 5
Q_ZERO_RTOL = 1e-10


def _mat(x):
    """2-D view used by Muon: conv filters (out, in, kh, kw) -> (out, in * kh * kw)."""
    return x.reshape(len(x), -1) if x.ndim == 4 else x


def ns5_f64(x):
    """Muon's NS5 (same quintic coefficients, 5 iterations, orientation rule) in float64, with the input normalised
    by its exact Frobenius norm. Muon divides by ||X|| + 1e-7, a divide-by-zero guard that makes NS5 slightly
    scale-dependent (NS5(0.95 G) != NS5(1.95 G) at the 1e-7/||G|| level); the algorithm relies on NS5 being
    scale-invariant (e.g. Q = 0 exactly at t = 2 for PostNS5), so the guard is replaced by an explicit X = 0 case."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = _mat(x).double()
    nrm = X.norm()
    if nrm.item() == 0.0:
        return torch.zeros_like(x, dtype=torch.float64)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / nrm
    for _ in range(NS_STEPS):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if tall:
        X = X.mT
    return X.reshape(x.shape)


def p_perp(a, x):
    """P_A^perp(X) = X - <X, A>_F / ||A||_F^2 * A; returns X if A = 0."""
    a_sq = a.square().sum()
    if a_sq.item() == 0.0:
        return x
    return x - ((x * a).sum() / a_sq) * a


class _ProjectedSOMA(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho, fsam_lambda=0.9, fsam_sigma=1.0):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        assert 0.0 <= fsam_lambda <= 1.0, f"Invalid fsam_lambda: {fsam_lambda}"
        # Re-use the base optimizer's parameter groups (Muon group + auxiliary AdamW group, same routing and
        # hyperparameters as the Muon baseline) so the LR scheduler drives eta exactly as for Muon.
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        super().__init__(self.param_groups, {"rho": rho, "fsam_lambda": fsam_lambda, "fsam_sigma": fsam_sigma})
        for g in self.param_groups:
            assert "use_muon" in g, "expects the param groups of SingleDeviceMuonWithAuxAdam"
        self.t = 0

    def state_dict(self):
        sd = super().state_dict()
        sd["t"] = self.t
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.t = state_dict.pop("t", 0)
        super().load_state_dict(state_dict)

    def _perturbation(self, group, p, state):
        raise NotImplementedError

    @torch.no_grad()
    def _perturb(self, bootstrap):
        """Computes E_i for every Muon matrix, saves W, and moves the weights to W + E."""
        for group in self.param_groups:
            if not group["use_muon"]:
                continue  # no Muon momentum -> no perturbation defined (E = 0)
            for p in group["params"]:
                state = self.state[p]
                if bootstrap:
                    if p.grad is None:
                        continue
                    g = p.grad.double()
                    d = g - group["fsam_sigma"] * (1.0 - group["fsam_lambda"]) * g
                    e = group["rho"] * ns5_f64(d)
                else:
                    e = self._perturbation(group, p, state)
                state["W"] = p.detach().clone()
                p.add_(e.to(p.dtype))

    @torch.no_grad()
    def _restore_and_update(self):
        """W <- saved W, then the Muon / auxiliary-AdamW update with G_adv (code shared with the Muon baseline)."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "W" in state:
                    p.copy_(state.pop("W"))
                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # same convention as SingleDeviceMuonWithAuxAdam
                lr, wd = group["lr"], group["weight_decay"]
                if group["use_muon"]:
                    beta = group["momentum"]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    g_adv = p.grad.detach().clone()          # muon_update modifies p.grad in place
                    if "M" not in state:
                        state["M"] = torch.zeros_like(p, dtype=torch.float64)
                    state["M"].mul_(beta).add_(g_adv.double())                       # M = beta * M + G_adv (raw)
                    state["O_prev"] = ns5_f64(g_adv.double() + beta * state["M"])    # O = NS5(N), N = G_adv + beta * M
                    # Muon update: m <- beta m + (1 - beta) G_adv, O = NS5(Nesterov), W -= eta * a * O.
                    update = muon_update(p.grad, state["momentum_buffer"], beta=beta)  # a_i is applied inside
                    p.add_(update.reshape(p.shape), alpha=-lr)
                    p.mul_(1.0 - lr * wd)                    # weight constraint, after the update (wd = 0 in v2)
                    state["G_prev"] = g_adv
                else:
                    # Auxiliary AdamW, identical to SingleDeviceMuonWithAuxAdam's non-Muon branch.
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"],
                                         group["betas"], group["eps"])
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, f"{type(self).__name__} requires a closure"
        closure = torch.enable_grad()(closure)
        self.t += 1
        if self.t == 1:
            loss = closure()                  # G at W (t = 1 only)
            self._perturb(bootstrap=True)
            self.zero_grad()
            closure()                         # G_adv at W + E, same minibatch
        else:
            self._perturb(bootstrap=False)
            self.zero_grad()
            loss = closure()                  # G_adv at W + E (the only gradient evaluation)
        self._restore_and_update()
        return loss


class SOMAPreNS5(_ProjectedSOMA):
    """SOMA-PreNS5: D_i = P_perp(M_i, G_prev_i) (projection against Muon's raw momentum, BEFORE NS5); E_i = rho NS5(D_i).
    Guarantee: D_i is orthogonal to M_i before NS5 (E_i itself need not be). Note: at t = 2, M = G_prev, so D = 0, E = 0."""

    def _perturbation(self, group, p, state):
        assert "G_prev" in state and "M" in state, "SOMA-PreNS5: t >= 2 needs G_prev and M from step t - 1"
        d = p_perp(state["M"], state["G_prev"].double())
        return group["rho"] * ns5_f64(d)


class OPSOMAPostNS5(_ProjectedSOMA):
    """OP-SOMA-PostNS5: D_i = G_prev_i - sigma (1 - beta) M_i; Z_i = rho NS5(D_i); Q_i = P_perp(O_prev_i, Z_i) (projection
    against the previous Nesterov-NS5 update, AFTER NS5); E_i = rho Q_i / ||Q_i||_op, or 0 if Q_i = 0. No NS5 after the
    projection, so E_i is orthogonal to O_prev_i. Note: at t = 2, D and N_1 are parallel, so Q = 0 and E = 0."""

    def _perturbation(self, group, p, state):
        assert "G_prev" in state and "M" in state and "O_prev" in state, \
            "OP-SOMA-PostNS5: t >= 2 needs G_prev, M and O_prev from step t - 1"
        d = state["G_prev"].double() - group["fsam_sigma"] * (1.0 - group["momentum"]) * state["M"]
        z = group["rho"] * ns5_f64(d)
        q = p_perp(state["O_prev"], z)
        if q.norm().item() <= Q_ZERO_RTOL * z.norm().item():
            return torch.zeros_like(q)
        return group["rho"] * q / torch.linalg.matrix_norm(_mat(q), ord=2)
