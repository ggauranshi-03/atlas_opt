"""Heavy-tailed stochastic gradient oracles.

Both oracles are gradients of a stochastic function f(W; xi) with E_xi[grad f] = grad F
(for alpha > 1), and both SAM passes of one step reuse the same xi (the analogue of
evaluating the same minibatch twice).

  label    : f = F_B(W) - 1/(bk) <E, X_B P(W)^T>,  E_ij = scale * zeta_ij.
             Gradient noise on P is -(1/(bk)) sum_i eps_i x_i^T (a sum of b rank-one heavy-tailed
             terms) and reaches every layer through the chain rule.
  additive : f = F_B(W) + sum_l <Xi_l, W_l>,  (Xi_l)_ij = scale * tau_l * zeta_ij, where tau_l is
             the RMS of the clean full-batch gradient of layer l at initialization. With a full
             batch this is exactly the oracle of Fatkhullin, Huebler & Lan (2025), Section 6.

zeta is two-sided Pareto(alpha) rescaled to have median |zeta| = 1 (Gaussian when alpha = inf),
so alpha changes the tail but not the typical noise magnitude.
"""
import math

import torch

from .objective import end_to_end, full_gradient, objective

GAUSSIAN_ABS_MEDIAN = 0.6744897501960817


def two_sided_pareto(shape, alpha, generator, dtype):
    if math.isinf(alpha):
        return torch.randn(shape, generator=generator, dtype=dtype) / GAUSSIAN_ABS_MEDIAN
    smallest = 2.0 ** -53 if dtype == torch.float64 else 2.0 ** -24
    uniform = torch.rand(shape, generator=generator, dtype=dtype).clamp_(min=smallest)
    signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).mul_(2).sub_(1).to(dtype)
    return signs * uniform.pow(-1.0 / alpha) / (2.0 ** (1.0 / alpha))


class Oracle:
    def __init__(self, problem, cfg, seeds, alpha, generator, dtype):
        self.problem = problem
        self.model = cfg.get("model", "label")
        self.scale = float(cfg.get("scale", 1.0))
        batch = cfg.get("batch_size")
        self.batch = None if batch is None or int(batch) >= problem.n else int(batch)
        self.seeds = seeds
        self.alpha = alpha
        self.generator = generator
        self.dtype = dtype
        if self.model == "additive":
            init = [w.unsqueeze(0) for w in problem.init]
            self.tau = [g.square().mean().sqrt().item() for g in full_gradient(init, problem)]
            
            cond_noise = float(cfg.get("cond_noise", 1.0))
            self.L = []
            self.R = []
            from .objective import _random_orthonormal, _geometric
            for shape in problem.shapes:
                out_dim, in_dim = shape
                if cond_noise > 1.0:
                    U_L = _random_orthonormal(out_dim, out_dim, generator, dtype)
                    ell = _geometric(out_dim, cond_noise, dtype)
                    ell = ell * out_dim / ell.sum()
                    L = (U_L * ell.sqrt()) @ U_L.T
                    
                    U_R = _random_orthonormal(in_dim, in_dim, generator, dtype)
                    r = _geometric(in_dim, cond_noise, dtype)
                    r = r * in_dim / r.sum()
                    R = (U_R * r.sqrt()) @ U_R.T
                else:
                    L = torch.eye(out_dim, dtype=dtype)
                    R = torch.eye(in_dim, dtype=dtype)
                self.L.append(L)
                self.R.append(R)
        elif self.model != "label":
            raise ValueError(f"Unknown noise model: {self.model}")

    def draw(self):
        """One stochastic sample xi_t for every seed."""
        sample = {}
        if self.batch is not None:
            sample["index"] = torch.randint(0, self.problem.n, (self.seeds, self.batch), generator=self.generator)
        rows = self.batch or self.problem.n
        if self.model == "label":
            sample["noise"] = self.scale * two_sided_pareto(
                (self.seeds, rows, self.problem.k), self.alpha, self.generator, self.dtype)
        else:
            sample["noise"] = []
            for tau, shape, L, R in zip(self.tau, self.problem.shapes, self.L, self.R):
                Z = two_sided_pareto((self.seeds, *shape), self.alpha, self.generator, self.dtype)
                Xi = L @ Z @ R.T
                sample["noise"].append(self.scale * tau * Xi)
        return sample

    def _batch(self, sample):
        if "index" not in sample:
            return self.problem.X, self.problem.Y
        return self.problem.X[sample["index"]], self.problem.Y[sample["index"]]

    def loss(self, Ws, sample):
        X, Y = self._batch(sample)
        if self.problem.loss_type == "l1":
            from .objective_l1 import objective_l1
            value = objective_l1(Ws, X, Y, self.problem.ridge)
        else:
            value = objective(Ws, X, Y, self.problem.ridge)
        if self.model == "label":
            outputs = X @ end_to_end(Ws).transpose(-1, -2)
            rows = outputs.shape[-2]
            value = value - (sample["noise"] * outputs).sum((-2, -1)) / (rows * self.problem.k)
        else:
            value = value + sum((xi * w).flatten(1).sum(1) for xi, w in zip(sample["noise"], Ws))
        return value

    def gradients(self, Ws, sample):
        params = [w.detach().requires_grad_(True) for w in Ws]
        with torch.enable_grad():
            value = self.loss(params, sample).sum()
            grads = torch.autograd.grad(value, params)
        return [g.detach() for g in grads]
