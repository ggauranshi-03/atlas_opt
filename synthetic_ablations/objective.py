"""Reduced-rank deep linear regression: the matrix objective used by every experiment.

    F(W_1..W_L) = 1/(2nk) * || X P(W)^T - Y ||_F^2   (+ ridge/2 ||W_1||_F^2 for L = 1),
    P(W)        = W_L ... W_1  (end-to-end matrix, k x d).

See README.md for the derivation of the exact optimum F* and why this objective is used.
"""
import math
from dataclasses import dataclass, field

import torch

from .linalg import bview, frob, spectral_norm, total_frob


@dataclass
class Problem:
    depth: int
    d: int
    k: int
    hidden: int
    n: int
    shapes: list
    X: torch.Tensor
    Y: torch.Tensor
    sigma: torch.Tensor
    P_star: torch.Tensor
    P_opt: torch.Tensor
    F_star: float
    ridge: float
    constraint: str
    radius: float
    init: list = field(default_factory=list)
    rank_budget: int = 0


def layer_shapes(depth, d, k, hidden):
    if depth == 1:
        return [(k, d)]
    return [(hidden, d)] + [(hidden, hidden)] * (depth - 2) + [(k, hidden)]


def _random_orthonormal(m, r, generator, dtype):
    q, upper = torch.linalg.qr(torch.randn(m, r, generator=generator, dtype=dtype))
    return q * torch.sign(torch.diagonal(upper)).unsqueeze(0)


def _geometric(count, condition, dtype):
    if count == 1:
        return torch.ones(1, dtype=dtype)
    exponents = torch.arange(count, dtype=dtype) / (count - 1)
    return condition ** (-exponents)


def end_to_end(Ws):
    product = Ws[0]
    for w in Ws[1:]:
        product = w @ product
    return product


def objective(Ws, X, Y, ridge=0.0):
    """Full-data objective, per seed. Ws are (S, m, n); X is (n, d) or (S, b, d)."""
    P = end_to_end(Ws)
    residual = X @ P.transpose(-1, -2) - Y
    n, k = residual.shape[-2], residual.shape[-1]
    value = residual.square().sum((-2, -1)) / (2.0 * n * k)
    if ridge:
        value = value + 0.5 * ridge * sum(frob(w) ** 2 for w in Ws)
    return value


def full_objective(Ws, problem):
    return objective(Ws, problem.X, problem.Y, problem.ridge)


def excess_risk(P, problem):
    """Population excess risk E_x ||(P - P*) x||^2 / (2k): exact, because x ~ N(0, Sigma)."""
    delta = P - problem.P_star
    return ((delta @ problem.sigma) * delta).sum((-2, -1)) / (2.0 * problem.k)


def balancedness(Ws):
    """sum_l ||W_{l+1}^T W_{l+1} - W_l W_l^T||_F. Zero for a balanced factorization (and for L = 1)."""
    total = torch.zeros(Ws[0].shape[0], dtype=Ws[0].dtype)
    for lower, upper in zip(Ws[:-1], Ws[1:]):
        gram_upper = upper.transpose(-1, -2) @ upper
        gram_lower = lower @ lower.transpose(-1, -2)
        total = total + frob(gram_upper - gram_lower)
    return total


def full_gradient(Ws, problem):
    params = [w.detach().clone().requires_grad_(True) for w in Ws]
    value = full_objective(params, problem).sum()
    return [g.detach() for g in torch.autograd.grad(value, params)]


def hessian_lambda_max(Ws, problem, iters=30, generator=None):
    """Top Hessian eigenvalue of F (by magnitude) per seed, via power iteration on HVPs."""
    params = [w.detach().clone().requires_grad_(True) for w in Ws]
    value = full_objective(params, problem).sum()
    grads = torch.autograd.grad(value, params, create_graph=True)
    v = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in params]
    norm = total_frob(v)
    v = [x / bview(norm, x) for x in v]
    estimate = torch.zeros(Ws[0].shape[0], dtype=Ws[0].dtype)
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        estimate = sum((h * x).flatten(1).sum(1) for h, x in zip(hv, v))
        norm = total_frob(hv).clamp(min=1e-300)
        v = [h / bview(norm, h) for h in hv]
    return estimate.detach()


def reference_optimum(X, Y, depth, hidden, ridge):
    """Exact minimizer of F over end-to-end matrices reachable by the architecture.

    L = 1, ridge > 0 : ridge regression, (X^T X + n k ridge I) P^T = X^T Y.
    ridge = 0        : reduced-rank regression with rank budget r = min(k, d) (L = 1) or
                       min(k, d, hidden) (L >= 2).  With Yhat = Pi_X Y the projection onto
                       col(X), the optimum is the rank-r SVD truncation of Yhat (Eckart-Young),
                       F* = [ ||Y - Yhat||^2 + sum_{i > r} sigma_i(Yhat)^2 ] / (2nk).
    """
    n, d = X.shape
    k = Y.shape[1]
    if ridge:
        if depth != 1:
            raise ValueError("ridge > 0 is only supported for depth 1 (exact optimum known)")
        system = X.T @ X + n * k * ridge * torch.eye(d, dtype=X.dtype)
        return torch.linalg.solve(system, X.T @ Y).T, min(k, d)
    rank_budget = min(k, d) if depth == 1 else min(k, d, hidden)
    X_pinv = torch.linalg.pinv(X)
    fitted = X @ (X_pinv @ Y)
    u, s, vh = torch.linalg.svd(fitted, full_matrices=False)
    truncated = (u[:, :rank_budget] * s[:rank_budget]) @ vh[:rank_budget]
    return (X_pinv @ truncated).T, rank_budget


def make_problem(cfg, dtype=torch.float64):
    depth = int(cfg["depth"])
    d, k, hidden, n = int(cfg["d"]), int(cfg["k"]), int(cfg.get("hidden", cfg["k"])), int(cfg["n_train"])
    teacher_rank = int(cfg.get("teacher_rank", min(k, d)))
    generator = torch.Generator().manual_seed(int(cfg.get("data_seed", 1729)))

    basis = _random_orthonormal(d, d, generator, dtype)
    eigenvalues = _geometric(d, float(cfg.get("cond_x", 1.0)), dtype)
    eigenvalues = eigenvalues * d / eigenvalues.sum()
    sigma = (basis * eigenvalues) @ basis.T
    X = (torch.randn(n, d, generator=generator, dtype=dtype) * eigenvalues.sqrt()) @ basis.T

    left = _random_orthonormal(k, teacher_rank, generator, dtype)
    alignment = cfg.get("teacher_alignment", "random")
    if alignment == "random":
        right = _random_orthonormal(d, teacher_rank, generator, dtype)
    elif alignment == "high_variance":
        right = basis[:, :teacher_rank]
    elif alignment == "low_variance":
        right = basis[:, -teacher_rank:]
    else:
        raise ValueError(f"Unknown teacher_alignment: {alignment}")
    singular = _geometric(teacher_rank, float(cfg.get("cond_teacher", 1.0)), dtype)
    P_star = (left * singular) @ right.T
    P_star = P_star / math.sqrt((((P_star @ sigma) * P_star).sum() / k).item())

    Y = X @ P_star.T
    label_noise_std = float(cfg.get("label_noise_std", 0.0))
    if label_noise_std:
        Y = Y + label_noise_std * torch.randn(n, k, generator=generator, dtype=dtype)

    ridge = float(cfg.get("ridge", 0.0))
    P_opt, rank_budget = reference_optimum(X, Y, depth, hidden, ridge)
    shapes = layer_shapes(depth, d, k, hidden)

    init_generator = torch.Generator().manual_seed(int(cfg.get("init_seed", 31415)))
    init_scale = float(cfg.get("init_scale", 1.0))
    init = [torch.randn(rows, cols, generator=init_generator, dtype=dtype) * (init_scale / math.sqrt(cols))
            for rows, cols in shapes]

    constraint = cfg.get("constraint", "spectral")
    radius = math.inf
    if constraint != "none":
        optimum_layer = spectral_norm(P_opt.unsqueeze(0)).item() ** (1.0 / depth)
        init_layer = max(spectral_norm(w.unsqueeze(0)).item() for w in init)
        if constraint == "entrywise":
            optimum_layer = P_opt.abs().max().item() ** (1.0 / depth)
            init_layer = max(w.abs().max().item() for w in init)
        radius = float(cfg.get("radius_factor", 3.0)) * max(optimum_layer, init_layer)

    problem = Problem(depth=depth, d=d, k=k, hidden=hidden, n=n, shapes=shapes, X=X, Y=Y,
                      sigma=sigma, P_star=P_star, P_opt=P_opt, F_star=0.0, ridge=ridge,
                      constraint=constraint, radius=radius, init=init, rank_budget=rank_budget)
    problem.F_star = _objective_of_product(P_opt, problem)
    return problem


def _objective_of_product(P, problem):
    residual = problem.X @ P.T - problem.Y
    value = residual.square().sum() / (2.0 * problem.n * problem.k)
    if problem.ridge:
        value = value + 0.5 * problem.ridge * P.square().sum()
    return value.item()
