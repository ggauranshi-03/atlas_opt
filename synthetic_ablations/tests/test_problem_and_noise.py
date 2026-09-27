"""Exactness of F*, correctness of the noise model, and the orthogonalization routines."""
import math

import torch

from synthetic_ablations.linalg import newton_schulz5, polar_svd
from synthetic_ablations.noise import Oracle, two_sided_pareto
from synthetic_ablations.objective import full_gradient, full_objective, make_problem

BASE = dict(d=16, k=8, hidden=8, n_train=80, teacher_rank=4, cond_x=20.0, cond_teacher=5.0, constraint="none")


def _descend(problem, steps=6000, lr=None):
    """Noiseless full-batch gradient descent from the fixed initialization."""
    Ws = [w.unsqueeze(0).clone() for w in problem.init]
    lr = lr or 0.5 / (torch.linalg.eigvalsh(problem.X.T @ problem.X / problem.n).max().item() / problem.k)
    for _ in range(steps):
        grads = full_gradient(Ws, problem)
        Ws = [w - lr * g for w, g in zip(Ws, grads)]
    return full_objective(Ws, problem).item()


def test_realizable_optimum_is_zero():
    problem = make_problem(dict(BASE, depth=2))
    assert problem.F_star < 1e-20
    assert torch.allclose(problem.P_opt, problem.P_star, atol=1e-8)


def test_ridge_optimum_matches_gradient_descent():
    problem = make_problem(dict(BASE, depth=1, label_noise_std=0.3, ridge=1e-3))
    assert problem.F_star > 0
    assert abs(_descend(problem) - problem.F_star) < 1e-9


def test_reduced_rank_optimum_matches_deep_linear_descent():
    # hidden = 2 < teacher rank 4: the bottleneck binds, F* > 0 comes from the Eckart-Young truncation.
    problem = make_problem(dict(BASE, depth=2, hidden=2, label_noise_std=0.1, init_scale=1.0))
    assert problem.rank_budget == 2 and problem.F_star > 1e-3
    reached = _descend(problem, steps=20000, lr=0.3)
    assert reached >= problem.F_star - 1e-10
    assert (reached - problem.F_star) / problem.F_star < 1e-4


def test_teacher_is_normalized_to_unit_output_variance():
    problem = make_problem(dict(BASE, depth=1))
    variance = ((problem.P_star @ problem.sigma) * problem.P_star).sum() / problem.k
    assert abs(variance.item() - 1.0) < 1e-10


def test_pareto_median_and_tail():
    generator = torch.Generator().manual_seed(0)
    for alpha in (1.1, 1.6, 3.0, math.inf):
        z = two_sided_pareto((400_000,), alpha, generator, torch.float64)
        assert abs(z.abs().median().item() - 1.0) < 0.01, alpha
        assert abs(z.sign().mean().item()) < 0.01
        if not math.isinf(alpha):
            # P(|zeta| > t) = t^-alpha / 2 for t >= 2^(-1/alpha)
            for t in (3.0, 10.0):
                expected = t ** (-alpha) / 2.0
                observed = (z.abs() > t).double().mean().item()
                binomial_sd = math.sqrt(expected * (1 - expected) / z.numel())
                assert abs(observed - expected) < 4 * binomial_sd, (alpha, t, observed, expected)


def test_label_oracle_is_unbiased():
    problem = make_problem(dict(BASE, depth=2))
    Ws = [w.unsqueeze(0).repeat(4000, 1, 1) for w in problem.init]
    oracle = Oracle(problem, dict(model="label", scale=1.0, batch_size=16), 4000, 3.0,
                    torch.Generator().manual_seed(1), torch.float64)
    grads = oracle.gradients(Ws, oracle.draw())
    clean = full_gradient([w[:1] for w in Ws], problem)
    for g, c in zip(grads, clean):
        error = (g.mean(0) - c[0]).norm() / c[0].norm()
        assert error < 0.1, error.item()


def test_additive_oracle_full_batch_is_clean_gradient_plus_noise():
    problem = make_problem(dict(BASE, depth=1))
    Ws = [w.unsqueeze(0).repeat(3, 1, 1) for w in problem.init]
    oracle = Oracle(problem, dict(model="additive", scale=0.5, batch_size=None), 3, 1.6,
                    torch.Generator().manual_seed(2), torch.float64)
    sample = oracle.draw()
    grads = oracle.gradients(Ws, sample)
    clean = full_gradient(Ws, problem)
    assert torch.allclose(grads[0] - clean[0], sample["noise"][0], atol=1e-12)


def test_orthogonalizers():
    g = torch.randn(5, 12, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    exact = torch.linalg.svdvals(polar_svd(g))
    assert torch.allclose(exact, torch.ones_like(exact), atol=1e-10)
    approx = torch.linalg.svdvals(newton_schulz5(g))
    assert approx.min() > 0.5 and approx.max() < 1.25
    tall = torch.linalg.svdvals(newton_schulz5(g.transpose(-1, -2)))
    assert torch.allclose(tall, approx, atol=1e-10)
