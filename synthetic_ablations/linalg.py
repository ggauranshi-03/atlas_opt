"""Batched matrix utilities. Every tensor carries a leading seed dimension S."""
import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def frob(t):
    """Per-seed Frobenius norm: (S, ...) -> (S,)."""
    return t.flatten(1).norm(dim=1)


def bview(x, like):
    """Reshape a per-seed vector (S,) so it broadcasts against `like` (S, m, n)."""
    return x.view(-1, *([1] * (like.dim() - 1)))


def total_frob(tensors):
    """Per-seed Frobenius norm of the concatenation of all layers."""
    return torch.stack([frob(t) ** 2 for t in tensors]).sum(0).sqrt()


def newton_schulz5(g, steps=5, eps=1e-7):
    """Quintic Newton-Schulz orthogonalization, identical to the atlas_opt optimizers
    (Frobenius pre-normalization with clamp, transpose when tall)."""
    a, b, c = NS_COEFFS
    x = g / bview(frob(g).clamp(min=eps), g)
    transpose = g.size(-2) > g.size(-1)
    if transpose:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        A = x @ x.transpose(-2, -1)
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transpose:
        x = x.transpose(-2, -1)
    return x


def polar_svd(g):
    """Exact polar factor U V^T (all non-zero singular values set to 1)."""
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    return u @ vh


def orthogonalize(g, method="ns5", steps=5):
    if method == "ns5":
        return newton_schulz5(g, steps=steps)
    if method == "svd":
        return polar_svd(g)
    raise ValueError(f"Unknown orthogonalization method: {method}")


def spectral_norm(t):
    return torch.linalg.matrix_norm(t, ord=2)


def project_spectral_ball(w, radius):
    """Euclidean projection onto {W : ||W||_2 <= radius}: clip singular values."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return (u * s.clamp(max=radius).unsqueeze(-2)) @ vh


def effective_rank(m, eps=1e-12):
    """Entropy effective rank of Roy & Vetterli (2007), per seed."""
    s = torch.linalg.svdvals(m)
    p = s / s.sum(-1, keepdim=True).clamp(min=eps)
    entropy = -(p * torch.log(p.clamp(min=eps))).sum(-1)
    return torch.exp(entropy)
