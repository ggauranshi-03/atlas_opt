import math
import torch
import torch.nn.functional as F
from .objective import Problem, layer_shapes, _random_orthonormal, _geometric
from .linalg import bview, frob, spectral_norm, total_frob

def make_problem_nn(cfg, dtype=torch.float64):
    depth = int(cfg.get("depth", 2))
    assert depth == 2, "NN objective currently only supports depth 2"
    d, k, hidden = int(cfg["d"]), int(cfg["k"]), int(cfg.get("hidden", cfg["k"]))
    n_train = int(cfg["n_train"])
    n_val = int(cfg.get("n_val", max(500, n_train // 4)))  # default validation set
    n_total = n_train + n_val
    generator = torch.Generator().manual_seed(int(cfg.get("data_seed", 1729)))

    # Generate X
    basis = _random_orthonormal(d, d, generator, dtype)
    eigenvalues = _geometric(d, float(cfg.get("cond_x", 1.0)), dtype)
    eigenvalues = eigenvalues * d / eigenvalues.sum()
    sigma = (basis * eigenvalues) @ basis.T
    X = (torch.randn(n_total, d, generator=generator, dtype=dtype) * eigenvalues.sqrt()) @ basis.T

    # Generate a teacher network to produce labels
    teacher_w1 = torch.randn(hidden, d, generator=generator, dtype=dtype) / math.sqrt(d)
    teacher_w2 = torch.randn(k, hidden, generator=generator, dtype=dtype) / math.sqrt(hidden)

    with torch.no_grad():
        logits = F.relu(X @ teacher_w1.T) @ teacher_w2.T
        if cfg.get("loss_type", "nn_ce") == "nn_ce":
            Y = logits.argmax(dim=-1)
        else:
            Y = logits

    # label_flip_prob is distinct from the regression objectives' label_noise_std (a Gaussian std):
    # here Y is a discrete class index, so noise is injected as a label flip, not additive noise.
    label_flip_prob = float(cfg.get("label_flip_prob", 0.0))
    if label_flip_prob > 0.0 and cfg.get("loss_type", "nn_ce") == "nn_ce":
        flip_mask = torch.rand(n_total, generator=generator, dtype=dtype) < label_flip_prob
        random_labels = torch.randint(0, k, (n_total,), generator=generator, dtype=Y.dtype)
        Y = torch.where(flip_mask, random_labels, Y)

    # Split train/val
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    ridge = float(cfg.get("ridge", 0.0))
    shapes = layer_shapes(depth, d, k, hidden)

    init_generator = torch.Generator().manual_seed(int(cfg.get("init_seed", 31415)))
    init_scale = float(cfg.get("init_scale", 1.0))
    init = [torch.randn(rows, cols, generator=init_generator, dtype=dtype) * (init_scale / math.sqrt(cols))
            for rows, cols in shapes]

    constraint = cfg.get("constraint", "spectral")
    radius = math.inf
    if constraint != "none":
        # Unlike the linear objectives (whose radius provably keeps the closed-form optimum
        # feasible, see objective.reference_optimum), there is no closed-form optimum for a
        # constrained ReLU network to reference here. This factor is a heuristic margin around
        # the initialization only, with no feasibility guarantee for any particular target network.
        init_layer = max(spectral_norm(w.unsqueeze(0)).item() for w in init)
        radius = float(cfg.get("radius_factor", 3.0)) * init_layer * 10.0

    # F* is the teacher network's own achieved loss on this exact (possibly label-flipped) training
    # set -- a genuine, directly-computed reference point, not a placeholder. It is not a proven
    # global minimum: cross-entropy has no closed form minimizer for a constrained ReLU network, and
    # a different point in weight space could in principle score lower. P_star/P_opt have no
    # analogue here (see Problem's docstring note) and are left as None.
    with torch.no_grad():
        f_star = objective_nn([teacher_w1.unsqueeze(0), teacher_w2.unsqueeze(0)], X_train, Y_train,
                              ridge=0.0, is_ce=True).item()

    problem = Problem(depth=depth, d=d, k=k, hidden=hidden, n=n_train, shapes=shapes, X=X_train, Y=Y_train,
                      sigma=sigma, F_star=f_star, ridge=ridge,
                      constraint=constraint, radius=radius, init=init, rank_budget=0,
                      loss_type=cfg.get("loss_type", "nn_ce"), X_val=X_val, Y_val=Y_val)

    return problem


def forward_nn(Ws, X):
    if X.dim() == 2:
        X = X.unsqueeze(0)
    w1, w2 = Ws[0], Ws[1]
    h = torch.relu(X @ w1.transpose(-1, -2))
    logits = h @ w2.transpose(-1, -2)
    return logits


def objective_nn(Ws, X, Y, ridge=0.0, is_ce=True):
    logits = forward_nn(Ws, X)
    if is_ce:
        S, n, k = logits.shape
        logits_flat = logits.reshape(S * n, k)
        if Y.dim() == 1:
            Y_flat = Y.unsqueeze(0).expand(S, n).reshape(S * n)
        elif Y.dim() == 2:
            Y_flat = Y.reshape(S * n)
        else:
            raise ValueError(f"Unexpected Y shape: {Y.shape}")
        loss_flat = F.cross_entropy(logits_flat, Y_flat, reduction='none')
        loss = loss_flat.reshape(S, n).mean(dim=1)
    else:
        if Y.dim() == 2:
            residual = logits - Y.unsqueeze(0)
        else:
            residual = logits - Y
        n, k = residual.shape[-2], residual.shape[-1]
        loss = residual.square().sum((-2, -1)) / (2.0 * n * k)

    if ridge:
        loss = loss + 0.5 * ridge * sum(frob(w) ** 2 for w in Ws)
    return loss
