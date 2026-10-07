import torch
from .linalg import frob
from .objective import end_to_end

def objective_l1(Ws, X, Y, ridge=0.0):
    """
    F(W) = (1 / nk) || X P(W)^T - Y ||_{1,entry} + (mu / 2) || W ||_F^2
    """
    P = end_to_end(Ws)
    residual = X @ P.transpose(-1, -2) - Y
    n, k = residual.shape[-2], residual.shape[-1]
    value = residual.abs().sum((-2, -1)) / (n * k)
    if ridge:
        value = value + 0.5 * ridge * sum(frob(w) ** 2 for w in Ws)
    return value
