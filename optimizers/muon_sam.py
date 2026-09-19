import torch

def newtonschulz5(G, steps=5, eps=1e-7):
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

class MuonSAM(torch.optim.Optimizer):
    """
    SpecSAM-Muon: two-pass SAM with a layerwise spectral inner step for matrix
    parameters and a single Euclidean inner step for the vector block.

    The outer step is delegated to the wrapped Muon/AdamW optimizer, which already
    implements momentum -> Newton-Schulz -> w*(1 - lr*wd) - eta*O_t.
    """
    def __init__(self, base_optimizer, rho=0.0015, rho_vector=0.01, ns_steps=5):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert rho_vector >= 0.0, f"Invalid rho_vector, should be non-negative: {rho_vector}"
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {"rho": rho, "rho_vector": rho_vector, "ns_steps": ns_steps}
        super(MuonSAM, self).__init__(self.param_groups, self.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 1D Global Grad Norm for AdamW parameters
        grad_norm_1d = self._grad_norm_1d()

        for group in self.param_groups:
            rho, rho_vector, ns_steps = group["rho"], group["rho_vector"], group["ns_steps"]
            scale_1d = rho_vector / grad_norm_1d

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()

                if p.ndim >= 2:
                    # 2D (Matrices): Spectral Geometry via Newton-Schulz.
                    # Ortho(g) already has unit spectral norm, so rho *is* the layer's radius.
                    g_2d = p.grad.float()
                    original_shape = g_2d.shape
                    g_2d = g_2d.reshape(g_2d.size(0), -1)

                    P_t = newtonschulz5(g_2d, steps=ns_steps).reshape(original_shape)
                    p.add_(P_t.to(p.dtype), alpha=rho)
                else:
                    # 1D (Vectors / AdamW): Euclidean Geometry
                    p.add_(p.grad * scale_1d.to(p.dtype))

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        # 1. Take gradient at w
        loss = closure()

        # 2. Add adversarial perturbation (climbs to w + e(w))
        self.first_step(zero_grad=True)

        # 3. Take gradient at w + e(w)
        closure()

        # 4. Step base optimizer using new gradients, and revert w + e(w) back to w
        self.second_step()

        return loss

    def _grad_norm_1d(self):
        # The non-matrix parameters form a single Euclidean block (Eq. 9 in the paper).
        shared_device = self.param_groups[0]["params"][0].device
        tensors = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None and p.ndim < 2
        ]
        if not tensors:
            return torch.tensor(1.0, device=shared_device)
        return torch.norm(torch.stack(tensors), p=2) + 1e-12
