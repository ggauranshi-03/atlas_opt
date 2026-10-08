"""Every optimizer variant as one point in a common design space.

A SAM-type step is   (perturbation source) x (perturbation geometry) x (outer optimizer):

  source    grad             fresh clean gradient g_t                          (+1 oracle call)
            friendly         g_t - sigma m_t,  m_t = lam m_{t-1} + (1-lam) g_t (+1 oracle call)
            random           Gaussian matrix                                   (no gradient)
            stale_update     previous Muon update O_{t-1} = NS5(nesterov_{t-1}) (single pass)
            stale_grad       previous adversarial gradient g~_{t-1}            (single pass)
            stale_momentum   Muon momentum buffer M_{t-1}                      (single pass)
            stale_friendly   g~_{t-1} - sigma m_{t-1},  m = EMA of g~          (single pass)
            stale_momentum_friendly  g~_{t-1} - sigma (1-beta) v_{t-1}         (single pass)
  geometry  frobenius        rho d / ||d||_F  per layer, or one global norm over all layers
            spectral         rho Ortho(d)   (operator-norm ball of radius rho, per layer)
            raw              rho d          (d is already orthogonal: Atlas)
  outer     sgd | adam | muon | nsgdm | clip_sgd

Stale sources bootstrap step 0 with a fresh clean gradient, exactly like the atlas_opt code.
"""
import math
from dataclasses import dataclass

import torch

from .linalg import bview, frob, frobenius_reject, orthogonalize, newton_schulz5, spectral_norm, total_frob


@dataclass(frozen=True)
class Spec:
    name: str
    label: str
    outer: str
    source: str = None
    geometry: str = None
    origin: str = "both"
    atlas_opt: str = "-"
    mnist: str = "-"

    @property
    def is_sam(self):
        return self.source is not None

    @property
    def frobenius_modes(self):
        return self.geometry == "frobenius"


SPECS = [
    # ---- baselines (no perturbation) ----
    Spec("sgd", "SGD", "sgd", origin="both", atlas_opt="torch.optim.SGD", mnist="sgd"),
    Spec("adam", "AdamW", "adam", origin="requested", atlas_opt="torch.optim.AdamW"),
    Spec("muon", "Muon", "muon", origin="both", atlas_opt="muon.py", mnist="muon"),
    Spec("nsgdm", "Normalized SGD-M (Frobenius twin of Muon)", "nsgdm", origin="added"),
    Spec("clip-sgd", "Clip-SGD", "clip_sgd", origin="added"),
    # ---- SAM with an SGD outer step ----
    Spec("sam-sgd", "SAM + SGD", "sgd", "grad", "frobenius", mnist="sam-sgd"),
    Spec("spectral-sam-sgd", "Spectral SAM + SGD", "sgd", "grad", "spectral", origin="added"),
    Spec("friendly-sam-sgd", "Friendly SAM + SGD", "sgd", "friendly", "frobenius",
         atlas_opt="fsam.py", mnist="friendly-sam-sgd"),
    Spec("spectral-friendly-sam-sgd", "Spectral Friendly SAM + SGD", "sgd", "friendly", "spectral",
         origin="atlas_opt", atlas_opt="fsam_ortho.py"),
    # ---- SAM with an AdamW outer step (fresh, two oracle calls per step) ----
    Spec("sam-adam", "SAM + AdamW", "adam", "grad", "frobenius", origin="added"),
    Spec("friendly-sam-adam", "Friendly SAM + AdamW", "adam", "friendly", "frobenius", origin="added"),
    # ---- SAM with a Muon outer step, fresh (two oracle calls per step) ----
    Spec("sam-muon", "SAM-Muon", "muon", "grad", "frobenius", origin="atlas_opt",
         atlas_opt="muon_sam_frob.py (per-layer)"),
    Spec("full-spectral-sam-muon", "Full Spectral SAM-Muon", "muon", "grad", "spectral",
         atlas_opt="muon_sam.py", mnist="full-spectral-sam-muon"),
    Spec("global-friendly-sam-muon", "Global Friendly SAM-Muon", "muon", "friendly", "frobenius",
         atlas_opt="fsam_muon.py (global)", mnist="global-friendly-sam-muon"),
    Spec("spectral-friendly-sam-muon", "Spectral Friendly SAM-Muon", "muon", "friendly", "spectral",
         atlas_opt="fsam_ortho_muon.py", mnist="spectral-friendly-sam-muon"),
    # ---- random directions (one oracle call per step) ----
    Spec("random-sam-muon", "Random SAM-Muon", "muon", "random", "frobenius",
         atlas_opt="atlas_random.py (per-layer)", mnist="random-sam-muon"),
    Spec("random-spectral-sam-muon", "Random Spectral SAM-Muon", "muon", "random", "spectral",
         origin="mnist", mnist="random-spectral-sam-muon"),
    # ---- stale / lazy directions (one oracle call per step after step 0) ----
    Spec("lazy-spectral-sam-muon", "Lazy Spectral SAM-Muon (Atlas)", "muon", "stale_update", "raw",
         atlas_opt="atlas_baseline.py", mnist="lazy-spectral-sam-muon"),
    Spec("stale-grad-sam-muon", "Stale-Gradient SAM-Muon", "muon", "stale_grad", "frobenius",
         origin="atlas_opt", atlas_opt="atlas_raw_grad.py (per-layer)"),
    Spec("stale-grad-spectral-sam-muon", "Stale-Gradient Spectral SAM-Muon", "muon", "stale_grad", "spectral",
         origin="added"),
    Spec("stale-momentum-sam-muon", "Stale-Momentum SAM-Muon", "muon", "stale_momentum", "frobenius",
         origin="atlas_opt", atlas_opt="muon_sam_stale.py (per-layer)"),
    Spec("stale-friendly-sam-muon", "Stale Friendly SAM-Muon", "muon", "stale_friendly", "frobenius",
         origin="added"),
    Spec("stale-momentum-friendly-sam-muon", "Stale Momentum-Friendly SAM-Muon", "muon",
         "stale_momentum_friendly", "frobenius", origin="added"),
    Spec("stale-friendly-spectral-sam-muon", "Stale Friendly Spectral SAM-Muon", "muon",
         "stale_friendly", "spectral", origin="mnist", mnist="stale-spectral-friendly-sam-muon"),
    Spec("stale-momentum-friendly-spectral-sam-muon", "Stale Momentum-Friendly Spectral SAM-Muon",
         "muon", "stale_momentum_friendly", "spectral", origin="mnist",
         mnist="stale-muon-momentum-spectral-sam-muon (sign-fixed)"),
    # ---- Orthogonally Projected SOMA (one oracle call per step after a two-call bootstrap) ----
    # PreNS5: the stale adversarial gradient is projected (Frobenius-orthogonal) away from the
    # current Muon momentum *before* NS5 orthogonalization -- source "op_soma_pre".
    Spec("op-soma-prens5", "OP-SOMA-PreNS5", "muon", "op_soma_pre", "spectral", origin="added"),
    # PostNS5: reuses the existing stale_momentum_friendly direction (g~_{t-1} - sigma(1-beta)M_{t-1}),
    # NS5-orthogonalizes it, then projects *after* NS5 against the previous Nesterov-NS5 update and
    # rescales to the operator-norm ball -- geometry "op_soma_post".
    Spec("op-soma-postns5", "OP-SOMA-PostNS5", "muon", "stale_momentum_friendly", "op_soma_post",
         origin="added"),
    # ---- Momentum-SAM (Becker et al. 2024, arXiv:2401.12033): SGD-momentum outer step whose
    # own momentum buffer v_t IS the (normalized) perturbation direction -- one oracle call per
    # step, same mu shared by the perturbation and the outer update. Requires optim.sgd_momentum
    # set to mu > 0 in the run's config (left at the shared default of 0.0 otherwise, which would
    # silently degenerate this into plain SAM+SGD). Uses "momentum_lookahead" (w~ = w - rho*v/||v||,
    # per the paper), NOT "stale_momentum" (w~ = w + rho*v/||v||, a different, ascent-style
    # algorithm used elsewhere in this file) -- see that source's docstring note.
    Spec("msam", "Momentum-SAM (MSAM)", "sgd", "momentum_lookahead", "frobenius", origin="added"),
]
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}
BASE_OUTERS = ("sgd", "adam", "muon", "nsgdm", "clip_sgd")


def variants(names, frobenius_modes, depth):
    """Expand algorithm names into (spec, frobenius_mode) pairs. With one matrix the two
    Frobenius modes coincide, so only 'global' is kept."""
    modes = ["global"] if depth == 1 else list(frobenius_modes)
    out = []
    for name in names:
        spec = SPEC_BY_NAME[name]
        if spec.frobenius_modes:
            out.extend((spec, mode) for mode in modes)
        else:
            out.append((spec, None))
    return out


def variant_label(spec, mode):
    if mode is None or spec.geometry != "frobenius":
        return spec.label
    return f"{spec.label} [{'global' if mode == 'global' else 'per-layer'} Frob]"


def variant_key(spec, mode):
    return spec.name if mode is None else f"{spec.name}@{mode}"


class OptimizerState:
    def __init__(self, Ws):
        zeros = [torch.zeros_like(w) for w in Ws]
        self.momentum = [z.clone() for z in zeros]
        self.adam_m = [z.clone() for z in zeros]
        self.adam_v = [z.clone() for z in zeros]
        self.friendly = [z.clone() for z in zeros]
        self.last_update = None
        self.last_outer = None
        self.t = 0

    def reset_seeds(self, mask):
        for group in (self.momentum, self.adam_m, self.adam_v, self.friendly,
                      self.last_update or [], self.last_outer or []):
            for tensor in group:
                tensor[mask] = 0.0


class Engine:
    """Applies one step of one variant to a batch of independent seeds."""

    def __init__(self, spec, frobenius_mode, rho, cfg, clip_reference=None, direction_generator=None):
        self.spec = spec
        self.mode = frobenius_mode or "global"
        self.rho = rho
        sam, optim = cfg["sam"], cfg["optim"]
        self.rho_units = sam.get("rho_units", "nominal")
        self.ortho = sam.get("ortho", "ns5")
        self.ns_steps = int(sam.get("ns_steps", 5))
        self.lam = float(sam.get("friendly_lambda", 0.9))
        self.sigma = float(sam.get("friendly_sigma", 1.0))
        self.beta = float(optim.get("muon_momentum", 0.95))
        self.sgd_beta = float(optim.get("sgd_momentum", 0.0))
        self.adam_betas = tuple(optim.get("adam_betas", (0.9, 0.999)))
        self.adam_eps = float(optim.get("adam_eps", 1e-8))
        self.weight_decay = float(optim.get("weight_decay", 0.0))
        self.clip_threshold = None if clip_reference is None else float(optim.get("clip_factor", 1.0)) * clip_reference
        self.direction_generator = direction_generator

    # ------------------------------------------------------------------ perturbation
    def _geometry(self, directions, state=None):
        rho, geometry = self.rho, self.spec.geometry
        if geometry == "frobenius":
            if self.mode == "global":
                scale = rho / (total_frob(directions) + 1e-12)
                eps = [d * bview(scale, d) for d in directions]
            else:
                eps = [d * bview(rho / (frob(d) + 1e-12), d) for d in directions]
        elif geometry == "spectral":
            eps = [rho * orthogonalize(d, self.ortho, self.ns_steps) for d in directions]
        elif geometry == "raw":
            eps = [rho * d for d in directions]
        elif geometry == "op_soma_post":
            # Z = rho * NS5(D); at bootstrap (no previous Nesterov-NS5 update yet) E = Z. Otherwise
            # project Z away from O_prev (Frobenius-orthogonal) and rescale to the operator-norm
            # ball of radius rho -- no second NS5 pass, per the algorithm's spec.
            o_prev = state.last_update
            z = [rho * orthogonalize(d, self.ortho, self.ns_steps) for d in directions]
            if o_prev is None:
                eps = z
            else:
                q = [frobenius_reject(o, zi) for o, zi in zip(o_prev, z)]
                eps = [qi * bview(rho / (spectral_norm(qi) + 1e-12), qi) for qi in q]
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
        if self.rho_units == "matched_frobenius":
            scale = rho / (total_frob(eps) + 1e-12)
            eps = [e * bview(scale, e) for e in eps]
        elif self.rho_units != "nominal":
            raise ValueError(f"Unknown rho_units: {self.rho_units}")
        return eps

    def _directions(self, Ws, state, clean):
        """Return (directions, extra oracle calls). `clean` evaluates the oracle at W_t."""
        source, t = self.spec.source, state.t
        if source == "grad":
            return clean(), 1
        if source == "friendly":
            g = clean()
            for m, gi in zip(state.friendly, g):
                m.mul_(self.lam).add_(gi, alpha=1.0 - self.lam)
            return [gi - self.sigma * m for gi, m in zip(g, state.friendly)], 1
        if source == "random":
            return [torch.randn(w.shape, generator=self.direction_generator, dtype=w.dtype) for w in Ws], 0
        if source == "stale_update":
            if state.last_update is not None:
                return [u.clone() for u in state.last_update], 0
            return [newton_schulz5(g, self.ns_steps) for g in clean()], 1
        if source == "stale_grad":
            if state.last_outer is not None:
                return [g.clone() for g in state.last_outer], 0
            return clean(), 1
        if source == "stale_momentum":
            if t > 0:
                return [v.clone() for v in state.momentum], 0
            return clean(), 1
        if source == "momentum_lookahead":
            # Momentum-SAM (Becker et al. 2024): perturbs by SUBTRACTING the momentum
            # direction (w~ = w - rho*v/||v||, a descent-direction lookahead/extrapolation),
            # the opposite sign from "stale_momentum" above (which ADDS it, an ascent-style
            # direction used by other, separately-tested algorithms). Do not merge these.
            if t > 0:
                return [-v for v in state.momentum], 0
            return [-g for g in clean()], 1
        if source in ("stale_friendly", "stale_momentum_friendly"):
            if state.last_outer is None:
                g = clean()
                return [gi - self.sigma * (1.0 - self.lam) * gi for gi in g], 1
            if source == "stale_friendly":
                ema = state.friendly
            else:
                ema = [(1.0 - self.beta) * v for v in state.momentum]
            return [g - self.sigma * m for g, m in zip(state.last_outer, ema)], 0
        if source == "op_soma_pre":
            if state.last_outer is None:
                g = clean()
                return [gi - self.sigma * (1.0 - self.lam) * gi for gi in g], 1
            # Project the stale adversarial gradient away from the current momentum (Frobenius-
            # orthogonal) before NS5; equivalent to projecting the "friendly" g~ - sigma(1-beta)M
            # direction, since the M-parallel term is removed by the projection either way.
            return [frobenius_reject(m, g) for m, g in zip(state.momentum, state.last_outer)], 0
        raise ValueError(f"Unknown source: {source}")

    # ------------------------------------------------------------------ outer update
    def _outer(self, Ws, grads, state, lr):
        outer, updates = self.spec.outer, []
        if outer == "sgd":
            for w, g, v in zip(Ws, grads, state.momentum):
                step = g
                if self.sgd_beta:
                    v.mul_(self.sgd_beta).add_(g)
                    step = v
                self._decay(w, lr)
                w.sub_(lr * step)
                updates.append(step)
        elif outer == "clip_sgd":
            norm = total_frob(grads)
            factor = torch.clamp(self.clip_threshold / (norm + 1e-12), max=1.0)
            for w, g in zip(Ws, grads):
                step = g * bview(factor, g)
                self._decay(w, lr)
                w.sub_(lr * step)
                updates.append(step)
        elif outer == "adam":
            beta1, beta2 = self.adam_betas
            t = state.t + 1
            for w, g, m, v in zip(Ws, grads, state.adam_m, state.adam_v):
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                step = (m / (1 - beta1 ** t)) / ((v / (1 - beta2 ** t)).sqrt() + self.adam_eps)
                self._decay(w, lr)
                w.sub_(lr * step)
                updates.append(step)
        elif outer in ("muon", "nsgdm"):
            for w, g, v in zip(Ws, grads, state.momentum):
                v.mul_(self.beta).add_(g)
                nesterov = g + self.beta * v
                if outer == "muon":
                    step = newton_schulz5(nesterov, self.ns_steps)
                else:
                    size = math.sqrt(min(w.shape[-2], w.shape[-1]))
                    step = nesterov * bview(size / (frob(nesterov) + 1e-12), nesterov)
                rows, cols = w.shape[-2], w.shape[-1]
                self._decay(w, lr)
                w.sub_(lr * math.sqrt(max(1.0, rows / cols)) * step)
                updates.append(step)
            state.last_update = [u.clone() for u in updates]
        else:
            raise ValueError(f"Unknown outer optimizer: {outer}")
        return updates

    def _decay(self, w, lr):
        if self.weight_decay:
            w.mul_(1.0 - lr * self.weight_decay)

    # ------------------------------------------------------------------ one step
    def step(self, Ws, state, oracle, sample, lr):
        """Advance Ws in place. Returns (perturbation, updates, oracle_calls)."""
        clean = lambda: oracle.gradients(Ws, sample)
        if not self.spec.is_sam:
            grads, calls, eps = clean(), 1, None
        else:
            directions, calls = self._directions(Ws, state, clean)
            eps = self._geometry(directions, state)
            grads = oracle.gradients([w + e for w, e in zip(Ws, eps)], sample)
            calls += 1
        updates = self._outer(Ws, grads, state, lr)
        if self.spec.source in ("stale_grad", "stale_friendly", "stale_momentum_friendly", "op_soma_pre"):
            state.last_outer = [g.clone() for g in grads]
        if self.spec.source == "stale_friendly":
            for m, g in zip(state.friendly, grads):
                m.mul_(self.lam).add_(g, alpha=1.0 - self.lam)
        state.t += 1
        return eps, updates, calls
