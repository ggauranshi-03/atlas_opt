"""Run one (variant, alpha, rho) configuration for a batch of independent seeds."""
import math

import torch

from .algorithms import Engine, OptimizerState, variant_key, variant_label
from .linalg import effective_rank, frob, project_spectral_ball, spectral_norm, total_frob
from .noise import Oracle
from .objective import (balancedness, end_to_end, excess_risk, full_gradient, full_objective,
                        hessian_lambda_max)

CURVE_METRICS = ("objective", "gap", "excess_risk", "dist_to_opt", "grad_norm", "perturbation_frob",
                 "perturbation_op", "update_frob", "sharpness_gap", "balancedness", "effective_rank",
                 "lambda_max", "train_acc", "val_acc", "val_loss", "cos_pert_grad", "cos_pert_momentum")


def alpha_key(alpha):
    return "inf" if math.isinf(alpha) else f"{alpha:g}"


def lr_factor(step, total, cfg_run, alpha):
    schedule = cfg_run.get("schedule", "cosine")
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        floor = float(cfg_run.get("min_lr_fraction", 0.01))
        progress = step / max(1, total - 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "power":
        power = cfg_run.get("schedule_power", 0.5)
        if power == "1/alpha":
            power = 0.5 if math.isinf(alpha) else 1.0 / alpha
        return (1.0 + step) ** (-float(power))
    raise ValueError(f"Unknown schedule: {schedule}")


def _project(Ws, problem):
    if problem.constraint == "spectral":
        return [project_spectral_ball(w, problem.radius) for w in Ws]
    if problem.constraint == "entrywise":
        return [w.clamp(-problem.radius, problem.radius) for w in Ws]
    return Ws


def _seed_base(cfg_run, alpha):
    alpha_code = 999_983 if math.isinf(alpha) else int(round(alpha * 1000))
    return int(cfg_run.get("seed_offset", 0)) * 1_000_003 + alpha_code


def run_variant(spec, mode, alpha, rho, lr, problem, cfg, seeds=None, seed_offset=None, steps=None):
    run = dict(cfg["run"])
    if seed_offset is not None:
        run["seed_offset"] = seed_offset
    S = int(seeds or run["seeds"])
    T = int(steps or run["steps"])
    log_every = int(run.get("log_every", 10))
    sharp_every = int(run.get("sharpness_every", 0))
    divergence_factor = float(run.get("divergence_factor", 1e3))
    measure_alignment = bool(run.get("measure_alignment", False))
    dtype = problem.X.dtype
    base = _seed_base(run, alpha)

    noise_generator = torch.Generator().manual_seed(base + 11)          # shared by all variants
    direction_generator = torch.Generator().manual_seed(base + 97_531)
    probe_generator = torch.Generator().manual_seed(base + 55_555)

    Ws = [w.unsqueeze(0).repeat(S, 1, 1).clone() for w in problem.init]
    oracle = Oracle(problem, cfg["noise"], S, alpha, noise_generator, dtype)
    clip_reference = total_frob(full_gradient(Ws, problem))[0].item()
    engine = Engine(spec, mode, rho, cfg, clip_reference, direction_generator)
    state = OptimizerState(Ws)

    F0 = full_objective(Ws, problem)
    threshold = divergence_factor * max(F0.max().item(), problem.F_star, 1e-12)
    alive = torch.ones(S, dtype=torch.bool)
    death_step = torch.full((S,), -1, dtype=torch.long)
    oracle_calls = 0
    log = []

    def record(step, eps, updates, prev):
        P = end_to_end(Ws)
        F = full_objective(Ws, problem)
        nan_col = torch.full((S,), math.nan, dtype=dtype)
        metrics = {
            "objective": F,
            "gap": F - problem.F_star,
            "grad_norm": total_frob(full_gradient(Ws, problem)),
            "effective_rank": effective_rank(P),
        }
        # excess_risk / dist_to_opt compare P against P_star / P_opt, and balancedness assumes the
        # exact linear conservation law of Arora et al. (2018) -- all three are specific to a
        # literal matrix-product model and are not defined for a nonlinear one (see Problem's
        # docstring note on P_star/P_opt). Report them as NaN rather than computing them against a
        # placeholder, which would silently move in the wrong direction as the model improves.
        if problem.P_star is not None:
            metrics["excess_risk"] = excess_risk(P, problem)
            metrics["dist_to_opt"] = frob(P - problem.P_opt)
            metrics["balancedness"] = balancedness(Ws)
        else:
            metrics["excess_risk"] = nan_col
            metrics["dist_to_opt"] = nan_col
            metrics["balancedness"] = nan_col
        if problem.loss_type.startswith("nn_ce"):
            from .objective_nn import forward_nn, objective_nn
            with torch.no_grad():
                logits_train = forward_nn(Ws, problem.X)
                pred_train = logits_train.argmax(dim=-1)
                metrics["train_acc"] = (pred_train == problem.Y.unsqueeze(0)).float().mean(dim=1)
                
                logits_val = forward_nn(Ws, problem.X_val)
                pred_val = logits_val.argmax(dim=-1)
                metrics["val_acc"] = (pred_val == problem.Y_val.unsqueeze(0)).float().mean(dim=1)
                metrics["val_loss"] = objective_nn(Ws, problem.X_val, problem.Y_val, ridge=0.0, is_ce=True)
        else:
            metrics["train_acc"] = nan_col
            metrics["val_acc"] = nan_col
            metrics["val_loss"] = nan_col
                
        zero = torch.zeros(S, dtype=dtype)
        if eps is None:
            metrics.update(perturbation_frob=zero, perturbation_op=zero, sharpness_gap=zero)
        else:
            metrics["perturbation_frob"] = total_frob(eps)
            metrics["perturbation_op"] = torch.stack([spectral_norm(e) for e in eps]).max(0).values
            perturbed = [p + e for p, e in zip(prev, eps)]
            metrics["sharpness_gap"] = full_objective(perturbed, problem) - full_objective(prev, problem)
        metrics["update_frob"] = zero if updates is None else total_frob(updates)
        # cos(perturbation, clean gradient) / cos(perturbation, momentum) at the moment the
        # perturbation was actually applied (see Engine.step) -- NaN for non-SAM baselines, and
        # whenever measure_alignment is off (the default: this costs one extra oracle call per
        # logged step for "stale"-sourced algorithms, so it's opt-in, never silently inflating
        # the cost of an ordinary sweep).
        metrics["cos_pert_grad"] = state.cos_pert_grad if state.cos_pert_grad is not None else nan_col
        metrics["cos_pert_momentum"] = state.cos_pert_momentum if state.cos_pert_momentum is not None else nan_col
        wants_sharpness = sharp_every and (step % sharp_every == 0 or step == T)
        metrics["lambda_max"] = (hessian_lambda_max(Ws, problem, int(run.get("sharpness_iters", 30)),
                                                     probe_generator)
                                 if wants_sharpness else torch.full((S,), math.nan, dtype=dtype))
        for name, value in metrics.items():
            value = value.detach().clone()
            value[~alive] = math.nan
            metrics[name] = value
        metrics["step"] = step
        metrics["oracle_calls"] = oracle_calls
        metrics["alive_fraction"] = alive.double().mean().item()
        log.append(metrics)

    record(0, None, None, None)
    for step in range(T):
        sample = oracle.draw()
        prev = [w.clone() for w in Ws]
        eps, updates, calls = engine.step(Ws, state, oracle, sample, lr * lr_factor(step, T, run, alpha),
                                          measure_alignment=measure_alignment)
        oracle_calls += calls

        finite = torch.stack([torch.isfinite(w).flatten(1).all(1) for w in Ws]).all(0)
        restore = ~finite | ~alive
        if restore.any():
            for w, p in zip(Ws, prev):
                w[restore] = p[restore]
            state.reset_seeds(restore)
        Ws[:] = _project(Ws, problem)
        F = full_objective(Ws, problem)
        blowup = alive & (~torch.isfinite(F) | (F > threshold))
        if blowup.any():
            for w, p in zip(Ws, prev):
                w[blowup] = p[blowup]
            state.reset_seeds(blowup)
        died = alive & (~finite | blowup)
        death_step[died] = step + 1
        alive &= ~died

        if (step + 1) % log_every == 0 or step + 1 == T:
            record(step + 1, eps, updates, prev)
        if not alive.any():
            break

    return log, alive, death_step


def summarize(log, alive, death_step, meta):
    """Aggregate per-step curves over seeds and emit per-seed final rows."""
    curves = []
    for entry in log:
        row = dict(meta, step=entry["step"], oracle_calls=entry["oracle_calls"],
                   alive_fraction=entry["alive_fraction"])
        for name in CURVE_METRICS:
            values = entry[name]
            finite = values[torch.isfinite(values)]
            if finite.numel() == 0:
                row.update({f"{name}_median": math.nan, f"{name}_q25": math.nan,
                            f"{name}_q75": math.nan, f"{name}_mean": math.nan})
                continue
            q = torch.quantile(finite, torch.tensor([0.25, 0.5, 0.75], dtype=finite.dtype))
            row.update({f"{name}_median": q[1].item(), f"{name}_q25": q[0].item(),
                        f"{name}_q75": q[2].item(), f"{name}_mean": finite.mean().item()})
        curves.append(row)

    last = log[-1]
    last_sharp = next((e for e in reversed(log) if torch.isfinite(e["lambda_max"]).any()), None)
    finals = []
    for seed in range(alive.numel()):
        row = dict(meta, seed=seed, diverged=int(not alive[seed]), death_step=int(death_step[seed]),
                   steps=last["step"], oracle_calls=last["oracle_calls"])
        for name in CURVE_METRICS:
            source = last_sharp if (name == "lambda_max" and last_sharp is not None) else last
            value = source[name][seed].item()
            if not alive[seed] and name in ("objective", "gap", "excess_risk", "dist_to_opt") and not math.isnan(value):
                value = math.inf
            row[f"final_{name}"] = value
        finals.append(row)
    return curves, finals


def describe(spec, mode, alpha, rho, lr, problem, cfg):
    return {
        "depth": problem.depth,
        "algorithm": spec.name,
        "variant": variant_key(spec, mode),
        "label": variant_label(spec, mode),
        "frobenius_mode": mode or "",
        "outer": spec.outer,
        "source": spec.source or "",
        "geometry": spec.geometry or "",
        "origin": spec.origin,
        "alpha": alpha,
        "rho": math.nan if rho is None else rho,
        "rho_units": cfg["sam"].get("rho_units", "nominal"),
        "lr": lr,
        "noise_model": cfg["noise"].get("model", "label"),
        "noise_scale": float(cfg["noise"].get("scale", 1.0)),
        "loss_type": problem.loss_type,
        "F_star": problem.F_star,
    }
