"""The functional engine must reproduce the real atlas_opt/optimizers implementations.

Trajectory tests run the atlas_opt optimizer and the engine side by side for several steps on the
same noiseless problem. Wrapper optimizers (MuonSAM, FSAMMuon, ...) delegate their outer step to
the bf16 library Muon, so for those only the perturbation is compared, which is what defines them.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimizers.atlas_baseline import AtlasOptimizer            # noqa: E402
from optimizers.atlas_raw_grad import AtlasOptimizerRaw         # noqa: E402
from optimizers.fsam import FSAM                                # noqa: E402
from optimizers.fsam_muon import FSAMMuon                       # noqa: E402
from optimizers.fsam_ortho import FSAMOrtho                     # noqa: E402
from optimizers.fsam_ortho_muon import FSAMOrthoMuon            # noqa: E402
from optimizers.muon import SingleDeviceMuonWithAuxAdam         # noqa: E402
from optimizers.muon_sam import MuonSAM                         # noqa: E402
from optimizers.muon_sam_frob import MuonSAMFrob                # noqa: E402
from optimizers.muon_sam_stale import MuonSAMStale              # noqa: E402

from synthetic_ablations.algorithms import SPEC_BY_NAME, Engine, OptimizerState  # noqa: E402
from synthetic_ablations.noise import Oracle                                   # noqa: E402
from synthetic_ablations.objective import full_gradient, make_problem, objective  # noqa: E402

RHO, LR, BETA, LAM, SIG = 0.07, 0.03, 0.9, 0.8, 1.0
CFG = dict(sam=dict(ns_steps=5, friendly_lambda=LAM, friendly_sigma=SIG, rho_units="nominal", ortho="ns5"),
           optim=dict(muon_momentum=BETA, weight_decay=0.0))


@pytest.fixture(scope="module")
def problem():
    return make_problem(dict(depth=1, d=20, k=12, n_train=60, teacher_rank=5, cond_x=30.0,
                             cond_teacher=4.0, label_noise_std=0.2, constraint="none"), torch.float32)


def _closure_for(param, problem, optimizer):
    def closure():
        optimizer.zero_grad()
        loss = objective([param.unsqueeze(0)], problem.X, problem.Y).sum()
        loss.backward()
        return loss
    return closure


def _engine_trajectory(name, problem, steps, lr=LR):
    Ws = [problem.init[0].unsqueeze(0).clone()]
    oracle = Oracle(problem, dict(model="label", scale=0.0), 1, 2.0, torch.Generator().manual_seed(0),
                    problem.X.dtype)
    engine = Engine(SPEC_BY_NAME[name], "global", RHO, CFG)
    state = OptimizerState(Ws)
    out = []
    for _ in range(steps):
        engine.step(Ws, state, oracle, oracle.draw(), lr)
        out.append(Ws[0][0].clone())
    return out


def _reference_trajectory(make_optimizer, problem, steps):
    param = torch.nn.Parameter(problem.init[0].clone())
    optimizer = make_optimizer(param)
    closure = _closure_for(param, problem, optimizer)
    out = []
    for _ in range(steps):
        optimizer.step(closure)
        out.append(param.detach().clone())
    return out


def _assert_trajectories(ours, theirs, tol=2e-4):
    for step, (a, b) in enumerate(zip(ours, theirs)):
        error = ((a - b).norm() / b.norm()).item()
        assert error < tol, f"step {step}: relative error {error:.2e}"


@pytest.mark.parametrize("name, factory", [
    ("lazy-spectral-sam-muon", lambda p: AtlasOptimizer([dict(params=[p], use_muon=True, lr=LR, weight_decay=0.0,
                                                               rho=RHO, momentum=BETA, nesterov=True, ns_steps=5)])),
    ("stale-grad-sam-muon", lambda p: AtlasOptimizerRaw([dict(params=[p], use_muon=True, lr=LR, weight_decay=0.0,
                                                               rho=RHO, momentum=BETA, nesterov=True, ns_steps=5)])),
    ("friendly-sam-sgd", lambda p: FSAM([p], lr=LR, rho=RHO, lam=LAM, sigma=SIG)),
    ("spectral-friendly-sam-sgd", lambda p: FSAMOrtho([p], lr=LR, rho=RHO, lam=LAM, sigma=SIG, ns_steps=5)),
])
def test_trajectory_matches(name, factory, problem):
    steps = 6
    _assert_trajectories(_engine_trajectory(name, problem, steps), _reference_trajectory(factory, problem, steps))


def _base_muon(param):
    return SingleDeviceMuonWithAuxAdam([dict(params=[param], lr=LR, momentum=BETA, weight_decay=0.0,
                                             use_muon=True)])


def _engine_perturbation(name, problem, gradients):
    """Perturbations the engine builds for a sequence of clean gradients (state carried over)."""
    engine = Engine(SPEC_BY_NAME[name], "global", RHO, CFG)
    Ws = [problem.init[0].unsqueeze(0).clone()]
    state = OptimizerState(Ws)
    out = []
    for g in gradients:
        directions, _ = engine._directions(Ws, state, lambda g=g: [g.unsqueeze(0)])
        out.append(engine._geometry(directions)[0][0])
        state.t += 1
    return out


@pytest.mark.parametrize("name, wrapper", [
    ("full-spectral-sam-muon", lambda base: MuonSAM(base, rho=RHO, rho_vector=RHO, ns_steps=5)),
    ("sam-muon", lambda base: MuonSAMFrob(base, rho=RHO, rho_vector=RHO)),
    ("global-friendly-sam-muon", lambda base: FSAMMuon(base, rho=RHO, fsam_lambda=LAM, fsam_sigma=SIG)),
    ("spectral-friendly-sam-muon", lambda base: FSAMOrthoMuon(base, rho=RHO, rho_vector=RHO, ns_steps=5,
                                                              fsam_lambda=LAM, fsam_sigma=SIG)),
])
def test_wrapper_perturbation_matches(name, wrapper, problem):
    generator = torch.Generator().manual_seed(7)
    gradients = [torch.randn(problem.init[0].shape, generator=generator) for _ in range(3)]
    param = torch.nn.Parameter(problem.init[0].clone())
    optimizer = wrapper(_base_muon(param))
    theirs = []
    for g in gradients:
        param.grad = g.clone()
        optimizer.first_step(zero_grad=True)
        old = optimizer.state[param]["old_p"]
        theirs.append((param.detach() - old).clone())
        param.data.copy_(old)
    for step, (a, b) in enumerate(zip(_engine_perturbation(name, problem, gradients), theirs)):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), f"step {step}"


def test_stale_momentum_perturbation_matches(problem):
    """Round 2 of MuonSAMStale perturbs along the Muon momentum; EMA vs sum form only rescales it."""
    param = torch.nn.Parameter(problem.init[0].clone())
    optimizer = MuonSAMStale(_base_muon(param), rho=RHO, rho_vector=RHO)
    optimizer.step(_closure_for(param, problem, optimizer))
    buffer = optimizer.base_optimizer.state[param]["momentum_buffer"].clone()
    optimizer.perturb_weights(use_stale=True)
    theirs = param.detach() - optimizer.state[param]["old_p"]

    engine = Engine(SPEC_BY_NAME["stale-momentum-sam-muon"], "global", RHO, CFG)
    state = OptimizerState([param.detach().unsqueeze(0)])
    state.t, state.momentum = 1, [buffer.unsqueeze(0) / (1.0 - BETA)]
    directions, calls = engine._directions(None, state, lambda: pytest.fail("stale method needs no gradient"))
    ours = engine._geometry(directions)[0][0]
    assert calls == 0
    assert torch.allclose(ours, theirs, rtol=1e-4, atol=1e-7)


def test_fresh_methods_use_two_oracle_calls_and_lazy_uses_one(problem):
    oracle = Oracle(problem, dict(model="label", scale=0.0), 1, 2.0, torch.Generator().manual_seed(0),
                    problem.X.dtype)
    for name, expected in (("full-spectral-sam-muon", [2, 2, 2]), ("lazy-spectral-sam-muon", [2, 1, 1]),
                           ("random-sam-muon", [1, 1, 1]), ("muon", [1, 1, 1])):
        Ws = [problem.init[0].unsqueeze(0).clone()]
        engine, state = Engine(SPEC_BY_NAME[name], "global", RHO, CFG), OptimizerState(Ws)
        engine.direction_generator = torch.Generator().manual_seed(1)
        calls = [engine.step(Ws, state, oracle, oracle.draw(), LR)[2] for _ in range(3)]
        assert calls == expected, (name, calls)
