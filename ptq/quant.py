"""Weight-only post-training quantization: RTN, GPTQ and AWQ.

GPTQ follows the reference implementation of Frantar et al. (IST-DASLab/gptq, gptq.py):
lazy-batch Cholesky updates, 1% dampening, block size 128, true-sequential layer order
(each layer's Hessian is collected with all preceding layers already quantized).

AWQ follows the reference implementation of Lin et al. (mit-han-lab/llm-awq, auto_scale.py /
auto_clip.py): activation-aware per-input-channel scale search (grid of 20 over s = x_mean^a),
followed by a weight clipping search, then round-to-nearest on the transformed weights.

All methods share one asymmetric min-max quantizer (range includes 0, as in the GPTQ reference),
so differences between methods are purely algorithmic. Weights are fake-quantized
(quantize -> dequantize) in FP32, which is the standard protocol for reporting PTQ accuracy.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


# ----------------------------------------------------------------------------- quantizer

def _minmax_params(x, maxq):
    """Per-row asymmetric params for x of shape [rows, cols]."""
    xmin = torch.minimum(x.min(dim=1).values, torch.zeros(1, device=x.device))
    xmax = torch.maximum(x.max(dim=1).values, torch.zeros(1, device=x.device))
    both_zero = (xmin == 0) & (xmax == 0)
    xmin[both_zero], xmax[both_zero] = -1.0, 1.0
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1)


def _fake_quant(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def quantize_weight(w, bits, group_size):
    """Round-to-nearest fake quantization of a 2D weight [out, in], grouped along `in`.
    group_size <= 0 means one group per output row (per-channel)."""
    maxq = 2 ** bits - 1
    rows, cols = w.shape
    gs = cols if group_size <= 0 else group_size
    pad = (-cols) % gs
    wp = F.pad(w, (0, pad)) if pad else w  # zero padding is inert: the range always includes 0
    g = wp.reshape(-1, gs)
    scale, zero = _minmax_params(g, maxq)
    return _fake_quant(g, scale, zero, maxq).reshape(rows, -1)[:, :cols]


# ----------------------------------------------------------------------------- layer views

def weight_2d(layer):
    w = layer.weight.data
    return w.flatten(1) if isinstance(layer, nn.Conv2d) else w


def set_weight_2d(layer, w2d):
    layer.weight.data.copy_(w2d.reshape(layer.weight.shape).to(layer.weight.dtype))


def inputs_2d(layer, x):
    """Input activations of `layer` as [tokens, in_features] matching weight_2d columns."""
    if isinstance(layer, nn.Conv2d):
        unfold = nn.Unfold(layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride)
        return unfold(x).permute(0, 2, 1).reshape(-1, weight_2d(layer).shape[1])
    return x.reshape(-1, x.shape[-1])


def conv1d_to_linear(model):
    """Replace HF GPT-2 Conv1D (y = x @ W + b, W: [in, out]) with an equivalent nn.Linear."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None, device=child.weight.device, dtype=child.weight.dtype)
                lin.weight.data.copy_(child.weight.data.t())
                if child.bias is not None:
                    lin.bias.data.copy_(child.bias.data)
                setattr(module, child_name, lin)
    return model


# ----------------------------------------------------------------------------- architecture specs

def get_arch(model):
    """Describe which layers are quantized and how AWQ scale groups are formed.

    LLMs: all linear layers inside transformer blocks are quantized; embeddings and the LM head stay
    in full precision (standard GPTQ/AWQ practice). AWQ groups follow the official llm-awq / AutoAWQ
    definitions (attention-output projection is not scaled for fused-QKV architectures, as in
    AutoAWQ's GPT-NeoX and GPTBigCode definitions).
    CNN: the three hidden conv layers are quantized; the input conv and the classifier stay in full
    precision (standard CNN PTQ practice of keeping first/last layers at high precision).
    """
    mtype = getattr(getattr(model, "config", None), "model_type", None)
    if mtype == "gpt2":
        blocks = list(model.transformer.h)
        spec = dict(
            kind="llm", blocks=blocks,
            block_layers=lambda b: {"attn.c_attn": b.attn.c_attn, "attn.c_proj": b.attn.c_proj,
                                    "mlp.c_fc": b.mlp.c_fc, "mlp.c_proj": b.mlp.c_proj},
            awq_groups=lambda b: [
                dict(prev=("ln", b.ln_1), layers=["attn.c_attn"], inspect=b.attn),
                dict(prev=("ln", b.ln_2), layers=["mlp.c_fc"], inspect=b.mlp),
                dict(prev=("act", b.mlp, "act"), layers=["mlp.c_proj"], inspect=None),
            ],
            no_clip=["attn.c_attn"],  # fused QKV: q/k are not clipped (llm-awq rule)
        )
    elif mtype == "gpt_neox":
        blocks = list(model.gpt_neox.layers)
        spec = dict(
            kind="llm", blocks=blocks,
            block_layers=lambda b: {"attention.query_key_value": b.attention.query_key_value,
                                    "attention.dense": b.attention.dense,
                                    "mlp.dense_h_to_4h": b.mlp.dense_h_to_4h,
                                    "mlp.dense_4h_to_h": b.mlp.dense_4h_to_h},
            awq_groups=lambda b: [
                dict(prev=("ln", b.input_layernorm), layers=["attention.query_key_value"], inspect=b.attention),
                dict(prev=("ln", b.post_attention_layernorm), layers=["mlp.dense_h_to_4h"], inspect=None),
                dict(prev=("act", b.mlp, "act"), layers=["mlp.dense_4h_to_h"], inspect=None),
            ],
            no_clip=["attention.query_key_value"],
        )
    elif mtype is None and hasattr(model, "layer1") and hasattr(model, "classifier"):
        spec = dict(kind="cnn", layers={"layer1.conv": model.layer1.conv,
                                        "layer2.conv": model.layer2.conv,
                                        "layer3.conv": model.layer3.conv})
    else:
        raise ValueError(f"Unsupported architecture: {type(model).__name__} ({mtype})")
    return spec


def quantized_layers(model):
    spec = get_arch(model)
    if spec["kind"] == "cnn":
        return dict(spec["layers"])
    out = {}
    for i, b in enumerate(spec["blocks"]):
        for n, l in spec["block_layers"](b).items():
            out[f"block{i}.{n}"] = l
    return out


# ----------------------------------------------------------------------------- calibration forward

@torch.no_grad()
def run_calib(model, calib, batch_size):
    """Forward the calibration set. `calib` is a tensor of token ids [N, T] or images [N, C, H, W]."""
    for i in range(0, calib.shape[0], batch_size):
        xb = calib[i:i + batch_size].to(next(model.parameters()).device)
        if xb.dtype == torch.long:
            model(input_ids=xb, use_cache=False)
        else:
            model(xb)


# ----------------------------------------------------------------------------- RTN

@torch.no_grad()
def quantize_rtn(model, bits, group_size):
    for layer in quantized_layers(model).values():
        set_weight_2d(layer, quantize_weight(weight_2d(layer).float(), bits, group_size))
    return model


# ----------------------------------------------------------------------------- GPTQ

class _GPTQLayer:
    def __init__(self, layer):
        self.layer = layer
        cols = weight_2d(layer).shape[1]
        self.H = torch.zeros(cols, cols, device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, x):
        n = x.shape[0]
        inp = inputs_2d(self.layer, x).t().float()
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def quantize(self, bits, group_size, blocksize=128, percdamp=0.01, actorder=False):
        maxq = 2 ** bits - 1
        W = weight_2d(self.layer).clone().float()
        cols = W.shape[1]
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W, H = W[:, perm], H[perm][:, perm]
            invperm = torch.argsort(perm)

        scale, zero = _minmax_params(W, maxq)  # per-channel params (used when group_size <= 0)
        Q = torch.zeros_like(W)
        idx = torch.arange(cols, device=W.device)
        damp = percdamp * torch.mean(torch.diag(H))
        Hinv = None
        for attempt in range(6):  # raise dampening if H is numerically indefinite
            Hd = H.clone()
            Hd[idx, idx] += damp * (10 ** attempt)
            try:
                L = torch.linalg.cholesky(Hd)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
                break
            except torch._C._LinAlgError:
                continue
        if Hinv is None:
            raise RuntimeError("GPTQ: Hessian is not positive definite even with heavy dampening")
        self.damp_attempts = attempt

        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if group_size > 0 and (i1 + i) % group_size == 0:
                    grp = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :group_size]
                    scale, zero = _minmax_params(grp, maxq)
                q = _fake_quant(w.unsqueeze(1), scale, zero, maxq).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]
        set_weight_2d(self.layer, Q)


@torch.no_grad()
def quantize_gptq(model, calib, bits, group_size, batch_size, actorder=False, log=print):
    """True-sequential GPTQ: layers are processed in forward order; each layer's Hessian is built
    from inputs produced by the network with all earlier layers already quantized."""
    layers = quantized_layers(model)
    for name, layer in layers.items():
        g = _GPTQLayer(layer)
        h = layer.register_forward_hook(lambda m, inp, out: g.add_batch(inp[0].detach()))
        run_calib(model, calib, batch_size)
        h.remove()
        g.quantize(bits, group_size, actorder=actorder)
        if getattr(g, "damp_attempts", 0):
            log(f"    [gptq] {name}: dampening increased {10 ** g.damp_attempts}x for PD Hessian")
        del g
    return model


# ----------------------------------------------------------------------------- AWQ

class ScaledActivation(nn.Module):
    """act(x) / s, as in llm-awq; used when the preceding op cannot absorb the scale."""
    def __init__(self, act, scales):
        super().__init__()
        self.act = act
        self.scales = nn.Parameter(scales.data.clone(), requires_grad=False)

    def forward(self, x):
        return self.act(x) / self.scales.view(*([1] * (x.dim() - 1)), -1)


class ScaledInputConv(nn.Module):
    """conv(x / s) with per-input-channel s; the CNN analogue of ScaledActivation."""
    def __init__(self, conv, scales):
        super().__init__()
        self.conv = conv
        self.scales = nn.Parameter(scales.data.clone(), requires_grad=False)

    def forward(self, x):
        return self.conv(x / self.scales.view(1, -1, 1, 1))


def _expand_scales(layer, scales):
    """Per-input-channel scales -> per-column scales of weight_2d (conv: C_in * kH * kW)."""
    if isinstance(layer, nn.Conv2d):
        return scales.repeat_interleave(layer.kernel_size[0] * layer.kernel_size[1])
    return scales


def _channel_mean_abs(layer, x):
    if isinstance(layer, nn.Conv2d):
        return x.abs().transpose(0, 1).reshape(x.shape[1], -1).float().mean(1)
    return x.abs().reshape(-1, x.shape[-1]).float().mean(0)


@torch.no_grad()
def _search_scale(layers, x_feat, run_inspect, bits, group_size, n_grid=20):
    """llm-awq _search_module_scale. `run_inspect()` returns the inspected module's output."""
    x_max = _channel_mean_abs(layers[0], x_feat)
    org_out = run_inspect()
    orig_w = [l.weight.data.clone() for l in layers]
    best_err, best_scales, best_ratio = float("inf"), None, -1
    for r in range(n_grid):
        ratio = r / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for l, w0 in zip(layers, orig_w):
            s = _expand_scales(l, scales).view(1, -1)
            set_weight_2d(l, quantize_weight(weight_2d(l).float() * s, bits, group_size) / s)
        err = (org_out - run_inspect()).float().pow(2).mean().item()
        if err < best_err:
            best_err, best_scales, best_ratio = err, scales.clone(), ratio
        for l, w0 in zip(layers, orig_w):
            l.weight.data.copy_(w0)
    if best_scales is None:
        raise RuntimeError("AWQ scale search produced no finite loss")
    return best_scales, best_ratio


@torch.no_grad()
def _search_clip(layer, x_feat, bits, group_size, n_grid=20, max_shrink=0.5, n_sample_token=512):
    """llm-awq auto_clip_layer: per-group symmetric clipping threshold minimizing output MSE."""
    w = weight_2d(layer).float()
    x = inputs_2d(layer, x_feat).float()
    co, ci = w.shape
    gs = ci if group_size <= 0 else group_size
    pad = (-ci) % gs
    if pad:
        w, x = F.pad(w, (0, pad)), F.pad(x, (0, pad))
    x = x[0::max(1, x.shape[0] // n_sample_token)]
    x = x.reshape(1, x.shape[0], -1, gs)
    w = w.reshape(co, 1, -1, gs)
    oc_bs = next(b for b in (256, 128, 64, 32, 16, 8, 4, 2, 1) if co % b == 0)
    best_all = []
    for ib in range(co // oc_bs):
        wb = w[ib * oc_bs:(ib + 1) * oc_bs]
        org_max = wb.abs().amax(dim=-1, keepdim=True)
        best_max = org_max.clone()
        min_errs = torch.full_like(org_max, 1e9)
        org_out = (x * wb).sum(dim=-1)
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i_s / n_grid)
            cur_w = torch.clamp(wb, -max_val, max_val)
            q_w = quantize_weight(cur_w.reshape(-1, gs), bits, gs).reshape(cur_w.shape)
            cur_out = (x * q_w).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            better = err < min_errs
            min_errs[better] = err[better]
            best_max[better] = max_val[better]
        best_all.append(best_max)
    best_max = torch.cat(best_all, dim=0)  # [co, 1, n_group, 1]
    wc = torch.clamp(w, -best_max, best_max).reshape(co, -1)[:, :ci]
    set_weight_2d(layer, wc)


def _capture_inputs(modules, fn):
    """Run fn() and return {name: concatenated first positional input} for each module."""
    feats = {n: [] for n in modules}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: feats[n].append(inp[0].detach()))
             for n, m in modules.items()]
    try:
        fn()
    finally:
        for h in hooks:
            h.remove()
    return {n: torch.cat(v, dim=0) for n, v in feats.items()}


@torch.no_grad()
def quantize_awq(model, calib, bits, group_size, batch_size, log=print):
    spec = get_arch(model)
    if spec["kind"] == "cnn":
        return _awq_cnn(model, spec, calib, bits, group_size, batch_size, log)
    return _awq_llm(model, spec, calib, bits, group_size, batch_size, log)


def _awq_cnn(model, spec, calib, bits, group_size, batch_size, log):
    layers = spec["layers"]
    feats = _capture_inputs(layers, lambda: run_calib(model, calib, batch_size))
    for name, conv in layers.items():
        x = feats[name]
        scales, ratio = _search_scale([conv], x, lambda: conv(x), bits, group_size)
        log(f"    [awq] {name}: best scale ratio {ratio}")
        conv.weight.data.mul_(scales.view(1, -1, 1, 1))
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, ScaledInputConv(conv, scales))
        # unfolded patches of 64 images (~65k tokens) are then subsampled to 512 tokens, as in llm-awq
        _search_clip(conv, x[:64] / scales.view(1, -1, 1, 1), bits, group_size)
        set_weight_2d(conv, quantize_weight(weight_2d(conv).float(), bits, group_size))
    return model


def _awq_llm(model, spec, calib, bits, group_size, batch_size, log):
    blocks = spec["blocks"]
    device = next(model.parameters()).device

    # Capture inputs (and remaining call args) of the first block for every calibration batch.
    block_inps, block_args, block_kwargs = [], [], []

    def pre_hook(mod, args, kwargs):
        if args:
            block_inps.append(args[0].detach())
            block_args.append(args[1:])
        else:
            block_inps.append(kwargs["hidden_states"].detach())
            block_args.append(())
        block_kwargs.append({k: v for k, v in kwargs.items() if k != "hidden_states"})
        raise _StopForward

    h = blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    for i in range(0, calib.shape[0], batch_size):
        try:
            model(input_ids=calib[i:i + batch_size].to(device), use_cache=False)
        except _StopForward:
            pass
    h.remove()

    def run_block(block, inps):
        return [_first(block(x, *a, **kw)) for x, a, kw in zip(inps, block_args, block_kwargs)]

    for bi, block in enumerate(blocks):
        named = spec["block_layers"](block)
        feats = _capture_inputs(named, lambda: run_block(block, block_inps))
        next_inps = run_block(block, block_inps)

        ratios = []
        for grp in spec["awq_groups"](block):
            lyrs = [named[n] for n in grp["layers"]]
            x = feats[grp["layers"][0]]
            if grp["inspect"] is None:
                run_inspect = lambda: lyrs[0](x)
            else:
                inspect = grp["inspect"]

                def run_inspect(inspect=inspect):
                    outs = []
                    hk = inspect.register_forward_hook(lambda m, i, o: outs.append(_first(o).detach()))
                    try:
                        run_block(block, block_inps)
                    finally:
                        hk.remove()
                    return torch.cat(outs, 0)

            scales, ratio = _search_scale(lyrs, x, run_inspect, bits, group_size)
            ratios.append(ratio)
            _apply_scale(grp["prev"], lyrs, scales)
            for n in grp["layers"]:
                feats[n] = feats[n] / scales.view(*([1] * (feats[n].dim() - 1)), -1)

        for n, l in named.items():
            if n not in spec["no_clip"]:
                _search_clip(l, feats[n], bits, group_size)
        for l in named.values():
            set_weight_2d(l, quantize_weight(weight_2d(l).float(), bits, group_size))
        log(f"    [awq] block {bi}: best scale ratios {ratios}")

        block_inps = next_inps
        del feats
    return model


class _StopForward(Exception):
    pass


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _apply_scale(prev, layers, scales):
    kind = prev[0]
    if kind == "ln":
        ln = prev[1]
        ln.weight.data.div_(scales)
        if ln.bias is not None:
            ln.bias.data.div_(scales)
    elif kind == "act":
        parent, attr = prev[1], prev[2]
        setattr(parent, attr, ScaledActivation(getattr(parent, attr), scales))
    else:
        raise ValueError(kind)
    for l in layers:
        l.weight.data.mul_(scales.view(1, -1))


# ----------------------------------------------------------------------------- entry point

def apply_ptq(model, method, bits, group_size, calib=None, batch_size=8, log=print):
    if method == "rtn":
        return quantize_rtn(model, bits, group_size)
    if method == "gptq":
        return quantize_gptq(model, calib, bits, group_size, batch_size, log=log)
    if method == "awq":
        return quantize_awq(model, calib, bits, group_size, batch_size, log=log)
    raise ValueError(f"Unknown PTQ method: {method}")
