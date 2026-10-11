"""READ-ONLY diagnostic: how large are the attention logits q.k/sqrt(d) in the base checkpoints?

For every base model (nanoGPT / Pythia-70M, pretrained with AdamW / Muon / SOMA, seeds 42-44, plus the released weights as reference) the maximum
causal attention logit per layer and head is measured on held-out FineWeb-Edu windows (512 tokens). Nothing is trained or modified; the only writes
are in this folder. The hook-based logit computation is validated against the model itself (softmax(logits) @ v must reproduce the input of the
attention output projection).

  python qk_diagnostic/qk_diagnostic.py --gpu 0
"""
import argparse, json, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ft_optimizer_study"))
import ft_study as S                 # noqa: E402  (chdirs to the repo, imports tools/forgetting.py as F)
F = S.F
from ptq import data as pdata        # noqa: E402

NW = 256                             # windows of 512 tokens -> 131k tokens


class Probe:
    """Registers hooks that record q, k (and v) of every attention layer and compares the recomputed attention output with the model's."""
    def __init__(self, model):
        self.model, self.cfg = model, model.config
        self.is_neox = self.cfg.model_type == "gpt_neox"
        layers = model.gpt_neox.layers if self.is_neox else model.transformer.h
        self.n = len(layers); self.qkv = {}; self.pre = {}; self.hooks = []
        for i, L in enumerate(layers):
            a = L.attention if self.is_neox else L.attn
            qkv = a.query_key_value if self.is_neox else a.c_attn
            out = a.dense if self.is_neox else a.c_proj
            self.hooks.append(qkv.register_forward_hook(lambda m, inp, o, i=i: self.qkv.__setitem__(i, o.detach())))
            self.hooks.append(out.register_forward_pre_hook(lambda m, inp, i=i: self.pre.__setitem__(i, inp[0].detach())))
        if self.is_neox:
            self.H = self.cfg.num_attention_heads; self.hd = self.cfg.hidden_size // self.H
            self.rot = int(self.hd * self.cfg.rotary_pct); base = getattr(self.cfg, "rotary_emb_base", 10000)
            self.inv = 1.0 / (base ** (torch.arange(0, self.rot, 2).float() / self.rot))
        else:
            self.H = self.cfg.n_head; self.hd = self.cfg.n_embd // self.H

    def qkv_heads(self, i):
        x = self.qkv[i].float(); B, T, _ = x.shape
        if self.is_neox:
            q, k, v = x.view(B, T, self.H, 3 * self.hd).chunk(3, dim=-1)     # NeoX: per head [q, k, v]
        else:
            q, k, v = (t.view(B, T, self.H, self.hd) for t in x.chunk(3, dim=-1))   # GPT-2: [Q | K | V]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                   # B,H,T,hd
        if self.is_neox:
            pos = torch.arange(T, device=x.device).float()
            fr = torch.outer(pos, self.inv.to(x.device)); emb = torch.cat([fr, fr], -1)
            cos, sin = emb.cos()[None, None], emb.sin()[None, None]
            def rope(t):
                r, p = t[..., :self.rot], t[..., self.rot:]
                h = self.rot // 2
                rh = torch.cat([-r[..., h:], r[..., :h]], -1)
                return torch.cat([r * cos + rh * sin, p], -1)
            q, k = rope(q), rope(k)
        return q, k, v

    def logits_and_check(self, i):
        q, k, v = self.qkv_heads(i); T = q.shape[2]
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)                  # B,H,T,T
        mask = torch.ones(T, T, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        o = torch.softmax(s, -1) @ v                                       # B,H,T,hd
        o = o.transpose(1, 2).reshape(o.shape[0], T, -1)
        err = (o - self.pre[i].float()).abs().max().item()
        mx = s.amax(dim=(0, 2, 3))                                         # per head, max over batch / queries / keys
        valid = s[:, :, :, :][:, :, mask]                                  # B,H,n_valid causal entries
        self.q999 = torch.quantile(valid[:, :, ::4].transpose(0, 1).reshape(self.H, -1).float(), 0.999, dim=1)   # per head, 99.9th percentile (every 4th entry)
        # max over non-first-token keys too (the first token is often an attention sink with extreme logits)
        s2 = s.clone(); s2[..., 0] = float("-inf")
        mx2 = s2[:, :, 1:].amax(dim=(0, 2, 3))
        return mx, mx2, err, self.q999


@torch.no_grad()
def measure(model, windows, device, bs=16):
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False     # re-asserted here: imports elsewhere switch TF32 back on
    torch.set_float32_matmul_precision("highest")
    model.eval(); pr = Probe(model)
    mx = torch.full((pr.n, pr.H), float("-inf")); mx2 = mx.clone(); q9 = mx.clone(); worst = 0.0
    for b in range(0, windows.shape[0], bs):
        ids = windows[b:b + bs].to(device, dtype=torch.long)
        model(input_ids=ids, use_cache=False)
        for i in range(pr.n):
            a, a2, err, qq = pr.logits_and_check(i)
            mx[i] = torch.maximum(mx[i], a.cpu()); mx2[i] = torch.maximum(mx2[i], a2.cpu()); q9[i] = torch.maximum(q9[i], qq.cpu()); worst = max(worst, err)
    for h in pr.hooks: h.remove()
    return mx, mx2, worst, q9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
    device = f"cuda:{a.gpu}"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False     # TF32 rounding corrupts logits of size 1e4-1e5
    torch.set_float32_matmul_precision("highest")
    out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
    rows = []
    for m in S.MODELS:
        from utils.helpers import build_model
        import yaml
        cfg = yaml.safe_load(open(os.path.join(S.REPO, F.MODELS[f"{m}_s42"]["config"])))
        ref, _, tok = build_model(cfg, device)                                  # released weights
        windows = pdata.lm_eval_sets(tok, F.SEQ)["fineweb"][:NW]
        todo = [("released", 0, ref.float())]
        for base in S.BASES:
            for seed in S.SEEDS:
                model, _, _ = S.load_base(m, base, seed, device)
                todo.append((S.BASES[base], seed, model))
        for name, seed, model in todo:
            mx, mx2, err, q9 = measure(model, windows, device)
            r = dict(model=m, base=name, seed=seed, max_logit=mx.max().item(), max_logit_excl_first_key=mx2.max().item(),
                     mean_over_heads_of_head_max=mx.mean().item(), p999_logit_max_over_heads=q9.max().item(), p999_mean_over_heads=q9.mean().item(), per_layer_max=mx.max(1).values.tolist(), per_head_max=mx.tolist(),
                     hook_vs_model_max_abs_err=err, windows=NW, tokens=NW * F.SEQ)
            rows.append(r)
            print(f"{m:9s} {name:8s} s{seed}: max logit {r['max_logit']:7.2f} | excl. first key {r['max_logit_excl_first_key']:7.2f} | "
                  f"mean head-max {r['mean_over_heads_of_head_max']:8.2f} | p99.9 (max head) {r['p999_logit_max_over_heads']:7.2f} | check err {err:.2e}", flush=True)
            if name != "released": del model
        json.dump(rows, open(os.path.join(out, "max_logits.json"), "w"), indent=1)
        del ref; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    os._exit(0)
