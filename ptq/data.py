"""Calibration and evaluation data for PTQ.

Language models (trained on the head of FineWeb-Edu sample-10BT shard 000):
  * calibration: random windows from shard 012 (never seen in training), GPTQ/AWQ standard of
    128 sequences; sequence length = training context (512).
  * eval: WikiText-2 test (standard GPTQ/AWQ protocol: "\\n\\n"-joined, non-overlapping windows)
    and a held-out FineWeb-Edu set of 2^20 tokens from shard 013 (in-distribution).
  Documents are concatenated without separators, exactly like data/loaders.py:CPTStreamDataset.
CIFAR-10:
  * calibration: random training images with the test-time transform (1024 images, as in
    AdaRound/BRECQ); eval: the full 10k test set.
"""
import os
import re
import torch
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ptq_cache")
FINEWEB = "HuggingFaceFW/fineweb-edu"
CALIB_SHARD = "sample/10BT/012_00000.parquet"
EVAL_SHARD = "sample/10BT/013_00000.parquet"
CALIB_POOL_TOKENS = 4 * 2 ** 20
FINEWEB_EVAL_TOKENS = 2 ** 20


def _tok_key(tokenizer):
    return re.sub(r"[^A-Za-z0-9]+", "_", tokenizer.name_or_path)


def _cached(name, build):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return torch.load(path)
    data = build()
    tmp = path + f".tmp{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)
    return data


def _stream_tokens(tokenizer, shard, n_tokens):
    ds = load_dataset(FINEWEB, data_files=shard, split="train", streaming=True)
    buf = []
    for ex in ds:
        text = ex.get("text", "")
        if text:
            buf.extend(tokenizer.encode(text, truncation=False))
        if len(buf) >= n_tokens:
            break
    return torch.tensor(buf[:n_tokens], dtype=torch.long)


def fineweb_calib_pool(tokenizer):
    return _cached(f"{_tok_key(tokenizer)}_fineweb012_pool{CALIB_POOL_TOKENS}.pt",
                   lambda: _stream_tokens(tokenizer, CALIB_SHARD, CALIB_POOL_TOKENS))


def lm_calibration(tokenizer, nsamples, seqlen, seed):
    pool = fineweb_calib_pool(tokenizer)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, pool.numel() - seqlen, (nsamples,), generator=g)
    return torch.stack([pool[s:s + seqlen] for s in starts.tolist()])


def lm_eval_sets(tokenizer, seqlen):
    def fineweb():
        return _stream_tokens(tokenizer, EVAL_SHARD, FINEWEB_EVAL_TOKENS)

    def wikitext():
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return torch.tensor(tokenizer("\n\n".join(test["text"]))["input_ids"], dtype=torch.long)

    key = _tok_key(tokenizer)
    sets = {
        "wikitext2": _cached(f"{key}_wikitext2_test.pt", wikitext),
        "fineweb": _cached(f"{key}_fineweb013_eval{FINEWEB_EVAL_TOKENS}.pt", fineweb),
    }
    return {k: v[: (v.numel() // seqlen) * seqlen].view(-1, seqlen) for k, v in sets.items()}


def cifar_eval_loader(batch_size):
    from data.loaders import get_cifar10_dataloaders
    _, val_loader = get_cifar10_dataloaders(batch_size=batch_size)
    return val_loader


def cifar_calibration(val_loader, nsamples, seed):
    from data.loaders import CIFAR10ArrowDataset
    train = CIFAR10ArrowDataset(load_dataset("cifar10")["train"], transform=val_loader.dataset.transform)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train), generator=g)[:nsamples].tolist()
    return torch.stack([train[i]["image"] for i in idx])
