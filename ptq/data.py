"""Calibration and evaluation data for PTQ.

Language models (trained on FineWeb-Edu shards disjoint from the ones below, see
data/loaders.py:fineweb_splits):
  * calibration: random windows from the "calib" shard, GPTQ/AWQ standard of 128 sequences;
    sequence length = training context (512).
  * eval: WikiText-2 test (standard GPTQ/AWQ protocol: "\\n\\n"-joined, non-overlapping windows)
    and a held-out FineWeb-Edu set of 2^20 tokens from the "eval" shard (in-distribution).
  Documents are concatenated without separators, exactly like data/loaders.py:CPTStreamDataset.
CIFAR-10:
  * calibration: random training images with the test-time transform (1024 images, as in
    AdaRound/BRECQ); eval: the full 10k official test set.
"""
import os
import re
import torch
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ptq_cache")
FINEWEB = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
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


def _shard(role):
    from data.loaders import fineweb_splits
    return fineweb_splits(DATASET_CONFIG)[role]


def _stem(shard):
    return os.path.basename(shard).split(".")[0]


def fineweb_calib_pool(tokenizer):
    shard = _shard("calib")
    return _cached(f"{_tok_key(tokenizer)}_fineweb{_stem(shard)}_pool{CALIB_POOL_TOKENS}.pt",
                   lambda: _stream_tokens(tokenizer, shard, CALIB_POOL_TOKENS))


def lm_calibration(tokenizer, nsamples, seqlen, seed):
    pool = fineweb_calib_pool(tokenizer)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, pool.numel() - seqlen, (nsamples,), generator=g)
    return torch.stack([pool[s:s + seqlen] for s in starts.tolist()])


def lm_eval_sets(tokenizer, seqlen):
    eval_shard = _shard("eval")

    def fineweb():
        return _stream_tokens(tokenizer, eval_shard, FINEWEB_EVAL_TOKENS)

    def wikitext():
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return torch.tensor(tokenizer("\n\n".join(test["text"]))["input_ids"], dtype=torch.long)

    key = _tok_key(tokenizer)
    sets = {
        "wikitext2": _cached(f"{key}_wikitext2_test.pt", wikitext),
        "fineweb": _cached(f"{key}_fineweb{_stem(eval_shard)}_eval{FINEWEB_EVAL_TOKENS}.pt", fineweb),
    }
    return {k: v[: (v.numel() // seqlen) * seqlen].view(-1, seqlen) for k, v in sets.items()}


def cifar_eval_loader(batch_size):
    from data.loaders import get_cifar10_test_loader
    return get_cifar10_test_loader(batch_size=batch_size)


def cifar_calibration(nsamples, seed):
    from data.loaders import CIFAR10ArrowDataset, CIFAR_TRANSFORM_TEST
    train = CIFAR10ArrowDataset(load_dataset("cifar10")["train"], transform=CIFAR_TRANSFORM_TEST)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train), generator=g)[:nsamples].tolist()
    return torch.stack([train[i]["image"] for i in idx])
