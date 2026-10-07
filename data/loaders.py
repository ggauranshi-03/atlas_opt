import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset
from datasets import load_dataset
import torchvision.transforms as T

FINEWEB = "HuggingFaceFW/fineweb-edu"
CIFAR_VAL_SIZE = 5000


class CPTStreamDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, max_len=256, text_column="text"):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_column = text_column

    def __iter__(self):
        buffer = []
        for example in self.hf_dataset:
            text = example.get(self.text_column, "")
            if not text: continue

            tokens = self.tokenizer.encode(text, truncation=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_len:
                chunk = buffer[:self.max_len]
                buffer = buffer[self.max_len:]

                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": tensor_chunk, "labels": tensor_chunk.clone()}


def fineweb_splits(dataset_config="sample-10BT"):
    """Disjoint parquet shards per role: all but the last three shards train the model; the last three
    are held out as training-time validation, PTQ calibration and final evaluation, respectively."""
    from huggingface_hub import HfApi
    prefix = "/".join(dataset_config.split("-", 1)) + "/"  # "sample-10BT" -> "sample/10BT/"
    shards = sorted(f for f in HfApi().list_repo_files(FINEWEB, repo_type="dataset")
                    if f.startswith(prefix) and f.endswith(".parquet"))
    if len(shards) < 4:
        raise ValueError(f"Need >= 4 shards under {prefix} for disjoint splits, found {len(shards)}")
    return {"train": shards[:-3], "val": shards[-3], "calib": shards[-2], "eval": shards[-1]}


def get_cpt_dataloaders(tokenizer, dataset_name=FINEWEB, dataset_config="sample-10BT", batch_size=16, max_len=256, seed=42):
    if dataset_name != FINEWEB:
        raise ValueError(f"Held-out shard splits are defined for {FINEWEB} only, got {dataset_name}")
    splits = fineweb_splits(dataset_config)
    print(f"  Streaming CPT corpus: {dataset_name} ({dataset_config}) (max_len={max_len}) | "
          f"train shards: {len(splits['train'])} | val shard: {splits['val']}")

    raw_train = load_dataset(dataset_name, data_files=splits["train"], split="train", streaming=True)
    raw_train = raw_train.shuffle(seed=seed, buffer_size=10_000)
    train_loader = DataLoader(CPTStreamDataset(raw_train, tokenizer, max_len=max_len), batch_size=batch_size, num_workers=0)

    raw_val = load_dataset(dataset_name, data_files=splits["val"], split="train", streaming=True)
    val_loader = DataLoader(CPTStreamDataset(raw_val, tokenizer, max_len=max_len), batch_size=batch_size, num_workers=0)

    print("  Streaming DataLoaders ready.")
    return train_loader, val_loader


class CIFAR10ArrowDataset(Dataset):
    def __init__(self, arrow_ds, transform=None):
        self.ds = arrow_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"]
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}


CIFAR_TRANSFORM_TRAIN = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
CIFAR_TRANSFORM_TEST = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_cifar10_dataloaders(batch_size=128):
    """Train on 45k training images; validate on a fixed held-out 5k split of the training set
    (used for model selection). The official test set is reserved for final reporting."""
    print("  Loading CIFAR-10 vision dataset...")
    raw_train = load_dataset("cifar10")["train"]
    # The split is fixed (seed 0) so every run and every seed shares the same validation images.
    perm = torch.randperm(len(raw_train), generator=torch.Generator().manual_seed(0)).tolist()
    val_idx, train_idx = perm[:CIFAR_VAL_SIZE], perm[CIFAR_VAL_SIZE:]
    train_ds = Subset(CIFAR10ArrowDataset(raw_train, transform=CIFAR_TRANSFORM_TRAIN), train_idx)
    val_ds = Subset(CIFAR10ArrowDataset(raw_train, transform=CIFAR_TRANSFORM_TEST), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    print(f"  CIFAR-10 Train: {len(train_ds):,} | Val (held-out from train): {len(val_ds):,}")
    return train_loader, val_loader


def get_cifar10_test_loader(batch_size=500):
    test_ds = CIFAR10ArrowDataset(load_dataset("cifar10")["test"], transform=CIFAR_TRANSFORM_TEST)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def infinite_batches(loader):
    """Yield batches forever, continuing where the previous epoch stopped instead of restarting."""
    while True:
        yield from loader
