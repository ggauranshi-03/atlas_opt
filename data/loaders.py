import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset
import torchvision.transforms as T

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

def get_cpt_dataloaders(tokenizer, dataset_name="HuggingFaceFW/fineweb-edu", dataset_config="sample-10BT", batch_size=16, max_len=256):
    print(f"  Streaming CPT corpus: {dataset_name} ({dataset_config}) (max_len={max_len})...")
    
    try:
        raw_train = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    except Exception as e:
        print(f"  [Warning] Failed to load config {dataset_config}, trying without config: {e}")
        raw_train = load_dataset(dataset_name, split="train", streaming=True)
        
    train_ds = CPTStreamDataset(raw_train, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0) 
    
    # Validation dataset fallback (some datasets don't have a split called validation)
    try:
        raw_val = load_dataset(dataset_name, name=dataset_config, split="validation", streaming=True)
    except Exception:
        raw_val = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        
    val_ds = CPTStreamDataset(raw_val, tokenizer, max_len=max_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0) 
    
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

def get_cifar10_dataloaders(batch_size=128):
    print("  Loading CIFAR-10 vision dataset...")
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        raw_ds = load_dataset("cifar10")
        train_ds = CIFAR10ArrowDataset(raw_ds["train"], transform=transform_train)
        val_ds   = CIFAR10ArrowDataset(raw_ds["test"], transform=transform_test)
    except Exception:
        import torchvision.datasets as dset
        train_dset = dset.CIFAR10(root="./data_cache", train=True, download=True, transform=transform_train)
        val_dset   = dset.CIFAR10(root="./data_cache", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_loader, val_loader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    print(f"  CIFAR-10 Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader
