import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Optional

# module level functions
def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Tokenizer (character level)
class CharTokenizer:
    def __init__(self, text: str):
        self.text = text
        self.vocab = sorted(list(set(text)))
        
        #mapping from all unique characters in our vocab -> integer
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[i] for i in tokens])
        
    
class charDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int, train_split: float = 0.9, train: bool = True):
        self.block_size = block_size
        self.train = train
        self.train_split = train_split
        # simple train/val split
        self.data_train = data[:int(train_split*len(data))]
        self.data_val = data[int(train_split*len(data)):]
    
    def __len__(self) -> int:
        n = len(self.data_train) if self.train else len(self.data_val)
        return n - self.block_size
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            return self.data_train[i:i+self.block_size], self.data_train[i+1:i+self.block_size+1]
        else: 
            return self.data_val[i:i+self.block_size], self.data_val[i+1:i+self.block_size+1]
        
        
def get_dataloaders(
    data_path: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_split: float = 0.9   
) -> tuple[DataLoader, DataLoader, CharTokenizer]:

    """
    Returns:
        tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    
    data = load_text(data_path)
    # Create Tokenizer
    tokenizer = CharTokenizer(data)
    data = torch.tensor(tokenizer.encode(data)) # encode the entire dataset, convert to tensors
    # Create Datasets
    train_dataset = charDataset(data, block_size, train_split)
    val_dataset = charDataset(data, block_size, train_split, train=False)
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, tokenizer
    
    