import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

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
        return [self.itos[i] for i in tokens]
        
    
class charDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int, train: bool = True):
        self.block_size = block_size
        self.train = train
        # simple train/val split
        self.data_train = data[:int(0.9*len(data))]
        self.data_val = data[int(0.9*len(data)):]
    
    def __len__(self) -> int:
        return len(self.data_train) if self.train else len(self.data_val)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            return self.data_train[idx:idx+self.block_size], self.data_train[idx+1:idx+self.block_size+1]