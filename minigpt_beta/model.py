import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from __future__ import annotations

from minigpt_beta.train import GPTConfig
    
class Head(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.qkv_dim = config.len_embed // config.num_heads
        self.query = nn.Linear(config.len_embed, self.qkv_dim, bias=False)
        self.key = nn.Linear(config.len_embed, self.qkv_dim, bias=False)
        self.value = nn.Linear(config.len_embed, self.qkv_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape               # batch size, block size, embedding dimension
        
        q = self.query(x)               # (B, T, qkv_dim)
        k = self.key(x)                 # (B, T, qkv_dim)
        v = self.value(x)               # (B, T, qkv_dim)
        
        scores: torch.Tensor = q @ torch.transpose(k, 1, 2)       # (B, T, qkv_dim) @ (B, qkv_dim, T) = (B, T, T)
        scores = scores * (self.qkv_dim ** 0.5)     #  / sqrt(d_k)
        # TODO: should this be qkv_dim or len_embed?
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")) 
        scores = F.softmax(scores, dim=-1)
        out = scores @ v                            # (B, T, T) @ (B, T, qkv_dim) = (B, T, qkv_dim)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.len_embed, config.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B, T, len_embed)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, len_embed)
        out = self.proj(out)
        return out
        
class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.proj1 = nn.Linear(config.len_embed, config.len_embed * 4)
        self.proj2 = nn.Linear(config.len_embed * 4, config.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B, T, len_embed)
        x = self.proj1(x)
        x = self.proj2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FFN(config)
        self.ln1 = nn.LayerNorm(config.len_embed)
        self.ln2 = nn.LayerNorm(config.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, vocab_size: int):
        super().__init__()
        self.block_size = config.block_size
        self.embed_table = nn.Embedding(vocab_size, config.len_embed)
        self.pos_enc = nn.Embedding(config.block_size, config.len_embed)
        self.transformer = nn.Sequential([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(config.len_embed)
        self.proj_final = nn.Linear(config.len_embed, vocab_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x = (B, T), y = (B, T)
        x = self.embed_table(x) # (B, T, len_embed)
        x = x + self.pos_enc(torch.arange(self.block_size)) # (B, T, len_embed) + (T, len_embed) = (B, T, len_embed)
        
        x = self.transformer(x) # (B, T, len_embed)
        x = self.ln_final(x)
        logits: torch.Tensor = self.proj_final(x) # (B, T, vocab_size)
        
        B, T, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = y.view(-1)
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self)
        pass
        
