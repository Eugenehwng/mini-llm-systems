import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class GPTConfig:
    # training/data hyperparameters
    block_size: int = 64 # maximum context length for predictions
    batch_size: int = 16 # number of sequences processed in parallel
    
    # model arch hyperparameters
    len_embed: int = 64 # embedding dimension
    # len_head: int = 64 # QKV dimension
    num_heads: int = 8 # in multi-head attention
    num_layers: int = 8 # number of transformer blocks
    
    dropout: float = 0.1 # dropout probability

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_dim = GPTConfig.len_embed // GPTConfig.num_heads
        self.query = nn.Linear(GPTConfig.len_embed, self.qkv_dim, bias=False)
        self.key = nn.Linear(GPTConfig.len_embed, self.qkv_dim, bias=False)
        self.value = nn.Linear(GPTConfig.len_embed, self.qkv_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(GPTConfig.block_size, GPTConfig.block_size)))
        

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
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(GPTConfig.num_heads)])
        self.proj = nn.Linear(GPTConfig.len_embed, GPTConfig.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B, T, len_embed)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, len_embed)
        out = self.proj(out)
        return out
        
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(GPTConfig.len_embed, GPTConfig.len_embed * 4)
        self.proj2 = nn.Linear(GPTConfig.len_embed * 4, GPTConfig.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B, T, len_embed)
        x = self.proj1(x)
        x = self.proj2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.ffn = FFN()
        self.ln1 = nn.LayerNorm(GPTConfig.len_embed)
        self.ln2 = nn.LayerNorm(GPTConfig.len_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.block_size = GPTConfig.block_size
        self.embed_table = nn.Embedding(vocab_size, GPTConfig.len_embed)
        self.pos_enc = nn.Embedding(GPTConfig.block_size, GPTConfig.len_embed)
        self.transformer = nn.Sequential(*[TransformerBlock() for _ in range(GPTConfig.num_layers)])
        self.ln_final = nn.LayerNorm(GPTConfig.len_embed)
        self.proj_final = nn.Linear(GPTConfig.len_embed, vocab_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        B, T = x.shape # x = (B, T), y = (B, T)
        
        x = self.embed_table(x) # (B, T, len_embed)
        x = x + self.pos_enc(torch.arange(T)) # (B, T, len_embed) + (T, len_embed) = (B, T, len_embed)
        
        x = self.transformer(x) # (B, T, len_embed)
        x = self.ln_final(x)
        logits: torch.Tensor = self.proj_final(x) # (B, T, vocab_size)
        
        if y is None: # inference
            return logits, None
        else: # training
            B, T, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    @torch.no_grad
    def generate(self, x: torch.Tensor, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            # cut x to last block_size tokens
            x_cond = x[:, -self.block_size:] # (B, T') where T' <= block_size
            # get the predictions
            logits, _ = self(x_cond) # (B, T', vocab_size)
            # get the last tokens
            logits = logits [:, -1, :] # (B, vocab_size)
            
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            
            next_tok = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            x = torch.cat([x, next_tok], dim=-1) # (B, T+1)
        
        return x
            
            
            
            
            
        
