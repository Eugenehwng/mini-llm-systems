from dataclasses import dataclass
from pathlib import Path
from minigpt_beta.data import CharTokenizer, get_dataloaders




DATA_PATH = Path(__file__).parent.parent / 'data' / 'input.txt'
data = DATA_PATH.read_text(encoding='utf-8')

# Sanity check
# print(data[:1000])
# tokenizer = CharTokenizer(data)
# print(tokenizer.vocab_size)
# print(tokenizer.encode('hello'))
# print(tokenizer.stoi)


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



train_loader, val_loader, tokenizer = get_dataloaders(data_path=DATA_PATH, block_size=GPTConfig.block_size, batch_size=GPTConfig.batch_size)

for data in train_loader:
    x, y = data
    print(x.shape, y.shape)
    print("--------------------------------")
    print(tokenizer.decode(x[0].view(-1).tolist()))
    print("--------------------------------")
    print(tokenizer.decode(y[0].view(-1).tolist()))
    break

for data in val_loader:
    x, y = data
    print(x.shape, y.shape)
    print("--------------------------------")
    print(tokenizer.decode(x[0].view(-1).tolist()))
    print("--------------------------------")
    print(tokenizer.decode(y[0].view(-1).tolist()))
    break