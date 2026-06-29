from dataclasses import dataclass
from pathlib import Path
import torch
from torchinfo import summary
from minigpt_beta.data import CharTokenizer, get_dataloaders
from minigpt_beta.model import GPT, GPTConfig

@dataclass
class TrainConfig:    
    max_new_tokens: int = 10000    
    max_iters: int = 5000
    lr: float = 1e-3
    eval_iters: int = 200
    eval_interval: int = 100
        
    
def make_iter(loader):
    while True:
        for batch in loader:
            yield batch
            
@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split, it in [('train', train_iter), ('val', val_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = next(it)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    
    c = TrainConfig()
    DATA_PATH = Path(__file__).parent.parent / 'data' / 'input.txt'
    data = DATA_PATH.read_text(encoding='utf-8')

    train_loader, val_loader, tokenizer = get_dataloaders(data_path=DATA_PATH, block_size=GPTConfig.block_size, batch_size=GPTConfig.batch_size)
    m = GPT(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(m.parameters(), TrainConfig.lr)
    
    train_iter = make_iter(train_loader)
    val_iter   = make_iter(val_loader)
    
    # summary(m, input_data=torch.randint(0, tokenizer.vocab_size, (GPTConfig.batch_size, GPTConfig.block_size)))

    m.train()
    for iter in range(c.max_iters):
        if iter % c.eval_interval == 0:
            losses = estimate_loss(m, c.eval_iters)
            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            print("-" * 50)
            ctx = torch.zeros((1, 1), dtype=torch.long)
            sample = m.generate(ctx, max_new_tokens=200)[0].tolist()
            print(f"sample: {tokenizer.decode(sample)}")
            print("-" * 50)
        
        xb, yb = next(train_iter)
        _, loss = m(xb, yb)
        
        optimizer.zero_grad(set_to_none=True) # clear previous iter's gradients
        loss.backward() # calculate gradients
        optimizer.step() # update parameters based on gradients