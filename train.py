import time
import torch
from tqdm import tqdm

from model import BLMConfig, BigramLanguageModel


# Hyperparameters
batch_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_interval = 250
eval_iters = 200

# TODO: move text helpers to common code
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"{vocab_size=}")
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"{len(train_data)=}", f"{len(val_data)=}")

# TODO: add default model config for different model sizes
config = BLMConfig(vocab_size=vocab_size, device="mps" if torch.backends.mps.is_available() else "cpu")
m = BigramLanguageModel(config)
model = m.to(config.device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# TODO: move text helpers to common code
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y =x.to(config.device), y.to(config.device)
    return x, y

# Using this context manager to tell pytorch that we don't want to call .backward() for this
# since we don't want to store all intermediate variables -> mem efficiency
@torch.no_grad()
def estimate_loss():
    final_losses = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        final_losses[split] = losses.mean()

    model.train()
    return final_losses

# TODO: try SGD
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

with tqdm(total=max_iters, desc=f"batch size={batch_size}") as pbar:
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.update(1)

val_loss = "{:.4f}".format(losses['val']).replace(".", "")
torch.save(model.state_dict(), f"model_resources/model_{int(time.time())}_{val_loss}.pth")
    
print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
