import torch
from model import BLMConfig, BigramLanguageModel


# TODO: move text helpers to common code
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = BLMConfig(vocab_size=vocab_size, device="cpu")
m = BigramLanguageModel(config)
model = m.to(config.device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# TODO: allow to specify model name or use latest
model.load_state_dict(torch.load('my_model.pth'))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

print("\ngenerated text:")
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
