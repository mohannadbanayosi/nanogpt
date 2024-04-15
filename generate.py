import os
import torch
from data.dataset import Dataset
from model import BLMConfig, BigramLanguageModel


dataset = Dataset()

config = BLMConfig(vocab_size=dataset.vocab_size, device="cpu")
m = BigramLanguageModel(config)
model = m.to(config.device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

model_directory = "model_resources"
model_path = sorted(os.listdir(model_directory))[-1]
print(f"Loading model from {model_directory}/{model_path}")
model.load_state_dict(torch.load(f"{model_directory}/{model_path}"))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

print("\ngenerated text:")
print(dataset.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
