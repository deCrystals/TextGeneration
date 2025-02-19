import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import os

def generate_text(model, tokenizer, seed_text, samples=1, max_len=200, top_p=0.6, temp=0.5):
    model.eval()
    generated_samples = []
    filter_value = -float("inf")
    
    with torch.no_grad():
        for _ in range(samples):
            seed_ids = torch.tensor(tokenizer.encode(seed_text)).unsqueeze(0)
            for _ in range(max_len):
                outputs = model(seed_ids, labels=seed_ids)
                logits = outputs.logits[:, -1, :] / temp
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_exclude = cumulative_probs > top_p
                sorted_exclude[..., 1:] = sorted_exclude[..., :-1].clone()
                sorted_exclude[..., 0] = 0
                indices_exclude = sorted_indices[sorted_exclude]
                logits[:, indices_exclude] = filter_value
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                seed_ids = torch.cat((seed_ids, next_token), dim=1)
                
                if next_token.item() == tokenizer.encode(".")[0]:
                    break
            generated_samples.append(tokenizer.decode(seed_ids[0].tolist()))
    return generated_samples