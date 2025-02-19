import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Data Preprocessing Class
class NewsDataset(Dataset):
    def __init__(self, data, end_text='news', truncate=False, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.news = []
        
        for item in data['description']:
            encoded_text = self.tokenizer.encode(f"<|{end_text}|>{item[:max_length]}")
            self.news.append(torch.tensor(encoded_text))
        
        if truncate:
            self.news = self.news[:30000]
        
    def __len__(self):
        return len(self.news)
    
    def __getitem__(self, index):
        return self.news[index]

# Load Dataset
data = pd.read_csv('/content/drive/MyDrive/Textgen/bbc.csv')
dataset = NewsDataset(data, truncate=True)

# Load Pre-trained GPT-2 Model and Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def train_gpt(dataset, model, tokenizer, batch_size=16, num_epochs=6, learning_rate=1e-5, 
              max_seq_length=500, warmup_steps=250, output_dir=".", output_prefix="news"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    news_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} Training...")
        for batch in tqdm(news_loader):
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
        
        torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"))
    return model

# Train the model
model = train_gpt(dataset, model, tokenizer)

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

# Evaluate Model with ROUGE
reference_text = "This is a reference text for evaluation."
generated_text = generate_text(model, tokenizer, "Sample seed text", samples=1)[0]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
scores = scorer.score(generated_text, reference_text)
print("ROUGE Scores:", scores)

# Evaluate Model with BLEU
smoothing = SmoothingFunction()
reference = nltk.word_tokenize(reference_text)
generated = nltk.word_tokenize(generated_text)
bleu_score = sentence_bleu([reference], generated, smoothing_function=smoothing.method1)
print("BLEU Score:", bleu_score)
