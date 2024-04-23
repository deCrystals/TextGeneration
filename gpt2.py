# importing the dependencies
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

#Data preprocessing
class New(Data):
    def __init__(self, end_text, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.news = []

        for item in data['description']:
          encoded_text = self.tokenizer.encode(f"<|{end_text}|>{item[:max_len]}")
          self.news.append(torch.tensor(encoded_text))

        if truncate:
            self.news = self.news[:30000]
        self.news_count = len(self.news)
    def __len__(self):
        return self.news_count

    def __getitem__(self, item):
        return self.news[item]
    
#loading the dataset
data = pd.read_csv('/content/drive/MyDrive/Textgen/bbc.csv')
data.head()

#calling the class New (data pre-processing )for Model building
dataset = New(data, truncate=True, gpt2_type="gpt2")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#function for training(fine-tune) the model
def train_gpt(data, model, tokenizer, batch_size=16, num_epoch=6, learning_rate=1e-5, max_seq_length=500, warmup_steps=250,
                gpt2_type="gpt2", output_dir=".", output_prefix="news"):
    accumsteps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    news_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loss = 0
    accumcount = 0
    input_tensor = None
    for i in range(num_epochs):
        print(f"Epoch Training {i}")
        print(loss)
        for idx, x in tqdm(enumerate(news_loader)):
            input_tensor = x.to(device)
            output = model(input_tensor, labels=input_tensor)
            loss = output.loss
            loss.backward()
            if (accumcount % accumsteps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
            accumcount += 1
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{i}.pt"),
            )
    return model
#building the model
model = gpt_train(dataset, model, tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#saving the model
model = torch.load('/content/drive/MyDrive/Textgen/news_uk.pt')



#Generating the text
def generate_text(model, tkn, seed, samples=1, max_len=200, top_prob=0.6, temp=0.5):
    model.eval()
    samples = []
    filter_value = -float("inf")
    with torch.no_grad():
        for i in trange(samples):
            seed_complete = False
            seed_ids = torch.tensor(tkn.encode(seed)).unsqueeze(0)
            for _ in range(max_len):
                outcomes = model(seed_ids, labels=seed_ids)
                loss, logits = outcomes[:2]
                logits = logits[:, -1, :] / (temp if temp > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_exclude = cumulative_probs > top_prob
                sorted_exclude[..., 1:] = sorted_exclude[..., :-1].clone()
                sorted_exclude[..., 0] = 0
                indices_exclude = sorted_indices[sorted_exclude]
                logits[:, indices_exclude] = filter_value
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                seed_ids = torch.cat((seed_ids, next_token), dim=1)
                if next_token.item() == tkn.encode("")[0]:
                    seed_complete = True
                if seed_complete:
                    samples.append(tkn.decode(seed_ids[0].tolist()))
                    break
            if not seed_complete:
                samples.append(tkn.decode(seed_ids[0].tolist()))
    return samples

# Evaluation using rouge 
from rouge_score import rouge_scorer
generated_text =gpt
reference_text = original2
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
# Calculate ROUGE scores
scores = scorer.score(generated_text, reference_text)
# Access individual ROUGE scores
rouge1_score = scores['rouge1']
rouge2_score = scores['rouge2']
rougeL_score = scores['rougeL']
rougeLsum_score = scores['rougeLsum']
# Print the scores
print('    lstm ROUGE SCORE at temp = 0.5')
print('=========================')
print("ROUGE-1 Score:", rouge1_score)
print("ROUGE-2 Score:", rouge2_score)
print("ROUGE-L Score:", rougeL_score)
print("ROUGE-Lsum Score:", rougeLsum_score)

# Evaluation using the BLEU
import statistics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
scores =[]
smoothing = SmoothingFunction()
reference = nltk.word_tokenize(original2)
generated = nltk.word_tokenize(gpt)
scores = (sentence_bleu(reference, generated, smoothing_function=smoothing.method1))
print('GPT BLEU score at Temperature')
print('++++++++++++++++++++++++++++++++++++++')
print(scores)

