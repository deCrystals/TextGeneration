# GPT-2 Text Generation

### 📌 Overview

This project fine-tunes OpenAI's GPT-2 model for text generation using a dataset of news descriptions. The model is trained to generate text based on a given seed, with evaluations performed using ROUGE and BLEU scores.

## Table of content
* Features
* Installation
* Dataset
* Training the model
* Evaluation
* Model Checkpoints

## ✨Features
✔️ Data preprocessing using a custom PyTorch Dataset class  
✔️ Fine-tuning GPT-2 using PyTorch and Hugging Face's Transformers Library  
✔️ Text generation with configurable parameters (temperature, top-p sampling)  
✔️ Model evaluation using ROUGE and BLEU scores


## 🔧 Installation

Ensure you have Python 3.8+ installed and then install dependencies:

pip install torch transformers tqdm pandas nltk rouge-score

## 📂 Dataset

The dataset consists of a CSV file (bbc.csv) with a description column containing news text.

## 🚀 Training the Model

To fine-tune the GPT-2 model, run:
```python
from train import train_gpt
train_gpt(dataset, model, tokenizer)
```
## 🔥 Generating Text

To generate text from a trained model:
```python
from generate import generate_text
generated_text = generate_text(model, tokenizer, "Sample seed text", samples=1)
print(generated_text)
```
## 📊 Evaluation

ROUGE and BLEU scores are computed using:
```python
from evaluate import evaluate_model
scores = evaluate_model(generated_text, reference_text)
print(scores)
```
Model Checkpoints

Model weights are saved per epoch as 
```python
news-{epoch}.pt 
```
inside  the output directory.



