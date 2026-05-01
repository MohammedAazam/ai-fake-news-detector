"""
AI Based Fake News Detection - Training Script 
Optimized for: preprocessed_welfake.csv (Enhanced WELFake Dataset)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import time

# 1. SETUP & CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128 
BATCH_SIZE = 32 
EPOCHS = 3
LEARNING_RATE = 2e-5

# 2. UPDATED DATA LOADING FOR KAGGLE
# Verified path from user screenshot
DATA_PATH = "/kaggle/input/datasets/mohammedaazam7/fakenewsdataset/preprocessed_welfake.csv" 

if not os.path.exists(DATA_PATH):
    print("---------------------------------------------------------------------------")
    print(f"❌ ERROR: Dataset not found at: {DATA_PATH}")
    print("---------------------------------------------------------------------------")
    raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
else:
    # Reading the preprocessed version
    df = pd.read_csv(DATA_PATH)
    
    # Standard WELFake columns are 'text' and 'label' (0=Real, 1=Fake)
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[['text', 'label']] 

    train_text, val_text, train_labels, val_labels = train_test_split(
        df['text'].values, df['label'].values, test_size=0.1, random_state=42
    )
    print(f"✅ Dataset Loaded: {len(df)} rows")

# 3. TOKENIZATION
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Calling the tokenizer directly is more robust than using .encode_plus()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Data Loaders
train_data_loader = DataLoader(NewsDataset(train_text, train_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(NewsDataset(val_text, val_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# 4. MODEL INITIALIZATION
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

# Updated optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * EPOCHS)

# 5. TRAINING LOOP
print(f"🚀 Starting Training on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, d in enumerate(train_data_loader):
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_data_loader)} | Loss: {loss.item():.4f}")
    
    print(f"✅ Epoch {epoch + 1} completed. Avg Loss: {total_loss/len(train_data_loader):.4f}")

# 6. SAVING THE MODEL (Download this folder after training)
os.makedirs("model_output", exist_ok=True)
model.save_pretrained("model_output")
tokenizer.save_pretrained("model_output")
print("🎉 Done! Download the 'model_output' folder from the 'Output' tab to use with your FastAPI backend.")