"""
AI Based Fake News Detection - RoBERTa Training Script
Optimized for: preprocessed_welfake.csv (Enhanced WELFake Dataset)
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. SETUP & CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'roberta-base'
MAX_LEN = 128
BATCH_SIZE = 16   # RoBERTa is larger, so smaller batch to avoid OOM
EPOCHS = 3
LEARNING_RATE = 2e-5

# 2. DATA LOADING
DATA_PATH = "/kaggle/input/datasets/mohammedaazam7/fakenewsdataset/preprocessed_welfake.csv"

if not os.path.exists(DATA_PATH):
    print("---------------------------------------------------------------------------")
    print(f"❌ ERROR: Dataset not found at: {DATA_PATH}")
    print("---------------------------------------------------------------------------")
    raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.dropna(subset=['text', 'label'], inplace=True)
df = df[['text', 'label']]

train_text, val_text, train_labels, val_labels = train_test_split(
    df['text'].values, df['label'].values, test_size=0.2, random_state=42  # 20% val
)
print(f"✅ Dataset Loaded: {len(df)} rows | Train: {len(train_text)} | Val: {len(val_text)}")

# 3. TOKENIZATION
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

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

train_data_loader = DataLoader(NewsDataset(train_text, train_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(NewsDataset(val_text, val_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# 4. MODEL INITIALIZATION
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * len(train_data_loader) * EPOCHS),  # 10% warmup
    num_training_steps=len(train_data_loader) * EPOCHS
)

# 5. TRAINING LOOP WITH VALIDATION ACCURACY + BEST MODEL SAVING
print(f"🚀 Starting RoBERTa Training on {DEVICE}...")

best_val_accuracy = 0

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    optimizer.zero_grad()

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

    avg_train_loss = total_loss / len(train_data_loader)

    # --- VALIDATE ---
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in val_data_loader:
            input_ids = d["input_ids"].to(DEVICE)
            attention_mask = d["attention_mask"].to(DEVICE)
            labels = d["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"✅ Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        os.makedirs("model_output_roberta", exist_ok=True)
        model.save_pretrained("model_output_roberta")
        tokenizer.save_pretrained("model_output_roberta")
        print(f"💾 Best model saved at epoch {epoch+1} with Val Accuracy: {val_accuracy:.2f}%")

print(f"🎉 Done! Best Val Accuracy: {best_val_accuracy:.2f}%")
print("📦 Download the 'model_output_roberta' folder from the Output tab.")