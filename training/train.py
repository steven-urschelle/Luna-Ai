# training/train.py
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model.luna_model import LunaAI
import torch
import torch.nn as nn
from transformers import AdamW

class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True,
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

def train_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):  # Adjust the number of epochs
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    dataset = TextDataset('data/dataset.csv')
    model = LunaAI()
    train_model(model, dataset)
