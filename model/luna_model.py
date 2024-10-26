# model/luna_model.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class LunaAI(nn.Module):
    def __init__(self):
        super(LunaAI, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 2)  # Adjust for number of classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits
