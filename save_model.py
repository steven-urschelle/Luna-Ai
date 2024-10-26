# save_model.py
from model.luna_model import LunaAI
from transformers import BertTokenizer

def save_model(model):
    model.save_pretrained('./luna_ai_model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained('./luna_ai_model')

if __name__ == "__main__":
    model = LunaAI()
    save_model(model)
