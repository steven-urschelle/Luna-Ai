# save_model.py
from model.luna_model import LunaAI
from transformers import BertTokenizer

def save_model(model, path='./luna_ai_model'):
    model.save_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(path)

if __name__ == "__main__":
    model = LunaAI(num_classes=2)  # Adjust num_classes if necessary
    save_model(model)
