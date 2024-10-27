# inference.py
import torch
from transformers import BertTokenizer
from model.luna_model import LunaAI

def predict(text, model_path='./luna_ai_model'):
    model = LunaAI(num_classes=2)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    encoding = tokenizer.encode_plus(text, return_tensors='pt')
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
    return prediction.item()

if __name__ == "__main__":
    sample_text = "Sample text to classify"
    prediction = predict(sample_text)
    print(f"Prediction: {prediction}")
