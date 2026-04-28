import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentReward:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.model.eval()
        print("reward model ready (neural)")

    def score(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].tolist()