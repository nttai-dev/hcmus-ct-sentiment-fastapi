from typing import Any
from omegaconf import OmegaConf
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class SentimentClassifier:
    """Wrapper around Hugging Face DistilBERT sentiment classification model."""

    def __init__(self, config_path: str = "./config.yaml") -> None:
        self.config = OmegaConf.load(config_path)
        self.model_path: str = str(self.config.model_path)
        self.max_length: int = int(self.config.get("max_length", 512))

        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        predicted_class_id = int(probs.argmax().item())
        label = str(self.model.config.id2label[predicted_class_id])
        confidence = round(float(probs[predicted_class_id].item()), 4)

        scores: dict[str, float] = {
            str(self.model.config.id2label[idx]): round(float(prob.item()), 4)
            for idx, prob in enumerate(probs)
        }

        return {
            "text": text,
            "label": label,
            "confidence": confidence,
            "scores": scores,
        }

    def __call__(self, text: str) -> dict[str, Any]:
        return self.predict(text)
