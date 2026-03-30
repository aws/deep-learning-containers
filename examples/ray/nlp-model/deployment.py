from ray import serve
from transformers import pipeline
import torch


@serve.deployment(num_replicas=1)
class DistilBERTSentiment:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
        )

    async def __call__(self, request):
        data = await request.json()
        text = data.get("text", "")
        results = self.classifier([text] if isinstance(text, str) else text)
        return {"predictions": results}


app = DistilBERTSentiment.bind()
