"""hf-serve custom handler that classifies text on Neuron in eager or compiled mode.

Package it as `handler.py` at the root of the model directory (or under `code/`)
and serve it with `TASK=custom TRUST_REMOTE_CODE=true`.

Environment variables:
    NEURON_INFERENCE_MODE: `eager` (default) or `compiled`, which wraps the model
        with `torch.compile(backend="neuron", dynamic=False)`.
    HANDLER_MODEL_ID: Hub model to load when the model directory has no
        `config.json` (default: distilbert-base-uncased-finetuned-sst-2-english).
"""

import os
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "neuron"
DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# Pad every request to one length so torch.compile sees static shapes.
SEQUENCE_LENGTH = 32


class EndpointHandler:
    def __init__(self, model_dir: str, **kwargs: Any) -> None:
        self.mode = os.getenv("NEURON_INFERENCE_MODE", "eager")
        if self.mode not in ("eager", "compiled"):
            raise ValueError(f"NEURON_INFERENCE_MODE must be eager or compiled, not {self.mode}")
        model_id = model_dir
        if not (Path(model_dir) / "config.json").is_file():
            model_id = os.getenv("HANDLER_MODEL_ID", DEFAULT_MODEL_ID)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.id2label = model.config.id2label
        model = model.to(DEVICE).eval()
        if self.mode == "compiled":
            model = torch.compile(model, backend="neuron", dynamic=False, fullgraph=True)
        self.model = model
        # Warm up so that compilation happens at load time, not on the first request.
        self("warmup")

    def __call__(self, data: Any) -> Dict[str, Any]:
        text = data.pop("inputs", data) if isinstance(data, dict) else data
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        device = logits.device.type
        scores = logits.float().softmax(dim=-1)[0].cpu()
        label_id = int(scores.argmax())
        return {
            "label": self.id2label[label_id],
            "score": float(scores[label_id]),
            "mode": self.mode,
            "device": device,
        }
