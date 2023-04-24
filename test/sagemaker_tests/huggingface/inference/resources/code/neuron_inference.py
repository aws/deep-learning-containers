import os
from transformers import AutoTokenizer, PretrainedConfig
import torch
import numpy as np
import torch.neuron
from sagemaker_huggingface_inference_toolkit import decoder_encoder

AWS_NEURON_TRACED_WEIGHTS_NAME = "neuron_traced_model.pt"

model_id2label = {"0": "NEGATIVE", "1": "POSITIVE"}

print("running Neuron... test")

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = torch.jit.load(os.path.join(model_dir, AWS_NEURON_TRACED_WEIGHTS_NAME))
    return {"model": model, "tokenizer": tokenizer}


def input_fn(input_data, content_type):
    decoded_input_data = decoder_encoder.decode(input_data, content_type)
    return decoded_input_data


def predict_fn(data, model):
    inputs = model["tokenizer"](
        data["inputs"], return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )
    with torch.no_grad():
        predictions = model["model"](*tuple(inputs.values()))[0]
        outputs = predictions.cpu().numpy()

    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
    return [{"label": model_id2label[str(item.argmax())], "score": item.max().item()} for item in scores]
