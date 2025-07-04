from optimum.neuron import NeuronModelForSequenceClassification
from transformers import AutoTokenizer

print("running neuronx encoder test...")


def model_fn(model_dir):
    model = NeuronModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def predict_fn(data, model):
    inputs = model["tokenizer"](data["inputs"], return_tensors="pt")
    logits = model["model"](**inputs).logits

    return {"label": model["model"].config.id2label[logits.argmax().item()]}
