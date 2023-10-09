from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForSequenceClassification

print("running neuronx encoder test...")


def model_fn(model_dir):
    model_id = "hf-internal-testing/tiny-random-DistilBertModel"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # TODO: Compile during the test, not possible without neuronx-cc writable access.
    # input_shapes = {"batch_size": 1, "sequence_length": 64}
    # compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}
    # model = NeuronModelForSequenceClassification.from_pretrained(
    #     model_id=model_id,
    #     export=True,
    #     **compiler_args,
    #     **input_shapes,
    # )

    model = NeuronModelForSequenceClassification.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def predict_fn(data, model):
    inputs = model["tokenizer"](data["inputs"], return_tensors="pt")
    logits = model["model"](**inputs).logits

    return {"label": model.config.id2label[logits.argmax().item()]}
