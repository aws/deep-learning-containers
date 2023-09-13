import torch
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

print("running neuronx decoder test...")


def model_fn(model_dir):
    model_id = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = NeuronModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,
        batch_size=1,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return {"model": model, "tokenizer": tokenizer}


def predict_fn(data, model):
    inputs = model["tokenizer"](data["inputs"], return_tensors="pt")
    with torch.inference_mode():
        sample_output = model.generate(
            **inputs,
            do_sample=True,
            min_length=128,
            max_length=256,
            temperature=0.7,
        )
        output = [model["tokenizer"].decode(tok) for tok in sample_output]

    return {"generated_text": output[0]}
