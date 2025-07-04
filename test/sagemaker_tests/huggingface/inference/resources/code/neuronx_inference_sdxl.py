from optimum.neuron import NeuronStableDiffusionXLPipeline
from transformers import AutoTokenizer

print("running neuronx sdxl test...")


def model_fn(model_dir):
    pipe = NeuronStableDiffusionXLPipeline.from_pretrained(model_dir, device_ids=[0, 1])
    return pipe


def predict_fn(data, pipe):
    prompt = data.pop("inputs", data)

    # run generation with parameters
    image = pipe(prompt).images[0]

    return {"generated_image": image}
