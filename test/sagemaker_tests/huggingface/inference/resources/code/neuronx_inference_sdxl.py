from optimum.neuron import NeuronStableDiffusionXLPipeline

print("running neuronx sdxl test...")


def model_fn(model_dir):
    # TODO: Compile during the test, not possible without neuronx-cc writable access.
    # model_id = "echarlaix/tiny-random-stable-diffusion-xl"
    # input_shapes = {"batch_size": 1, "height": 64, "width": 64}
    # compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}
    # pipe = NeuronStableDiffusionXLPipeline.from_pretrained(
    #     model_id=model_id, export=True, **compiler_args, **input_shapes, device_ids=[0, 1]
    # )

    pipe = NeuronStableDiffusionXLPipeline.from_pretrained(model_dir, device_ids=[0, 1])
    return pipe


def predict_fn(data, pipe):
    prompt = data.pop("inputs", data)

    # run generation with parameters
    image = pipe(prompt).images[0]

    return {"generated_image": image}
