import base64
from io import BytesIO
from einops import rearrange
import json
from pathlib import Path
from PIL import Image
from pytorch_lightning import seed_everything
import numpy as np
from sagemaker_inference.errors import BaseInferenceToolkitError
import sgm
from sgm.inference.api import (
    ModelArchitecture,
    SamplingParams,
    SamplingPipeline,
    Sampler,
)
from sgm.inference.helpers import (
    get_input_image_tensor,
    embed_watermark,
)
import os


def model_fn(model_dir, context=None):
    sgm_path = os.path.dirname(sgm.__file__)
    config_path = os.path.join(sgm_path, "configs/inference")
    if not os.path.exists(config_path):
        config_path = os.path.join(sgm_path, "../configs/inference")
    # Change this to SD_2_1_768 if you downloaded the 768 model
    pipeline = SamplingPipeline(
        ModelArchitecture.SD_2_1, model_path=model_dir, config_path=config_path
    )

    return pipeline


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        model_input = json.loads(request_body)
        if not "text_prompts" in model_input:
            raise BaseInferenceToolkitError(400, "Invalid Request", "text_prompts missing")
        return model_input
    else:
        raise BaseInferenceToolkitError(
            400, "Invalid Request", "Content-type must be application/json"
        )


def predict_fn(data, model, context=None):
    # Only a single positive and optionally a single negative prompt are supported by this example.
    prompts = []
    negative_prompts = []
    if "text_prompts" in data:
        for text_prompt in data["text_prompts"]:
            if "text" not in text_prompt:
                raise BaseInferenceToolkitError(
                    400, "Invalid Request", "text missing from text_prompt"
                )
            if "weight" not in text_prompt:
                text_prompt["weight"] = 1.0
            if text_prompt["weight"] < 0:
                negative_prompts.append(text_prompt["text"])
            else:
                prompts.append(text_prompt["text"])

    if len(prompts) != 1:
        raise BaseInferenceToolkitError(
            400,
            "Invalid Request",
            "One prompt with positive or default weight must be supplied",
        )
    if len(negative_prompts) > 1:
        raise BaseInferenceToolkitError(
            400, "Invalid Request", "Only one negative weighted prompt can be supplied"
        )

    seed = 0
    height = 1024
    width = 1024
    sampler_name = "DPMPP2MSampler"
    cfg_scale = 7.0
    steps = 40
    init_image = None
    image_strength = 0.35

    if "height" in data:
        height = data["height"]
    if "width" in data:
        width = data["width"]
    if "sampler" in data:
        sampler_name = data["sampler"]
    if "cfg_scale" in data:
        cfg_scale = data["cfg_scale"]
    if "steps" in data:
        steps = data["steps"]
    if "seed" in data:
        seed = data["seed"]
        seed_everything(seed)
    if "init_image" in data:
        if "image_strength" in data:
            image_strength = data["image_strength"]
        try:
            init_image_bytes = BytesIO(base64.b64decode(data["init_image"]))
            init_image_bytes.seek(0)
            if init_image_bytes is not None:
                init_image = get_input_image_tensor(Image.open(init_image_bytes))
        except Exception as e:
            raise BaseInferenceToolkitError(400, "Invalid Request", "Unable to decode init_image")

    try:
        if init_image is not None:
            img_height, img_width = init_image.shape[2], init_image.shape[3]
            output = model.image_to_image(
                params=SamplingParams(
                    width=img_width,
                    height=img_height,
                    steps=steps,
                    sampler=Sampler(sampler_name),
                    scale=cfg_scale,
                    img2img_strength=image_strength,
                ),
                image=init_image,
                prompt=prompts[0],
                negative_prompt=negative_prompts[0] if len(negative_prompts) > 0 else "",
            )
        else:
            output = model.text_to_image(
                params=SamplingParams(
                    width=width,
                    height=height,
                    steps=steps,
                    sampler=Sampler(sampler_name),
                    scale=cfg_scale,
                ),
                prompt=prompts[0],
                negative_prompt=negative_prompts[0] if len(negative_prompts) > 0 else "",
            )

        if isinstance(output, (tuple, list)):
            samples, samples_z = output
        else:
            samples = output
            samples_z = None

        samples = embed_watermark(samples)
        images = []
        for sample in samples:
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
            image_bytes = BytesIO()
            Image.fromarray(sample.astype(np.uint8)).save(image_bytes, format="PNG")
            image_bytes.seek(0)
            images.append(image_bytes.read())

        return images

    except ValueError as e:
        raise BaseInferenceToolkitError(400, "Invalid Request", str(e))


def output_fn(prediction, accept):
    # This only returns a single image since that's all the example code supports
    if accept != "image/png":
        raise BaseInferenceToolkitError(400, "Invalid Request", "Accept header must be image/png")
    return prediction[0], accept
