"""Documentation global variables for mkdocs-macros-plugin."""

from constants import GLOBAL_CONFIG
from image_config import (
    build_ecr_uri,
    get_latest_image_uri,
    load_repository_images,
    sort_by_version,
)


def _get_latest_ray_uri(platform: str, accelerator: str) -> str:
    """Get latest Ray image URI filtered by platform and accelerator."""
    images = load_repository_images("ray")
    matching = [
        img
        for img in images
        if img.get("platform") == platform and img.get("accelerator") == accelerator
    ]
    if not matching:
        raise ValueError(f"Image not found for ray with platform={platform}, accelerator={accelerator}")
    latest = sort_by_version(matching)[0]
    account = latest.get("example_ecr_account", GLOBAL_CONFIG["example_ecr_account"])
    return build_ecr_uri(account, "ray", latest.display_tag, "us-west-2")


def define_env(env):
    """Define variables for mkdocs-macros-plugin."""
    # Expose all global config variables to mkdocs macros
    for key, value in GLOBAL_CONFIG.items():
        if isinstance(value, str):
            env.variables[key] = value

    # Image helpers
    env.variables["images"] = {
        "latest_pytorch_training_ec2": get_latest_image_uri("pytorch-training", "ec2"),
        "latest_vllm_sagemaker": get_latest_image_uri("vllm", "sagemaker"),
        "latest_sglang_sagemaker": get_latest_image_uri("sglang", "sagemaker"),
        "latest_ray_ec2_gpu": _get_latest_ray_uri("ec2", "gpu"),
        "latest_ray_ec2_cpu": _get_latest_ray_uri("ec2", "cpu"),
        "latest_ray_sagemaker_gpu": _get_latest_ray_uri("sagemaker", "gpu"),
        "latest_ray_sagemaker_cpu": _get_latest_ray_uri("sagemaker", "cpu"),
    }
