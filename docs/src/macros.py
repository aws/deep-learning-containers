"""Documentation global variables for mkdocs-macros-plugin."""

from constants import GLOBAL_CONFIG
from image_config import get_latest_image_uri


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
        "latest_ray_ec2_gpu": get_latest_image_uri("ray", "ec2", "gpu"),
        "latest_ray_ec2_cpu": get_latest_image_uri("ray", "ec2", "cpu"),
        "latest_ray_sagemaker_gpu": get_latest_image_uri("ray", "sagemaker", "gpu"),
        "latest_ray_sagemaker_cpu": get_latest_image_uri("ray", "sagemaker", "cpu"),
    }
