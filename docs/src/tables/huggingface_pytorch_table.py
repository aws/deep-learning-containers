"""HuggingFace PyTorch table generation."""

import re

from utils import render_table

REPO_KEYS = [
    "huggingface-pytorch-training",
    "huggingface-pytorch-inference",
    "huggingface-pytorch-inference-neuronx",
    "huggingface-pytorch-training-neuronx",
    "huggingface-pytorch-trcomp-training",
]
DISPLAY_NAMES = {
    "huggingface-pytorch-training": "HuggingFace PyTorch Training",
    "huggingface-pytorch-inference": "HuggingFace PyTorch Inference",
    "huggingface-pytorch-inference-neuronx": "HuggingFace PyTorch Inference (Neuronx)",
    "huggingface-pytorch-training-neuronx": "HuggingFace PyTorch Training (Neuronx)",
    "huggingface-pytorch-trcomp-training": "HuggingFace PyTorch Training Compiler",
}
COLUMNS = ["Framework", "Transformers", "Platform", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse HuggingFace PyTorch tag formats.

    Standard: 2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04
    Neuronx: 2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
    """
    result = {
        "version": "",
        "transformers": "",
        "accelerator": "",
        "python": "",
        "cuda": "",
        "sdk": "",
        "platform": "",
    }

    # Standard format
    match = re.match(
        r"^(\d+\.\d+\.\d+)-"  # pytorch version
        r"transformers(\d+\.\d+\.\d+)-"  # transformers version
        r"(cpu|gpu|neuronx)-"  # accelerator
        r"(py\d+)-"  # python
        r"(?:(cu\d+|sdk[\d.]+)-)?",  # cuda or sdk (optional)
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["transformers"] = match.group(2)
        accel = match.group(3)
        result["accelerator"] = "Neuron" if accel == "neuronx" else accel.upper()
        result["python"] = match.group(4)
        extra = match.group(5) or ""
        if extra.startswith("cu"):
            result["cuda"] = extra
        elif extra.startswith("sdk"):
            result["sdk"] = extra

    if tag.endswith("-sagemaker"):
        result["platform"] = "SageMaker"
    elif tag.endswith("-ec2"):
        result["platform"] = "EC2"

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for HuggingFace PyTorch repositories."""
    images = yaml_data.get("images", {})
    sections = []

    for repo_key in REPO_KEYS:
        tags = images.get(repo_key, [])
        if not tags:
            continue

        rows = []
        for tag in tags:
            parsed = parse_tag(tag)
            rows.append(
                [
                    f"PyTorch {parsed['version']}",
                    parsed["transformers"],
                    parsed["platform"],
                    parsed["python"],
                    parsed["cuda"] or parsed["sdk"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
