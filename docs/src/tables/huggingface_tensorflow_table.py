"""HuggingFace TensorFlow table generation."""

import re

from constants import TABLE_HEADER
from utils import render_table

REPO_KEYS = [
    "huggingface-tensorflow-training",
    "huggingface-tensorflow-inference",
]
DISPLAY_NAMES = {
    "huggingface-tensorflow-training": "HuggingFace TensorFlow Training",
    "huggingface-tensorflow-inference": "HuggingFace TensorFlow Inference",
}
COLUMNS = ["Framework", "Transformers", "Platform", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse HuggingFace TensorFlow tag format.

    Example: 2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04
    """
    result = {
        "version": "",
        "transformers": "",
        "accelerator": "",
        "python": "",
        "cuda": "",
        "platform": "",
    }

    match = re.match(
        r"^(\d+\.\d+\.\d+)-"  # tensorflow version
        r"transformers(\d+\.\d+\.\d+)-"  # transformers version
        r"(cpu|gpu)-"  # accelerator
        r"(py\d+)-"  # python
        r"(?:(cu\d+)-)?",  # cuda (optional)
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["transformers"] = match.group(2)
        result["accelerator"] = match.group(3).upper()
        result["python"] = match.group(4)
        result["cuda"] = match.group(5) or ""

    if tag.endswith("-sagemaker"):
        result["platform"] = "SageMaker"
    elif tag.endswith("-ec2"):
        result["platform"] = "EC2"

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for HuggingFace TensorFlow repositories."""
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
                    f"TensorFlow {parsed['version']}",
                    parsed["transformers"],
                    parsed["platform"],
                    parsed["python"],
                    parsed["cuda"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"{TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
