"""TensorFlow table generation."""

import re

from constants import TABLE_HEADER
from utils import render_table

REPO_KEYS = [
    "tensorflow-training",
    "tensorflow-inference",
]
DISPLAY_NAMES = {
    "tensorflow-training": "TensorFlow Training",
    "tensorflow-inference": "TensorFlow Inference",
}
COLUMNS = ["Framework", "Platform", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse TensorFlow tag format: 2.19.0-cpu-py312-ubuntu22.04-sagemaker"""
    result = {"version": "", "accelerator": "", "python": "", "cuda": "", "os": "", "platform": ""}

    match = re.match(
        r"^(\d+\.\d+\.\d+)-"  # version
        r"(cpu|gpu)-"  # accelerator
        r"(py\d+)-"  # python
        r"(?:(cu\d+)-)?",  # cuda (optional)
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["accelerator"] = match.group(2).upper()
        result["python"] = match.group(3)
        result["cuda"] = match.group(4) or ""

    if tag.endswith("-ec2"):
        result["platform"] = "EC2"
    elif tag.endswith("-sagemaker"):
        result["platform"] = "SageMaker"
    elif tag.endswith("-eks"):
        result["platform"] = "EKS"
    elif tag.endswith("-ecs"):
        result["platform"] = "ECS"

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for TensorFlow repositories."""
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
