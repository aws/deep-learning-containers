"""Base containers table generation."""

import re

from utils import render_table

REPO_KEYS = ["base"]
DISPLAY_NAMES = {"base": "Base"}
COLUMNS = ["Version", "Platform", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse base container tag format: 13.0.0-gpu-py312-cu130-ubuntu22.04-ec2"""
    result = {"version": "", "accelerator": "", "python": "", "cuda": "", "platform": ""}

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

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for base containers."""
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
                    parsed["version"],
                    parsed["platform"],
                    parsed["python"],
                    parsed["cuda"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
