"""vLLM table generation."""

import re

from utils import render_table

REPO_KEYS = ["vllm"]
DISPLAY_NAMES = {"vllm": "vLLM"}
COLUMNS = ["Version", "Platform", "Python", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse vLLM tag format: 0.13-gpu-py312 or 0.13-gpu-py312-ec2"""
    result = {"version": "", "accelerator": "", "python": "", "platform": ""}

    match = re.match(
        r"^(\d+\.\d+)-"  # version
        r"(cpu|gpu)-"  # accelerator
        r"(py\d+)"  # python
        r"(?:-(ec2|sagemaker|eks|ecs))?$",  # platform (optional)
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["accelerator"] = match.group(2).upper()
        result["python"] = match.group(3)
        result["platform"] = (match.group(4) or "").upper()

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for vLLM repositories."""
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
                    f"vLLM {parsed['version']}",
                    parsed["platform"],
                    parsed["python"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
