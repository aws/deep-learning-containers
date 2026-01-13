"""DJL/LMI table generation."""

import re

from utils import render_table

REPO_KEYS = ["djl-inference"]
DISPLAY_NAMES = {"djl-inference": "DJL Inference"}
COLUMNS = ["Version", "Engine", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse DJL tag formats.

    GPU: 0.36.0-lmi18.0.0-cu128 or 0.33.0-tensorrtllm0.21.0-cu128
    CPU: 0.36.0-cpu-full
    """
    result = {"version": "", "engine": "", "cuda": "", "accelerator": ""}

    # GPU with LMI
    match = re.match(r"^(\d+\.\d+\.\d+)-lmi([\d.]+)-(cu\d+)$", tag)
    if match:
        result["version"] = match.group(1)
        result["engine"] = f"LMI {match.group(2)}"
        result["cuda"] = match.group(3)
        result["accelerator"] = "GPU"
        return result

    # GPU with TensorRT-LLM
    match = re.match(r"^(\d+\.\d+\.\d+)-tensorrtllm([\d.]+)-(cu\d+)$", tag)
    if match:
        result["version"] = match.group(1)
        result["engine"] = f"TensorRT-LLM {match.group(2)}"
        result["cuda"] = match.group(3)
        result["accelerator"] = "GPU"
        return result

    # CPU
    match = re.match(r"^(\d+\.\d+\.\d+)-cpu-full$", tag)
    if match:
        result["version"] = match.group(1)
        result["engine"] = "CPU Full"
        result["accelerator"] = "CPU"
        return result

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for DJL repositories."""
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
                    parsed["engine"],
                    parsed["cuda"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
