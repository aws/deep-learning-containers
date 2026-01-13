"""AutoGluon table generation."""

import re

from utils import render_table

REPO_KEYS = ["autogluon-training", "autogluon-inference"]
DISPLAY_NAMES = {
    "autogluon-training": "AutoGluon Training",
    "autogluon-inference": "AutoGluon Inference",
}
COLUMNS = ["Framework", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse AutoGluon tag format: 1.4.0-gpu-py311-cu124-ubuntu22.04"""
    result = {"version": "", "accelerator": "", "python": "", "cuda": ""}

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

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for AutoGluon repositories."""
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
                    f"AutoGluon {parsed['version']}",
                    parsed["python"],
                    parsed["cuda"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
