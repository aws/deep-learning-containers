"""Triton table generation."""

import re

from utils import render_table

REPO_KEYS = ["sagemaker-tritonserver"]
DISPLAY_NAMES = {"sagemaker-tritonserver": "NVIDIA Triton Server for SageMaker"}
COLUMNS = ["Version", "Python", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse Triton tag format: 25.09-py3"""
    result = {"version": "", "python": ""}

    match = re.match(r"^([\d.]+)-(py\d+)$", tag)
    if match:
        result["version"] = match.group(1)
        result["python"] = match.group(2)

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for Triton repositories."""
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
                    f"Triton {parsed['version']}",
                    parsed["python"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"## {display_name}\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
