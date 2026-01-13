"""SGLang table generation."""

import re

from constants import TABLE_HEADER
from utils import render_table

REPO_KEYS = ["sglang"]
DISPLAY_NAMES = {"sglang": "SGLang"}
COLUMNS = ["Version", "Python", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse SGLang tag format: 0.5-gpu-py312"""
    result = {"version": "", "accelerator": "", "python": ""}

    match = re.match(
        r"^(\d+\.\d+)-"  # version
        r"(cpu|gpu)-"  # accelerator
        r"(py\d+)$",  # python
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["accelerator"] = match.group(2).upper()
        result["python"] = match.group(3)

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for SGLang repositories."""
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
                    f"SGLang {parsed['version']}",
                    parsed["python"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"{TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
