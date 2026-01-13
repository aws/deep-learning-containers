"""Neuron (non-HuggingFace) table generation."""

import re

from constants import TABLE_HEADER
from utils import render_table

REPO_KEYS = [
    "pytorch-inference-neuronx",
    "pytorch-training-neuronx",
    "tensorflow-inference-neuronx",
]
DISPLAY_NAMES = {
    "pytorch-inference-neuronx": "PyTorch Inference (Neuronx)",
    "pytorch-training-neuronx": "PyTorch Training (Neuronx)",
    "tensorflow-inference-neuronx": "TensorFlow Inference (Neuronx)",
}
COLUMNS = ["Framework", "SDK", "Python", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse Neuron tag format: 2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04"""
    result = {"version": "", "sdk": "", "python": "", "accelerator": "Neuron"}

    match = re.match(
        r"^(\d+\.\d+\.\d+)-"  # framework version
        r"neuronx-"  # neuronx marker
        r"(py\d+)-"  # python
        r"sdk([\d.]+)-",  # sdk version
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["python"] = match.group(2)
        result["sdk"] = match.group(3)

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for Neuron repositories."""
    images = yaml_data.get("images", {})
    sections = []

    for repo_key in REPO_KEYS:
        tags = images.get(repo_key, [])
        if not tags:
            continue

        # Determine framework from repo key
        if "pytorch" in repo_key:
            framework_name = "PyTorch"
        elif "tensorflow" in repo_key:
            framework_name = "TensorFlow"
        else:
            framework_name = "Unknown"

        rows = []
        for tag in tags:
            parsed = parse_tag(tag)
            rows.append(
                [
                    f"{framework_name} {parsed['version']}",
                    parsed["sdk"],
                    parsed["python"],
                    parsed["accelerator"],
                    f"`{tag}`",
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(f"{TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
