# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Neuron (non-HuggingFace) table generation."""

import re

from constants import AVAILABLE_IMAGES_TABLE_HEADER
from utils import build_ecr_url, render_table

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
COLUMNS = ["Framework", "Python", "SDK", "Accelerator", "Platform", "Example URL"]


def parse_tag(tag: str) -> dict:
    """Parse Neuron tag format: 2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04"""
    result = {"version": "", "sdk": "", "python": ""}

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
                    parsed["python"],
                    parsed["sdk"],
                    "Neuron",
                    "SageMaker",
                    build_ecr_url(repo_key, tag),
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(
            f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows)
        )

    return "\n\n".join(sections)
