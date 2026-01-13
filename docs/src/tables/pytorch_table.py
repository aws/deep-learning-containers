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
"""PyTorch table generation."""

import re

from constants import TABLE_HEADER
from utils import render_table

REPO_KEYS = [
    "pytorch-training",
    "pytorch-inference",
    "pytorch-training-arm64",
    "pytorch-inference-arm64",
]
DISPLAY_NAMES = {
    "pytorch-training": "PyTorch Training",
    "pytorch-inference": "PyTorch Inference",
    "pytorch-training-arm64": "PyTorch Training (ARM64)",
    "pytorch-inference-arm64": "PyTorch Inference (ARM64)",
}
COLUMNS = ["Framework", "Platform", "Python", "CUDA", "Accelerator", "Tag"]


def parse_tag(tag: str) -> dict:
    """Parse PyTorch tag format: 2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"""
    result = {"version": "", "accelerator": "", "python": "", "cuda": "", "os": "", "platform": ""}

    # Pattern: version-accelerator-python-[cuda]-os-[platform]
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

    # Extract platform from end
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
    """Generate tables for PyTorch repositories."""
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
                    f"PyTorch {parsed['version']}",
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
