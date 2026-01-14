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
"""vLLM table generation."""

import re

from constants import AVAILABLE_IMAGES_TABLE_HEADER
from utils import build_ecr_url, build_public_registry_note, render_table

REPO_KEYS = ["vllm", "vllm-arm64"]
DISPLAY_NAMES = {"vllm": "vLLM", "vllm-arm64": "vLLM (ARM64)"}
COLUMNS = ["Framework", "Python", "CUDA", "Accelerator", "Platform", "Example URL"]


def parse_tag(tag: str) -> dict:
    """Parse vLLM tag format: 0.13.0-gpu-py312-cu129-ubuntu22.04-ec2"""
    result = {"version": "", "accelerator": "", "python": "", "cuda": "-", "platform": ""}

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
        result["cuda"] = match.group(4) or "-"

    if tag.endswith("-ec2"):
        result["platform"] = "EC2, ECS, EKS"
    elif tag.endswith("-sagemaker"):
        result["platform"] = "SageMaker"

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for vLLM repositories."""
    images = yaml_data.get("images", {})
    sections = []

    for repo_key in REPO_KEYS:
        repo_data = images.get(repo_key, {})
        tags = repo_data.get("tags", [])
        if not tags:
            continue

        rows = []
        for tag in tags:
            parsed = parse_tag(tag)
            rows.append(
                [
                    f"vLLM {parsed['version']}",
                    parsed["python"],
                    parsed["cuda"],
                    parsed["accelerator"],
                    parsed["platform"],
                    build_ecr_url(repo_key, tag),
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        section = f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n"
        if repo_data.get("public_registry"):
            section += "\n" + build_public_registry_note(repo_key) + "\n"
        section += render_table(COLUMNS, rows)
        sections.append(section)

    return "\n\n".join(sections)
