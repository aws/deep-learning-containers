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
"""DJL/LMI table generation."""

import re

from constants import AVAILABLE_IMAGES_TABLE_HEADER
from utils import build_ecr_url, render_table

REPO_KEYS = ["djl-inference"]
DISPLAY_NAMES = {"djl-inference": "DJL Inference"}
COLUMNS = ["Framework", "Python", "CUDA", "Engine", "Accelerator", "Platform", "Example URL"]


def parse_tag(tag: str) -> dict:
    """Parse DJL tag formats.

    GPU: 0.36.0-lmi18.0.0-cu128 or 0.33.0-tensorrtllm0.21.0-cu128
    CPU: 0.36.0-cpu-full
    """
    result = {"version": "", "engine": "", "cuda": "-", "accelerator": "", "python": "-"}

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
                    f"DJLServing {parsed['version']}",
                    parsed["python"],
                    parsed["cuda"],
                    parsed["engine"],
                    parsed["accelerator"],
                    "SageMaker",
                    build_ecr_url(repo_key, tag),
                ]
            )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        sections.append(
            f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows)
        )

    return "\n\n".join(sections)
