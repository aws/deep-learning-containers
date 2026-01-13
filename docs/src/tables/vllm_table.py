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

from constants import TABLE_HEADER
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
        sections.append(f"{TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
