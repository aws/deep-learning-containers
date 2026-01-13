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
"""Triton table generation."""

import re

from constants import TABLE_HEADER
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
        sections.append(f"{TABLE_HEADER} {display_name}\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
