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
"""Common utilities for documentation generation."""

import yaml


def load_yaml(path: str) -> dict:
    """Load and return YAML data."""
    with open(path) as f:
        return yaml.safe_load(f)


def render_table(headers: list[str], rows: list[list[str]], prefix: str = "") -> str:
    """Convert headers and rows to markdown table string."""
    if not rows:
        return ""
    header_line = f"{prefix}" + "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return f"\n{prefix}".join([header_line, separator] + row_lines)


def read_template(path: str) -> str:
    """Read template file content."""
    with open(path) as f:
        return f.read()


def write_output(path: str, content: str) -> None:
    """Write generated markdown to file."""
    with open(path, "w") as f:
        f.write(content)
