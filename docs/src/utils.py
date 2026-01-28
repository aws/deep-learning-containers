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

import logging
import os
import subprocess
from pathlib import Path

import yaml
from constants import GLOBAL_CONFIG, TABLES_DIR
from packaging.version import Version

LOGGER = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Load YAML file and return parsed content."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_table_config(table_path: str) -> dict:
    """Load table column configuration for a repository.

    Raises:
        FileNotFoundError: If table config file does not exist.
    """
    path = TABLES_DIR / f"{table_path}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Table config not found: {path}")
    return load_yaml(path)


def load_jinja2(path: str | Path) -> str:
    """Load and return Jinja2 template file content."""
    with open(path) as f:
        return f.read()


def write_output(path: str | Path, content: str) -> None:
    """Write content to output file."""
    with open(path, "w") as f:
        f.write(content)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render headers and rows as a markdown table string."""
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator] + row_lines)


def build_ecr_uri(account: str, repository: str, tag: str, region: str = "<region>") -> str:
    """Build ECR URI string."""
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


def build_public_ecr_uri(repository: str, tag: str) -> str:
    """Build public ECR URI string."""
    return f"public.ecr.aws/deep-learning-containers/{repository}:{tag}"


def parse_version(version_str: str | None) -> Version:
    """Parse version string safely, returning Version("0") on failure."""
    try:
        return Version(version_str or "0")
    except Exception:
        return Version("0")


def clone_git_repository(git_repository: str, target_dir: str | Path) -> None:
    """Clone a git repository to target directory if it doesn't exist."""
    if os.path.exists(target_dir):
        return
    subprocess.run(["git", "clone", "--depth", "1", git_repository, target_dir], check=True)

    def version_key(img: dict) -> Version:
        try:
            return Version(img.get("version", "0"))
        except Exception:
            return Version("0")


def get_framework_order() -> list[str]:
    """Derive framework order from table_order, collapsing framework groups."""
    table_order = GLOBAL_CONFIG.get("table_order", [])
    framework_groups = GLOBAL_CONFIG.get("framework_groups", {})

    # Build reverse mapping: repo -> framework group
    repo_to_group = {}
    for group, repos in framework_groups.items():
        for repo in repos:
            repo_to_group[repo] = group

    seen = set()
    result = []
    for repo in table_order:
        key = repo_to_group.get(repo, repo)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result
